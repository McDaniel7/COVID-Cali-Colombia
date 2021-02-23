import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
import random
import arrow

import torch
import torch.optim as optim

# import geopandas as gpd
# from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon
# from shapely.ops import cascaded_union


class StdDiffusionKernel(object):
    """
    Kernel function including the diffusion-type model proposed by Musmeci and
    Vere-Jones (1992).
    """
    def __init__(self, C=1., beta=1., sigma_x=1., sigma_y=1.):
        self.C       = C
        self.beta    = beta
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def nu(self, t, s, his_t, his_s):
        delta_s = s - his_s
        delta_t = t - his_t
        delta_x = delta_s[:, 0]
        delta_y = delta_s[:, 1]
        return torch.exp(- self.beta * delta_t) * \
            (self.C / (2 * np.pi * self.sigma_x * self.sigma_y * delta_t)) * \
            torch.exp((- 1. / (2 * delta_t)) * \
                ((torch.pow(delta_x, 2) / torch.pow(self.sigma_x, 2)) + \
                (torch.pow(delta_y, 2) / torch.pow(self.sigma_y, 2))))


class TorchETAS(torch.nn.Module):
    """
    PyTorch Module for ETAS
    """

    def __init__(self, obs, region, keep_latest_t = None, kernel = "StdDiffusion", data_dim = 3):
        """
        Denote the number of observations as N, data dim = 3(time, x, y)

        Args:
        - obs:    event observations    [ N, data_dim ]
        - region: a shapely Polygon
        """
        torch.nn.Module.__init__(self)
        # data
        self.obs    = torch.Tensor(np.array(obs)) # [ K, N ]
        self.n = torch.max(self.obs[:, 0]) # max time stamp
        self.region = region
        # configurations
        self.N = obs.shape[0]
        self.kpt = keep_latest_t
        # parameters
        self.Lambda0   = torch.nn.Parameter(torch.Tensor(1).uniform_(0, .01)[0]) # [ 1 ]
        self.C = torch.nn.Parameter(torch.Tensor(1).uniform_(.01, 1)[0])
        self.Beta  = torch.nn.Parameter(torch.Tensor(1).uniform_(1, 3)[0])           # [ 1 ]
        self.Sigmax = torch.nn.Parameter(torch.Tensor(1).uniform_(.01, .1)[0]) # [ 1 ]
        self.Sigmay = torch.nn.Parameter(torch.Tensor(1).uniform_(.01, .1)[0]) # [ 1 ]
        if kernel == "StdDiffusion": self.kernel = StdDiffusionKernel(C=self.C, beta=self.Beta,
                                                                      sigma_x=self.Sigmax, 
                                                                      sigma_y=self.Sigmay)
    
    def _Lambda0(self, _t, _s):
        """
        Background rate at time `t`, location `s`
        """
        return self.Lambda0
    
    def _exp_kernel(self):
        """
        Exponential kernel
        """
        return self.C * torch.exp(-self.Beta * torch.arange(1, self.n, 1))

    def _lambda(self, _t, _s):
        """
        Conditional intensity function at time `t`, location `s`

        Args:
        - _t:  index of time, e.g., 0, 1, ..., N (integer)
        - _s:  location of the case, [x, y]
        Return:
        - lam: a lambda value at time `t`, location `s`
        """
        if _t > 1:
            # self-exciting effect
            t = torch.ones(1) * _t
            s = torch.ones(2) * _s
            if self.kpt is not None:
                his_idx = (self.obs[:, 0] >= t - self.kpt) & (self.obs[:, 0] < t)
                his_t = self.obs[his_idx, 0]
                his_s = self.obs[his_idx, 1:]
            else:
                his_idx = self.obs[:, 0] < t
                his_t = self.obs[his_idx, 0]
                his_s = self.obs[his_idx, 1:]
            ker = self.kernel.nu(t, s, his_t, his_s) # [ N(_t) ]
            lam    = self.Lambda0 + torch.sum(ker)    # [ 1 ]
        else:
            lam    = self.Lambda0
        return lam
    
    def _lambda_t(self, _t, ps):
        """
        Conditional intensity function of points `ps` at time `t`, calculated in vector form

        Args:
        - _t:  index of time, e.g., 0, 1, ..., N (integer)
        - ps:  location of the points
        Return:
        - lam: a vector lambda value of `ps` at time `t`
        """
        n_ps = ps.shape[0]
        t = torch.ones(1) * _t
        
        if _t > 1:
            if self.kpt is not None:
                his_idx = (self.obs[:, 0] >= t - self.kpt) & (self.obs[:, 0] < t)
                his_t = self.obs[his_idx, 0]
                his_s = self.obs[his_idx, 1:]
            else:    
                his_idx = self.obs[:, 0] < t
                his_t = self.obs[his_idx, 0]
                his_s = self.obs[his_idx, 1:]
            
            delta_t = t - his_t
            M = torch.div(self.C * torch.exp(-self.Beta * delta_t), self.Sigmax * self.Sigmay * 2 * np.pi * delta_t)
            
            delta_s = torch.Tensor(ps).unsqueeze(-1).expand(n_ps, 2, his_t.shape[0]) - his_s.T.expand(n_ps, 2, his_t.shape[0])
            st1 = torch.div(torch.square(delta_s),
                            torch.square(torch.stack([self.Sigmax, self.Sigmay], dim=0)).expand(his_t.shape[0], 2).T * -2 * delta_t)
            st2 = torch.exp(torch.sum(st1, dim=1))
            
            kers = torch.mm(st2, M.unsqueeze(-1))
            return kers
        else:
            return torch.ones(ps.shape[0]) * self.Lambda0
    
    def _spa_tem_int(self):
        """
        Spatio-Temporal integration in log-likelihood, calculated in vector form

        Return:
        - lam: Spatio-Temporal Integral estimation
        """
        C = self.Lambda0 * self.region.area * self.n
        exp_ker = self._exp_kernel()
        
        def F_t(_t):
            t = torch.ones(1) * _t
            if _t > 1:
                if self.kpt is not None:
                    his_idx = (self.obs[:, 0] >= t - self.kpt) & (self.obs[:, 0] < t)
                    his_t = self.obs[his_idx, 0]
                else:    
                    his_idx = self.obs[:, 0] < t
                    his_t = self.obs[his_idx, 0]
                    
                delta_t = t - his_t
                values, counts = torch.unique(delta_t, return_counts = True)
            
                return torch.sum(counts * exp_ker[:len(values)])
            
            else:
                return torch.zeros(1)[0]
        
        F = [F_t(_t) for _t in np.arange(1, self.n + 1, 1)]
        F = torch.sum(torch.stack(F, dim = 0))
#         F = 0
#         for _t in np.arange(1, self.n + 1, 1):
#             F = F + F_t(_t)
        
        lam = C + F
        
        return lam
    
    def _log_likelihood(self):
        """
        Log likelihood function at time `T` = n

        Return:
        - loglik: a vector of log likelihood value at location k = 0, 1, ..., K [ K ]
        - lams:   
        """
#         # lambda values from 0 to N
#         lams0    = [ self._mu(t) for t in np.arange(self.N) ]     # ( N, [ K ] )
#         lams1    = [ self._lambda(t) for t in np.arange(self.N) ] # ( N, [ K ] )
#         lams0    = torch.stack(lams0, dim=1)                      # [ K, N ]
#         lams1    = torch.stack(lams1, dim=1)                      # [ K, N ]
#         Nloglams = self.obs * torch.log(lams0 + lams1 + 1e-5)     # [ K, N ]
#         # log-likelihood function
#         loglik   = (Nloglams - lams0 - lams1).sum()

        lam_t_1 = [torch.sum(torch.log(self._lambda_t(_t, self.obs[self.obs[:, 0] == _t, 1:]))) for _t in np.arange(1, self.n + 1, 1)]
        lams1 = torch.sum(torch.stack(lam_t_1, dim = 0))
    
#         lams1 = 0
#         for _t in np.arange(1, self.n + 1, 1):
#             lams1 = lams1 + torch.sum(torch.log(self._lambda_t(_t, self.obs[self.obs[:, 0] == _t, 1:])))
        
        lams2 = self._spa_tem_int()
    
#         lams1 = 0
#         A = torch.ones(1)[0] * self.region.area
#         MC_points = self._MC_sample()
#         for _t in np.arange(1, self.n + 1, 1):
#             print(_t)
#             lams1 += torch.sum(self._lambda_t(_t, MC_points) * A)
#         lams1 = lams1 / self.nMC
        
        loglik = lams1 - lams2

        return loglik, lams1, lams2

    def forward(self):
        """
        customized forward function
        """
        # calculate data log-likelihood
        return self._log_likelihood()
