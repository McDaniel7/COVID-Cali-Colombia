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

    def __init__(self, obs, region, n_MC = 10000, kernel = "StdDiffusion", data_dim = 3):
        """
        Denote the number of observations as N, data dim = 3(time, x, y)

        Args:
        - obs:    event observations    [ N, data_dim ]
        - region: a shapely Polygon
        """
        torch.nn.Module.__init__(self)
        # data
        self.obs    = torch.Tensor(np.array(obs)) # [ K, N ]
        self.n = torch.max(self.obs[:, 0])
        self.region = region
        # configurations
        self.N = obs.shape[0]
        self.nMC = n_MC
        # parameters
        self.Lambda0   = torch.nn.Parameter(torch.Tensor(1).uniform_(0, .01)[0]) # [ 1 ]
        self.C = torch.nn.Parameter(torch.Tensor(1).uniform_(0.5, 2)[0])
        self.Beta  = torch.nn.Parameter(torch.Tensor(1).uniform_(1, 3)[0])           # [ 1 ]
        self.Sigmax = torch.nn.Parameter(torch.Tensor(1).uniform_(0.5, 2)[0]) # [ 1 ]
        self.Sigmay = torch.nn.Parameter(torch.Tensor(1).uniform_(0.5, 2)[0]) # [ 1 ]
        if kernel == "StdDiffusion": self.kernel = StdDiffusionKernel(C=self.C, beta=self.Beta,
                                                                      sigma_x=self.Sigmax, 
                                                                      sigma_y=self.Sigmay)  # useless
    
    def _Lambda0(self, _t, _s):
        """
        Background rate at time `t`, location `s`
        """
        return self.Lambda0
    
    def _MC_sample(self):
        """
        Monte Carlo Sampling to estimate the integral term in log-likelihood

        Return:
        - lam: a lambda value at time `t`, location `s`
        """
        m = 0
        smp = []
        x_left = self.region.bounds[0]
        x_right = self.region.bounds[2]
        y_lower = self.region.bounds[1]
        y_upper = self.region.bounds[3]
        while m < self.nMC:
            xs = torch.Tensor(int(self.nMC / 2)).uniform_(x_left, x_right)
            ys = torch.Tensor(int(self.nMC / 2)).uniform_(y_lower, y_upper)
            points = MultiPoint(list(zip(xs, ys)))
            tupVerts = self.region.exterior
            p = Path(tupVerts)
            if_cnt = p.contains_points(points)
            smp.extend(list(zip(xs[if_cnt], ys[if_cnt])))
            m += sum(if_cnt)
        
        return np.array(smp[:self.nMC])

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
            his_idx = self.obs[:, 0] < t
            his_t = self.obs[his_idx, 0]
            s = torch.ones(2) * _s
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
            his_idx = self.obs[:, 0] < t
            his_t = self.obs[his_idx, 0]
            his_s = self.obs[his_idx, 1:]
            
            delta_t = t - his_t
            M = self.C * torch.exp(-self.Beta * delta_t) / (self.Sigmax * self.Sigmay * 2 * np.pi * delta_t)
            
            delta_s = torch.Tensor(ps).unsqueeze(-1).expand(n_ps, 2, his_t.shape[0]) - his_s.T.expand(n_ps, 2, his_t.shape[0])
            st1 = torch.div(torch.square(delta_s), torch.square(torch.stack([self.Sigmax, self.Sigmay], dim=0)).expand(his_t.shape[0], 2).T)
            st2 = torch.exp(torch.div(torch.sum(st1, dim=1), -2 * delta_t))
            
            kers = torch.mm(st2, M.unsqueeze(-1))
            return kers
        else:
            return torch.ones(ps.shape[0]) * self.Lambda0
        
    def Int_Est(self):
        """
        Integral term estimation in log-likelihood using Monte Carlo Sampling

        Return:
        - lam: Integral Estimation
        """
        A = torch.ones(1)[0] * self.region.area
        MC_points = self._MC_sample()
        
        lam_t = [torch.sum(self._lambda_t(_t, MC_points) * A) for _t in np.arange(1, self.n + 1, 1)]
        lam_t = torch.stack(lam_t, dim = 0)
        lam = torch.sum(lam_t) / self.nMC
        
        return lams2
        
    def _log_likelihood(self):
        """
        Log likelihood function at time `T` = n

        Return:
        - loglik: a vector of log likelihood value at location k = 0, 1, ..., K [ K ]
        - lams1: The sum term of history
        - lams2: The integral term over spatio-temporal space
        """

        lam_t_1 = [torch.sum(torch.log10(self._lambda_t(_t, self.obs[self.obs[:, 0] == _t, 1:]))) for _t in np.arange(1, self.n + 1, 1)]
        lam_t_1 = torch.stack(lam_t_1, dim = 0)
        lams1 = torch.sum(lam_t_1)
        lams2 = self.Int_Est()
        
        loglik = lams1 + lams2

        return loglik, lams1, lams2

    def forward(self):
        """
        customized forward function
        """
        # calculate data log-likelihood
        return self._log_likelihood()
    
    
def train(model, niter=1000, lr=1e-1, log_interval=50):
    """training procedure for one epoch"""
    # coordinates of K locations
    ##coords    = locs[:, :2]
    # define model clipper to enforce inequality constraints
    clipper1  = NonNegativeClipper()
    ##clipper2  = ProximityClipper(coords, k=k)
    # NOTE: gradient for loss is expected to be None, 
    #       since it is not leaf node. (it's root node)
    logliks = []
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    for _iter in range(niter):
        try:
            model.train()
            optimizer.zero_grad()           # init optimizer (set gradient to be zero)
            loglik, _, _ = model()
            # objective function
            loss         = - loglik
            loss.backward()                 # gradient descent
            optimizer.step()                # update optimizer
            model.apply(clipper1)
            ##model.apply(clipper2)
            # log training output
            logliks.append(loglik.item())
            if _iter % log_interval == 0 and _iter != 0:
                print("[%s] Train batch: %d\tLoglik: %.3e" % (arrow.now(), 
                    _iter / log_interval, 
                    sum(logliks) / log_interval))
                logliks = []
        except KeyboardInterrupt:
            break



class NonNegativeClipper(object):
    """
    References:
    https://discuss.pytorch.org/t/restrict-range-of-variable-during-gradient-descent/1933
    https://discuss.pytorch.org/t/set-constraints-on-parameters-or-layers/23620/3
    """

    def __init__(self):
        pass

    def __call__(self, module):
        """enforce non-negative constraints"""
        # TorchHawkes
        if hasattr(module, 'Lambda0'):
            Lambda0 = module.Lambda0.data
            module.Lambda0.data = torch.clamp(Lambda0, min=0.)
        if hasattr(module, 'Alpha'):
            Alpha = module.Alpha.data
            module.Alpha.data = torch.clamp(Alpha, min=0.)
        if hasattr(module, 'Beta'):
            Beta  = module.Beta.data
            module.Beta.data  = torch.clamp(Beta, min=0.)
        if hasattr(module, 'C'):
            C  = module.C.data
            module.C.data = torch.clamp(C, min=0.)
        if hasattr(module, 'Sigmax'):
            Sigmax  = module.Sigmax.data
            module.Sigmax.data = torch.clamp(Sigmax, min=0.)
        if hasattr(module, 'Sigmay'):
            Sigmay  = module.Sigmay.data
            module.Sigmay.data = torch.clamp(Sigmay, min=0.)
