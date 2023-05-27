import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
import arrow
import random
import pickle
import cProfile
import sys
from datetime import datetime

from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde
from scipy.stats import chi2
from scipy.spatial.distance import pdist, squareform
from statsmodels.graphics.tsaplots import plot_pacf

from matplotlib.path import Path
import branca
import folium
from tqdm import tqdm

import geopandas as gpd
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon
from descartes.patch import PolygonPatch
from shapely.ops import cascaded_union

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullFormatter
import matplotlib

########### Point process ############

class TorchKernelPP(torch.nn.Module):
    """
    PyTorch Module for Spatio-temporal Point Process
    """

    def __init__(self, obs, region_attr, kernel, keep_latest_t = np.inf, discretize = True, data_dim = 3, 
                 edo_learn = True, exo_learn = False, bg_learn = True):
        """
        Denote the number of observations as N, data dim = 3(time, x, y)

        Args:
        - obs:    event observations    [ T, max(daily_number), 2 ]
        - region_area: area of a shapely Polygon
        """
        torch.nn.Module.__init__(self)
        # data
        self.obs    = obs # Original data matrix, [N, data_dim]
        self.ts = torch.unique(self.obs[:, 0]).int().cpu().numpy()
        # configurations
        self.usegpu = False
        self.region_area = region_attr[0] # Tensor float
        self.xmin = region_attr[1] # Tensor float
        self.xmax = region_attr[2] # Tensor float
        self.ymin = region_attr[3] # Tensor float
        self.ymax = region_attr[4] # Tensor float
        self.n = np.int(torch.max(obs[:, 0]).item())
        self.N = obs.shape[0]
        self.kpt = keep_latest_t
        self.discretize = discretize
        self.data_dim = data_dim
        self.bg_learn = bg_learn
        self.exo_learn = exo_learn
        self.edo_learn = edo_learn
        # parameters
        bg = torch.mean(torch.unique(self.obs[:, 0], return_counts = True)[1].float()).item()
        self.Lambda0 = torch.nn.Parameter(torch.FloatTensor([bg*5])[0]) # [ 1 ]
        #self.Lambda0   = torch.nn.Parameter(torch.Tensor(1).uniform_(2000, 3000)[0]) # [ 1 ]
        self.kernel = kernel
        # pre-compute
        self.data_transformation()

    def data_transformation(self):
        """
        Transform a N*2 data matrix into model-desired formation
        """
        t_value, t_counts = torch.unique(self.obs[:, 0], return_counts = True)
        t_value = t_value.int()
        trsfm_obs = torch.zeros(torch.max(t_value), torch.max(t_counts), self.data_dim, device=device)
        mask = torch.zeros(torch.max(t_value), torch.max(t_counts), device=device)
        for i in range(len(t_value)):
            t = t_value[i]
            idx = self.obs[:, 0] == t
            n = sum(idx)
            trsfm_obs[t-1, :n, :] = self.obs[idx, :]
            mask[t-1, :n] = 1
        mask = mask > 0

        self.trsfm_obs = trsfm_obs
        self.data_mask = mask

    def _Lambda0(self, _t, _s):
        """
        Background rate at time `t`, location `s`
        """
        return self.Lambda0
    
    def _Time_Decay_Kernel(self):
        """
        Time decay kernel
        """
        self.time_decay_ker = self.kernel._Time_decay_kernel()
    
    def _lambda_t(self, t, s, his_t, his_s):
        """
        Conditional intensity function of current events `s` at time `t`, calculated in vector form

        Args:
        - t:  index of time, e.g., 0, 1, ..., N (integer)
        - s:  location of current events [N, 2]
        - idx: index of current events. [N]
        - his_t: history time vector. [M]
        - his_s: history location matrix. [M, 2]
        - his_idx: index of inspected history events. [M]
        Return:
        - lam: a vector lambda value of `s` at time `t`
        """
        N = s.shape[0]
        M = his_s.shape[0]
        if self.usegpu: device = "cuda:0"
        else: device = "cpu"

        if M > 0:
            kers = self.kernel.nu(t, s, his_t, his_s)

            return kers + self.Lambda0
        else:
            return torch.ones(N, device=device) * self.Lambda0
    
    def Overall_int(self, t):
        """
        Spatio-Temporal integration of conditional intensity function at time T

        Return:
        - lam: Spatio-Temporal Integral estimation
        """
        C = self.Lambda0 * self.region_area
        
        if t > 1:
            his_t = self.trsfm_obs[(max(t - self.kpt - 1, 0)):(t-1), :, 0][self.data_mask[(max(t - self.kpt - 1, 0)):(t-1), :]].clone()
            delta_t = t - his_t
            values, counts = torch.unique(delta_t, return_counts = True)

            F = torch.sum(counts * self.time_decay_ker[1:(len(values)+1)])
        else:
            F = 0
        
        lam = C + F
        
        return lam

    def discrete_spa_tem_int(self):

        bg = self.Lambda0 * self.region_area * np.max(self.ts)
        edo = self.kernel.discrete_spa_tem_int()

        return bg + edo
    
    def _log_likelihood(self):
        """
        Log likelihood function at time `T` = n

        Return:
        - loglik: a vector of log likelihood value at location k = 0, 1, ..., K [ K ]
        - lams:   
        """
        if hasattr(self.kernel, "precompute_features"):
            self.kernel.precompute_features()

        lams1 = [torch.sum(torch.log(self._lambda_t(t,
                               self.trsfm_obs[t-1, :, 1:][self.data_mask[t-1, :]].clone(),
                               self.trsfm_obs[(max(t - self.kpt - 1, 0)):(t-1), :, 0][self.data_mask[(max(t - self.kpt - 1, 0)):(t-1), :]].clone(),
                               self.trsfm_obs[(max(t - self.kpt - 1, 0)):(t-1), :, 1:][self.data_mask[(max(t - self.kpt - 1, 0)):(t-1), :]].clone()) + 1e-5)) \
                     for t in self.ts]
        lams1 = torch.sum(torch.stack(lams1, dim = 0))
        if self.discretize == True:
            lams2 = self.discrete_spa_tem_int()
        else:
            # TODO: Continuous temporal model
            pass
     
        loglik = lams1 - lams2

        return loglik, lams1, lams2

    def forward(self):
        """
        customized forward function
        """
        # calculate data log-likelihood
        return self._log_likelihood()
    

class TorchKernelPP_Exogenous(torch.nn.Module):
    """
    PyTorch Module for Spatio-temporal Point Process
    """

    def __init__(self, obs, region_attr, kernel, exogenous_model, keep_latest_t = 3, discretize = True, data_dim = 3,
                 edo_learn = True, exo_learn = True, bg_learn = True):
        """
        Denote the number of observations as N, data dim = 3(time, x, y)

        Args:
        - obs:     Tensor, Original data matrix, [N, data_dim]
        - ldmk:     Tensor, Exogenous landmark factor, [M, 2]
        - region_area: area of a shapely Polygon
        """
        torch.nn.Module.__init__(self)
        # data
        self.obs    = obs # Original data matrix, [N, data_dim]
        self.ts = torch.unique(self.obs[:, 0]).int().cpu().numpy()
        #self.ldmk   = ldmk # landmarks in the area, [M, 2]
        # configurations
        self.usegpu = False
        self.region_area = region_attr[0] # Tensor float
        self.xmin = region_attr[1] # Tensor float
        self.xmax = region_attr[2] # Tensor float
        self.ymin = region_attr[3] # Tensor float
        self.ymax = region_attr[4] # Tensor float
        self.N = self.obs.shape[0]
        self.kpt = keep_latest_t
        self.discretize = discretize
        self.data_dim = data_dim
        self.edo_learn = edo_learn
        self.exo_learn = exo_learn
        self.bg_learn = bg_learn
        # parameters
        bg = torch.mean(torch.unique(self.obs[:, 0], return_counts = True)[1].float()).item()
        self.Lambda0 = torch.nn.Parameter(torch.FloatTensor([bg/1])[0], requires_grad=False) # [ 1 ]
        #self.Lambda0   = torch.nn.Parameter(torch.Tensor(1).uniform_(.1, 1)[0]) # [ 1 ]
        self.kernel  = kernel
        self.exogenous = exogenous_model
        # pre-compute
        self.data_transformation()

    def data_transformation(self):
        """
        Transform a N*2 data matrix into model-desired formation
        """
        t_value, t_counts = torch.unique(self.obs[:, 0], return_counts = True)
        t_value = t_value.int()
        trsfm_obs = torch.zeros(torch.max(t_value), torch.max(t_counts), self.data_dim, device=device)
        mask = torch.zeros(torch.max(t_value), torch.max(t_counts), device=device)
        for i in range(len(t_value)):
            t = t_value[i]
            idx = self.obs[:, 0] == t
            n = sum(idx)
            trsfm_obs[t-1, :n, :] = self.obs[idx, :]
            mask[t-1, :n] = 1
        mask = mask > 0

        self.trsfm_obs = trsfm_obs
        self.data_mask = mask

    def _Lambda0(self, _t, _s):
        """
        Background rate at time `t`, location `s`
        """
        return self.Lambda0
    
    def _Time_Decay_Kernel(self):
        """
        Time decay kernel
        """
        self.time_decay_ker = self.kernel._Time_decay_kernel()
    
    def _triggering_t(self, t, s, his_t, his_s):
        """
        Triggering effect term in conditional intensity function of current events `s` at time `t`, calculated in vector form

        Args:
        - t:  index of time, e.g., 0, 1, ..., N (integer)
        - s:  location of current events [N, 2]
        - idx: index of current events. [N]
        - his_t: history time vector. [M]
        - his_s: history location matrix. [M, 2]
        - his_idx: index of inspected history events. [M]
        Return:
        - lam: a vector lambda value of `s` at time `t`
        """
        N = s.shape[0]
        M = his_s.shape[0]
        if self.usegpu: device = "cuda:0"
        else: device = "cpu"

        if M > 0:
            # if self.kpt is not None:
            #     his_idx = (self.obs[:, 0] >= t - self.kpt) & (self.obs[:, 0] < t)
            #     his_t = self.obs[his_idx, 0]
            #     his_s = self.obs[his_idx, 1:]
            # else:    
            #     his_idx = self.obs[:, 0] < t
            #     his_t = self.obs[his_idx, 0]
            #     his_s = self.obs[his_idx, 1:]

            kers = self.kernel.nu(t, s, his_t, his_s)

            return kers
        else:
            return torch.zeros(N, device=device)

    def _Exogenous_promotion(self, t, s):
        """
        Exogenous effect term of landmarks in conditional intensity function of current events `s` at time `t`, calculated in vector form

        Args:
        - t:  index of time, e.g., 0, 1, ..., N (integer)
        - s:  location of current events [N, 2]
        Return: a vector of exogenous effect value of `s` at time `t`
        """

        return self.exogenous.Exogenous_calculation(s)

    def _lambda_t(self, t, s, his_t, his_s):

        bkg = self._Lambda0(t, s)
        kers = self._triggering_t(t, s, his_t, his_s)
        exo = self._Exogenous_promotion(t, s)

        return bkg + kers + exo

    def Overall_int(self, t):
        """
        Spatio-Temporal integration of conditional intensity function at time T

        Return:
        - lam: Spatio-Temporal Integral estimation
        """
        C = self.Lambda0 * self.region_area
        
        if t > 1:
            his_t = self.trsfm_obs[(max(t - self.kpt - 1, 0)):(t-1), :, 0][self.data_mask[(max(t - self.kpt - 1, 0)):(t-1), :]].clone()
            delta_t = t - his_t
            values, counts = torch.unique(delta_t, return_counts = True)

            F = torch.sum(counts * self.time_decay_ker[1:(len(values)+1)])
        else:
            F = 0
        
        lam = C + F + self.exogenous.Exogenous_in_integral(t)
        
        return lam

    def discrete_spa_tem_int(self):
        bg = self.Lambda0 * self.region_area * np.max(self.ts)
        edo = self.kernel.discrete_spa_tem_int()
        exo = self.exogenous.Exogenous_in_integral(1) * np.max(self.ts)

        return bg + edo + exo
    
    def _log_likelihood(self):
        """
        Log likelihood function at time `T` = n

        Return:
        - loglik: a vector of log likelihood value at location k = 0, 1, ..., K [ K ]
        - lams:   
        """
        if hasattr(self.kernel, "precompute_features"):
            self.kernel.precompute_features()

        lams1 = [torch.sum(torch.log(self._lambda_t(t,
                               self.trsfm_obs[t-1, :, 1:][self.data_mask[t-1, :]].clone(),
                               self.trsfm_obs[(max(t - self.kpt - 1, 0)):(t-1), :, 0][self.data_mask[(max(t - self.kpt - 1, 0)):(t-1), :]].clone(),
                               self.trsfm_obs[(max(t - self.kpt - 1, 0)):(t-1), :, 1:][self.data_mask[(max(t - self.kpt - 1, 0)):(t-1), :]].clone()) + 1e-5)) \
                     for t in self.ts]
        lams1 = torch.sum(torch.stack(lams1, dim = 0))
        if self.discretize == True:
            lams2 = self.discrete_spa_tem_int()
        else:
            # TODO: Continuous temporal model
            pass
     
        loglik = lams1 - lams2

        return loglik, lams1, lams2

    def forward(self):
        """
        customized forward function
        """
        # calculate data log-likelihood
        return self._log_likelihood()