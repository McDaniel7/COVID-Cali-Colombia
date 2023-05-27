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


########## Kernel ############

def weight_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data,0.01)
    elif isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0,1)
        m.bias.data.normal_(0,1)
        
        
def weight_update(m, rate):
    m.weight.data = m.weight.data + torch.ones(m.weight.data.shape) * rate


'''Temporal kernel'''

class _Gaussian_kernel(torch.nn.Module):
    """
    Temporal Gaussian decaying kernel
    """

    def __init__(self, C, Sigma):
        super(_Gaussian_kernel, self).__init__()
        
        self.C  = torch.nn.Parameter(torch.FloatTensor([C])[0])
        self.Sigma  = torch.nn.Parameter(torch.FloatTensor([Sigma])[0])

    def integration(self, obs):
        """
        return closed form integration w.r.t. data obs
        """
        integral = np.sqrt(2 * np.pi) * self.C * self.Sigma * torch.sum((torch.distributions.Normal(0, 1).cdf((obs[:, 0].max() + 1 - obs[:, 0]) / self.Sigma) - 0.5))

        return integral

    def grid_evaluation(self, obs):
        """
        return kernel evaluations on time grids
        """
        return self.C * torch.exp(-1/2/torch.square(self.Sigma) * torch.square(torch.arange(0, obs[:, 0].max(), 1, device=obs.device)))
        
    def forward(self, x):

        return self.C * torch.exp(-1/2/torch.square(self.Sigma) * torch.square(x))


class _Weibull_kernel(torch.nn.Module):
    """
    Temporal Weibull decaying kernel
    """

    def __init__(self, Lamb, K):
        super(_Weibull_kernel, self).__init__()
        
        self.Lamb  = torch.nn.Parameter(torch.FloatTensor([Lamb])[0])
        self.K  = torch.nn.Parameter(torch.FloatTensor([K])[0])

    def integration(self, obs):
        """
        return closed form integration w.r.t. data obs
        """
        integral = torch.sum(torch.distributions.weibull.Weibull(self.Lamb, self.K).cdf(obs[:, 0].max() + 1 - obs[:, 0]))

        return integral

    def grid_evaluation(self, obs):
        """
        return kernel evaluations on time grids
        """
        x = torch.square(torch.arange(0, obs[:, 0].max(), 1, device=obs.device))
        return self.K / self.Lamb * (x / self.Lamb) ** (self.K - 1) * torch.exp(-(x / self.Lamb) ** self.K)
        
    def forward(self, x):
        """
        return kernel evaluation at x
        """
        return self.K / self.Lamb * (x / self.Lamb) ** (self.K - 1) * torch.exp(-(x / self.Lamb) ** self.K)


class _Gamma_kernel(torch.nn.Module):
    """
    Temporal Gamma decaying kernel
    """

    def __init__(self, Alpha, Beta):
        super(_Gamma_kernel, self).__init__()
        
        self.Alpha  = torch.nn.Parameter(torch.FloatTensor([Alpha])[0])
        self.Beta  = torch.nn.Parameter(torch.FloatTensor([Beta])[0])

    def integration(self, obs):
        """
        return closed form integration w.r.t. data obs
        """
        grids = torch.linspace(1e-5, obs[:, 0].max().item() + 1, 100)
        h = (obs[:, 0].max().item() + 1) / 100
        dens  = torch.exp(torch.distributions.gamma.Gamma(self.Alpha, self.Beta).log_prob(grids))
        cdf = torch.cumsum(dens * h,dim=0)      # [ 100 ]

        int_t = obs[:, 0].max() + 1 - obs[:, 0]
        int_start_idx = torch.div(int_t, h, rounding_mode="floor").long()
        int_end_idx   = int_start_idx + 1
        int_start     = cdf[int_start_idx]      # [ n_obs ]
        int_end       = cdf[int_end_idx]        # [ n_obs ]
        int_prop      = torch.remainder(int_t, h) / h
                                                          # [ n_obs ]
        integral  = torch.sum(torch.lerp(int_start, int_end, int_prop))  # [ n_obs ]

        return integral

    def grid_evaluation(self, obs):
        """
        return kernel evaluations on time grids
        """
        x = torch.square(torch.arange(1e-5, obs[:, 0].max(), 1, device=obs.device))
        return torch.exp(torch.distributions.gamma.Gamma(self.Alpha, self.Beta).log_prob(x))
        
    def forward(self, x):
        """
        return kernel evaluation at x
        """
        return torch.exp(torch.distributions.gamma.Gamma(self.Alpha, self.Beta).log_prob(x))


class _Rayleigh_kernel(torch.nn.Module):
    """
    Temporal Rayleigh decaying kernel
    """

    def __init__(self, Sigma):
        super(_Rayleigh_kernel, self).__init__()
        
        self.Sigma  = torch.nn.Parameter(torch.FloatTensor([Sigma])[0])

    def integration(self, obs):
        """
        return closed form integration w.r.t. data obs
        """
        integral = torch.sum(1 - torch.exp(-(obs[:, 0].max() + 1 - obs[:, 0]) ** 2 / 2 / self.Sigma ** 2))

        return integral

    def grid_evaluation(self, obs):
        """
        return kernel evaluations on time grids
        """
        x = torch.square(torch.arange(0, obs[:, 0].max(), 1, device=obs.device))
        return x / self.Sigma ** 2 * torch.exp(-x ** 2 / 2 / self.Sigma ** 2)
        
    def forward(self, x):
        """
        return kernel evaluation at x
        """
        return x / self.Sigma ** 2 * torch.exp(-x ** 2 / 2 / self.Sigma ** 2)


'''Deep spatial kernel'''
class NeuralNet_for_FocusPoints_Multikernel(torch.nn.Module):
    '''
    The neural network used to generate focus points and kernel weights for each location s.
    Input shape = 2, Output Shape = 3sa
    '''
    # The network contains three fully-connected layers and one non-linear activation layer with
    # an input shape of 2 and output shape of 2.
    def __init__(self):
        super(NeuralNet_for_FocusPoints_Multikernel, self).__init__()
        
        self.fc1 = torch.nn.Linear(2, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 3)
        self.scale = torch.nn.Parameter(torch.Tensor([0.1, 0.1, 1]), requires_grad = False)
        
    def forward(self, x):
        
        x = self.fc1(x)
        x = F.softplus(x)
        x = self.fc2(x)
        x = F.softplus(x)
        x = self.fc3(x)
        # x = (2 * torch.sigmoid(x) - 1)
        # x[:2] = x[:2] * 0.1
        x = (2 * torch.sigmoid(x) - 1) * self.scale
        
        return x

class NeuralNet_for_ExogenousEffect(torch.nn.Module):
    '''
    The neural network used to generate exogenous effect for a landmark s.
    Assume bivariate Gaussian with independent elements.
    Input shape = 2, Output Shape = 1
    '''
    # The network contains three fully-connected layers and one non-linear activation layer with
    # an input shape of 2 and output shape of 2.
    def __init__(self):
        super(NeuralNet_for_ExogenousEffect, self).__init__()
        
        self.fc1 = torch.nn.Linear(2, 16)
        self.fc2 = torch.nn.Linear(16, 8)
        self.fc3 = torch.nn.Linear(8, 1)
        
    def forward(self, x):
        
        # ReLU activation function
        
        x = self.fc1(x)
        x = F.softplus(x)
        x = self.fc2(x)
        x = F.softplus(x)
        x = self.fc3(x)
        x = torch.abs(2 * torch.sigmoid(x) - 1)
        
        return x

class StdDiffusionKernel(torch.nn.Module):
    """
    Kernel function including the diffusion-type model proposed by Musmeci and
    Vere-Jones (1992).
    """
    def __init__(self, obs):
        super(StdDiffusionKernel, self).__init__()
        self.obs    = obs # Original data matrix, [N, 3]
        self.n = torch.max(self.obs[:, 0])
        self.C      = torch.nn.Parameter(torch.Tensor(1).uniform_(100, 1000)[0]) # [ 1 ]
        self.Beta   = torch.nn.Parameter(torch.Tensor(1).uniform_(0.1, 1)[0])           # [ 1 ]
        self.Sigmax = torch.nn.Parameter(torch.Tensor(1).uniform_(1, 10)[0]) # [ 1 ]
        self.Sigmay = torch.nn.Parameter(torch.Tensor(1).uniform_(1, 10)[0]) # [ 1 ]
        self.usegpu = False

    def _Time_decay_kernel(self):
        """
        Time decay kernel
        """
        if self.usegpu: device = "cuda:0"
        else: device = "cpu"

        return self.C * torch.exp(-self.Beta * torch.arange(1, torch.max(self.obs[:, 0]), 1, device=device))
    
    def discrete_spa_tem_int(self):
        
        edo = -self.C / self.Beta * torch.sum(torch.exp(-self.Beta * (self.n + 1 - self.obs[:, 0])) - 1)

        return edo

    def nu(self, t, s, his_t, his_s):

        N = s.shape[0]
        M = his_s.shape[0]

        delta_t = t - his_t
        W = torch.div(self.C * torch.exp(-self.Beta * delta_t), self.Sigmax * self.Sigmay * 2 * np.pi * delta_t)
            
        delta_s = s.unsqueeze(-1).expand(N, 2, M) - his_s.T.expand(N, 2, M)
        st1 = torch.div(torch.square(delta_s), torch.square(torch.stack([self.Sigmax, self.Sigmay], dim=0)).expand(M, 2).T \
                        * -2 * delta_t)
            
        return torch.mm(torch.exp(torch.sum(st1, dim=1)), W.unsqueeze(-1))


class Multi_NN_NonstationaryKernel_3(torch.nn.Module):
    """
    Multi-component non-stationary Gaussian kernel, time and space decoupled.
    The focus points of one standard ellipse (OSE) are generated by Neural Network.
    """
    def __init__(self, obs, region_attr, temp_kernel, n_comp = 3, keep_latest_t = 3, data_dim = 3):
        super(Multi_NN_NonstationaryKernel_3, self).__init__()
        # data
        self.obs = obs # Original data matrix, [N, 3]
        self.n = torch.max(self.obs[:, 0])
        self.region_area = region_attr[0] # Tensor float
        self.xmin = region_attr[1] # Tensor float
        self.xmax = region_attr[2] # Tensor float
        self.ymin = region_attr[3] # Tensor float
        self.ymax = region_attr[4] # Tensor float
        self.obs_normalized = self._Normalization(self.obs[:, 1:])
        self.obs_normalized = torch.hstack((self.obs[:, 0].reshape(-1, 1), self.obs_normalized))
        # configuration
        self.usegpu = False
        self.n_comp = n_comp
        self.net1 = NeuralNet_for_FocusPoints_Multikernel()
        self.net2 = NeuralNet_for_FocusPoints_Multikernel()
        self.net3 = NeuralNet_for_FocusPoints_Multikernel()
        self.net1.apply(weight_init)
        self.net2.apply(weight_init)
        self.net3.apply(weight_init)
        self.temp_kernel = temp_kernel
        self.A     = torch.nn.Parameter(torch.Tensor([0.1])[0], requires_grad = False) # [ 1 ]
        self.tau_z   = torch.nn.Parameter(torch.Tensor([30])[0], requires_grad = False) # [ 1 ]
        self.data_dim = data_dim
        self.kpt = keep_latest_t
        self.precompute = False
        self.new_psi = False
        # precompute
        self.data_transformation()

    def data_transformation(self):
        """
        Transform a N*3 data matrix into model-desired formation
        """
        t_value, t_counts = torch.unique(self.obs_normalized[:, 0], return_counts = True)
        t_value = t_value.int()
        trsfm_obs = torch.zeros(torch.max(t_value)+1, torch.max(t_counts), self.data_dim, device=device)
        mask = torch.zeros(torch.max(t_value)+1, torch.max(t_counts), device=device)
        for i in range(len(t_value)):
            t = t_value[i]
            idx = self.obs_normalized[:, 0] == t
            n = sum(idx)
            trsfm_obs[t-1, :n, :] = self.obs_normalized[idx, :]
            mask[t-1, :n] = 1
        mask = mask > 0

        self.trsfm_obs = trsfm_obs
        self.data_mask = mask

    def _Time_decay_kernel(self):
        """
        Time decay kernel
        """

        return self.temp_kernel.grid_evaluation(self.obs)
    
    def _Normalization(self, ps):
        normalized = torch.zeros_like(ps)
        normalized[..., 0] = 2 * (ps[..., 0] - self.xmin) / (self.xmax - self.xmin) - 1
        normalized[..., 1] = 2 * (ps[..., 1] - self.ymin) / (self.ymax - self.ymin) - 1
    
        return normalized
    
    def precompute_features(self):
        """
        Here this function is used to precompute Focus points.
        """
        self.psis = []
        self.ws = []
        out1 = self.net1(self.trsfm_obs[:, :, 1:])
        self.psis.append(out1[:, :, :-1])
        self.ws.append(out1[:, :, -1])
        out2 = self.net2(self.trsfm_obs[:, :, 1:])
        self.psis.append(out2[:, :, :-1])
        self.ws.append(out2[:, :, -1])
        out3 = self.net3(self.trsfm_obs[:, :, 1:])
        self.psis.append(out3[:, :, :-1])
        self.ws.append(out3[:, :, -1])
        self.psis = torch.stack(self.psis, dim=0) # [n_comp, T, max(daily_number), 2]
        self.ws = torch.stack(self.ws, dim=-1) # [T, max(daily_number), n_comp]
        self.precompute = True

    def discrete_spa_tem_int(self):

        return self.temp_kernel.integration(self.obs)

    def Cov_para(self, psi):
        """
        Generate parameters of the covariance matrix.

        - psi: coordinate of focus points. [N, 2]
        """
        norm_psi = torch.norm(psi, dim=-1)
        a = torch.sqrt(torch.sqrt(4 * torch.square(self.A) + torch.pow(norm_psi, 4) * np.pi ** 2) / 2 / np.pi \
                   + (torch.square(psi[..., 0]) - torch.square(psi[..., 1])) / 2)
        b = torch.sqrt(torch.sqrt(4 * torch.square(self.A) + torch.pow(norm_psi, 4) * np.pi ** 2) / 2 / np.pi \
                   + (torch.square(psi[..., 1]) - torch.square(psi[..., 0])) / 2)
        a = torch.clamp(a, min = 0.)
        b = torch.clamp(b, min = 0.)
        alpha = torch.arctan(psi[..., 1] / psi[..., 0])
        rho_a_b = torch.square(norm_psi) * torch.cos(alpha) * torch.sin(alpha) / torch.square(self.tau_z)

        return rho_a_b, a/self.tau_z, b/self.tau_z
  
    def nu(self, t, s, his_t, his_s):
        """
        Vectorized trigger function computation. 

        - t: present time, a scalar.
        - s: location matrix of current events. [N, 2]
        - his_t: history time vector. [M]
        - his_s: history location matrix. [M, 2]
        - psi: if given, should be 2-d tensor. [L, 2]

        Return:
        - A N-length vector of trigger function value
        """
        N = s.shape[0]
        M = his_s.shape[0]
        # A_prime = self.A / torch.square(self.tau_z)
        t_p = np.int(np.ceil(t))
        if not self.precompute:
            raise Exception("No available focus points!")
        elif self.new_psi:
            his_psi = self.psis[:, (max(t_p - self.kpt - 1, 0)):(t_p-1), :, :][:, self.data_mask[(max(t_p - self.kpt - 1, 0)):(t_p-1), :]].clone()
            his_w = self.ws[(max(t_p - self.kpt - 1, 0)):(t_p-1), :, :][self.data_mask[(max(t_p - self.kpt - 1, 0)):(t_p-1), :]].clone()
            psi = []
            w = []
            out1 = self.net1(self._Normalization(s))
            psi.append(out1[:, :-1])
            w.append(out1[:, -1])
            out2 = self.net2(self._Normalization(s))
            psi.append(out2[:, :-1])
            w.append(out2[:, -1])
            out3 = self.net3(self._Normalization(s))
            psi.append(out3[:, :-1])
            w.append(out3[:, -1])
            psi = torch.stack(psi, dim=0) # [n_comp, N, 2]
            w = torch.stack(w, dim=-1) # [N, n_comp]
        else:
            his_psi = self.psis[:, (max(t_p - self.kpt - 1, 0)):(t_p-1), :, :][:, self.data_mask[(max(t_p - self.kpt - 1, 0)):(t_p-1), :]].clone()
            his_w = self.ws[(max(t_p - self.kpt - 1, 0)):(t_p-1), :, :][self.data_mask[(max(t_p - self.kpt - 1, 0)):(t_p-1), :]].clone()
            psi = self.psis[:, (t_p-1), :, :][:, self.data_mask[(t_p-1), :]].clone() # [n_comp, N, 2]
            w = self.ws[(t_p-1), :, :][self.data_mask[(t_p-1), :]].clone() # [N, n_comp]
        delta_t = t - his_t
        rho_a_b, a, b = self.Cov_para(psi)
        his_rho_a_b, his_a, his_b = self.Cov_para(his_psi)

        sigma_x = torch.sqrt(torch.square(a.T.unsqueeze(1).unsqueeze(-1).repeat(1, M, 1, self.n_comp)) \
                  + torch.square(his_a.T.unsqueeze(1).unsqueeze(0).repeat(N, 1, self.n_comp, 1)))
        sigma_y = torch.sqrt(torch.square(b.T.unsqueeze(1).unsqueeze(-1).repeat(1, M, 1, self.n_comp)) \
                  + torch.square(his_b.T.unsqueeze(1).unsqueeze(0).repeat(N, 1, self.n_comp, 1)))
        rho = (rho_a_b.T.unsqueeze(1).unsqueeze(-1).repeat(1, M, 1, self.n_comp) \
                  + his_rho_a_b.T.unsqueeze(1).unsqueeze(0).repeat(N, 1, self.n_comp, 1)) / sigma_x / sigma_y
    
        time_decay = self.temp_kernel(delta_t)

        weight = torch.matmul((torch.exp(w).T / torch.sum(torch.exp(w), dim = 1)).T.unsqueeze(-1).unsqueeze(1).repeat(1, M, 1, 1),
                              (torch.exp(his_w).T / torch.sum(torch.exp(his_w), dim = 1)).T.unsqueeze(-2).unsqueeze(0).repeat(N, 1, 1, 1)) # [N, M, n_comp, n_comp]

        std_x = (s[:, 0].unsqueeze(1).repeat(1, M) - his_s[:, 0].unsqueeze(0).repeat(N, 1)).unsqueeze(-1).unsqueeze(-1) / sigma_x
        std_y = (s[:, 1].unsqueeze(1).repeat(1, M) - his_s[:, 1].unsqueeze(0).repeat(N, 1)).unsqueeze(-1).unsqueeze(-1) / sigma_y

        return torch.mm(time_decay.reshape(1, -1),
                        (((0.5 / (np.pi * sigma_x * sigma_y * torch.sqrt(1-torch.square(rho))) \
                           * torch.exp(-0.5 / (1-torch.square(rho)) * ( \
                            torch.square(std_x) + torch.square(std_y) \
                            - 2 * rho * std_x * std_y) ) ) *
                        weight).sum(-1).sum(-1)).T)