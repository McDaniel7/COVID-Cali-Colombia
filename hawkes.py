import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
import random
import arrow

import torch
import torch.optim as optim

class TorchHawkes(torch.nn.Module):
    """
    PyTorch Module for Hawkes Processes
    """

    def __init__(self, obs):
        """
        Denote the number of time units as N, the number of locations as K

        Args:
        - obs:    event observations    [ N, K ]
        """
        torch.nn.Module.__init__(self)
        # data
        self.obs    = torch.Tensor(obs) # [ K, N ]
        # configurations
        self.K, self.N = obs.shape
        # parameters
        self.Mu0   = self.obs.mean(1) / 10 + 1e-5                                    # [ K ]
        self.Beta  = torch.nn.Parameter(torch.Tensor(self.K).uniform_(1, 3))           # [ K ]
        self.Alpha = torch.nn.Parameter(torch.Tensor(self.K, self.K).uniform_(0, .01)) # [ K, K ]
    
    def _mu(self, _t):
        """
        Background rate at time `t`
        """
        return self.Mu0

    def _lambda(self, _t):
        """
        Conditional intensity function at time `t`

        Args:
        - _t:  index of time, e.g., 0, 1, ..., N (integer)
        Return:
        - lam: a vector of lambda value at time t and location k = 0, 1, ..., K [ K ]
        """
        if _t > 0:
            # current time and the past 
            t      = torch.ones(_t, dtype=torch.int32) * _t      # [ t ]
            tp     = torch.arange(_t)                            # [ t ]
            # self-exciting effect
            kernel = self.__exp_kernel(self.Beta, t, tp, self.K) # [ K, t ]
            Nt     = self.obs[:, :_t].clone()                    # [ K, t ]
            lam    = torch.mm(self.Alpha, Nt * kernel).sum(1)    # [ K ]
            lam    = torch.nn.functional.softplus(lam)           # [ K ]
        else:
            lam    = torch.zeros(self.K)
        return lam
        
    def _log_likelihood(self):
        """
        Log likelihood function at time `T`
        
        Args:
        - tau:    index of start time, e.g., 0, 1, ..., N (integer)
        - t:      index of end time, e.g., 0, 1, ..., N (integer)

        Return:
        - loglik: a vector of log likelihood value at location k = 0, 1, ..., K [ K ]
        - lams:   a list of historical conditional intensity values at time t = tau, ..., t
        """
        # lambda values from 0 to N
        lams0    = [ self._mu(t) for t in np.arange(self.N) ]     # ( N, [ K ] )
        lams1    = [ self._lambda(t) for t in np.arange(self.N) ] # ( N, [ K ] )
        lams0    = torch.stack(lams0, dim=1)                      # [ K, N ]
        lams1    = torch.stack(lams1, dim=1)                      # [ K, N ]
        Nloglams = self.obs * torch.log(lams0 + lams1 + 1e-5)     # [ K, N ]
        # log-likelihood function
        loglik   = (Nloglams - lams0 - lams1).sum()
        return loglik, lams0, lams1

    def forward(self):
        """
        customized forward function
        """
        # calculate data log-likelihood
        return self._log_likelihood()

    @staticmethod
    def __exp_kernel(Beta, t, tp, K):
        """
        Args:
        - Beta:  decaying rate [ K ]
        - t, tp: time index    [ t ]
        """
        delta_t = t - tp                              # [ t ]
        delta_t = delta_t.unsqueeze(0).repeat([K, 1]) # [ K, t ]
        Beta    = Beta.unsqueeze(1)                   # [ K, 1 ]
        return Beta * torch.exp(- delta_t * Beta)
    
    
class TorchHawkes_train_mu(TorchHawkes):
    """
    PyTorch Module for Hawkes Processes
    """

    def __init__(self, obs):
        """
        Denote the number of time units as N, the number of locations as K

        Args:
        - obs:    event observations    [ N, K ]
        """
        torch.nn.Module.__init__(self)
        # data
        self.obs    = torch.Tensor(obs) # [ K, N ]
        # configurations
        self.K, self.N = obs.shape
        # parameters
        self.Mu0   = torch.nn.Parameter(torch.Tensor(self.K).uniform_(0, .01))                                   # [ K ]
        self.Beta  = torch.nn.Parameter(torch.Tensor(self.K).uniform_(1, 3))           # [ K ]
        self.Alpha = torch.nn.Parameter(torch.Tensor(self.K, self.K).uniform_(0, .01)) # [ K, K ]

def __exp_kernel(Beta, t, tp, K):
        """
        Args:
        - Beta:  decaying rate [ K ]
        - t, tp: time index    [ t ]
        """
        delta_t = t - tp                              # [ t ]
        delta_t = delta_t.unsqueeze(0).repeat([K, 1]) # [ K, t ]
        Beta    = Beta.unsqueeze(1)                   # [ K, 1 ]
        return Beta * torch.exp(- delta_t * Beta)

def _lambda(model, _t):
        """
        Conditional intensity function at time `t`

        Args:
        - _t:  index of time, e.g., 0, 1, ..., N (integer)
        Return:
        - lam: a vector of lambda value at time t and location k = 0, 1, ..., K [ K ]
        """
        if _t > 0:
            # current time and the past 
            t      = torch.ones(_t, dtype=torch.int32) * _t      # [ t ]
            tp     = torch.arange(_t)                            # [ t ]
            # self-exciting effect
            kernel = __exp_kernel(model.Beta, t, tp, model.K) # [ K, t ]
            Nt     = model.obs[:, :_t].clone()                    # [ K, t ]
            lam    = torch.mm(model.Alpha, Nt * kernel).sum(1)    # [ K ]
        else:
            lam    = torch.zeros(model.K)
        return lam

def _lambda_all(model):
    lams0    = [ model._mu(t) for t in np.arange(model.N) ]     # ( N, [ K ] )
    lams1    = [ _lambda(model, t) for t in np.arange(model.N) ] # ( N, [ K ] )
    lams0    = torch.stack(lams0, dim=1)                      # [ K, N ]
    lams1    = torch.stack(lams1, dim=1)                      # [ K, N ]

    return (lams0 + lams1).data.numpy()