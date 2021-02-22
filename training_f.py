import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import arrow

import torch
import torch.optim as optim


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