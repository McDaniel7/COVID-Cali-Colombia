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
from tqdm import tqdm

"""training functions"""

def train(model, niter=1000, lr=1e-1, log_interval=50, tol = 10):
  """training procedure for one epoch"""
  # define model clipper to enforce inequality constraints
  clipper1  = NonNegativeClipper()
  logliks = []
  best_lglk = -np.inf
  prev_lglk = -np.inf
  no_incre = 0
  converge = 0
  _lr = lr
  opts = []
  if model.bg_learn:
    opts.append(optim.Adadelta([model.Lambda0], lr=_lr))
  if model.exo_learn:
    opts.append(optim.Adadelta(model.exogenous.parameters(), lr=_lr))
  if model.edo_learn:
    opts.append(optim.Adadelta(model.kernel.parameters(), lr=_lr))

  llk_out = []
  for _iter in range(niter):
    try:
      model.train()
      for optimizer in opts:
        optimizer.zero_grad()           # init optimizer (set gradient to be zero)
      loglik, _, _ = model()
      if np.isnan(loglik.item()): 
        print("Loglikelihood:", loglik)
        for name, parameters in model.named_parameters():
            print(name)
            print(parameters)
        break
      # objective function
      loss         = - loglik
      loss.backward()                 # gradient descent
      for optimizer in opts:
        optimizer.step()                # update optimizer
      model.apply(clipper1)
      # log training output
      logliks.append(loglik.item())
      if loglik > best_lglk:
          best_lglk = loglik.item()
          no_incre = 0
      else:
          no_incre += 1
      if no_incre == 50:
          print("Learning rate decrease!")
          _lr = _lr / np.sqrt(10)
          if model.bg_learn:
            opts.append(optim.Adadelta([model.Lambda0], lr=_lr))
          if model.exo_learn:
            opts.append(optim.Adadelta(model.exogenous.parameters(), lr=_lr))
          if model.edo_learn:
            opts.append(optim.Adadelta(model.kernel.parameters(), lr=_lr))
          no_incre = 0
          best_lglk = -np.inf
      if np.abs(loglik.item() - prev_lglk) > tol:
          converge = 0
      else:
          converge += 1
      prev_lglk = loglik.item()
      if _iter % log_interval == 0:
        print("[%s] Train batch: %d\tLoglik: %.3e\t stag: %d converge: %d" % (arrow.now(), 
          _iter / log_interval, 
          sum(logliks) / log_interval,
          no_incre,
          converge))
        llk_out.append(sum(logliks) / log_interval)
        logliks = []
      if converge == 30:
          return llk_out
      
    except KeyboardInterrupt:
      return llk_out
      break

  return llk_out

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
    if hasattr(module, 'Lambda0'):
      Lambda0 = module.Lambda0.data
      module.Lambda0.data = torch.clamp(Lambda0, min=0.)
    # TorchMultiHawkes
    if hasattr(module, 'Alpha'):
      Alpha = module.Alpha.data
      module.Alpha.data = torch.clamp(Alpha, min=1e-5)
    if hasattr(module, 'Beta'):
      Beta  = module.Beta.data
      module.Beta.data  = torch.clamp(Beta, min=1e-5)
    # TorchKernel
    if hasattr(module, 'C'):
      C  = module.C.data
      module.C.data = torch.clamp(C, min=1e-5)
    if hasattr(module, 'K'):
      K  = module.K.data
      module.K.data = torch.clamp(K, min=1e-5)
    if hasattr(module, 'Lamb'):
      Lamb  = module.Lamb.data
      module.Lamb.data = torch.clamp(Lamb, min=1e-5)
    if hasattr(module, 'Sigmax'):
      Sigmax  = module.Sigmax.data
      module.Sigmax.data = torch.clamp(Sigmax, min=1e-5)
    if hasattr(module, 'Sigmay'):
      Sigmay  = module.Sigmay.data
      module.Sigmay.data = torch.clamp(Sigmay, min=1e-5)
    if hasattr(module, 'Sigma'):
      Sigma  = module.Sigma.data
      module.Sigma.data = torch.clamp(Sigma, min=1e-5)
    if hasattr(module, 'tau_z'):
      tau_z = module.tau_z.data
      module.tau_z.data = torch.clamp(tau_z, min=1e-5)
    if hasattr(module, 'ample'):
      ample = module.ample.data
      module.ample.data = torch.clamp(ample, min=1e-5)
    if hasattr(module, 'len_scale'):
      len_scale = module.len_scale.data
      module.len_scale.data = torch.clamp(len_scale, min=1e-6, max=1e-3)
    # Exogenous Effect
    if hasattr(module, 'gamma'):
      gamma  = module.gamma.data
      module.gamma.data = torch.clamp(gamma, min=1e-5)
    if hasattr(module, 'exo_dist'):
      exo_dist  = module.exo_dist.data
      module.exo_dist.data = torch.clamp(exo_dist, min=1e-5)