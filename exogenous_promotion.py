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

########### Exogenous promotion ############

class Exogenous_Promotion(torch.nn.Module):
    """
    Pytorch module of exogenous promotion of landmarks.
    """
    def __init__(self, landmark, data_dim = 3):
        super(Exogenous_Promotion, self).__init__()
        self.ldmk   = landmark # landmarks in the area, [M, 2]
        # configurations
        self.M = self.ldmk.shape[0]
        self.data_dim = data_dim
        # parameters
        self.gamma  = torch.nn.Parameter(torch.zeros(self.M), requires_grad = False) # [ 1 ]
        self.exo_dist = torch.nn.Parameter(torch.Tensor(self.M).uniform_(.1, 1))
    
    def Exogenous_calculation(self, s):
        """
        Calculate the exogenous effect of landmarks at location s.
        """
        N = s.shape[0]
        sq_norm = torch.square(torch.norm(s.unsqueeze(1).repeat(1, self.M, 1) - self.ldmk.unsqueeze(0).repeat(N, 1, 1), dim=-1))
        exo = (self.gamma * 1 / 2 / np.pi * torch.exp(sq_norm * -0.5 / torch.square(self.exo_dist)) / torch.square(self.exo_dist)).sum(1).reshape(1, -1)

        return exo

    def Exogenous_in_integral(self, t):
        return torch.sum(self.gamma)
