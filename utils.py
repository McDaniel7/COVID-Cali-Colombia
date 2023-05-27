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

import geopandas as gpd
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon
from descartes.patch import PolygonPatch
from shapely.ops import cascaded_union

"""# Utils"""

def datelist(beginDate, endDate):
    date_l=[datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start=beginDate, end=endDate))]
    return date_l

def with_in_region(ps, region):
    tupVerts = region.exterior
    p = Path(tupVerts)
    return p.contains_points(ps)

def MC_sample(n, region):
    n_ps = 0
    points = []
    x_min, y_min, x_max, y_max = region.bounds
    tupVerts = region.exterior
    p = Path(tupVerts)
    
    while n_ps < n:
      xs = np.random.uniform(x_min, x_max, size = n)
      ys = np.random.uniform(y_min, y_max, size = n)
      ps = MultiPoint(list(zip(xs, ys)))
      cnt = p.contains_points(ps)
      points.extend(np.array(list(zip(xs, ys)))[cnt])
      n_ps += sum(cnt)
        
    return points[:n]
