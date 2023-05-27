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

import geopandas as gpd
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon
from descartes.patch import PolygonPatch
from shapely.ops import cascaded_union

from utils import *
from kernels import _Gaussian_kernel, Multi_NN_NonstationaryKernel_3
from exogenous_promotion import Exogenous_Promotion
from point_process import TorchKernelPP_Exogenous
from training_functions import train
 
############ Load data #############

c19cc = pd.read_csv("./Data/COVID-19_Cali_Colombia_Contained.txt", sep = "\t")
c19cc["time"] = c19cc["time"] + 1
Cali_polys = []
for i in range(1, 23):
    Cali_polys.append(gpd.read_file("./Data/Colombia_geojson/Comuna/comuna"+str(i)+"_geojson.json").iloc[0, 0][0][0])

Cali_polys_list = MultiPolygon(Cali_polys)

Cali = cascaded_union(Cali_polys_list)
obs = torch.Tensor(np.array(c19cc.iloc[:, [3, 1, 2]]))
obs_week = torch.Tensor.clone(obs)
obs_week[:, 0] = torch.Tensor(np.int_((obs_week[:, 0] - 1) / 7) + 1)

df_townhall = pd.read_csv("./Data/Colombia_geojson/townhall.csv", sep = ",")
df_church = pd.read_csv("./Data/Colombia_geojson/churches.csv", sep = ",", encoding= 'unicode_escape')
df_school = pd.read_csv("./Data/Colombia_geojson/schools.csv", sep = ",", encoding= 'unicode_escape')

townhall = torch.Tensor(df_townhall[["x", "y"]].values)[with_in_region(torch.Tensor(df_townhall[["x", "y"]].values), Cali)]
church = torch.Tensor(df_church[["x", "y"]].values)[with_in_region(torch.Tensor(df_church[["x", "y"]].values), Cali)]
school = torch.Tensor(df_school[["x", "y"]].values)[with_in_region(torch.Tensor(df_school[["x", "y"]].values), Cali)]

landmarks = torch.cat((townhall, church, school), dim=0)


############ Train model ############

seed = np.random.randint(1, 100000, 1)
print("seed:", seed)
torch.manual_seed(seed[0])
usegpu = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

temp_kernel = _Gaussian_kernel(C=1, Sigma=1.5)
kernel = Multi_NN_NonstationaryKernel_3(obs_week.to(device), region_attr=torch.Tensor([Cali.area, Cali.bounds[0], Cali.bounds[2], Cali.bounds[1], Cali.bounds[3]]).to(device),
                                        temp_kernel=temp_kernel, keep_latest_t=3)
kernel.to(device)
exogenous_model = Exogenous_Promotion(landmarks.to(device))
exogenous_model.to(device)
model_weekly = TorchKernelPP_Exogenous(obs_week.to(device), region_attr=torch.Tensor([Cali.area, Cali.bounds[0], Cali.bounds[2], Cali.bounds[1], Cali.bounds[3]]).to(device),
                      kernel = kernel, exogenous_model = exogenous_model, keep_latest_t=3, bg_learn=False, exo_learn=True, edo_learn = True)

model_weekly.usegpu = True
model_weekly.kernel.usegpu = True

model_weekly.kernel.net1.scale = torch.nn.Parameter(torch.Tensor([0.1, 0.1, 1]))
model_weekly.kernel.net2.scale = torch.nn.Parameter(torch.Tensor([0.1, 0.1, 1]))
model_weekly.kernel.net3.scale = torch.nn.Parameter(torch.Tensor([0.1, 0.1, 1]))
model_weekly.kernel.A = torch.nn.Parameter(torch.Tensor([0.35]))
model_weekly.to(device)

Exo_llk = train(model_weekly, niter=1000, lr=1, log_interval=1, tol=1)
print("[%s] saving model..." % arrow.now())
if usegpu:
    torch.save(model_weekly.cpu().state_dict(), "./Result/saved_models/Multi_3_Nonstat_covid_cali_weekly_ds_" + \
               str(np.array(model_weekly.kernel.A)) + "_" + str(np.array(model_weekly.kernel.tau_z)) + "_" + str(seed[0]) + ".pt")
else:
    torch.save(model_weekly.state_dict(), "./Result/saved_models/Multi_3_Nonstat_covid_cali_weekly_ds_" + \
               str(model_weekly.kernel.A) + "_" + str(model_weekly.kernel.tau_z) + "_" + str(seed[0]) + ".pt")