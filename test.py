'''
Get test results for sherds from the same break.
'''

import os
import gc
import sys
import h5py
import math
import json
import pickle
import random
import trimesh
import pymeshlab
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
# utils
sys.path.insert(1, 'utils')
from utils import *

# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

### .sh file Variables ###
vessel = sys.argv[1]
test_break = sys.argv[2]
test_path = sys.argv[3]
rotate_phi = int(sys.argv[4])
separate_files = int(sys.argv[5])

### compiling and loading weights ###
rot = PotNet(6)
trans = PotNet(2)
compile_nets(rot, trans, euclidean_distance_loss)

rot.load_weights(f'./models/{vessel}/best_rot_model.hdf5')
trans.load_weights(f'./models/{vessel}/best_trans_model.hdf5')

metrics = {'filename': [], 'msePTC': [], 'rmsePTC': [], 'distCNTRD': [], 'rmsePTC (x,y,z)': [], 'stdPTC': []}
T_dict = {}


### Begin ###
mesh_tuple = 0
os.mkdir(f'./results/{vessel}_results_{test_break}/')
os.mkdir(f'./results/{vessel}_results_{test_break}/files/')

for i in range(1, len(os.listdir(test_path)) + 1):
  print(f'Predicting {test_break}_frac_{i}')
  mesh = trimesh.load(f'{test_path}/{test_break}_frac_{i}.stl')

  if type(mesh) == trimesh.base.Trimesh and mesh.bounding_box.volume > 0.00001:
    poisson_points = poisson(f'{test_path}/{test_break}_frac_{i}.stl')

    r, theta, phi, yz = yz_normalization(poisson_points)
    canon = generate_canon(yz)
    T_unnorm2canon = kabsch(canon, poisson_points)

    pred_rot = rot.predict(canon[None, :, :], verbose=0)[0]
    pred_trans = trans.predict(canon[None, :, :], verbose=0)[0]
        
    T_pred = T_matrix(pred_rot, pred_trans)
    
    # send canon cloud to pred position
    points4dim = np.array([np.append(point, 1) for point in canon])
    predicted_cloud = (points4dim @ T_pred.T)[:,:3]
    
    # compare the same exact cloud in two different positions
    get_metrics(f'{test_break}_frac_{i}', metrics, ['dist_centroid', 'mse_ptc'], yz, predicted_cloud)
    
    # save matrices
    T_dict[f'{test_break}_frac_{i}'] = {'unnorm2canon': T_unnorm2canon.tolist(), 'T_pred': T_pred.tolist()}

    # get rotation y matrix for -phi angle
    T_rot_y = T_phi(-phi, [0,0])

    # visualize results
    mesh.apply_transform(T_unnorm2canon)
    mesh.apply_transform(T_pred)
    if rotate_phi:
      mesh.apply_transform(T_rot_y)
    if separate_files:
      # export function supports .stl or .ply
      mesh.export(f'./results/{vessel}_results_{test_break}/files/{vessel}_{test_break}_{i}.stl')
    mesh_tuple += mesh

if not separate_files:
  mesh_tuple.export(f'./results/{vessel}_results_{test_break}/files/{vessel}_{test_break}.stl')

# exporting data to xlsx file
df = pd.DataFrame.from_dict(metrics)
df.sort_values('filename', inplace=True, ignore_index=True)
df.to_csv(f'./results/{vessel}_results_{test_break}/{vessel}_real_metrics.csv')

# exporting T matrices dict
with open(f'./results/{vessel}_results_{test_break}/{vessel}_T_dict.txt','w') as js:
  json.dump(T_dict, js)

#
