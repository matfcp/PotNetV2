'''
1. produce a poisson-disk sample for each sherd from the train set;
2. produce the yz-normalized position and canon position for each
sherd from the train set;
3. save the matrices containing the points sets;
'''

import os
import sys
import pickle
import trimesh
import numpy as np
import pymeshlab
import random
import gc
# utils
sys.path.insert(1, 'utils')
from utils import *

min_volume = 0.00001

vessel = 'LV'
path = f'datasets/{vessel}_dataset/files/train/'
lenght_path = len(os.listdir(path))
print('path lenght: ', lenght_path)
path = [os.path.join(path, f'file{i}') for i in range(1, lenght_path + 1)]

# Processing:
canon_points = []
T = []

print('Starting...')
for i in range(1, lenght_path + 1):
  for j in range(1, len(os.listdir(path[i-1])) + 1):
    if i % 100 == 0 and j == 1:
      print(f'Reached file{i}')

    mesh = trimesh.load(f'{path[i-1]}/file{i}_frac_{j}.stl')

    if type(mesh) == trimesh.base.Trimesh and mesh.bounding_box.volume > min_volume:
      ms = f'{path[i-1]}/file{i}_frac_{j}.stl'
      poisson_vertices = poisson(ms)
      poisson_points.append(poisson_vertices)

      r, theta, phi, yz = yz_normalization(poisson_vertices)
      canon = generate_canon(yz)

      yz_points.append(yz)
      canon_points.append(canon)
      targets.append([r, theta, phi])

      k = kabsch(yz, canon)
      T.append(k)
      class_array.append(vessel_class)

canon_points = np.array(canon_points, dtype='object')
T = np.array(T, dtype='object')

path_save = f'train_data/{vessel}/'

array_file = open(path_save + 'canon_points.npy', 'wb')
np.save(array_file, canon_points)
array_file.close()

array_file = open(path_save + 'T.npy', 'wb')
np.save(array_file, T)
array_file.close()

#
