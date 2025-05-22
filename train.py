import os
import gc
import h5py
import math
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
import keras.backend as K
import sys
sys.path.insert(1, 'utils')
from utils import *

# Tensorflow compiler flags
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Tensorflow GPU usage
print("\n>>> Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')), '\n')

### .sh file Variables ###
vessel = sys.argv[1]
net = sys.argv[2]

data_path = f'./train_data/{vessel}/'
save_path = f'./models/{vessel}/'

### Loading files ###
array_file = open(data_path + 'canon_points.npy', 'rb')
points = np.load(array_file, allow_pickle=True)
points = points.astype(np.float32)

array_file = open(data_path + 'T.npy', 'rb')
targets = np.load(array_file, allow_pickle=True)
targets = targets.astype(np.float32)

if net == 'trans':
  # trans targets
  targets = [[T[1,3], T[2,3]] for T in targets]
else:
  #rot targets
  targets = [np.concatenate((np.array(T[:3,:3][:,0]), np.array(T[:3,:3][:,1]))) for T in targets]

print('Files loaded!')


### Splitting data ###
train_points, valid_points, train_targets, valid_targets = train_test_split(points, targets, train_size=0.9, shuffle=True)

def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.00005, 0.00005, dtype=tf.float32)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label


train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_targets))
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_points, valid_targets))

train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(128)
valid_dataset = valid_dataset.shuffle(len(valid_points)).map(augment).batch(128)

print('Dataset splitted!')
print(f'train: {len(train_dataset)}\nvalidation: {len(valid_dataset)}')


points = 0
targets = 0
train_points = 0
valid_points = 0
train_targets = 0
valid_targets = 0
gc.collect()

### Compiling model ###
if net == 'trans':
  model = PotNet(2)
else:
  model = PotNet(6)

model.compile(
  loss=sqrd_euclidean_distance_loss,
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
  metrics=[sqrd_euclidean_distance_loss],
)

### Training the model ###

#early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path + f'best_{net}_model.hdf5', monitor='val_loss', verbose=1,
    save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

#weights = tf.keras.callbacks.ModelCheckpoint('best_model_weights.hdf5', monitor='val_loss', verbose=1,
#    save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')

history_logger = tf.keras.callbacks.CSVLogger(save_path + f'history_{net}.csv', separator=",", append=True)

# get the start time
st = time.time()

if net == 1:
  history = model.fit(train_dataset, epochs=1000, validation_data=valid_dataset, callbacks=[checkpoint, history_logger], verbose=1)
else:
  history = model.fit(train_dataset, epochs=1500, validation_data=valid_dataset, callbacks=[checkpoint, history_logger], verbose=1)

print('Success!')

# get the end time
et = time.time()
elapsed_time = et - st
elapsed_time = elapsed_time / 3600
print('Execution time:', elapsed_time, 'hours')

# memory usage
#print(tf.config.experimental.get_memory_info('GPU:0'))
