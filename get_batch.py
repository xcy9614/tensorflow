#coding=utf-8                                                                                                                                                                                                                                                                 

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

batch_start = 0
batch_size = 2
n_steps = 3

def get_batch():
    global batch_size
    global batch_start
    global n_steps
    col = [[] for i in range(6)]
    col = np.loadtxt('getbatch2.csv', delimiter=',')
    # print(batch_start)
    batch_num = int(len(col)/batch_size)
    # print(batch_num)
    features = col[:,0:2]
    labels = col[:,2:6]
    features_batch = []
    labels_batch = []
    for i in range(batch_size):
        start_num = batch_num*i + batch_start
        print(start_num)
        if (start_num + n_steps) >= len(col):
            start_num = start_num % len(col)
        features_batch.append(features[start_num:(start_num+n_steps)])
        labels_batch.append(labels[start_num:(start_num+n_steps)])
        
    # print(features)
    # print(labels)

    return features_batch, labels_batch


for i in range(10):
    f,l = get_batch()
    print(f)
    print(l)
    
    batch_start += n_steps