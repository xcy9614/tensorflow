#加载包
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
# import tensorflow.contrib.learn.datasets.base as base
# import tf.contrib.learn.python.learn.datasets.base as base


# 数据集名称，数据集要放在你的工作目录下
IRIS_TRAINING = "ContentNewLinkAllSample.csv"
# IRIS_TEST = "iris_test.csv"

# 数据集读取，训练集和测试集
training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
print(training_set)
# test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#     filename=IRIS_TEST,
#     target_dtype=np.int,
#     features_dtype=np.float32)

# 特征
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=234)]

