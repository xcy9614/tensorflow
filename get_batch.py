#coding=utf-8                                                                                                                                                                                                                                                                 

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

input_num = 10152
# batch_size = 4
# batch_num = int(input_num/batch_size)

def readMyFileFormat(file_name_queue):
    reader = tf.TextLineReader()
    key, value = reader.read(file_name_queue)

    record_defaults = [[0.] for i in range(6)]
    col = [[]for i in range(6)]
    col = tf.decode_csv(value,record_defaults=record_defaults)
    features = tf.stack(col[0:3])
    label = tf.stack(col[3:6])

    return features, label

def inputPipeLine(fileNames = ['getbatch.csv'], batchSize = 4, numEpochs = None):
    global input_num
    batch_num = int(input_num/batchSize)
    file_name_queue = tf.train.string_input_producer(fileNames, num_epochs = numEpochs)
    example, label = readMyFileFormat(file_name_queue)
    print(example)
    # example = tf.transpose(example)
    # label = tf.transpose(label)
    # print(example)
    # example_batch= example[0:3]
    # label_batch = label[0:3]
    # print('******************************')
    # print(example_batch)
    example_batch, label_batch = tf.train.batch([example, label], batch_size = batchSize)
    # print(example_batch)
    return example_batch, label_batch

featureBatch, labelBatch = inputPipeLine(['getbatch2.csv'], batchSize = 4)
featureBatch1, labelBatch1 = inputPipeLine(['getbatch.csv'],batchSize = 4)
with tf.Session() as sess:
    # Start populating the filename queue.                                                                                                                                                                                                                                    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(7):
        example, label = sess.run([featureBatch, labelBatch])
        example1, label1 = sess.run([featureBatch1, labelBatch1])
        print(tf.concat([example,example1],0))
        # print(example1)

    coord.request_stop()
    coord.join(threads)
    sess.close()

