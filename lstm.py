#coding=utf-8                                                                                                                                                                                                                                                                 

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

def readMyFileFormat(file_name_queue):
    reader = tf.TextLineReader()
    key, value = reader.read(file_name_queue)

    record_defaults = [[0.] for i in range(81)]
    col = [[]for i in range(81)]
    col = tf.decode_csv(value,record_defaults=record_defaults)
    features = tf.stack(col[0:18])
    label = col[18:81]

    return features, label

def inputPipeLine(fileNames = ['train.csv'], batchSize = 4, numEpochs = None):
    file_name_queue = tf.train.string_input_producer(fileNames, num_epochs = numEpochs)
    example, label = readMyFileFormat(file_name_queue)
    example_batch, label_batch = tf.train.batch([example, label], batch_size = batchSize)
    return example_batch, label_batch

# featureBatch, labelBatch = inputPipeLine(['train.csv'], batchSize = 10)
# with tf.Session() as sess:
#     # Start populating the filename queue.                                                                                                                                                                                                                                    
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     for i in range(5):
#         print(featureBatch.eval())
#     coord.request_stop()
#     coord.join(threads)


# # Parameters
learning_rate = 0.001
training_iters = 500000
batch_size = 500 #批处理数量
# display_step = 10

# # Network Parameters
# n_input = 234
# n_steps = 1 # timesteps
# n_hidden = 128 # hidden layer num of features
# n_classes = 2 # MNIST total classes (0-9 digits)

# # tf Graph input
# x = tf.placeholder("float", [None, n_steps,n_input])
# y = tf.placeholder("float", [None, n_classes])

# # Define weights
# weights = {
#     'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
# }
# biases = {
#     'out': tf.Variable(tf.random_normal([n_classes]))
# }


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps,1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

# pred = RNN(x, weights, biases)

# # Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# # Evaluate model
# correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# # Initializing the variables
# init = tf.global_variables_initializer()
# batch_x, batch_y = inputPipeLine(["ContentNewLinkAllSample.csv"], batchSize = batch_size)
# # Launch the graph
# with tf.Session() as sess:
#     sess.run(init)
#     step = 1
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)

#     # Keep training until reach max iterations
#     while step * batch_size < training_iters:
#         # print(step)
        
#         # print(batch_x,batch_y)
#         # print(1)
#         batchx, batchy = sess.run([batch_x,batch_y])
#         # Reshape data to get 28 seq of 28 elements
#         batchx = batchx.reshape((batch_size, n_steps,n_input))
#         # Run optimization op (backprop)
#         # print(2)
#         sess.run(optimizer, feed_dict={x: batchx, y: batchy})
#         # print(3)
#         if step % display_step == 0:
#             # Calculate batch accuracy
#             acc = sess.run(accuracy, feed_dict={x: batchx, y: batchy})
#             # Calculate batch loss
#             loss = sess.run(cost, feed_dict={x: batchx, y: batchy})
#             print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
#                   "{:.6f}".format(loss) + ", Training Accuracy= " + \
#                   "{:.5f}".format(acc))
#         step += 1
#     print("Optimization Finished!")
#     coord.join(threads)
#     sess.close()
