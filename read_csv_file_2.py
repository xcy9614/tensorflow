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
    label = tf.stack(col[18:81])

    return features, label

def inputPipeLine(fileNames = ['train.csv'], batchSize = 4, numEpochs = None):
    file_name_queue = tf.train.string_input_producer(fileNames, num_epochs = numEpochs)
    example, label = readMyFileFormat(file_name_queue)
    example_batch, label_batch = tf.train.batch([example, label], batch_size = batchSize)
    return example_batch, label_batch

# featureBatch, labelBatch = inputPipeLine(["ContentNewLinkAllSample.csv"], batchSize = 4)
# with tf.Session() as sess:
#     # Start populating the filename queue.                                                                                                                                                                                                                                    
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)

#    # Retrieve a single instance:                                                                                                                                                                                                                                             
#     try:
#         #while not coord.should_stop():                                                                                                                                                                                                                                       
#         # while True:
#             example, label = sess.run([featureBatch, labelBatch])
#             print(example)
#     except tf.errors.OutOfRangeError:
#         print ('Done reading')
#     finally:
#         coord.request_stop()

#     coord.join(threads)
#     sess.close()


# Parameters
learning_rate = 0.001
# training_iters = 500000
batch_size = 16 #批处理数量
# display_step = 10

# Network Parameters
n_input = 18
n_output = 63
n_steps = 30 # timesteps
n_hidden = 128 # hidden layer num of features


# tf Graph input
# shape (, 30, 18)
x = tf.placeholder("float", [None, n_steps, n_input])

y = tf.placeholder("float", [None, n_steps, n_output])

# Define weights
weights = {
    # shape (18, 128)
    'in': tf.Variable(tf.random_normal([n_input,n_hidden])),
    # shape (128, 63)
    'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
}
biases = {
    # shape (128, )
    'in': tf.Variable(tf.constant(0.1, shape = [n_hidden,])),
    # shape (63, )
    'out': tf.Variable(tf.constant(0.1, shape = [n_output,]))
}


def RNN(x, weights, biases):
    # shape => (batch_size * timesteps, input_size)
    x = tf.reshape(x, [-1, n_input])
    # shape => (batch_size * timesteps, n_hidden)
    x_in = tf.matmul(x, weights['in']) + biases['in']
    # shape => (batch_size, timesteps, n_hidden)
    x_in = tf.reshape(x_in, [-1, n_steps, n_hidden])
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype = tf.float32)
    # Get lstm cell output
    # outputs => (batch_size, timesteps, n_hidden)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x_in,initial_state=init_state,time_major = False, dtype=tf.float32)
    # outputs => (batch_size * timesteps, n_hidden)
    outputs = tf.reshape(outputs,[-1,n_hidden])
    # Linear activation, using rnn inner loop last output
    # return => (batch_size * timesteps, n_output)
    return tf.matmul(outputs, weights['out']) + biases['out']

pred = RNN(x, weights, biases)
# print(pred)
# Define loss and optimizer
cost = tf.reduce_mean(tf.square(tf.subtract(pred,tf.reshape(y,[-1,n_output]))))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# # Evaluate model
# correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
batch_x, batch_y = inputPipeLine(["train.csv"], batchSize = batch_size*n_steps)
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Keep training until reach max iterations
    while step < 1000:
        # print(step)
         
        # print(batch_x,batch_y)
        # print(1)
        # if step == 1 or step == 2 or step == 3:
        # print(batch_x.eval())
        # print(batch_x, batch_y)
        batchx, batchy = sess.run([batch_x,batch_y])
        # print(batchx.eval())
        # print(2)
        # Reshape data to get 28 seq of 28 elements
        batchx = batchx.reshape([batch_size, n_steps, n_input])
        batchy = batchy.reshape([batch_size, n_steps, n_output])
        # Run optimization op (backprop)
        # print(2)
        # print('******************************')
        # print(batch_x, batchy)
        # if step == 1:
        #     feed_dict = {x: batchx, y: batchy}
        # else:
        #     feed_dict = {x: batchx, y: batchy, }
                
        sess.run(optimizer, feed_dict={x: batchx, y: batchy})
        # print(3)
        if step % 10 == 0:
            # Calculate batch accuracy
            # acc = sess.run(accuracy, feed_dict={x: batchx, y: batchy})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batchx, y: batchy})
            print(loss)
            # print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                #   "{:.6f}".format(loss) + ", Training Accuracy= " + \
                #   "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    coord.request_stop()
    coord.join(threads)
    # sess.close()
