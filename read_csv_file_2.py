#coding=utf-8                                                                                                                                                                                                                                                                 

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn


# Parameters
learning_rate = 0.001
# training_iters = 500000
batch_size = 16 #批处理数量
batch_start = 0

# Network Parameters
n_input = 18
n_output = 63
n_steps = 48 # timesteps
n_hidden = 128 # hidden layer num of features
col = [[] for i in range(81)]
col = np.loadtxt('train.csv', delimiter=',')
# final_state = lstm_cell.zero_state(batch_size, dtype = tf.float32)

def get_batch():
    global batch_size
    global batch_start
    global n_steps
    global col
    # print(len(col))
    batch_num = int(len(col)/batch_size)
    features = col[:,0:18]
    labels = col[:,18:81]
    features_batch = []
    labels_batch = []
    for i in range(batch_size):
        start_num = (batch_num*i + batch_start) % len(col)
        # print(start_num)
        if (start_num + n_steps) >= len(col):
            start_num = (start_num+n_steps) % len(col)
        # print(start_num)
        features_batch.append(features[start_num:(start_num+n_steps)])
        labels_batch.append(labels[start_num:(start_num+n_steps)])
    # shape => (batch_size, n_steps, n_input/n_output)
    return features_batch, labels_batch

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
    return (tf.matmul(outputs, weights['out']) + biases['out']), states

pred, final_state = RNN(x, weights, biases)
# print(pred)
# Define loss and optimizer
cost = tf.reduce_mean(tf.square(tf.subtract(pred,tf.reshape(y,[-1,n_output]))))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# # Evaluate model
# correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
# saver = tf.train.Saver(tf.global_variables())
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1

    # Keep training until reach max iterations
    while step < 1000:
        # batchx, batchy = sess.run([batch_x,batch_y])
        # print(batchx.eval())
        # print(2)
        # Reshape data to get 28 seq of 28 elements
        # batchx = batchx.reshape([batch_size, n_steps, n_input])
        # batchy = batchy.reshape([batch_size, n_steps, n_output])
        # if step == 1:
        #     feed_dict = {x: batchx, y: batchy}
        # else:
        #     feed_dict = {x: batchx, y: batchy, }
        batchx, batchy = get_batch()
        sess.run(optimizer, feed_dict={x: batchx, y: batchy})
        # print(3)
        if step % 100 == 0:
            # Calculate batch accuracy
            # acc = sess.run(accuracy, feed_dict={x: batchx, y: batchy})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batchx, y: batchy})
            print(loss)
            # saver.save(sess,'train.model')
        step += 1
        batch_start += n_steps

    print("Optimization Finished!")

    # test_x = col[:,0:18]
    # test_y = col[:,18:81]
    # test_x = test_x[0:10140]
    # test_y = test_y[0:10140]
    # test_x = tf.reshape(test_x,[-1, n_steps, n_input])
    # test_y = tf.reshape(test_y,[-1, n_steps, n_output])
    # testx, testy = sess.run([test_x,test_y])
    batch_start = 0
    testx, testy = get_batch()
    # np.savetxt('testx.csv', testx)
    # np.savetxt('testy.csv', testy)
    print(testx)
    loss = sess.run(cost, feed_dict = {x:testx, y:testy})
    pred = sess.run(pred, feed_dict = {x:testx, y:testy})
    print('&&&&&&&&&&&&&&&&&&&&&&'+str(batch_start))
    print(loss)
    # print(pred)
    np.savetxt('predicty.csv',pred,delimiter=',')
    sess.close()
