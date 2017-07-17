# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.

Run this script on tensorflow r0.10. Errors appear when using lower versions.
"""
import json

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

# json.dumps()

BATCH_START = 0
TIME_STEPS = 30
BATCH_SIZE = 16
INPUT_SIZE = 18
OUTPUT_SIZE = 66
CELL_SIZE = 100
LR = 0.001
EPOCH_NUM = 14

col = np.loadtxt('train.csv', delimiter=',')
features = col[:, 0:18]
labels = col[:, 18:84]
length = len(col) // BATCH_SIZE
training_features = np.zeros([BATCH_SIZE, length, INPUT_SIZE])
training_labels = np.zeros([BATCH_SIZE, length, OUTPUT_SIZE])
for i in range(BATCH_SIZE):
    training_features[i] = features[i * length:(i + 1) * length]
    training_labels[i] = labels[i * length:(i + 1) * length]
print(training_labels.shape)
print(training_features.shape)
print(labels.shape)
print(features.shape)


def get_batch():
    start = 0
    while start + TIME_STEPS < training_features.shape[1]:
        yield training_features[:, start: start + TIME_STEPS, :], training_labels[:, start: start + TIME_STEPS, :]
        start += TIME_STEPS


class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
            # self.keep_prob = tf.placeholder(tf.float32, name='keep-prob')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_input_layer(self, ):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size, ])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=1.0)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size])
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

    def compute_cost(self):
        with tf.name_scope('average_cost'):
            self.debug_value = tf.square(tf.subtract(self.pred, tf.reshape(self.ys, [-1, self.output_size])))
            self.cost = tf.reduce_mean(tf.sqrt(self.debug_value),
                                       name='average_cost')
            tf.summary.scalar('cost', self.cost)

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1., )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


if __name__ == '__main__':
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12

    init = tf.global_variables_initializer()
    sess.run(init)
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'
    for epoch in range(EPOCH_NUM):
        state = sess.run([model.cell_init_state])
        for step, (batchx, batchy) in enumerate(get_batch()):
            # print("x shape: ", batchx.shape)
            # print("y shape: ", batchy.shape)
            # print(batchx)
            # print(batchy)
            feed_dict = {
                model.xs: batchx,
                model.ys: batchy,
                model.cell_init_state: state  # use last state as the initial state for this run
            }
            _, cost, state, pred, debug_value = sess.run(
                [model.train_op, model.cost, model.cell_final_state, model.pred, model.cell_outputs],
                feed_dict=feed_dict)
            if step % 10 == 0:
                print(step // 10, 'training cost', cost)
                # print(debug_value)
                result = sess.run(merged, feed_dict)
                writer.add_summary(result, step)
            BATCH_START += TIME_STEPS

        state = sess.run([model.cell_init_state])
        testx = col[2001:(2001+TIME_STEPS*BATCH_SIZE), 0:18]
        testy = col[2001:(2001+TIME_STEPS*BATCH_SIZE), 18:84]
        test_x = np.reshape(testx, (-1, TIME_STEPS, INPUT_SIZE))
        test_y = np.reshape(testy, (-1, TIME_STEPS, OUTPUT_SIZE))
        spred, loss = sess.run([model.pred, model.cost],
                               feed_dict={model.xs: test_x, model.ys: test_y, model.cell_init_state: state})
        # print(np.array(spred.tolist()).shape())
        print("test cost", loss)
    predict_y = []
    # for i in range(2):
    # testx, testy = get_batch(1,0,1200)
    testx = col[2001:(2001+TIME_STEPS*BATCH_SIZE), 0:18]
    testy = col[2001:(2001+TIME_STEPS*BATCH_SIZE), 18:84]
    test_x = np.reshape(testx, (-1, TIME_STEPS, INPUT_SIZE))
    test_y = np.reshape(testy, (-1, TIME_STEPS, OUTPUT_SIZE))
    spred, loss = sess.run([model.pred, model.cost], feed_dict={model.xs: test_x, model.ys: test_y})
    # print(np.array(spred.tolist()).shape())
    print(loss)
    spred = np.reshape(spred, (-1, 22, 3))
    predict_y.extend(spred.tolist())

    xx = np.array(predict_y)
    # print(np.shape(xx))
    file = open('predict', 'w')
    count = 1
    for item in predict_y:
        file.write(str(count) + '\n')
        temp = json.dumps(item[0])
        file.write(temp + '\n')
        temp = json.dumps(item[1:])
        file.write(temp + '\n')
        count += 1
        # for i in range(len(pred)):
