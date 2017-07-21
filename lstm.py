import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.keras import layers
from musigma import MUSIGMA_A
# json.dumps()


TIME_STEPS = 73
BATCH_SIZE = 18
INPUT_SIZE = 21
OUTPUT_SIZE = 66
CELL_SIZE = 64
LR = 0.001
EPOCH_NUM = 5

col = np.loadtxt('train.csv', delimiter=',')
features = col[:, 0:21]
labels = col[:, 21:87]
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
            self.lr = tf.placeholder(tf.float32)
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
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

    def add_input_layer(self, ):
        # masking_x = layers.Masking(mask_value=0.,)
        # masking_x = tf.contrib.keras.layers.Masking(mask_value = 0., input_shape = (-1,self.n_steps,self.input_size))
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size, ])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        temp = tf.layers.dense(l_in_y, 6, activation=None)
        self.l_in_y = tf.reshape(temp, [-1, self.n_steps, 6], name='2_3D')

    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=1.0)
        lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        lstm_cell2 = rnn.DropoutWrapper(cell=lstm_cell2,output_keep_prob=1.0)
        mlstm_cell = rnn.MultiRNNCell(cells=[lstm_cell,lstm_cell2],state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = mlstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            mlstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)
        temp = tf.reshape(self.cell_outputs, [-1, self.cell_size])
        self.l_out_x = tf.layers.dense(temp,self.cell_size,activation=None)

    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        # l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size])
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(self.l_out_x, Ws_out) + bs_out

    def compute_cost(self):
        with tf.name_scope('average_cost'):
            self.y_temp = tf.reshape(self.ys, [-1, self.output_size])
            self.mask = tf.abs(tf.sign(self.y_temp[:,25]),name='mask')
            # self.mask = [1]*(self.batch_size*self.n_steps-1)+[0.0]*1
            self.debug_value = tf.square(tf.subtract(self.pred, tf.reshape(self.ys, [-1, self.output_size])))
            temp = tf.sqrt(tf.reduce_sum(self.debug_value,1))
            self.masked_lossed = self.mask * temp
            # masked_lossed = temp
            self.cost = tf.reduce_mean(self.masked_lossed,
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
    init = tf.global_variables_initializer()
    sess.run(init)
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'
    for epoch in range(EPOCH_NUM):
        print('epoch'+str(epoch))
        
        state = sess.run([model.cell_init_state])
        LR = LR / 2
        print('LR'+str(LR))
        for step, (batchx, batchy) in enumerate(get_batch()):
            # print("x shape: ", batchx.shape)
            # print("y shape: ", batchy.shape)
            # print(batchx)
            # print(batchy)
            feed_dict = {
                model.xs: batchx,
                model.ys: batchy,
                model.cell_init_state: state,  # use last state as the initial state for this run
                model.lr: LR
            }
            _, cost, state, pred, debug_value, mask_info, mask, yy = sess.run(
                [model.train_op, model.cost, model.cell_final_state, model.pred, model.cell_outputs, model.masked_lossed, model.mask, model.y_temp],
                feed_dict=feed_dict)
            if step % 10 == 0:
                print(step // 10, 'training cost', cost)
                # print()
                # print(mask)
                # print(mask_info)
                # print(debug_value)
                # print(batchy)
                result = sess.run(merged, feed_dict)
                writer.add_summary(result, step)
            

        state = sess.run([model.cell_init_state])
        testx = col[2001:(2001+TIME_STEPS*BATCH_SIZE), 0:21]
        testy = col[2001:(2001+TIME_STEPS*BATCH_SIZE), 21:87]
        test_x = np.reshape(testx, (-1, TIME_STEPS, INPUT_SIZE))
        test_y = np.reshape(testy, (-1, TIME_STEPS, OUTPUT_SIZE))
        spred, loss = sess.run([model.pred, model.cost],
                               feed_dict={model.xs: test_x, model.ys: test_y, model.cell_init_state: state, model.lr: LR})
        # print(np.array(spred.tolist()).shape())
        print("test cost", loss)
    # predict_y = []
    # for i in range(2):
    # testx, testy = get_batch(1,0,1200)
    n_start = 0
    testx = col[n_start:(n_start+TIME_STEPS*BATCH_SIZE), 0:21]
    testy = col[n_start:(n_start+TIME_STEPS*BATCH_SIZE), 21:87]
    test_x = np.reshape(testx, (-1, TIME_STEPS, INPUT_SIZE))
    test_y = np.reshape(testy, (-1, TIME_STEPS, OUTPUT_SIZE))
    spred, loss = sess.run([model.pred, model.cost], feed_dict={model.xs: test_x, model.ys: test_y})
    # print(np.array(spred.tolist()).shape())
    for j in range(len(spred)):
        for i in range(0,66):
            spred[j][i]=spred[j][i]*MUSIGMA_A[i][1]+MUSIGMA_A[i][0]
    print(loss)
    spred = np.reshape(spred, (-1, 22, 3))
    predict_y = spred.tolist()
        
    xx = np.array(predict_y)
    # print(np.shape(xx))
    file = open('predict', 'w')
    count = 1
    for item in predict_y:
        file.write(str(count) + '\n')
        temp = json.dumps(item[0])
        file.write(temp + '\n')
        temp2 = json.dumps(item[1:])
        file.write(temp2 + '\n')
        count += 1
        # for i in range(len(pred)):
    
