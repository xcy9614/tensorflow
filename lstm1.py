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
TIME_STEPS = 40
BATCH_SIZE = 16
INPUT_SIZE = 19
OUTPUT_SIZE = 66
CELL_SIZE = 128
LR = 0.001
EPOCH_NUM = 10

musigma=[[0.49176511941726975, 18.310384478652672],
[92.354439052097263, 6.9480504292986058],
[10.594724445475443, 36.511141891316321],
[-5.5480235862754244, 6.9610482024571549],
[45.209462779391274, 47.388416762210781],
[-0.10780900531709391, 8.6433423137490237],
[2.5635241499812467, 6.9604587873439963],
[52.643493950387317, 28.205747142593761],
[0.43030644518092281, 7.8625317010150493],
[0.046234131910453641, 3.7544724045008437],
[21.752407107566302, 11.393179182060246],
[-4.0852521183299721, 5.7258153337756124],
[-8.4482229059047604, 7.7654080605187961],
[44.32149523305921, 8.3643843364890724],
[-3.6743878716223066, 8.9127893112892771],
[-17.57800056668442, 17.057356510833337],
[28.919160144597612, 10.023684344341463],
[-2.4675778445260703, 15.048376857867828],
[-19.159828686433652, 18.283856587305518],
[29.414053529976762, 14.183758211448213],
[1.7627742735307352, 16.235531963966451],
[-18.921412095684595, 24.63340681761003],
[27.356143329264206, 19.634748147706194],
[5.8313863813639619, 17.995744588659232],
[-3.3481687220886989, 12.505992182499506],
[10.900590618537795, 13.526101154718676],
[-2.3918049623019657, 9.6619854376512322],
[-9.1743237506739757, 11.208761707832677],
[-25.445064681015388, 16.673420118076159],
[1.3861729954738762, 10.804485592913863],
[-10.667074655386527, 11.726099428363376],
[-59.149591990344391, 22.491856838983274],
[-1.9915279981104643, 11.598964976866441],
[-6.3321984040668209, 13.10534239870389],
[-74.700712887557756, 21.228829471885845],
[-6.6908084624752977, 13.021920121154118],
[-9.266974785993515, 13.588461683201942],
[-79.940744260134466, 16.262072279808624],
[4.6056322546560562, 12.858129064517071],
[3.1817660616502819, 15.421080584691861],
[44.591514298086032, 10.51871292584938],
[-2.4010827164004076, 10.324040585436084],
[15.467844933559501, 18.937063149711836],
[31.01584133574524, 12.671211111155973],
[0.91616274027069333, 13.923997468688423],
[18.764559151069829, 18.141540733627082],
[30.621639100480074, 16.28484802703813],
[5.9643896275059189, 15.727653853991022],
[18.147656686391546, 25.137823376108301],
[30.470783631489283, 23.261366473776146],
[11.407258656633424, 17.508564070856544],
[7.3333252590376565, 10.4428103826189],
[4.47243742763985, 17.313395695773497],
[-2.8303511745097323, 12.444854145296544],
[6.8077286593685704, 12.785098826835814],
[-43.705981702179585, 16.777738616072128],
[4.5396047343147359, 10.733803982744412],
[10.971370907508565, 11.226767580354876],
[-60.986415509441748, 20.873428278980526],
[-0.7424881652852855, 10.651835275112871],
[7.4146459597205761, 12.539990052239975],
[-75.874003495692918, 19.470198528615807],
[-6.5042994337601057, 12.408514166715042],
[10.555529751870361, 11.903764539291798],
[-80.785809535381645, 14.659348227794586],
[5.0142376784016953, 12.502855391222015]]

col = np.loadtxt('train.csv', delimiter=',')
features = col[:, 0:19]
labels = col[:, 19:85]
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
            _, cost, state, pred, debug_value = sess.run(
                [model.train_op, model.cost, model.cell_final_state, model.pred, model.cell_outputs],
                feed_dict=feed_dict)
            if step % 10 == 0:
                print(step // 10, 'training cost', cost)
                # print(debug_value)
                # print(batchy)
                result = sess.run(merged, feed_dict)
                writer.add_summary(result, step)
            BATCH_START += TIME_STEPS

        state = sess.run([model.cell_init_state])
        testx = col[2001:(2001+TIME_STEPS*BATCH_SIZE), 0:19]
        testy = col[2001:(2001+TIME_STEPS*BATCH_SIZE), 19:85]
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
    testx = col[n_start:(n_start+TIME_STEPS*BATCH_SIZE), 0:19]
    testy = col[n_start:(n_start+TIME_STEPS*BATCH_SIZE), 19:85]
    test_x = np.reshape(testx, (-1, TIME_STEPS, INPUT_SIZE))
    test_y = np.reshape(testy, (-1, TIME_STEPS, OUTPUT_SIZE))
    spred, loss = sess.run([model.pred, model.cost], feed_dict={model.xs: test_x, model.ys: test_y})
    # print(np.array(spred.tolist()).shape())
    for j in range(len(spred)):
        for i in range(0,66):
            spred[j][i]=spred[j][i]*musigma[i][1]+musigma[i][0]
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
    
