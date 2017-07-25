import tensorflow as tf
import numpy as np
from lstm import LSTMRNN
from musigma import *
import json
TIME_STEPS = 73
BATCH_SIZE = 18
INPUT_SIZE = 21
OUTPUT_SIZE = 66
CELL_SIZE = 64
LR = 0.001
EPOCH_NUM = 200
col = np.loadtxt('show.csv', delimiter=',')
features = col[:, 0:21]
print(len(col))
length = len(col) // BATCH_SIZE
training_features = np.zeros([BATCH_SIZE, length, INPUT_SIZE])
for i in range(BATCH_SIZE):
    training_features[i] = features[i * length:(i + 1) * length]

# init = tf.global_variables_initializer()
model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
spred=[]
for i in range(0,len(col)//(TIME_STEPS*BATCH_SIZE)):
    n_start = i*TIME_STEPS*BATCH_SIZE
    testx = col[n_start:(n_start+TIME_STEPS*BATCH_SIZE), 0:21]
    test_x = np.reshape(testx, (-1, TIME_STEPS, INPUT_SIZE))
    testy=np.zeros([TIME_STEPS*BATCH_SIZE,66])
    test_y = np.reshape(testy, (-1, TIME_STEPS, OUTPUT_SIZE))
    saver = tf.train.Saver() # 声明tf.train.Saver类用于保存模型
    with tf.Session() as sess:
        saver.restore(sess, "save/model.ckpt") 
        spred_t, loss = sess.run([model.pred, model.cost], feed_dict={model.xs: test_x, model.ys: test_y})
    print(loss)
    if len(spred) == 0:
        spred=spred_t
    else:
        spred=np.concatenate((spred,spred_t),axis=0)
# print(np.array(spred.tolist()).shape())
for j in range(len(spred)):
    for i in range(0,66):
        spred[j][i]=spred[j][i]*MUSIGMA_A[i][1]+MUSIGMA_A[i][0]
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