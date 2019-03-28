import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data

# data = unpickle(trainData[0])
# dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
# data[b'data'] ==> (10000, 3072)
# data[b'labels']==> 10000, list range 0-9
# data[b'batch_label'] ==> b'training batch 1 of 5'

class CifarData():
    def __init__(self):

        self.trainData = ['../data/cifar-10-batches-py/data_batch_1', '../data/cifar-10-batches-py/data_batch_2',
                          '../data/cifar-10-batches-py/data_batch_3', '../data/cifar-10-batches-py/data_batch_4',
                          '../data/cifar-10-batches-py/data_batch_5']

        self.testData = ['../data/cifar-10-batches-py/test_batch']

        self.trainBatchCtr=0

    def nextBatch(self, batchSize, shuffle=False):

        if shuffle:
            self.trainBatchCtr = np.random.randint(0,len(self.trainData)-1)
        else:
            self.trainBatchCtr+=1
            self._checkBatchCtr()
        dataRaw = CifarData.unpickle(self.trainData[self.trainBatchCtr])
        idx = np.random.randint(0,10000, batchSize)
        # Start from first batch
        X = dataRaw[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8") / 255.0
        y_true = CifarData.onehottrue(dataRaw[b'labels'])

        return X[idx,:,:,:],y_true[idx,:]


    def testBatch(self):
        dataRaw = CifarData.unpickle(self.testData[0])
        # Start from first batch
        X = dataRaw[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8") / 255.0
        y_true = CifarData.onehottrue(dataRaw[b'labels'])
        return X,y_true

    def _checkBatchCtr(self):
        if self.trainBatchCtr > len(self.trainData)-1:
            self.trainBatchCtr = 0

    @staticmethod
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    @staticmethod
    def onehottrue(rawLabels):
        imax, jmax  = len(rawLabels), max(rawLabels)+1
        y_true = np.zeros((imax,jmax))
        for n,hot in zip(range(imax),rawLabels):
            y_true[n,hot] = 1
        return y_true





### Init weights ###
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init_random_dist)

## Init Bias ###
def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

### CONV2D ###
def conv2d(x, W):
    # x = [batch, H, W, Channels]
    # W = Kernel, [filter H, filter W, Channels IN, Channels OUT]
    # Padding SAME == Keep the same size as the input
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

### Pooling ###
def max_pooling_2by2(x):
    # x = [batch, H, W, Channels]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

### Convulutional Layer ###
def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W)+b)

### Normal layer ###
def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer,W) + b




### Placeholders ###
x = tf.placeholder(tf.float32, shape=[None,32,32,3])
y_true = tf.placeholder(tf.float32, shape=[None,10])

### Model ###
### Conv1 ###
convo_1 = convolutional_layer(x, [5,5,3,32])
convo_1_maxpool = max_pooling_2by2(convo_1)

### Conv2 ###
convo_2 = convolutional_layer(convo_1_maxpool, [5,5,32,64])
convo_2_maxpool = max_pooling_2by2(convo_2)

### Dense regions start here ###
flattened = tf.reshape(convo_2_maxpool,(-1,8*8*64))

### Dense1 ###
dense1 = tf.nn.relu(normal_full_layer(flattened,1024))

### Dropout ###
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(dense1, keep_prob=hold_prob)

### Dense2 ###
logits = normal_full_layer(full_one_dropout,10)

### Cross Entropy ###
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_true)

### optimizer ###
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

### Init ###
init = tf.global_variables_initializer()

### Initialize Data object ###

cifar = CifarData()
batchSize = 100
steps = 5000

loss_log = []
acc_log = []
acc_it_log =[]

with tf.Session() as sess:
    sess.run(init)

    for i in range(steps):
        batch_x, batch_y = cifar.nextBatch(batchSize)
        _, ce = sess.run([optimizer,cross_entropy], feed_dict={x: batch_x,
                                                              y_true: batch_y,
                                                              hold_prob: 0.5})
        loss_log.append(ce)

        if i%100 == 0:
            print("On Step: {}".format(i))
            print("Accuracy:")

            matches = tf.equal(tf.argmax(logits,1),tf.argmax(y_true,1))
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))

            test_x, test_y = cifar.testBatch()

            accuracy = sess.run(acc, feed_dict={x: test_x,
                                                y_true: test_y,
                                                hold_prob: 1.0})
            acc_log.append(accuracy)
            acc_it_log.append(i)
            print(accuracy)
            print('\n')

    plt.plot(range(steps), loss_log, 'r-')
    plt.plot(acc_it_log, acc_log, 'b-')
    plt.show()






