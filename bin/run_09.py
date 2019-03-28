import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
#X, y = mnist.train.next_batch(batchSize)


batchSize = 100
nepoches = 1000

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
x = tf.placeholder(tf.float32, shape=[None,784])
y_true = tf.placeholder(tf.float32, shape=[None,10])


### Layers ##
x_image = tf.reshape(x, shape=[-1,28,28,1]) # Reshape flatten out array into image again

convo_1 = convolutional_layer(x_image, shape=[5,5,1,32])
# First two dimensions are the patch size (5x5)
# Third: Color channel
# Fourth: Features going to compute (output)
convo_1_pooling = max_pooling_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling, shape=[5,5,32,64])
convo_2_pooling = max_pooling_2by2(convo_2)

print(convo_1.get_shape())
print(convo_1_pooling.get_shape())
print(convo_2.get_shape())
print(convo_2_pooling.get_shape())

# convo_1 => (?, 28, 28, 32)
# convo_1_pooling => (?, 14, 14, 32)
# convo_2 => (?, 14, 14, 64)
# convo_2_pooling => (?, 7, 7, 64)


convo_2_flat = tf.reshape(convo_2_pooling,[-1, 7*7*64])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

### Dropout ###
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

### Prediction ###
y_pred = normal_full_layer(full_one_dropout, 10)

### Loss ###
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

### Optimizer ###
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

### Init ###
init = tf.global_variables_initializer()

steps = 5000

loss_log = []
acc_log = []
acc_it_log =[]

with tf.Session() as sess:
    sess.run(init)

    for i in range(steps):
        batch_x, batch_y = mnist.train.next_batch(50)
        _, ce = sess.run([optimizer,cross_entropy], feed_dict={x: batch_x,
                                                              y_true: batch_y,
                                                              hold_prob: 0.5})
        loss_log.append(ce)

        if i%100 == 0:
            print("On Step: {}".format(i))
            print("Accuracy:")

            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))
            accuracy = sess.run(acc, feed_dict={x: mnist.test.images,
                                                y_true: mnist.test.labels,
                                                hold_prob: 1.0})
            acc_log.append(accuracy)
            acc_it_log.append(i)
            print(accuracy)
            print('\n')

    plt.plot(range(steps), loss_log, 'r-')
    plt.plot(acc_it_log, acc_log, 'b-')
    plt.show()