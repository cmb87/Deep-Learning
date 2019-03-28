import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

batchSize = 100
nepoches = 1000

def xavier_init(shape):
    return tf.random_normal(shape=shape,mean=0.0,stddev=1.0/shape[0])


### Build Graph ###

Xph = tf.placeholder(tf.float32,shape=[None, 28*28])
Yph = tf.placeholder(tf.float32,shape=[None, 10])
Whl = tf.Variable(initial_value=xavier_init([28*28, 10]))
bhl = tf.Variable(initial_value=tf.zeros(10))

logits = (tf.matmul(Xph,Whl) + bhl)

### Define loss ###
probs = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Yph)
loss = tf.reduce_mean(probs)

### Define training ###
train = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)

### Define accuracy ###
#print("----------")
#print(logits.get_shape())
#print(Yph.get_shape())
#print(probs.get_shape())
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(Yph,1)),tf.float32))

### Start Session ###
loss_log, acc_log = [], []
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for ep in range(nepoches):

        X, y = mnist.train.next_batch(batchSize)

        acc, error,_ = sess.run([accuracy,loss,train], feed_dict={Xph:X,
                                                                  Yph:y})

        loss_log.append(error)
        acc_log.append(acc)
        print("Epoche: {} Accuracy: {:.4} Loss: {:.4}".format(ep, acc ,error))

    plt.plot(range(nepoches), loss_log, 'b-')
    plt.plot(range(nepoches), acc_log,'r-')
    plt.show()