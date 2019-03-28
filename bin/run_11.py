import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data





### Constants ###
num_inputs = 2
num_neurons = 3

### Placeholders ###
x0 = tf.placeholder(tf.float32, shape=[None, num_inputs])
x1 = tf.placeholder(tf.float32, shape=[None, num_inputs])

### Variables ###
Wx = tf.Variable(tf.random_normal(shape=[num_inputs,num_neurons])) # Attached to x0
Wy = tf.Variable(tf.random_normal(shape=[num_neurons,num_neurons]))
b = tf.Variable(tf.zeros([1,num_neurons]))

### Graph ###

y0 = tf.tanh(tf.matmul(x0,Wx) + b)
y1 = tf.tanh(tf.matmul(y0,Wy) + tf.matmul(x1,Wx) + b)

### init ###
init = tf.global_variables_initializer()

###Create Data###

# Timestamp 0
x0_batch = np.array([[0,1],[2,3],[4,5]])

# Timestamp 1
x1_batch = 100+np.array([[0,1],[2,3],[4,5]])


with tf.Session() as sess:

    sess.run(init)

    y0_output_vals, y1_output_vals = sess.run([y0,y1], feed_dict={x0:x0_batch,
                                                                  x1:x1_batch})

    print(y0_output_vals)

    print(y1_output_vals)