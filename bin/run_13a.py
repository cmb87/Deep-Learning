import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data

# The static_rnn() function creates an unrolled RNN network by chaining
# cells. The following code creates the exact same model as the previous one
# (run_13.py):

n_inputs = 3
n_neurons = 5

# First we create the input placeholders, as before.
X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

# Then we create a BasicRNNCell, which you can think of as a factory that
# creates copies of the cell to build the unrolled RNN (one for each time step).
# Then we call static_rnn(), giving it the cell factory and the input tensors,
# and telling it the data type of the inputs (this is used to create
# the initial state matrix, which by default is full of zeros).
# The static_rnn() function calls the cell factory’s __call__() function
# once per input, creating two copies of the cell (each containing a layer
# of five recurrent neurons), with shared weights and bias terms, and
# it chains them just like we did earlier. The static_rnn() function
# returns two objects. The first is a Python list containing
# the output tensors for each time step. The second is a tensor containing
# the final states of the network. When you are using basic cells,
# the final state is simply equal to the last output.

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1],
                                                dtype=tf.float32)
Y0, Y1 = output_seqs

init = tf.global_variables_initializer()


# This mini-batch contains four instances, each with an input sequence
# composed of exactly two inputs. At the end, Y0_val and Y1_val contain
# the outputs of the network at both time steps for all neurons and all
# instances in the mini-batch:
# Mini-batch:        instance 0,instance 1,instance 2,instance 3
X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})


print(Y0_val)

# That wasn’t too hard, but of course if you want to be able to run an RNN
# over 100 time steps, the graph is going to be pretty big. Now let’s look
# at how to create the same model using TensorFlow’s RNN operations.

