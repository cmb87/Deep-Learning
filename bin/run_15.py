import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

n_inputs = 28
n_outputs = 10
n_steps = 28
n_neurons = 60
nepoches = 3000
n_layers = 2

### Placeholders ###
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])
keep_prob = tf.placeholder_with_default(0.8,shape=())


cells = [tf.contrib.rnn.GRUCell(num_units=n_neurons) for layer in range(n_layers)]
cells_drop = [tf.contrib.rnn.DropoutWrapper(cell,input_keep_prob=keep_prob) for cell in cells]

multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells_drop)

# 'state' is a tensor of shape [batch_size, cell_state_size] The final state

# 'rnn_outputs' is a tensor of shape [batch_size, max_time, 256]
# 'states' is a N-tuple where N is the number of LSTMCells containing a
# tf.contrib.rnn.LSTMStateTuple for each cell
rnn_outputs, states =tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

print(states[0].get_shape()) #==> (?, 60)
print(rnn_outputs.get_shape()) #==> (?, 28, 60)

### Loss and optimizer ###
logits = tf.layers.dense(states[0], n_outputs)
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y))

optimize = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

prediction = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

### Start Session ###
loss_log, acc_log = [], []
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for ep in range(nepoches):

        X_batch, y_batch = mnist.train.next_batch(30)

        loss, _, acc = sess.run([cross_entropy,optimize,accuracy], feed_dict={X:X_batch.reshape(-1,28,28),
                                                                              y:y_batch,
                                                                              keep_prob:0.8})
        loss_log.append(loss)
        acc_log.append(acc)

        if ep % 100 == 0:

            print("Epoche: {} LossTrain: {:.6f} Accuracy: {:.6f}".format(ep, loss, acc))

    plt.plot(range(nepoches), loss_log, 'b-')
    plt.plot(range(nepoches), acc_log,'r-')
    plt.show()