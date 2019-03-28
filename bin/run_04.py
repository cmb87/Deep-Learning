import tensorflow as tf
import numpy as np
np.random.seed(101)
tf.set_random_seed(101)



nfeatures = 10
n_dense_neurons = 3

x = tf.placeholder(tf.float32,(None, nfeatures))

W = tf.Variable(tf.random_normal([nfeatures, n_dense_neurons]))
b = tf.Variable(tf.zeros([n_dense_neurons]))

z = tf.matmul(x, W) + b
a = tf.sigmoid(z)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    layer_out = sess.run(a, feed_dict={x:np.random.random([1,nfeatures])})

    print(layer_out)
