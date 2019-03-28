import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(101)
tf.set_random_seed(101)



x_data = np.linspace(0,10,10) + np.random.uniform(-1.5, 1.5, 10)
X_data = np.vstack((x_data,np.ones(10))).T

y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)




X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])
w = tf.Variable(tf.random_normal([2,1]))

#w = tf.get_variable("tf_var_initialized_from_np",initializer=np.asarray([1, 0]))

y_hat = tf.matmul(X, w)

loss = tf.reduce_mean(tf.square(y_hat-Y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.05).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    plt.plot(x_data, y_label, 'o')
    for it in range(20):
        error_sum = 0.0

        for xd, yd in zip(x_data, y_label):
            yhat, error, weights, _ = sess.run([y_hat, loss, w, optimizer], feed_dict={X: np.asarray([xd, 1]).reshape(1,2),
                                                                                       Y: np.asarray([yd]).reshape(1,1)})
            error_sum +=error/10


        print("Iteration {} Loss {:.2f} Weights {}".format(it, error_sum, weights))

    yhat, error, weights, _ = sess.run([y_hat, loss, w, optimizer], feed_dict={X: X_data,
                                                                               Y: y_label.reshape(10, 1)})

    plt.plot(x_data,yhat,'r-')
    plt.show()