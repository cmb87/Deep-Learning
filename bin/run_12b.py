import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data



class TimeSeriesData():
    def __init__(self, n_points, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.n_points = n_points
        self.resolution = (xmax-xmin)/n_points

        self.x_data = np.linspace(xmin, xmax, n_points)
        self.y_true = np.sin(self.x_data)

    def ret_true(self, x_series):
        """
        Convinience Function
        :param x_series:
        :return:
        """
        return np.sin(x_series)

    def next_batch(self, batchSize, steps, return_batch_ts=False):

        # Grab a random starting point for each batch of data
        rand_start = np.random.rand(batchSize,1)

        # Convert to on time series
        ts_start = rand_start*(self.xmax-self.xmin - (steps*self.resolution))

        # Create batch time series on the x-axis
        batch_ts = ts_start + np.arange(0.0,steps+1)*self.resolution

        # Create the Y data for the time series x axis from previous step
        y_batch = np.sin(batch_ts)

        # Fromatting for RNN
        if return_batch_ts:
            return y_batch[:,:-1].reshape(-1, steps,1), y_batch[:,1:].reshape(-1,steps,1), batch_ts
        else:
            return y_batch[:,:-1].reshape(-1, steps,1), y_batch[:,1:].reshape(-1,steps,1)


tsData = TimeSeriesData(250, 0,10)
num_time_steps = 30

y1,y2,ts = tsData.next_batch(1,num_time_steps,True)

if False:
    plt.plot(tsData.x_data, tsData.y_true, label='sin(t)')
    plt.plot(ts.flatten()[1:],y2.flatten(),'*', label='Single training instance')
    plt.plot(ts.flatten()[:-1],y1.flatten(),'.')

    plt.legend()
    plt.tight_layout()
    plt.show()

### Training Data ###

train_inst = np.linspace(5,5+tsData.resolution*(num_time_steps+1), num_time_steps+1)

if False:
    plt.title('A training instance')
    plt.plot(train_inst[:-1], tsData.ret_true(train_inst[:-1]),'bo', markersize=15, alpha=0.5, label="TrainingInstance")
    plt.plot(train_inst[1:], tsData.ret_true(train_inst[1:]),'ko', markersize=7, alpha=0.5, label="Target")

    plt.legend()
    plt.show()

### Build the model ###

num_inputs = 1
num_neurons = 100
num_outputs = 1
learning_rate = 0.0001
num_iterations = 2000
batchSize = 1

### Placeholders ###

X = tf.placeholder(tf.float32,shape=[None,num_time_steps,num_inputs])
y = tf.placeholder(tf.float32,shape=[None,num_time_steps,num_outputs])

### RNN Build the model ###

#aux = tf.contrib.rnn.BasicRNNCell(num_units=num_neurons,activation=tf.nn.relu)
aux = tf.contrib.rnn.GRUCell(num_units=num_neurons,activation=tf.nn.relu)

cell = tf.contrib.rnn.OutputProjectionWrapper(aux, output_size=num_outputs) # Just one output

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

### Loss ###
loss = tf.reduce_mean(tf.square(outputs-y))

### optimizer ###
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

### Init ###
init = tf.global_variables_initializer()

saver = tf.train.Saver()


### Predict future ###
with tf.Session() as sess:
    saver.restore(sess, "./190327_rnn_timeSeriesModel")

    zero_seq_seed = [0.0 for i in range(num_time_steps)]



    X_new = np.sin(train_inst[:-1].reshape(-1,num_time_steps,num_inputs))
    y_pred = sess.run(outputs, feed_dict={X:X_new})

plt.title("Testing the Model")

# Training Instance
plt.plot(train_inst[:-1],np.sin(train_inst[:-1]),"bo", markersize=15, alpha=0.5, label="Training Instance")


# Target to Predict
plt.plot(train_inst[1:],np.sin(train_inst[1:]),"ko", markersize=10, alpha=0.5, label="target")


# Models prediction
plt.plot(train_inst[1:],y_pred[0,:,0],"r.", markersize=10, alpha=0.5, label="Predictions")

plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.show()



