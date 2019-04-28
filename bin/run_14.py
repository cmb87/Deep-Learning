import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



class MilkData():
    def __init__(self):

        self.df = pd.read_csv('../data/04-Recurrent-Neural-Networks/monthly-milk-production.csv')
        self.df['Month'] = pd.to_datetime(self.df['Month'])
        # df.columns ==> Index(['Month', 'Milk Production'], dtype='object')

        # Split data into test and training set
        split_date = pd.datetime(1973, 1, 1)
        self.df_training = self.df.loc[self.df['Month'] <= split_date]
        self.df_test = self.df.loc[self.df['Month'] > split_date]

        # Standardize data
        self.min = np.asarray(self.df_training['Milk Production'][:].min())
        self.max = np.asarray(self.df_training['Milk Production'][:].max())

        self.df_training['Milk Production'] = (self.df_training['Milk Production']-self.min)/ (self.max-self.min)
        self.df_test['Milk Production'] = (self.df_test['Milk Production']-self.min) / (self.max-self.min)


    def next_batch(self, nsteps, returnTime=False):

        ir = np.random.randint(0,self.df_training.shape[0]-nsteps-1)

        x = np.asarray(self.df_training['Milk Production'][ir:ir+nsteps]).reshape(-1,nsteps,1)
        y = np.asarray(self.df_training['Milk Production'][ir+1:ir+nsteps+1]).reshape(-1,nsteps,1)

        tx = self.df_training['Month'][ir:ir+nsteps]
        ty = self.df_training['Month'][ir+1:ir+nsteps+1]

        if returnTime:
            return x,y,tx,ty
        else:
            return x,y


    def get_testData(self, nsteps, returnTime=False):

        ir = np.random.randint(0, self.df_test.shape[0] - nsteps - 1)

        x = np.asarray(self.df_test['Milk Production'][ir:ir+nsteps]).reshape(-1,nsteps,1)
        y = np.asarray(self.df_test['Milk Production'][ir+1:ir+nsteps+1]).reshape(-1,nsteps,1)

        tx = self.df_test['Month'][ir:ir+nsteps]
        ty = self.df_test['Month'][ir+1:ir+nsteps+1]

        if returnTime:
            return x,y,tx,ty
        else:
            return x,y




### Load Data Object ####
data = MilkData()

if False:
    x,y,tx,ty = data.next_batch(30, returnTime=True)
    #plt.plot(data.df_training['Month'], data.df_training['Milk Production'], 'b-')
    #plt.plot(data.df_test['Month'], data.df_test['Milk Production'], 'g-')
    plt.plot(tx,x[0,:,0],'bo', markersize=15, alpha=0.5,label='Training Instance')
    plt.plot(ty,y[0,:,0],'ko', markersize=7, alpha=0.5,label='Target')
    plt.xticks(rotation='vertical', fontsize=8)
    plt.legend()
    plt.show()


### Build Graph ####
n_steps = 30
n_inputs = 1
n_outputs = 1
n_neurons = 100
nepoches = 2000

### Placeholders ###
X = tf.placeholder(tf.float32, [None, n_steps,n_inputs])
Y = tf.placeholder(tf.float32, [None, n_steps,n_outputs])

### The network ###
#basic_cell = tf.contrib.rnn.OutputProjectionWrapper(
 ##  tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),output_size=n_outputs)
#    tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.relu),output_size=n_outputs)

basic_cell = tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.relu)
rnn_outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

# print(states.get_shape()) ==> (?, 100)
# print(rnn_outputs.get_shape()) ==> (?, 30, 100)
stacked_rnn_outputs = tf.reshape(rnn_outputs,[-1, n_neurons])

#print(stacked_rnn_outputs.get_shape()) # (?, 100)

stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)

#print(stacked_outputs.get_shape()) # (?, 1)

outputs = tf.reshape(stacked_outputs, [-1,n_steps, n_outputs])

### The loss ###
mse = tf.reduce_mean(tf.square(outputs-Y))

### The optimizer ###
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(mse)

### Initializer ###
init = tf.global_variables_initializer()

### Train it ###
with tf.Session() as sess:
    init.run()

    for ep in range(nepoches):
        X_batch, Y_batch, tx, ty = data.next_batch(n_steps, returnTime=True)
        Ypred_train, loss, _ = sess.run( [outputs, mse, optimizer], feed_dict={X: X_batch,
                                                                               Y: Y_batch})

        if ep%100 == 0:
            loss_test = 0.0

            X_test, Y_test, tx_test, ty_test = data.get_testData(n_steps, returnTime=True)
            Ypred_test, loss_test = sess.run([outputs, mse], feed_dict={X: X_test,
                                                                        Y: Y_test,
                                                                        })

            plt.plot(data.df["Month"].iloc[0],-0.05,'-')
            plt.plot(data.df["Month"].iloc[-1], 1.05, '-')

            plt.plot(ty, Y_batch[0, :, 0], '-ko', alpha=0.5, markersize=8, label="Train Target")
            plt.plot(ty, Ypred_train[0, :, 0], '-ro', alpha=0.5, markersize=5, label="Train Predicted")

            plt.plot(ty_test, Y_test[0, :, 0], '-ko', alpha=0.5, markersize=8, label="Test Target")
            plt.plot(ty_test, Ypred_test[0, :, 0], '-ro', alpha=0.5, markersize=5, label="Test Predicted")


            print("Epoche: {} LossTrain: {:.6f} LossTest: {:.6f}".format(ep, loss, loss_test))

            plt.title("Epoche: {} LossTrain: {:.6f} LossTest: {:.6f}".format(ep, loss, loss_test))
            plt.savefig("rnn_train_ep{}.png".format(ep))
            plt.legend()
            plt.close()


    X_test, Y_test = data.get_testData(n_steps, returnTime=False)
    sequence = X_test[0,:,0].tolist()
   # sequence = [0.] * n_steps


    for iteration in range(600):
        X_batch = np.array(sequence[-n_steps:]).reshape(1, n_steps,1)

        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        sequence.append(y_pred[0, -1, 0])

    plt.plot(range(len(sequence)),sequence)
    plt.show()