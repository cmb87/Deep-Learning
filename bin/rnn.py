import copy, numpy as np
import sys
np.random.seed(0)

class RNN:
    def __init__(self):

        self.n_neurons = 16
        self.n_input = 2
        self.n_output = 1

        self.W_hh = 2 * np.random.random((self.n_neurons, self.n_neurons)) - 1 # previous hidden current hidden state matrix
        self.W_xh = 2 * np.random.random((self.n_input, self.n_neurons)) - 1 # input state to current hidden state
        self.W_hy = 2 * np.random.random((self.n_neurons, self.n_output)) - 1 # current hidden state to output y

    # forward pass of a vanilla RNN.
    def step(self, x):
        # update the hidden state
        self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
        # compute the output vector
        y = np.dot(self.W_hy, self.h)
        return y



# Going deep. RNNs are neural networks and everything works monotonically
# better (if done right) if you put on your deep learning hat and
# start stacking models up like pancakes. For instance, we can form a 2-layer
# recurrent network as follows:

rnn1 = RNN()
rnn2 = RNN()
y1 = rnn1.step(x)
y = rnn2.step(y1)

#In other words we have two separate RNNs: One RNN is receiving the input vectors
# and the second RNN is receiving the output of the first RNN as its input.
# Except neither of these RNNs know or care - it’s all just vectors coming in
# and going out, and some gradients flowing through each module during backpropagation.