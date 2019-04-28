import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

class AppleData():
    def __init__(self):
        df = pd.read_csv('../data/apple_stock/AAPL.csv')
        #print(df.columns) # ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        #print(df.describe())
        df['Date'] = pd.to_datetime(df['Date'])

        split_date = pd.datetime(2017, 1, 1)

        self.df_training = df.loc[df['Date'] <= split_date]
        self.df_test = df.loc[df['Date'] > split_date]



        # Standardize data
        self.min = np.asarray(self.df_training['Open'].min())
        self.max = np.asarray(self.df_training['Open'].max())


        #self.df_training.loc[:,'Open'] = (self.df_training.loc[:,'Open'] -self.min)/ (self.max-self.min)
        #self.df_test.loc[:,'Open']  = (self.df_test.loc[:,'Open'] -self.min) / (self.max-self.min)

        #self.df_training.loc[:,'Open'] = (self.df_training['Open'] -self.min)/ (self.max-self.min)
       # self.df_test.loc[:,'Open']  = (self.df_test['Open'] -self.min) / (self.max-self.min)

    def next_batch(self, nsteps, returnTime=False):

        ir = np.random.randint(0, self.df_training.shape[0] - nsteps - 1)

        x = np.asarray(self.df_training['Open'][ir:ir + nsteps]).reshape(-1, nsteps, 1)
        y = np.asarray(self.df_training['Open'][ir + 1:ir + nsteps + 1]).reshape(-1, nsteps, 1)

        tx = self.df_training['Date'][ir:ir + nsteps]
        ty = self.df_training['Date'][ir + 1:ir + nsteps + 1]

        if returnTime:
            return x, y, tx, ty
        else:
            return x, y

    def get_testData(self, nsteps, returnTime=False):

        ir = np.random.randint(0, self.df_test.shape[0] - nsteps - 1)

        x = np.asarray(self.df_test['Open'][ir:ir + nsteps]).reshape(-1, nsteps, 1)
        y = np.asarray(self.df_test['Open'][ir + 1:ir + nsteps + 1]).reshape(-1, nsteps, 1)

        tx = self.df_test['Date'][ir:ir + nsteps]
        ty = self.df_test['Date'][ir + 1:ir + nsteps + 1]

        if returnTime:
            return x, y, tx, ty
        else:
            return x, y



data = AppleData()




