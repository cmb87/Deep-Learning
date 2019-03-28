import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('../data/02-TensorFlow-Basics/cal_housing_clean.csv')

features = ['housingMedianAge', 'totalRooms', 'totalBedrooms', 'population',
            'households', 'medianIncome']

labels = ['medianHouseValue']


# Normalize data
data[features] = data[features].apply(lambda x: (x - x.mean())/(x.std()))
#data[features] = data[features].apply(lambda x: (x - x.min())/(x.max() - x.min()))

X = data.drop('medianHouseValue', axis=1)
y = data[labels[0]]

if False:
    for f in data.columns:
        data[f].hist(bins=20)
        plt.title(f)
        plt.show()



# Split data
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)


# Build TF model

feat = []
for f in features:
    feat.append(tf.feature_column.numeric_column(f))


input_func_train = tf.estimator.inputs.pandas_input_fn(x=xtrain, y=ytrain,
                                                 batch_size=30,
                                                 num_epochs=1000,
                                                 shuffle=True)

input_func_test = tf.estimator.inputs.pandas_input_fn(x=xtest, y=ytest,
                                                 batch_size=30,
                                                 num_epochs=1,
                                                 shuffle=False)


model = tf.estimator.DNNRegressor(feature_columns=feat,
                                  hidden_units=[15,15,10],
                                  label_dimension=1)

model.train(input_fn=input_func_train, steps=20000)

res = model.evaluate(input_fn=input_func_test)

print(res)

input_func_pred = tf.estimator.inputs.pandas_input_fn(x=xtest,
                                                 batch_size=30,
                                                 num_epochs=1,
                                                 shuffle=False)

preds = list(model.predict(input_fn=input_func_pred))

average = 0
for p, ytrue in zip(preds,ytest):
    average += np.sqrt((p['predictions'][0]-ytrue)**2)/len(preds)

print(average)

