import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

diabetes = pd.read_csv('../data/02-TensorFlow-Basics/pima-indians-diabetes.csv')

# Clean data
print(diabetes.columns)
cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree']

# Normalize data
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min())/(x.max() - x.min()))

# Create feature columns
num_preg = tf.feature_column.numeric_column('Number_pregnant')
glu_con = tf.feature_column.numeric_column('Glucose_concentration')
blood_pressure = tf.feature_column.numeric_column('Blood_pressure')
triceps = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

# Categorial
assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group', ['A', 'B', 'C', 'D'])
assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=10) # for situations, where there are too many groups to type out

# plot
#diabetes['Age'].hist(bins=20)
#plt.show()

# Convert continous numeric to categorial variable ==> This is where domain knowledge comes into the game
age_bucket = tf.feature_column.bucketized_column(age, boundaries=[20, 30, 40, 50, 60, 70, 80])

# DNNS classifier wont like assigned_group, transform to embedded_group
embedded_group_col = tf.feature_column.embedding_column(assigned_group, dimension=4)


# Train test split
x_data = diabetes.drop('Class', axis=1)
labels = diabetes['Class']

X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.3, random_state=101)

# Model input
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train,
                                                 batch_size=10,
                                                 num_epochs= 1000,
                                                 shuffle=True)

# Model
#feat_cols = [num_preg, glu_con, blood_pressure, triceps, insulin, bmi,
 #            pedigree, assigned_group, age_bucket]

feat_cols = [num_preg, glu_con, blood_pressure, triceps, insulin, bmi,
             pedigree, embedded_group_col, age_bucket]

#model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)
model = tf.estimator.DNNClassifier(hidden_units=[10, 20, 20, 10],
                                   feature_columns=feat_cols, n_classes=2)

# Train model
model.train(input_fn=input_func, steps=1000)

# Get test error
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test,
                                                      batch_size=10,
                                                      num_epochs=1,
                                                      shuffle=False)
results = model.evaluate(eval_input_func)
print(results)


pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,
                                                      batch_size=10,
                                                      num_epochs=1,
                                                      shuffle=False)

mypreds = list(model.predict(pred_input_func))
print(mypreds)

