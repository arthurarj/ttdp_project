import pickle as pkl
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from IPython.display import clear_output

import tensorflow as tf
from tensorflow import keras

# Load data
output_scaler = pkl.load(open('../features_extracted/scalers/out_scaler.scl', 'rb'))
dataset = pkl.load(open('../features_extracted/taxi_trip_data_normalized_10M.dat', 'rb'))
dataset.head()

dataset['denorm_duration'] = output_scaler.inverse_transform(dataset.duration)
dataset.drop(dataset[dataset.denorm_duration == 0].index, inplace=True)
dataset.reset_index(inplace=True)
dataset.drop('index', axis=1, inplace=True)
dataset.drop('denorm_duration', axis=1, inplace=True)
dataset.shape

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(dataset[dataset.columns[:-1]], 
                                                    dataset[dataset.columns[[-1]]],
                                                    test_size=0.3, random_state=42)
test_size = len(y_test)
train_size = len(y_train)

y_test_denorm = output_scaler.inverse_transform(y_test)

# Build model

def build_model():
    model = keras.Sequential([
    keras.layers.Dense(100, activation  = tf.nn.relu,
                           input_shape = (X_train.shape[1],)),
    keras.layers.Dense(50, activation=tf.nn.relu),
    keras.layers.Dense(25, activation=tf.nn.relu),
    keras.layers.Dense(1)
    ])

    optimizer = tf.train.AdamOptimizer(0.01)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae'])
    return model

model = build_model()
model.summary()

# Training

EPOCHS = 10
    
# Store training stats
history = model.fit(X_train, y_train, 
                    epochs=EPOCHS,
                    validation_split=0.2, 
                    verbose=1)
                    #, callbacks=[PrintDot()])


# Testing

y_pred = model.predict(X_test)
y_pred_denorm = output_scaler.inverse_transform(y_pred)

np.concatenate((y_pred_denorm, y_test_denorm),axis=1);
100*np.mean(np.abs(y_pred_denorm - y_test_denorm)/y_test_denorm)