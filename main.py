
#Author : t.me/Jelofen1962
#MIT Copy-Right, you can use this model but just With the CREDIT


import numpy as np
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os

# Load the dataset
dataset = loadtxt('src/data.txt', delimiter=',')
X = dataset[:, 0:23]
y = dataset[:, 23]

# Normalize the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load or initialize the Keras model
if os.path.exists('src/model.keras'):
    model = load_model('src/model.keras')
    print("Loaded model from disk")
else:
    # Define a more complex Keras model with batch normalization
    model = Sequential()
    model.add(Dense(256, input_shape=(23,), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    print("Initialized new model")

# Compile the Keras model with advanced learning rate scheduling
optimizer = Adam(learning_rate=0.0001)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error'])

# Callbacks
checkpoint = ModelCheckpoint('src/model.keras', monitor='val_mean_absolute_error', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_mean_absolute_error', patience=20, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.5, patience=5, verbose=1, mode='min')  # Adjusted patience
callbacks_list = [checkpoint, early_stopping, reduce_lr]

# Fit the Keras model with larger batch size and advanced callbacks
history = model.fit(X_train, y_train, epochs=200, batch_size=128, validation_split=0.2, callbacks=callbacks_list)

# Evaluate the Keras model on the test set
_, mean_absolute_error = model.evaluate(X_test, y_test)
accuracy = (1 - mean_absolute_error / y.mean()) * 100
print('Test Accuracy: %.2f%%' % accuracy)

# Save the accuracy to a file
with open('accuracy.txt', 'a') as f:
    f.write(f'Test Accuracy: {accuracy:.2f}%\n')
