"""
John Fahringer jrf115@zips.uakron.edu
Big Data Programming _ Project2
"""
from __future__ import print_function
import keras
import datetime
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

# Use Keras to create and compare deep learning models for the MNIST Fashion data set.

# Create saved models to be reused, and create logs for visualizing in TensorBoard.
# Use the MNIST Fashion data set for training and validation.

### 1 ###
#  Create a Sequential fully connected model. Create logs for accuracy and loss
#  of training and validation data sets. Save the model in ‘hdf5’ format. Name the saved
#  model as ‘f_mnist.h5’
fashion_mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

do_Sequential = True
if do_Sequential:
    Seq_model = Sequential()
    Seq_model.add(Flatten(input_shape=(28, 28)))
    Seq_model.add(Dense(512, activation='relu'))
    Seq_model.add(Dense(10, activation="softmax"))
    Seq_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train the model
    Seq_history = Seq_model.fit(x=x_train,
                             y=y_train,
                             epochs=5,
                             validation_data=(x_test, y_test),
                             callbacks=[tensorboard_callback])

    Seq_Dict = Seq_history.history

    Seq_acc = Seq_history.history['accuracy']
    Seq_val_acc = Seq_history.history['val_accuracy']
    Seq_loss = Seq_history.history['loss']
    Seq_val_loss = Seq_history.history['val_loss']


    # Save model
    Seq_model.reset_metrics()
    Seq_model.save('Models/f_mnist.h5')
