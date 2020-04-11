"""
John Fahringer jrf115@zips.uakron.edu
Big Data Programming _ Project2
"""
from __future__ import print_function
import keras
import keras.backend as K
from keras.datasets import fashion_mnist
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import datetime

# Use Keras to create and compare deep learning models for the MNIST Fashion data set.

# Create saved models to be reused, and create logs for visualizing in TensorBoard.
# Use the MNIST Fashion data set for training and validation.

### 1 ###
#  Create a Sequential fully connected model. Create logs for accuracy and loss
#  of training and validation data sets. Save the model in ‘hdf5’ format. Name the saved
#  model as ‘f_mnist.h5’

print("\n\n\n__Sequential__")
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("Shape of fashion_mnist:", x_train.shape)
print("Num of labels in data:", len(y_train))
print("The labels...:", y_train)
print("Shape of testing data:", x_test.shape)
print("The test set contains this amount of data:", len(y_test))

Seq_model = Sequential()
Seq_model.add(Flatten(input_shape=(28, 28)))
Seq_model.add(Dense(512, activation='relu'))
Seq_model.add(Dense(10, activation="softmax"))
Seq_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir="logs/fit/sequent_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
s_tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
Seq_history = Seq_model.fit(x=x_train,
                         y=y_train,
                         epochs=50,
                         validation_data=(x_test, y_test),
                         callbacks=[s_tensorboard_callback])

Seq_Dict = Seq_history.history

Seq_acc = Seq_history.history['accuracy']
Seq_val_acc = Seq_history.history['val_accuracy']
Seq_loss = Seq_history.history['loss']
Seq_val_loss = Seq_history.history['val_loss']


# Save model
Seq_model.reset_metrics()
Seq_model.save('Models/f_mnist.h5')



### 2 ###
#  Create a cov2Dnet model. Create logs for accuracy and loss of training and
#  validation data sets. Save the model in ‘hdf5’ format. Name the saved model as
#  ‘cov_f_mnist.h5’.

print("\n\n\n__CNN__")
K.clear_session()
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Reshape data'
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
x_train, x_test = x_train / 255, x_test / 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print("Shape of fashion_mnist:", x_train.shape)
print("Num of labels in data:", len(y_train))
print("The labels...:", y_train)
print("Shape of testing data:", x_test.shape)
print("The test set contains this amount of data:", len(y_test))

Cov_model = Sequential()
Cov_model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1)))
Cov_model.add(Activation('relu'))

Cov_model.add(Conv2D(64, (3, 3), activation='relu'))
Cov_model.add(MaxPooling2D(pool_size=(2, 2)))
Cov_model.add(Dropout(0.25))
Cov_model.add(Flatten())

Cov_model.add(Dense(128))
Cov_model.add(Activation('relu'))
Cov_model.add(Dropout(0.5))

Cov_model.add(Dense(10))
Cov_model.add(Activation('softmax'))

Cov_model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])

c_log_dir="logs/fit/conv_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=c_log_dir, histogram_freq=1)

Cov_history = Cov_model.fit(x_train, y_train,
                            batch_size=128,
                            epochs=50,
                            verbose=2,
                            validation_data=(x_test, y_test),
                            callbacks=[tensorboard_callback])

Cov_model.reset_metrics()
Cov_model.save('Models/cov_f_mnist.h5')




### 3 ###
# List and compare the two models in terms of accuracy and hyperparameters.

print("\nThe MNIST Sequential fully connected model")
print(Seq_model.summary())
print("\nThe MNIST Cov2Dnet Model")
print(Cov_model.summary(), '\n')


### 4 ###
# Create a cov2Dnet model for the CIFAR-10 data set. Create logs for
# accuracy and loss of training and validation data sets. Save the model in ‘hdf5’
# format. Name the saved model as ‘cifar-10.h5’. Compare the performance of your
# model to the Cifar-10 2019 top benchmark of 99.3%.
print('\n\n\n__CIFAR-10__')
K.clear_session()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print("Num of labels in data:", len(y_train))
print("The labels...:", y_train)
print("The test set contains this amount of data:", len(y_test))

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

Cov_model = Sequential()
Cov_model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
Cov_model.add(Activation('relu'))

Cov_model.add(Conv2D(64, (3, 3), activation='relu'))
Cov_model.add(MaxPooling2D(pool_size=(2, 2)))
Cov_model.add(Dropout(0.25))
Cov_model.add(Flatten())

Cov_model.add(Dense(128))
Cov_model.add(Activation('relu'))
Cov_model.add(Dropout(0.5))

Cov_model.add(Dense(10))
Cov_model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

Cov_model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

c_log_dir="logs/fit/cifar_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=c_log_dir, histogram_freq=1)

Cov_history = Cov_model.fit(x_train, y_train,
                            batch_size=128,
                            epochs=50,
                            verbose=2,
                            validation_data=(x_test, y_test),
                            callbacks=[tensorboard_callback])

Cov_model.reset_metrics()
Cov_model.save('Models/cifar-10.h5')


#########################################################
### Display accuracy charts for the first two models. ###
#########################################################

