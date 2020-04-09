"""
John Fahringer jrf115@zips.uakron.edu
Big Data Programming _ Project2
All rights reserved
"""
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os


### 1 ###
#  Create a Sequential fully connected model. Create logs for accuracy and loss
#  of training and validation data sets. Save the model in ‘hdf5’ format. Name the saved
#  model as ‘f_mnist.h5’
