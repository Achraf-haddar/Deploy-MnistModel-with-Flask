# Python 2/3 compatibility
from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# mini batch gradient descent ftw
batch_size = 128
# 10 different characters
num_classes = 10
# very short training time
epochs = 12

# input image dimensions 28x28 pixel images
img_rows, img_cols = 28, 28

# The data downloaded, shuffled and split between train and test
# If only all datasets were this easy to import and format
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# This assumes our data format
# For 3D data, "channels_last" assumes (conv_dim1, conv_dim2, conv_dim3, channels) while
# "channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3)
if K.image_data_format() == "channels_first":
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# more reshaping
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

