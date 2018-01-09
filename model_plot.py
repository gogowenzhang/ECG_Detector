from myfunctions import add_conv_blocks
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Bidirectional, LSTM
from keras import backend as K

model = Sequential()

# Convolutional layer
model = add_conv_blocks(model, 4, 6, (None, 33, 1))

# Feature aggregation across time
model.add(Lambda(lambda x: K.mean(x, axis=[1,2])))


# Linear classifier
model.add(Dense(4, activation='softmax'))

plot_model(model, to_file='cnn_model.png')
