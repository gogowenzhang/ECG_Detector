import numpy as np
import keras
import pickle
import pandas as pd
import wfdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Bidirectional, LSTM
from keras import backend as K
from sklearn.metrics import confusion_matrix
from myfunctions import add_conv_blocks

print('Loading data...')
with open('x_s_whole.pkl', 'rb') as f:
    x_s = pickle.load(f)
    
with open('y_whole.pkl', 'rb') as f:
    y = pickle.load(f)


# Hyperparameter
batch_size = 20
epochs = 10

# Model CNN
print('Build model...')

model = Sequential()

# Convolutional layer
model = add_conv_blocks(model, 6, 4, initial_input_shape=(140, 33, 1))
print model.output_shape

# Feature aggregation across time
model.add(Lambda(lambda x: K.mean(x, axis=1)))
model.add(Dropout(0.15))
print model.output_shape

model.add(Flatten())
model.add(Dropout(0.15))
print model.output_shape

# Linear classifier
model.add(Dense(4, activation='softmax'))
model.add(Dropout(0.015))
print model.output_shape

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

print('Train...')
model.fit(x_s, y, batch_size=batch_size,
          epochs=epochs, verbose=1)


print('Evaluation...')
y_predict = model.predict(x_s)

acc = np.mean(y_predict.argmax(axis=1) == y.argmax(axis=1))
print 'accuracy', acc

print confusion_matrix(y.argmax(axis=1), y_predict.argmax(axis=1))
