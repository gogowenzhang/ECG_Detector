import numpy as np
import keras
import pickle
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
epochs = 100

# Model CNN
print('Build model...')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(140, 33, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer = keras.optimizers.Adam(),
              metrics=['accuracy'])

print('Train...')
model.fit(x_s, y, batch_size=batch_size,
          epochs=epochs, verbose=1)


print('Evaluation...')
y_predict = model.predict(x_s)

acc = np.mean(y_predict.argmax(axis=1) == y.argmax(axis=1))
print 'accuracy', acc

print confusion_matrix(y.argmax(axis=1), y_predict.argmax(axis=1))
