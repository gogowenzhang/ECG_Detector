import numpy as np
import keras
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Bidirectional, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import backend as K
from sklearn.metrics import confusion_matrix
from myfunctions import add_conv_blocks

print('Loading data...')
with open('x_s_whole.pkl', 'rb') as f:
    x_s = pickle.load(f)
    
with open('y_whole.pkl', 'rb') as f:
    y = pickle.load(f)

obs_n = x_s.shape[0]
train_n = int(x_s.shape[0] * 0.8)
ind_selected = np.random.choice(obs_n, train_n,replace=False)
ind_not_selected = [i for i in range(obs_n) if i not in ind_selected]
x_train = x_s[ind_selected]
x_test = x_s[ind_not_selected]
y_train = y[ind_selected]
y_test = y[ind_not_selected]

# Hyperparameter
batch_size = 20
epochs = 10

model = Sequential()

# Convolutional layer
model = add_conv_blocks(model, 6, 4, initial_input_shape=(140, 33, 1))
print model.output_shape

# Feature aggregation across time
model.add(Reshape((9, 480)))
print model.output_shape

# LSTM layer
model.add(Bidirectional(LSTM(200), merge_mode='ave'))
model.add(Dropout(0.15))
print model.output_shape

# Linear classifier
model.add(Dense(4, activation='softmax'))
print model.output_shape

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train, batch_size=batch_size,
          epochs=epochs, verbose=1)


print('Evaluation...')
y_test_predict = model.predict(x_test)

acc = np.mean(y_test_predict.argmax(axis=1) == y_test.argmax(axis=1))
print 'accuracy', acc

print confusion_matrix(y_test.argmax(axis=1), y_test_predict.argmax(axis=1))
