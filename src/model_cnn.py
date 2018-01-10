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

MODEL_NAME = 'cnn3'

print('Loading data...')
with open('x_s_whole.pkl', 'rb') as f:
    x_s = pickle.load(f)
    
with open('y_whole.pkl', 'rb') as f:
    y = pickle.load(f)

# Split train and test
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
epochs = 100

# Model CNN
print('Build model...')

model = Sequential()

# Convolutional layer
model = add_conv_blocks(model, 4, 6, initial_input_shape=(140, 33, 1))
print model.output_shape

# Feature aggregation across time
model.add(Lambda(lambda x: K.mean(x, axis=1)))
print model.output_shape

model.add(Flatten())
print model.output_shape

# Linear classifier
model.add(Dense(4, activation='softmax'))
print model.output_shape

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

print('Train...')
# saved_model_file = './trained_model_{}.h5'.format(MODEL_NAME)
# We use early stopping to avoid spending time on overfitting our model
#  early_stopping = EarlyStopping(monitor='val_loss', patience=3)
# save model at checkpoints when loss function improved
#  checkpoint = ModelCheckpoint(saved_model_file, monitor='val_loss', save_best_only=True, verbose=1)
# and keep logs for visualisation with TensorBoard
#  tensorboard = TensorBoard('./tensorboard_logs', histogram_freq=1)

model.fit(x_train, y_train, batch_size=batch_size,
          epochs=epochs, verbose=1)

print('Evaluation...')
y_test_predict = model.predict(x_test)
y_train_predict = model.predict(x_train)

acc = np.mean(y_test_predict.argmax(axis=1) == y_test.argmax(axis=1))
print 'test accuracy', acc

print 'train confusion_matrix', confusion_matrix(y_test.argmax(axis=1), y_train_predict.argmax(axis=1))

print 'test confusion_matrix', confusion_matrix(y_test.argmax(axis=1), y_test_predict.argmax(axis=1))
