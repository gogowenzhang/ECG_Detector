from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.layers import Dropout
from keras import backend as K
import scipy
import numpy as np


# Convolutional layers
def add_conv_blocks(model, block_size, block_count, initial_input_shape):
    channels = 32
    for i in range(block_count):
        for j in range(block_size):
            if (i, j) == (0, 0):
                conv = Conv2D(channels, kernel_size=(5, 5),
                              input_shape=initial_input_shape, padding='same')
            else:
                conv = Conv2D(channels, kernel_size=(5, 5), padding='same')
            model.add(conv)
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.15))
            if j == block_size - 2:
                channels += 32
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.15))
    return model


# Metrics
def K_f1_score(y_true, y_pred):
    '''
    Customized f1 score for Keras model log
    Input: y_true, y_pred
    Output: f1 score
    '''
    y_pred = K.argmax(y_pred, axis=1)
    y_true = K.argmax(y_true, axis=1)
    result = []
    for i in range(3):
        denom = (K.sum(K.cast(K.equal(y_true, i), dtype='float32')) + K.sum(K.cast(K.equal(y_pred, i), dtype='float32'))) + K.epsilon()
        num = K.sum(K.cast(K.equal(y_true, i), dtype='float32') * K.cast(K.equal(y_pred, i), dtype='float32'))
        result.append(2.0 * num / denom)
    return (result[0] + result[1] + result[2]) / 3.0





# To spectrogram
def to_spectrogram(signal):
    _, _, Sxx = scipy.signal.spectrogram(signal, fs=300, window=('tukey', 0.25), 
                                     nperseg=64, noverlap=0.5, return_onesided=True)

    return Sxx.T