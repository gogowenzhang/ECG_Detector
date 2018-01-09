from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.layers import Dropout
from keras import backend as K
import scipy


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
def f1_score_single(y_true, y_pred):
    y_true = set(y_true)
    y_pred = set(y_pred)
    cross_size = len(y_true & y_pred)
    if cross_size == 0: return 0.
    p = 1. * cross_size / len(y_pred)
    r = 1. * cross_size / len(y_true)
    return 2 * p * r / (p + r)
    
def f1_score(y_true, y_pred):
    return K.mean([f1_score_single(x, y) for x, y in zip(y_true, y_pred)])

# To spectrogram
def to_spectrogram(signal):
    _, _, Sxx = scipy.signal.spectrogram(signal, fs=300, window=('tukey', 0.25), 
                                     nperseg=64, noverlap=0.5, return_onesided=True)

    return Sxx.T