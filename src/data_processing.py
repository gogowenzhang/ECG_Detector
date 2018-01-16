import numpy as np
import pandas as pd
import sys
import wfdb
import matplotlib.pyplot as plt
import scipy
import keras
import pickle
from myfunctions import to_spectrogram

def load_data(signal_filename, label_filename):
    '''
    Load signal file and label file. One-hot encoding of labels. 
    Input: signal filename, label filename
    Output: list of signals, numpy array of labels
    '''

    signals = []
    records_file = open(signal_filename, 'r')
    for record_name in records_file:
        record = wfdb.rdsamp('data/training2017/{}'.format(record_name.strip())) 
        d_signal = record.adc()[:,0]
        signals.append(d_signal)

    labels = pd.read_csv(label_filename, header=None, names=['id', 'label'])
    y = labels['label']
    y[y=='N'] = 0
    y[y=='A'] = 1
    y[y=='O'] = 2
    y[y=='~'] = 3
    y = keras.utils.to_categorical(y)
    return signals, y


def trim_signal(signals, y):
    '''
    Trim signal into same length.  
    (EAD shows that length is not assciated with outcome)
    Input: list of signals, y
    Output: numpy array of signals of same length and their labels
    '''
    # Subselect length == 9000
    lenghts = np.array([a.shape[0] for a in signals])
    signals_9000 = np.array([a for a in signals if len(a)==9000])
    y_9000 = y[lenghts==9000]

    # Cut longer records into 9000
    signals_c = np.array([a[:9000] for a in signals if (len(a) > 9000) &  (len(a) < 18000)])
    y_c = y[((lenghts > 9000) &  (lenghts < 18000))]

    # Divid records 18000 into two parts
    signals_d_1 = np.array([a[:9000] for a in signals if len(a) == 18000])
    signals_d_2 = np.array([a[9000:] for a in signals if len(a) == 18000])
    y_d_1 = y[lenghts==18000]
    y_d_2 = y[lenghts==18000]

    # Combine all the signals
    lst = [signals_c, signals_d_1, signals_d_2]
    signals_whole = signals_9000
    for sig in lst:
        signals_whole = np.concatenate((signals_whole, sig), axis=0)

    lst = [y_c, y_d_1, y_d_2]
    y_whole = y_9000
    for sig in lst:
        y_whole = np.concatenate((y_whole, sig), axis=0)
    return signals_whole, y_whole


def fouier_transformation(signals_whole):
    '''
    Transform 1-D array signals into 2-D array as spectrogram.
    Take log and standardize spectrogram. 
    Input: 1-D array signals, labels
    Output: 2-D array spectrograms
    '''
    spectrogram = np.apply_along_axis(to_spectrogram, 1, signals_whole)

    # Log transformation and Standardizer
    log_spectrogram = np.log(spectrogram + 1)

    centers = log_spectrogram.mean(axis=(1,2))
    stds = log_spectrogram.std(axis=(1,2))
    log_spectrogram_s = np.array([(x - c) / d for x, c, d in zip(log_spectrogram, centers, stds)])
    x_s = log_spectrogram_s[..., np.newaxis]
    return x_s


if __name__ == '__main__':
    signal_filename = sys.argv[1]
    label_filename = sys.argv[2]
    signals, y = load_data(signal_filename, label_filename)
    signals_whole, y_whole = trim_signal(signals, y)
    x_s = fouier_transformation(signals_whole)

    with open('./data/processed_data.pkl', 'wb') as f:
        pickle.dump((x_s, y_whole), f)

