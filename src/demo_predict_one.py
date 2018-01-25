from keras.models import load_model
import numpy as np
import wfdb
import sys
from data_processing import fouier_transformation
from myfunctions import K_f1_score


def load_data(record_name):
    '''
    Load signal file
    Input: signal file
    Output: list of signals
    '''    
    record = wfdb.rdsamp('data/training2017/{}'.format(record_name.strip())) 
    signal = record.adc()[:,0]
    return signal

def trim_signal(signal):
    '''
    Trim signal into same length.  
    Input: list of signals
    Output: numpy array of signals of same length
    '''
    # Subselect length == 9000
    if len(signal) < 9000:
        print 'Signal too short, try another one'

    if len(signal) == 9000:
        return signal

    if len(signal) > 9000:
        return signal[:9000]


def predict(model, x):
    '''
    Make prediction of signals
    Input: model, processed signals
    Output: probabities of each class
    '''
    pred_prob = model.predict(x)
    return pred_prob


if __name__ == '__main__':
    model_file = sys.argv[1]
    record_name = sys.argv[2]
    model = load_model(model_file)
    signal = load_data(record_name)
    x = fouier_transformation(signal)
    pred_prob = predict(model, x)
    print pred_prob

