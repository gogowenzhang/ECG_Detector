from keras.models import load_model
import numpy as np
import wfdb
import sys
from data_processing import fouier_transformation


def load_data(filename):
    '''
    Load signal file
    Input: signal file
    Output: list of signals
    '''
    signals = []
    records_file = open(filename, 'r')
    for record_name in records_file:
        record = wfdb.rdsamp('data/training2017/{}'.format(record_name.strip())) 
        d_signal = record.adc()[:,0]
        signals.append(d_signal)
    return signals


def trim_signal(signals):
    '''
    Trim signal into same length.  
    Input: list of signals
    Output: numpy array of signals of same length
    '''
    # Subselect length == 9000
    lenghts = np.array([a.shape[0] for a in signals])
    signals_9000 = np.array([a for a in signals if len(a)==9000])

    # Cut longer records into 9000
    signals_c = np.array([a[:9000] for a in signals if (len(a) > 9000) &  (len(a) < 18000)])
    
    # Divid records 18000 into two parts
    signals_d_1 = np.array([a[:9000] for a in signals if len(a) == 18000])
    signals_d_2 = np.array([a[9000:] for a in signals if len(a) == 18000])
    
    # Combine all the signals
    lst = [signals_c, signals_d_1, signals_d_2]
    signals_whole = signals_9000
    for sig in lst:
        signals_whole = np.concatenate((signals_whole, sig), axis=0)
    
    return signals_whole



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
    signal_file = sys.argv[2]
    model = load_model(model_file)
    signals = load_data(signal_file)
    x = fouier_transformation(signals)
    pred_prob = predict(model, x)

    # save prediction to csv
    pred_prob.tofile('./predictions/pred.csv', sep=',', format='%10.2f')
