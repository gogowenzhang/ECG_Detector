# ECG_Detector

This is an automatic electrocardiogram (ECG) detector. 

There are various types of cardiac arrhymias. Among them, atrial fibrillation (AF) is the most common one.  This disease is associated with significant mortality and morbidity through an increasing risk of heart failure, dementia, and stoke. AF remains problematic, because it may be episodic and often episodes have no symptoms. 

This dectector focuses on identifying AF from other kinds of records, namely normal sinus rhythm, other abnormal rhythms and noisy recordings. 

A convolutional nework is a neural network that is specialized for processing a grid of values, such as an image. A recurrent neural network is a neural network that is specialized for processing a sequence of values, such as time series. After fouier transformation, the waveform turns into spectrogram, which is a 2-D image and also time-related. So I tried CNN and RNN (LSTM) separately and jointly in explorating the final model.  

Neural Network Models implemented with Keras and used Tensorflow backend. 

### Dataset
Total 8331 single short ECG recordings with 30s length were collected (thanks to AliveCor). These recordings were labeled in four classes: normal(59%) , AF(9%), other(30%), and noise(2%). 
![ecg](https://github.com/gogowenzhang/ECG_Detector/blob/master/img/ecg_new.png)

### Data Processing
Transform 1-D array waveform into 2-D array tensor through fouier transformation. 
Log transformation and standardization were applied to spectrogram before passed into model. 

### Model Architecture



### Requirements
* numpy==1.12.0
* Keras==2.0.1
* scikit-learn==0.17.1
* h5py==2.6.0
* tensorflow==1.0.1
* wfdb==1.3.9


### How to Run
Install python requirements:
```
pip install --requirement requirements.txt
```

To process data:
```
python src/data_processing.py path_to_signal_file path_to_label_file
```

To train and save model:
```
python src/model_cnn.py  path_to_processed_data  epochs  path_to_save_evaluation_file
```

To predict and save predictions:
```
python src/predict.py model_file signal_file
```






