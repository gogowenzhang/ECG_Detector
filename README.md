# ECG_Detector

This is an automatic electrocardiogram (ECG) detector. 

There are various types of cardiac arrhymias. Among them, atrial fibrillation (AF) is the most common one.  This disease is associated with significant mortality and morbidity through an increasing risk of heart failure, dementia, and stoke. AF remains problematic, because it may be episodic and often episodes have no symptoms. 

This dectector focuses on identifying AF from other kinds of records, namely normal sinus rhythm, other abnormal rhythms and noisy recordings. 

ECG recordings are usually stored as 1-D waveform, which displays changes in signal's amplitude over time. Through fouier transformation, the waveform transforms into spectrogram, which displays signals by time and by frequency. Amplitude is then represented with colors or brightness. Then we can regard the input spectrogram not as one long vector but as an image. 

A deep convolutional network, which is specialized for processing an image, were trained to classify the spectrograms. 

Neural Network Models implemented with Keras and used Tensorflow backend. Models were trained on AWS EC2 P2 instance with NVIDIA Tesla® K80 GPUs.  


### Dataset
Total 8331 single short ECG recordings with 30s length were collected (thanks to AliveCor). These recordings were labeled in four classes: normal(59%) , AF(9%), other(30%), and noise(2%). 
![ecg](https://github.com/gogowenzhang/ECG_Detector/blob/master/img/ecg_new.png)

### Data Processing
Transform 1-D array waveform into 2-D array tensor through fouier transformation. 
Log transformation and standardization were applied to spectrogram before passed into model. 

### Model Architecture

Convolutional layers are arranged in blocks. For each block there are four convolutional layers, following each convolutional layer, there are one normalization layer, one relu activation layer and one dropout layer. At the end of each block, one max pooling layer is placed. 

Following the convolutional layers, a customized layer is applied to take average of features across time. Then there is a flatten layer to reduce dimension before passing to classifer layer. 

A standard linear layer with Softmax is placed to compute the class probabilities. 


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






