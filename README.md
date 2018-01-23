# ECG_Detector

This is an automatic electrocardiogram (ECG) detector using convolutional neural networks. 

There are various types of cardiac arrhymias. Among them, atrial fibrillation (AF) is most common.  The disease is associated with significant mortality and morbidity through an increasing risk of heart failure, dementia, and stoke. Detection for AF remains problematic, because the signal may be episodic and often episodes have no symptoms. 

This dectector focuses on identifying AF from other kinds of records, namely normal sinus rhythm, other abnormal rhythms and noisy recordings. 

ECG recordings are usually stored as 1-D waveform, which displays changes in amplitude of electical activies of the heart over time. Through Fourier transformation, the waveform are turned into spectrogram - a 2-D visual representation of the spectrum of frequencies with amplitude represented by brightness.

A deep convolutional network, which is specialized for processing an image, were trained to classify the spectrograms. 

I used the [keras](https://keras.io/) package with Tensorflow as backend to train the model on AWS p2.xlarge instance with NVIDIA Tesla® K80 GPUs.  


### Dataset
Total 8331 single short ECG recordings with 30s length were collected (thanks to AliveCor). These recordings were labeled in four classes: normal(59%) , AF(9%), other(30%), and noise(2%). 

<img src="https://github.com/gogowenzhang/ECG_Detector/blob/master/img/ecg_new.png" width='600' height='600'>

### Data Processing
Transform 1-D waveform into 2-D spetrogram by Fourier transformation. 
Log transformation and standardization were applied to spectrograms before passed into model. 

<img src="https://github.com/gogowenzhang/ECG_Detector/blob/master/img/spectrogram.png" width='600' height='150'>

### Model Architecture

Convolutional layers are arranged in blocks. For each block there are four convolutional layers, following each convolutional layer, there is one normalization layer, one relu activation layer and one dropout layer. A max pooling layer is added at the end of each block. 

Following the convolutional layers, a customized layer is added to take average of features across time. Then there is a flatten layer to reduce dimension before passing to classifer(fully-connected) layer. 

A standard linear layer with Softmax is used to compute the class probabilities. 

<img src="https://github.com/gogowenzhang/ECG_Detector/blob/master/img/nn.png" width="450" height="600">


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






