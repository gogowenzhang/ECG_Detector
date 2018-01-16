import numpy as np
import keras
import pickle
import sys
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras import backend as K
from sklearn.metrics import confusion_matrix
from myfunctions import add_conv_blocks
from myfunctions import f1_score


def load_data(filename):
    '''
    Load data from pickle file, then split data to training set and test set.
    Input: pickle filename
    Output: training data and testing data
    '''

    with open(filename, 'rb') as f:
        x, y = pickle.load(f)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)
    return x_train, x_test, y_train, y_test


def generator(x_train, x_test, y_train, y_test):
    '''
    Initialize data generator for model training
    Input: training data and testing data
    Output: training generator and testing generator
    '''

    imagegen = ImageDataGenerator()
    train_generator = imagegen.flow(x_train, y_train, batch_size=20)
    test_generator = imagegen.flow(x_test, y_test, batch_size=20)
    return train_generator, test_generator


def model_fit(train_generator, test_generator, epochs):
    '''
    load training data and testing data, compile and train CNN model, return training history
    Parameters
    Input: train_generator, test_generator
    epochs: number of epochs for training
    Output: training history parameters

    '''
    model = Sequential()

    # Convolutional layer
    model = add_conv_blocks(model, 4, 6, initial_input_shape=(140, 33, 1))

    # Feature aggregation across time
    model.add(Lambda(lambda x: K.mean(x, axis=1)))

    model.add(Flatten())

    # Linear classifier
    model.add(Dense(4, activation='softmax'))


    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy', f1_score])


    model.fit_generator(train_generator,
                        validation_data=test_generator,
                        epochs=epochs, verbose=1)
    return model

if __name__ == '__main__':
    filename = sys.argv[1]
    epochs = int(sys.argv[2])
    evaluation_result_path = sys.argv[3]
    x_train, x_test, y_train, y_test = load_data(filename)
    train_generator, test_generator = generator(x_train, x_test, y_train, y_test)
    model = model_fit(train_generator, test_generator, epochs)
    model.save('model/cnn_model.h5')

    print('Evaluation...')
    y_predict = model.predict_generator(test_generator).argmax(axis=1)
    y_test = y_test.argmax(axis=1)

    f = open(evaluation_result_path, 'w')
    f.write('model: CNN, epochs: {} \n confusion_matrix: \n {}'.format(epochs, confusion_matrix(y_test, y_predict)))
    f.close()    

