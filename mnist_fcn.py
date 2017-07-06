'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
from os.path import isfile
from mnist_cnn import mnist_cnn, train_mnist, num_classes, num_hidden_layer, load_data
import numpy as np

def transfer_weights(model_path='mnist_cnn.h5'):
    if not isfile(model_path):
        train_mnist(mnist_cnn())
    
    model_fcn = mnist_fcn()
    model_cnn = load_model(model_path)
    index = {}
    for layer in model_fcn.layers:
        if layer.name:
            index[layer.name] = layer

    for layer in model_cnn.layers:
        weights = layer.get_weights()
        if layer.name == 'conv3' or layer.name == 'conv4':
            weights[0] = np.reshape(weights[0], index[layer.name].get_weights()[0].shape)
        if index.has_key(layer.name):
            index[layer.name].set_weights(weights)

    model_fcn.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model_fcn.save('mnist_fcn.h5')
    return model_fcn
            

def mnist_fcn():
    if K.image_data_format() == 'channels_first':
        input_shape = (1, None, None)
    else:
        input_shape = (None, None, 1)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape,
                     name='conv1'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=1, padding='valid',
        name='conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(num_hidden_layer, (12, 12), activation='relu', strides=1, padding='valid',
        name='conv3'))
    model.add(Dropout(0.5))
    model.add(Conv2D(num_classes, (1, 1), activation='softmax', name='conv4'))

    return model


if __name__ == '__main__':
    model = transfer_weights()

    (x_train, y_train), (x_test, y_test) = load_data()
    if K.image_data_format() == 'channels_first':
        y_test = y_test.reshape((-1, num_classes, 1, 1))
    else:
        y_test = y_test.reshape((-1, 1, 1, num_classes))
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
