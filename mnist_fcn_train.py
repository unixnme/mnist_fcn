# use keras to train mnist fcn
import keras
from keras import regularizers
from keras.datasets.mnist import load_data
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Reshape
from keras.models import load_model
from os.path import isfile
from mnist_cnn import mnist_cnn, train_mnist, num_classes, num_hidden_layer, load_data, img_rows, img_cols
import numpy as np

num_classes = 11
batch_size = 128
epochs = 10

def create_model(input_shape=(28,28,1)):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='elu',
                     input_shape=input_shape,
                     name='conv1', padding='same'))
    # model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu', strides=1, padding='same',
                     name='conv2'))
    # model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(num_hidden_layer, kernel_size=(7, 7), activation='elu',
               strides=(1, 1), padding='same', name='conv3'))
    # model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.5))
    model.add(Conv2D(num_classes, kernel_size=(1, 1), activation='softmax'))
    model.add(Reshape((49, num_classes,)))

    return model


def train_model(model):
    (x_train, y_train), (x_test, y_test) = load_data()
    y_train = np.argmax(y_train, axis=-1)
    y_test = np.argmax(y_test, axis=-1)

    training_epochs = 1
    num_train_examples = len(x_train)
    steps_per_epochs = int(np.floor(num_train_examples / float(batch_size)))

    gen = get_batch(x_train, y_train, batch_size)
    gen_test = get_batch(x_test, y_test, len(x_test))
    x_test, y_test = next(gen_test)

    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit_generator(generator=gen, steps_per_epoch=steps_per_epochs,
                        validation_data=(x_test, y_test), epochs=epochs)

    eval = model.evaluate(x_test, y_test, len(x_test))
    print eval

    return model

def get_batch(x, y, batch_size):
    # generator to yield batches
    num_samples = len(x)
    indices = np.arange(num_samples)
    while True:
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = min(num_samples, start + batch_size)
            idx = indices[start:end]

            x_batch, temp = x[idx], y[idx]
            y_batch = np.zeros((len(idx), 7, 7), dtype=np.int64)
            for i in range(len(idx)):
                y_batch[i,:,:] = 10
                y_batch[i,2:5,2:5] = 0.
                y_batch[i,2:5,2:5] = temp[i]

            yield x_batch, y_batch.reshape(-1, 49)


if __name__ == '__main__':
    model = create_model()
    model = train_model(model)