'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras import backend as K
from pso import Pso
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.optimizers as optim
from tensorflow.python.keras import regularizers
import logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)

batch_size = 128
num_classes = 10
epochs = 100

# input image dimensions
img_rows, img_cols = 32, 32

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

def getPrecision(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # TP
    N = (-1) * K.sum(K.round(K.clip(y_true - K.ones_like(y_true), -1, 0)))  # N
    TN = K.sum(K.round(K.clip((y_true - K.ones_like(y_true)) * (y_pred - K.ones_like(y_pred)), 0, 1)))  # TN
    FP = N - TN
    precision = TP / (TP + FP + K.epsilon())  # TT/P
    return precision


def getRecall(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # TP
    P = K.sum(K.round(K.clip(y_true, 0, 1)))
    FN = P - TP  # FN=P-TP
    recall = TP / (TP + FN + K.epsilon())  # TP/(TP+FN)
    return recall

def func(x):
  learning_rate, moment = x[0], x[1]

  model = Sequential()
  model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(120))
  model.add(Dense(84))
  model.add(Dense(num_classes, activation='softmax'))

  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=moment),
                metrics=['accuracy', getPrecision, getRecall])

  cp = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')];

  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_data=(x_test, y_test),
            # callbacks=cp
            )

  score = model.evaluate(x_test, y_test, verbose=0)

  # loss, val
  # print('current config:',x,'val:',score[1])
  logging.info("cifar10 current config: %.4f, %.4f, %.4f, %.4f, %.4f" % (x[0], x[1], score[1], score[2], score[3]))
  return score[1]

##################################################################
pso = Pso(swarmsize=100,maxiter=100)
# n,sf,sp,l
bp,value = pso.run(func,[0.001, 0.45],[0.3, 0.95])

v = func(bp);

##################################################################

print('Test loss:', bp)
print('Test accuracy:', value,v)