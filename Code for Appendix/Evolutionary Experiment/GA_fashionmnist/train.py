from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=3)

def get_fashionmnist():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    batch_size = 64
    # Get the data.
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    classes = np.unique(y_train)
    nb_classes = len(classes)

    nRows,nCols = x_train.shape[1:]
    nDims = 1

    x_train = x_train.reshape(x_train.shape[0], nRows, nCols, nDims)
    x_test = x_test.reshape(x_test.shape[0], nRows, nCols, nDims)
    input_shape = (nRows, nCols, nDims)
    #print(input_shape)
    # Change to float datatype
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Scale the data to lie between 0 to 1
    x_train /= 255
    x_test /= 255

    # Change the labels from integer to categorical data
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

   

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

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

def compile_model(network, nb_classes, input_shape):
    
    # Get our network parameters.
    # nb_layers = network['nb_layers']
    # activation = network['activation']
    learning_rate = network['learning_rate']
    # weight_decay = network['weight_decay']
    momentum = network['momentum']

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))

    # sgd = SGD(lr=learning_rate, momentum=momentum, decay=weight_decay)
    sgd = SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy', getPrecision, getRecall])

    return model

def train_and_score(network, dataset):
    
    if dataset == 'fashion_mnist':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_fashionmnist()
    

    model = compile_model(network, nb_classes, input_shape)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10,  # using early stopping, so no real limit
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1], score[2], score[3]  # 3 is recall, 2 is precision, 1 is accuracy. 0 is loss.
