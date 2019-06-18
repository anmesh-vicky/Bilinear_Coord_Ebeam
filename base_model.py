"""
A Keras model with six layers implemented with a 
"""
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout,AveragePooling2D
from keras.utils.vis_utils import plot_model
from keras import metrics

class base_model:

    def __init__(self, learning_rate, epochs, batch_size, dropout, kernel_size):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        #self.activations = activations
        self.dropout = dropout
        self.kernel_size = kernel_size


    def build_model(self):
        model = Sequential()
        model.add(Conv2D(30, kernel_size = 5,padding = 'valid', activation = 'relu', input_shape = (128,128,3)))
        model.add(Conv2D(20, kernel_size = 5,padding = 'valid', activation = 'relu'))
        model.add(Conv2D(20, kernel_size = 5,padding = 'valid', activation = 'relu'))
        model.add(Conv2D(20, kernel_size = 5,padding = 'valid', activation = 'relu'))
        model.add(Conv2D(20, kernel_size = 5,padding = 'valid', activation = 'relu'))
        model.add(Conv2D(20, kernel_size = 5,padding = 'valid', activation = 'relu'))
        model.add(Conv2D(30, kernel_size = 5,padding = 'valid', activation = 'relu'))
       # model.add(Dropout(0.2))
        model.add(Conv2D(20, kernel_size = 5, padding = 'valid',activation = 'relu'))
       # model.add(Dropout(0.2))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
        model.add(Flatten())
        model.add(Dense(128,activation = 'relu'))
       # model.add(Dropout(0.2))
        model.add(Dense(3,activation = 'softmax'))
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [metrics.categorical_accuracy])
        
        return model

