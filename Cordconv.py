import keras as k
import numpy as np
from keras.layers import Input, Conv2D, Flatten, Softmax,Dropout,AveragePooling2D,Dense,BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam,SGD
from coord import CoordinateChannel2D

class CordConv:
	def __init__(self,learning_rate,dropout,kernel_size):
		self.learning_rate = learning_rate
		self.dropout = dropout
		self.kernel_size = kernel_size

	def build_model(self):
		ip = Input(shape=(128,128,3))
		x = CoordinateChannel2D()(ip)
		x = Conv2D(32,self.kernel_size,padding='valid',activation="relu")(ip)
		x = Conv2D(32,self.kernel_size,padding='valid',activation="relu")(x)
#		x = BatchNormalization()(x)
	#	x = Conv2D(40,self.kernel_size,padding='valid',activation="relu")(x)
		x = Dropout(self.dropout)(x, training=True)
		x = Conv2D(32,self.kernel_size,padding='valid',activation="relu")(x)
	#	x = Conv2D(40,self.kernel_size,padding='valid',activation='relu')(x)
		x = Dropout(self.dropout)(x, training=True)
	#	x = Conv2D(40,self.kernel_size,padding='valid',activation="relu")(x)
		x = AveragePooling2D(pool_size=(2,2))(x)
		x = Flatten()(x)
		x = Dense(128,activation='relu')(x)
		x = Dense(128,activation='relu')(x)
		x = Dropout(self.dropout)(x, training=True)
		x = Dense(5)(x)
		x = Softmax(axis=-1)(x)
		model = Model(ip, x)
		optimizer = Adam(lr=0.0001)
		model.compile(optimizer, 'categorical_crossentropy', metrics=['accuracy'])
		return model

