import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Convolution2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,Adam

from parameters import *


def build_dense(input_shape,lr):
	###build model by keras
	model = Sequential()

	#model.add(Flatten(input_shape=(agg_num,fea_dim)))
	model.add(Dropout(0.1,input_shape=input_shape)) # passing input_shape as an arg to 1st layer is crucial (see https://github.com/tensorflow/tensorflow/issues/25175)
	#model.add(Dropout(0.1,input_shape=(agg_num*fea_dim,)))

	model.add(Dense(1000))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	model.add(Dense(500))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	model.add(Dense(n_classes))
	model.add(Activation('sigmoid'))

	sgd = SGD(lr=lr, decay=0, momentum=0.9)
	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
	# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model



