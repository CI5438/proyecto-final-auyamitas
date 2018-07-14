import numpy as np 
import pandas as pd 

from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop

def getLSTMmodel(layers, inputShape, activation, loss, metrics):

	model = Sequential()

	# Primera capa
	if len(layers) > 1:
		model.add(LSTM(layers[0], input_shape=inputShape, return_sequences=True))
		model.add(Dropout(0.5))
	else:
		model.add(LSTM(layers[0], input_shape=inputShape, return_sequences=False))
		model.add(Dropout(0.5))		

	# Capas intermedias
	for i in layers[1:-1]:
		model.add(LSTM(i, return_sequences=True))
		model.add(Dropout(0.5))

	# Ultima capa LSTM
	if len(layers) > 1:
		model.add(LSTM(layers[-1], return_sequences=False))
		model.add(Dropout(0.2))

	# Capa de salida
	model.add(Dense(1))
	model.add(Activation(activation))

	rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
	model.compile(loss=loss, optimizer=rms, metrics=[metrics])

	return model

def classifyFromPrediction(x, predictions):
	yAprox = []
	for i in range(len(x)):
		if x[i, 0, -1] < predictions[i]:
			yAprox.append(1)
		else:
			yAprox.append(0)

	return yAprox