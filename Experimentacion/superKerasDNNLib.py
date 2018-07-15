# Universidad Simón Bolívar
# Abril-Julio 2018
# Inteligencia Artificial II
# Prof. Ivette Carolina Martínez 
# Proyecto Final
# Libreria de Redes Neuronales Profundas en Keras

# Autores:
    # Lautaro Villalón 12-10427
    # Yarima Luciani 13-10770
    # Benjamin Amos 12-10240

import numpy as np 
import pandas as pd 

from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import Adam

# Devuelve model DNN Dense
def getDNNmodel(layers, inputDim, activation, loss, metrics):

	model = Sequential()

	# Primera capa
	model.add(Dense(layers[0], input_dim=inputDim))
	#model.add(Dropout(0.5))	

	# Capas intermedias
	for i in layers[1:]:
		model.add(Dense(i, activation=activation))
		#model.add(Dropout(0.5))

	# Capa de salida
	model.add(Dense(1))
	model.add(Activation(activation))

	rms = Adam(lr=0.0001)
	model.compile(loss=loss, optimizer=rms, metrics=[metrics])

	return model

# Obtiene clasificacion de prediccion diferencial
def classifyDiffFromPrediction(x, predictions):
	yAprox = []
	for i in range(len(x)):
		if 0 < predictions[i]:
			yAprox.append(1)
		else:
			yAprox.append(0)

	return yAprox
