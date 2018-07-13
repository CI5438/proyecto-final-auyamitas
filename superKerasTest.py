import numpy as np 
import pandas as pd 

# from subprocess import check_output

# from sklearn.cross_validation import  train_test_split
# import time 
<<<<<<< HEAD
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import newaxis

from dataLib import *


def createDataSet(df, lookBack=30):
	x = []
	y = []

	for i in range(len(df) - lookBack-3):
		a = df[i:(i + lookBack), 2:]
		x.append(a)
		y.append(df[i + lookBack + 2, 2])

	return np.array(x), np.array(y)
=======
#from numpy import newaxis
>>>>>>> c8152f8466a480b91838949c355c9b5dc64f8e5c

import matplotlib.pyplot as plt
from superDataLib import *
from superKerasRNNLib import *

def main():
	lookBack = 60 
	percentage = 0.9
	iterations = 500

<<<<<<< HEAD
	# Obtenemos la y real
	testCase = 'Mark1 Data/Communication Services Sector/Communication Services Sector.test.18.txt'
	df = readData(testCase, ';')
	y = df[df.columns[-1]]
=======
	# Obtenemos la y real (sector, lookBack)
	y = classifiedY('Communication Services Sector', lookBack)
>>>>>>> c8152f8466a480b91838949c355c9b5dc64f8e5c

	# Dividimos la y real en entrenamiento y prueba
	trainSize = int(len(y)*percentage)

	y1 = y[:trainSize]
	y2 = y[trainSize:]

	# Preparamos el archivo del indice (sector, scale?, scaleRange)
	df = prepareData('Communication Services Sector', True, (0, 1))

<<<<<<< HEAD
	scaler = MinMaxScaler(feature_range=(0, 1))    # mejor manera?
	df = scaler.fit_transform(df)

	lookBack = 18
=======
	# Obtenemos DataSet
>>>>>>> c8152f8466a480b91838949c355c9b5dc64f8e5c
	x, y = createDataSet(df, lookBack)

<<<<<<< HEAD
	xTrain = np.reshape(xTrain, (xTrain.shape[0], 4, xTrain.shape[1]))
	xTest = np.reshape(xTest, (xTest.shape[0], 4, xTest.shape[1]))
=======
	# Dividimos DataSet (x, y, trainSize, features)
	xTrain, yTrain, xTest, yTest = splitDataSet(x, y, trainSize, 1)
>>>>>>> c8152f8466a480b91838949c355c9b5dc64f8e5c

	# Creamos el modelo: (layers, inputSize, activation, loss, metrics)

<<<<<<< HEAD
	model.add(LSTM(50, input_shape=(4,18), return_sequences=True))
	model.add(Dropout(0.5))

	model.add(LSTM(200, return_sequences=True))
	model.add(Dropout(0.5))

	model.add(LSTM(10, return_sequences=False))
	model.add(Dropout(0.5))

	model.add(Dense(1))
	model.add(Activation('linear'))

	model.compile(loss="msle", optimizer="rmsprop", metrics=["accuracy"])
=======
	model = getLSTMmodel([50, 200, 10], (1, 60), 'linear', 'msle', 'accuracy')
>>>>>>> c8152f8466a480b91838949c355c9b5dc64f8e5c

	# Entrenamos
	model.fit(xTrain, yTrain, batch_size=xTrain.shape[0], epochs=iterations, validation_data=(xTest, yTest))

	# Predecimos conjunto de entrenamiento
	trainPredictions = model.predict(xTrain)
	yAproxTrain = classifyFromPrediction(xTrain, trainPredictions)

	# Predecimos conjunto de prueba
	predictions = model.predict(xTest)
	yAproxTest = classifyFromPrediction(xTest, predictions)

	# Obtenemos errores de clasificacion
	errorTrain = sum(abs(y1 - np.array(yAproxTrain)))*100/len(y1)
	errorTest = sum(abs(y2 - np.array(yAproxTest)))*100/len(y2)


	print("Training error: "+str(errorTrain)+" Testing error: "+str(errorTest))

	# Guardamos prediccion total
	f = open("test.txt", "w")

	for i in yAproxTrain:
		f.write(str(i)+"\n")

	for i in yAproxTest:
		f.write(str(i)+"\n")

	f.close()


if __name__ == '__main__':
	main()

