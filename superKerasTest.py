import numpy as np 
import pandas as pd 

from subprocess import check_output
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.cross_validation import  train_test_split
import time 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import newaxis

from dataLib import *


def createDataSet(df, lookBack=30):
	x = []
	y = []
	
	for i in range(len(df) - lookBack-3):
		a = df[i:(i + lookBack), 2]
		x.append(a)
		y.append(df[i + lookBack + 2, 2])

	return np.array(x), np.array(y)


def main():

	# Obtenemos la y real
	testCase = 'Mark1 Data/Communication Services Sector/Communication Services Sector.test.30.txt'
	df = readData(testCase, ';')
	y = df[df.columns[-1]]


	trainSize = int(len(df)*0.9)
	y1 = y[:trainSize]
	y2 = y[trainSize:]

	# Preparamos el archivo del indice
	file = 'Mark1 Data/Communication Services Sector/Communication Services Sector - Index.txt'
	sectorName = file.split('/')[1]
	df = readData(file, ';')

	scaler = MinMaxScaler(feature_range=(0, 1))    # mejor manera?
	df = scaler.fit_transform(df)

	lookBack = 30 
	x, y = createDataSet(df, lookBack)
	#xTest, yTest = createDataSet(test, lookBack)

	xTrain = x[:trainSize, :]
	xTest = x[trainSize:, :]
	yTrain = y[:trainSize]
	yTest = y[trainSize:]

	xTrain = np.reshape(xTrain, (xTrain.shape[0], 1, xTrain.shape[1]))
	xTest = np.reshape(xTest, (xTest.shape[0], 1, xTest.shape[1]))

	# Creamos el modelo
	model = Sequential()

	model.add(LSTM(50, input_shape=(1,30), return_sequences=True))
	model.add(Dropout(0.5))

	model.add(LSTM(200, return_sequences=True))
	model.add(Dropout(0.5))

	model.add(LSTM(10, return_sequences=False))
	model.add(Dropout(0.5))

	model.add(Dense(1))
	model.add(Activation('linear'))

	model.compile(loss="msle", optimizer="rmsprop", metrics=["accuracy"])

	# Entrenamos
	model.fit(xTrain, yTrain, batch_size=xTrain.shape[0], epochs=500, validation_data=(xTest, yTest))

	# Predecimos conjunto de entrenamiento
	predictions = model.predict(xTrain)


	yAproxTrain = []
	for i in range(len(xTrain)):
		if xTrain[i, 0, -1] < predictions[i]:
			yAproxTrain.append(1)
		else:
			yAproxTrain.append(0)

	# Predecimos conjunto de prueba
	testPredictions = model.predict(xTest)

	yAproxTest = []
	for i in range(len(xTest)):
		if xTest[i, 0, -1] < predictions[i]:
			yAproxTest.append(1)
		else:
			yAproxTest.append(0)

	errorTrain = sum(abs(y1 - np.array(yAproxTrain)))*100/len(y1)
	errorTest = sum(abs(y2 - np.array(yAproxTest)))*100/len(y2)

	print("Training error: "+str(errorTrain)+" Testing error: "+str(errorTest))


	f = open("test.txt", "w")

	for i in yAproxTrain:
		f.write(str(i)+"\n")

	f.close()


if __name__ == '__main__':
	main()

