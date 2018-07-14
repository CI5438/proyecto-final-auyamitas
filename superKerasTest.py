import numpy as np 
import pandas as pd 

# from subprocess import check_output

# from sklearn.cross_validation import  train_test_split
# import time 
#from numpy import newaxis

import matplotlib.pyplot as plt
from superDataLib import *
from superKerasRNNLib import *

def main():
	lookBack = 30
	daysAfter = 1
	percentage = 0.9
	iterations = 500

	# Obtenemos la y real (sector, lookBack)
	y = classifiedY('Communication Services Sector', lookBack)

	# Dividimos la y real en entrenamiento y prueba
	trainSize = int(len(y)*percentage)

	y1 = y[:trainSize]
	y2 = y[trainSize:]

	# Preparamos el archivo del indice (sector, scale?, scaleRange)
	df = prepareData('Communication Services Sector', True, (0, 1))

	# Obtenemos DataSet
	x, y = createDataSet(df, lookBack, daysAfter)

	# Dividimos DataSet (x, y, trainSize, features)
	xTrain, yTrain, xTest, yTest = splitDataSet(x, y, trainSize, 1)

	# Creamos el modelo: (layers, inputSize, activation, loss, metrics)

	model = getLSTMmodel([50, 200, 10], (1, 30), 'linear', 'msle', 'accuracy')

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

	f = open("test.txt", "a")

	print("Training error: "+str(errorTrain)+" Testing error: "+str(errorTest))

	# Guardamos prediccion total
	f = open("test.txt", "w")

	for i in yAproxTrain:
		f.write(str(i)+"\n")

	for i in yAproxTest:
		f.write(str(i)+"\n")

	f.close()


if __name__ == '__main__':

	# daysTest = [1,2,3,4,5,7,10,15,20,25,30,45,60]
	# for day in daysTest:
	main()

