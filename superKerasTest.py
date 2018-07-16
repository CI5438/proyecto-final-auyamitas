import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from superDataLib import *
from superKerasRNNLib import *


def main():
	lookBack = 45
	daysAfter = 2
	percentage = 0.9
	iterations = 20

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

	# Dividimos DataSet (x, y, trainSize, features, reshape)
	xTrain, yTrain, xTest, yTest = splitDataSet(x, y, trainSize, 1, True)

	# Creamos el modelo: (layers, inputSize, activation, loss, metrics)

	model = getLSTMmodel([50, 200, 10], (1, lookBack),
	                     'linear', 'msle', 'accuracy')

	# Entrenamos
	model.fit(xTrain, yTrain,
	          batch_size=xTrain.shape[0], epochs=iterations, validation_data=(xTest, yTest))

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

	trainValues = xTrain[:, 0, -1]

	plt.plot([i for i in range(len(trainValues))], trainValues, color='b')
	plt.plot([i for i in range(len(trainPredictions))],
	         trainPredictions, color='r')
	plt.show()

	testValues = xTest[:, 0, -1]

	plt.plot([i for i in range(len(testValues))], testValues, color='b')
	plt.plot([i for i in range(len(predictions))], predictions, color='r')
	plt.show()

	plt.plot([i for i in range(len(y1))], y1, 'b-')
	plt.plot([i for i in range(len(yAproxTrain))], yAproxTrain, 'r-')
	plt.show()

	plt.plot([i for i in range(len(y2))], y2, 'b-')
	plt.plot([i for i in range(len(yAproxTest))], yAproxTest, 'r-')
	plt.show()


if __name__ == '__main__':

	# daysTest = [1,2,3,4,5,7,10,15,20,25,30,45,60]
	# for day in daysTest:
	main()
