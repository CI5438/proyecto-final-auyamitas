# Universidad Simón Bolívar
# Abril-Julio 2018
# Inteligencia Artificial II
# Prof. Ivette Carolina Martínez 
# Proyecto Final
# Algoritmo de Predicción de Precio y Tendencia de un Indice

# Autores:
    # Lautaro Villalón 12-10427
    # Yarima Luciani 13-10770
    # Benjamin Amos 12-10240

import sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from superDataLib import *
from superKerasRNNLib import *

# Obtiene argumentos de la linea de comandos
def getArgs():
    try:
        sector = sys.argv[1]
    except:
        print("Ingrese el Sector a Producir el Indice:")
        sector = input()

    try:
        lookBack = int(sys.argv[2])
    except:
        print("Ingrese el Intervalo de Dias por Muestra:")
        lookBack = int(input())

    try:
        daysAfter = int(sys.argv[3])
    except:
        print("Ingrese los Dias a Predecir despues del Ultimo:")
        daysAfter = int(input())

    try:
        percentage = float(sys.argv[4])
    except:
        print("Ingrese el Porcentaje de Entrenamiento:")
        percentage = float(input())

    try:
        iters = int(sys.argv[5])
    except:
        print("Ingrese la Cantidad de Iteraciones Deseadas:")
        iters = int(input())

    return sector, lookBack, daysAfter, percentage, iters

# Corrida del algoritmo de prediccion
def main():
	# Mejor de los Casos:
	# lookBack = 1
	# daysAfter = 2
	# percentage = 0.9
	# iterations = 200
	# sector = 'Communication Services Sector'

	sector, lookBack, daysAfter, percentage, iterations = getArgs()

	# Preparamos el archivo del indice (sector, scale?, scaleRange)
	df = prepareData(sector, True, (0, 1))

	# Obtenemos DataSet
	x, y, classY = createDataSet(df, lookBack, daysAfter)

	# Dividimos DataSet (x, y, classY, trainSize, features, reshape)
	trainSize = int(len(y)*percentage)

	xTrain, yTrain, xTest, yTest, classY1, classY2 = splitDataSet(x, y, classY, trainSize, 1, True)

	# Creamos el modelo: (layers, inputSize, activation, loss, metrics)

	model = getLSTMmodel([50, 200, 10], (1, lookBack), 'linear', 'msle', 'accuracy')

	# Entrenamos
	model.fit(xTrain, yTrain, batch_size=xTrain.shape[0], epochs=iterations, validation_data=(xTest, yTest))

	# Predecimos conjunto de entrenamiento
	trainPredictions = model.predict(xTrain)
	yAproxTrain = classifyFromPrediction(xTrain, trainPredictions)

	# Predecimos conjunto de prueba
	predictions = model.predict(xTest)
	yAproxTest = classifyFromPrediction(xTest, predictions)

	# Obtenemos errores de clasificacion
	errorTrain = sum(abs(classY1 - yAproxTrain))*100/len(classY1)
	errorTest = sum(abs(classY2 - yAproxTest))*100/len(classY2)

	print("Training error: "+str(errorTrain)+" Testing error: "+str(errorTest))

	# Ploteamos resultados
	plt.plot([i for i in range(len(yTrain))], yTrain, color='b')
	plt.plot([i for i in range(len(trainPredictions))], trainPredictions, color='r')
	plt.show()

	plt.plot([i for i in range(len(yTest))], yTest, color='b')
	plt.plot([i for i in range(len(predictions))], predictions, color='r')
	plt.show()

	plt.plot([i for i in range(len(classY1))], classY1, 'b-')
	plt.plot([i for i in range(len(yAproxTrain))], yAproxTrain, 'r-')
	plt.show()

	plt.plot([i for i in range(len(classY2))], classY2, 'b-')
	plt.plot([i for i in range(len(yAproxTest))], yAproxTest, 'r-')
	plt.show()


	# Guardamos prediccion total
	# f = open("test.txt", "w")

	# for i in range(len(yAproxTrain)):
	# 	f.write(str(yTrain[i])+"->"+str(trainPredictions[i])+"\n")

	# for i in range(len(yAproxTest)):
	# 	f.write(str(yTrain[i])+"->"+str(predictions[i])+"\n")

	# f.close()


if __name__ == '__main__':

	main()

