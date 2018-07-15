# Universidad Simón Bolívar
# Abril-Julio 2018
# Inteligencia Artificial II
# Prof. Ivette Carolina Martínez 
# Proyecto Final
# Libreria de Manipulación de Datos

# Autores:
    # Lautaro Villalón 12-10427
    # Yarima Luciani 13-10770
    # Benjamin Amos 12-10240

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Lectura de datos CSV
def readData(file, sep):
    dataSet = pd.read_csv(file, sep=sep, comment="#", header=None)
    return dataSet

# Abre un archivo y escala los datos en el rango deseado
def prepareData(sector, scale, scaleRange):
	file = 'Mark1 Data/'+sector+'/'+sector+' - Index.txt'
	df = readData(file, ';')

	if scale:
		scaler = MinMaxScaler(feature_range=(0, 1))    # mejor manera?
	
		return scaler.fit_transform(df)
	else:
		return np.array(df)

# Crea los casos de prueba con su X, Y y Y de clasificacion
def createDataSet(df, lookBack, daysAfter):
	x = []
	y = []
	yClass = []
	
	for i in range(len(df) - lookBack-daysAfter+1):
		a = df[i:(i + lookBack), 2]
		x.append(a)
		y.append(df[i + lookBack + daysAfter-1, 2])

		if df[i+lookBack+daysAfter-1, 2] > df[i+lookBack-1, 2]:
			yClass.append(1)
		else:
			yClass.append(0)

	return np.array(x), np.array(y), np.array(yClass)

# Divide el caso de prueba en entrenamiento y prueba
def splitDataSet(x, y, classY, trainSize, features, reshape=True):
	# Dividimos los datos a aproximar en entrenamiento y prueba
	xTrain = x[:trainSize, :]
	xTest = x[trainSize:, :]
	yTrain = y[:trainSize]
	yTest = y[trainSize:]

	classY1= classY[:trainSize]
	classY2 = classY[trainSize:]

	if reshape:
		# Redimensionamos los datos de entrenamiento y prueba en 3 dimensiones
		xTrain = np.reshape(xTrain, (xTrain.shape[0], features, xTrain.shape[1]))
		xTest = np.reshape(xTest, (xTest.shape[0], features, xTest.shape[1]))

	return xTrain, yTrain, xTest, yTest, classY1, classY2
	