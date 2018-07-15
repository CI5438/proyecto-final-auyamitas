import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Lectura de datos CSV
def readData(file, sep):
    dataSet = pd.read_csv(file, sep=sep, comment="#", header=None)
    return dataSet

def createDataSet(df, lookBack, daysAfter):
	x = []
	y = []
	
	for i in range(len(df) - lookBack-daysAfter+1):
		a = df[i:(i + lookBack), 2]
		x.append(a)
		y.append(df[i + lookBack + daysAfter-1, 2])

	return np.array(x), np.array(y)

def createDataSetDiff(df, lookBack, daysAfter):
	x = []
	y = []
	
	for i in range(len(df) - lookBack-daysAfter):
		a = df[i:(i + lookBack), 2] - df[i:(i + lookBack), 1]
		x.append(a)
		b = df[i + lookBack + daysAfter-1, 2] - df[i + lookBack + daysAfter-1, 1]
		y.append(b)

	return np.array(x), np.array(y)

def classifiedY(sector, caseSize):

	testCase = 'Mark1 Data/'+sector+'/'+sector+'.test.'+str(caseSize)+'.txt'
	df = readData(testCase, ';')

	return df[df.columns[-1]]

def prepareData(sector, scale, scaleRange):
	file = 'Mark1 Data/'+sector+'/'+sector+' - Index.txt'
	df = readData(file, ';')

	if scale:
		scaler = MinMaxScaler(feature_range=(0, 1))    # mejor manera?
	
		return scaler.fit_transform(df)
	else:
		return np.array(df)

def splitDataSet(x, y, trainSize, features, reshape=True):
	# Dividimos los datos a aproximar en entrenamiento y prueba
	xTrain = x[:trainSize, :]
	xTest = x[trainSize:, :]
	yTrain = y[:trainSize]
	yTest = y[trainSize:]

	if reshape:
		# Redimensionamos los datos de entrenamiento y prueba en 3 dimensiones
		xTrain = np.reshape(xTrain, (xTrain.shape[0], features, xTrain.shape[1]))
		xTest = np.reshape(xTest, (xTest.shape[0], features, xTest.shape[1]))

	return xTrain, yTrain, xTest, yTest
	