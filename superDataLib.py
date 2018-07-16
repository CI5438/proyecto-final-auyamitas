import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from testCaseGenerator import getCases
from indexGenerator import getIndex

# Lectura de datos CSV


def readData(file, sep=";"):
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

    for i in range(len(df) - lookBack-daysAfter+1):
        a = df[i:(i + lookBack), 2] - df[i:(i + lookBack), 1]
        x.append(a)

        if df[i+lookBack-1, 2] < df[i+lookBack+daysAfter-1, 2]:
            y.append(1)
        else:
            y.append(0)

        # b = df[i + lookBack + daysAfter-1, 2] - df[i + lookBack + daysAfter-1, 1]
        # y.append(b)

    return np.array(x), np.array(y)


def classifiedY(sector, caseSize, type=None):
    try:
        if (not type or type == 'trend'):
            testCase = 'Mark1 Data/'+sector+'/' + \
                sector+'.test.'+str(caseSize)+'.txt'
        elif (type == 'price'):
            testCase = 'Mark1 Data/'+sector+'/' + \
                sector+'.priceTest.'+str(caseSize)+'.txt'
    except FileNotFoundError:
        raise FileNotFoundError('Error: Archivo no encontrado')
    df = readData(str(testCase), ';')

    return df[df.columns[-1]]


def readIndex(sector):
    file = 'Mark1 Data/'+sector+'/'+sector+' - Index.txt'
    df = readData(file, ";")
    return df


def prepareData(sector, scale, scaleRange):
    # file = 'Mark1 Data/'+sector+'/'+sector+' - Index.txt'
    # df = readData(file, ';')
    df = readIndex(sector)

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
        xTrain = np.reshape(
            xTrain, (xTrain.shape[0], features, xTrain.shape[1]))
        xTest = np.reshape(xTest, (xTest.shape[0], features, xTest.shape[1]))

    return xTrain, yTrain, xTest, yTest


def getCorrelations(sector, interval, daysAfter):
	# Verificamos que hay indice
	try:
		index = readIndex(sector)
	except:
		print("El indice no existe. Creando..")
		getIndex(sector, 1, 1)
		index = readIndex(sector)

	# Casos unicamente con tendencia
	getCases(sector, interval, 0, daysAfter, 0, '')
	caseTrends = classifiedY(sector, interval)

	# Casos unicamente con precios
	getCases(sector, interval, 1, daysAfter, 0, '')
	casePrices = classifiedY(sector, interval, 'price')

	rowsDiff = index.shape[0] - caseTrends.shape[0]

	if rowsDiff > 0:
		index = index.drop(index.index[-rowsDiff:])

	index['price_balance'] = index[4] - index[1]

    # Concatenando el indice con las tendencias
	frames = [index, caseTrends]
	indexWithTrends = pd.concat(frames, axis=1)
	indexWithTrends.columns = [
        'date',
		'opening_price',
		'highest_price', 
		'lower_price', 
		'closing_price', 
		'volume',
		'price_balance', 
		'trend'
	]

    # Concatenando el indice con los precios
    frames = [index, casePrices]
    indexWithPrices = pd.concat(frames, axis=1)
    indexWithPrices.columns = [
        'date', 
		'opening_price', 
		'highest_price', 
		'lower_price', 
		'closing_price', 
		'volume', 
		'price'
	]

    # Correlacion del indice con tendencias
	corrTrend = indexWithTrends.drop('date', axis=1).corr()

    # # Correlacion del indice con precios
    corrPrices = indexWithPrices.drop('date', axis=1).corr()

	return indexWithTrends, corrTrend, corrPrices
