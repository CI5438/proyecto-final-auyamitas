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
from Experimentacion.testCaseGenerator import getCases
from indexGenerator import getIndex

# Lectura de datos CSV


def readData(file, sep=";"):
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

    classY1 = classY[:trainSize]
    classY2 = classY[trainSize:]

    if reshape:
        # Redimensionamos los datos de entrenamiento y prueba en 3 dimensiones
        xTrain = np.reshape(
            xTrain, (xTrain.shape[0], features, xTrain.shape[1]))
        xTest = np.reshape(xTest, (xTest.shape[0], features, xTest.shape[1]))

    return xTrain, yTrain, xTest, yTest, classY1, classY2

# Obtiene dos datasets con todos los features ademas de agregar
# una columna de balance de precio y retorna estos datasets, uno en funcion
# de la tendencia y otro del precio, y las matrices de correlacion de cada uno

def getCorrelations(sector, interval, daysAfter):
	# Verificamos que hay indice
	indexFile = 'Mark1 Data/'+sector+'/'+sector+' - Index.txt'
	try:
		index = readData(indexFile, ';')
	except:
		print("El indice no existe. Creando..")
		getIndex(sector, 1, 1)
		index = readData(indexFile, ';')

		print('Indice obtenido. Obteniendo tendencias y precios..')
	trendFile = 'Mark1 Data/'+sector+'/'+sector+'.test.'+str(interval)+'.txt'
	priceFile = 'Mark1 Data/'+sector+'/' + \
		sector+'.priceTest.'+str(interval)+'.txt'

	# Casos unicamente con tendencia
	getCases(sector, interval, 0, daysAfter, 0, '')
	caseTrends = readData(trendFile, ';')[2]

	# Casos unicamente con precios
	getCases(sector, interval, 1, daysAfter, 0, '')
	casePrices = readData(priceFile, ';')[2]

	print('Tendencias y precios obtenidos. Reduciendo el dataset por la prediccion de ' + str(daysAfter) + ' dias despues')
	rowsDiff = index.shape[0] - caseTrends.shape[0]

	if rowsDiff > 0:
		index = index.drop(index.index[-rowsDiff:])

	print('Calculando columna de diferencia de precios..')
	index['price_balance'] = index[4] - index[1]

	# Labeling
	columns = [
		'Date',
		'Opening Price',
		'Highest Price',
		'Lower Price',
		'Closing Price',
		'Volume',
		'Price Balance'
	]

	print('Creando datasets..')
	# Concatenando el indice con las tendencias
	frames = [index, caseTrends]
	indexWithTrends = pd.concat(frames, axis=1)
	indexWithTrends.columns = columns + ['Trend']

	# Concatenando el indice con los precios
	frames = [index, casePrices]
	indexWithPrices = pd.concat(frames, axis=1)
	indexWithPrices.columns = columns + ['Price']

	print('Calculando correlaciones..')
	# Correlacion del indice con tendencias
	corrTrend = indexWithTrends.drop('Date', axis=1).corr()

	# # Correlacion del indice con precios
	corrPrices = indexWithPrices.drop('Date', axis=1).corr()

	print('Listo!')
	return indexWithTrends, indexWithPrices, corrTrend, corrPrices

# Guarda en el directorio Resultados/Correlaciones/Correlaciones 'sector'/
# las matrices de correlacion y las imagenes de todas las correlaciones entre
# features y tendencia/precio, ademas de las graficas de matriz de correlacion
def saveCorrelations(interval, daysAfter):
	import matplotlib.pyplot as plt
	import os

	sectors = os.listdir('Mark1 Data')
	sectors.remove('DJIA')

	for sector in sectors:
		indexTrends, indexPrices, trendsCorr, pricesCorr = getCorrelations(
			sector, interval, daysAfter)
		trendsCorr.to_csv('Resultados/Correlaciones/Correlaciones ' +
							sector + '/trend_correlation.txt', sep=';')
		pricesCorr.to_csv('Resultados/Correlaciones/Correlaciones ' +
							sector + '/price_correlation.txt', sep=';')
		for feature in indexTrends.drop('Date', axis=1):
			if (feature == 'Trend'):
				break
			title = "Gráfico de dispersión de " + feature + " vs Tendencia"
			indexTrends.plot.scatter(x=feature, y='Trend', c=[
				'green'], s=3, legend=True, title=title)
			plt.savefig('Resultados/Correlaciones/Correlaciones ' +
						sector + "/" + feature + " vs Trend.png")
			plt.close()
		for feature in indexPrices.drop('Date', axis=1):
			if (feature == 'Price'):
				break
			title = "Gráfico de dispersión de " + feature + " vs Precio"
			indexPrices.plot.scatter(x=feature, y='Price', c=[
				'green'], s=3, legend=True, title=title)
			plt.savefig('Resultados/Correlaciones/Correlaciones ' +
						sector + "/" + feature + " vs Price.png")
			plt.close()
		save_corrMatrix_plot(trendsCorr, 'Trend', sector)
		save_corrMatrix_plot(pricesCorr, 'Price', sector)

# Permite guardar la matriz de correlacion con un estilo agradable

def save_corrMatrix_plot(corrMatrix, classifier, sector):
	import matplotlib
	import matplotlib.pyplot as plt
	matplotlib.style.use('ggplot')

	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(corrMatrix, vmin=-1, vmax=1)
	fig.colorbar(cax)
	ticks = np.arange(0, 7, 1)
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)
	ax.set_xticklabels(corrMatrix.columns)
	ax.set_yticklabels(corrMatrix.columns)
	plt.savefig('Resultados/Correlaciones/Correlaciones ' +
				sector + "/Correlation Matrix for " + classifier+".png")
	plt.close()
