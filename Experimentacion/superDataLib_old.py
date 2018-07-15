# Universidad Simón Bolívar
# Abril-Julio 2018
# Inteligencia Artificial II
# Prof. Ivette Carolina Martínez 
# Proyecto Final
# Libreria de Carga de Datos Experimentacion

# Autores:
    # Lautaro Villalón 12-10427
    # Yarima Luciani 13-10770
    # Benjamin Amos 12-10240

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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

def classifiedY(sector, caseSize):

	testCase = 'Mark1 Data/'+sector+'/'+sector+'.test.'+str(caseSize)+'.txt'
	df = readData(testCase, ';')

	return df[df.columns[-1]]