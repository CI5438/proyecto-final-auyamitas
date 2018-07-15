# Universidad Simón Bolívar
# Abril-Julio 2018
# Inteligencia Artificial II
# Prof. Ivette Carolina Martínez 
# Proyecto Final
# Prueba con Algoritmo en Keras

# Autores:
    # Lautaro Villalón 12-10427
    # Yarima Luciani 13-10770
    # Benjamin Amos 12-10240

import numpy as np
from deepNeuralLib import *
from dataLib import *
import matplotlib.pyplot as plt

def main():
	# file = sys.argv[1]
	file = 'Mark1 Data/Consumer Cyclical Sector/Consumer Cyclical Sector.test.30.txt'

	sectorName = file.split('/')[1]

	df = readData(file, ';')

	xNorm, x, y = prepareData(df, 27)

	xTrain = xNorm[:2081, 1:]
	yTrain = y[:2081, :]
	xTest = xNorm[2075:, 1:]
	yTest = y[2075:, :]

	print(yTest)

	x = np.array([[i] for i in range(200)])
	y = np.array([[i**3] for i in range(200)])

	xTrain = x[:150, :]
	yTrain = y[:150, :]
	xTest = x[150:, :]
	yTest = y[150:, :]

	n_hidden_layers = 2
	hidden_size = [200, 120]
	out_classes = 1

	learning_rate = 0.001
	epochs = 20

	deepNN = KerasDeepNN(hidden_size, xTrain.shape[0], 1, 1)

	error = deepNN.train(xTrain, yTrain, xTest, yTest, epochs)

	yAprox = deepNN.predict(xTest)
	print(yAprox)
	print(yTest, yAprox)
	print(accuracyVector(yTest, yAprox))

	print("SQD ERROR:")

	plt.plot(error.history['loss'], label='train')
	plt.plot(error.history['val_loss'], label='test')
	plt.legend()
	plt.show()

	f = open("test2.txt", "w")

	for i in range(len(yTest)):
		f.write(str(yTest[i])+"->"+str(yAprox[i]) +
				" = "+str(yAprox[i].round())+"\n")

	f.close()


if __name__ == '__main__':
    main()
