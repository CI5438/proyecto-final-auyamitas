import numpy as np
from deepNeuralLib import *
from dataLib import *
import matplotlib.pyplot as plt

def main():
	file = 'Mark1 Data/Technology Sector/Technology Sector.test.30.txt'
	sectorName = file.split('/')[1]

	df = readData(file, ';')

	xNorm, x, y = prepareData(df, 27)

	xTrain = x[:1985, :]
	yTrain = y[:1985, :]
	xTest = x[1985:, :]
	yTest = y[1985:, :]

	x = np.array([[i] for i in range(200)])
	y = np.array([[i**3] for i in range(200)])

	xTrain = x[:150, :]
	yTrain = y[:150, :]
	xTest = x[150:, :]
	yTest = y[150:, :]

	n_hidden_layers = 2
	hidden_size = [200, 120]
	out_classes = 1

	learning_rate = 0.01
	epochs = 200
m
	print(xTrain)
	print(yTrain)

	deepNN = KerasDeepNN(hidden_size, xTrain.shape[0], 1, 1)

	error = deepNN.train(xTrain, yTrain, xTest, yTest, epochs)

	yAprox = deepNN.predict(xTrain)

	print(accuracyVector(yTrain, yAprox))

	print("SQD ERROR:")
	
	plt.plot(error.history['loss'], label='train')
	plt.plot(error.history['val_loss'], label='test')
	plt.legend()
	plt.show()


	f = open("test2.txt", "w")

	for i in range(len(yTest)):
		f.write(str(yTest[i])+"->"+str(yAprox[i])+" = "+str(yAprox[i].round())+"\n")

	f.close()



if __name__ == '__main__':
	main()