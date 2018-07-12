import numpy as np
from deepNeuralLib import *
from dataLib import *
import matplotlib.pyplot as plt

def main():
	file = 'Mark1 Data/Technology Sector/Technology Sector.priceTest.30.txt'
	sectorName = file.split('/')[1]

	df = readData(file, ';')

	xNorm, x, y = prepareData(df, 60)

	xTrain = xNorm[:1985, 1:]
	yTrain = y[:1985, :]
	xTest = xNorm[1985:, 1:]
	yTest = y[1985:, :]

	n_hidden_layers = 2
	hidden_size = [180, 120]
	out_classes = 1

	learning_rate = 0.01
	epochs = 20

	deepNN = KerasDeepNN(hidden_size, xTrain.shape[0], 2, 30)

	error = deepNN.train(xTrain, yTrain, xTest, yTest, epochs)

	yAprox = deepNN.predict(xTest)

	print(accuracyVector(yTest, yAprox))

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