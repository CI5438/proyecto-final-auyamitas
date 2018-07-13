import numpy as np
from deepNeuralLib import *
from dataLib import *
import matplotlib.pyplot as plt

def main():
	file = 'Mark1 Data/Technology Sector/Technology Sector.test.30.txt'
	sectorName = file.split('/')[1]

	df = readData(file, ';')

	xNorm, x, y = prepareData(df, 27)

	x = np.array([[i] for i in range(200)])
	x = np.c_[np.ones(len(x)), x]
	y = np.array([[i**3] for i in range(200)])


	n_hidden_layers = 1
	hidden_size = [180]
	out_classes = 1

	learning_rate = 0.01
	epochs = 1000

	deepNN = DeepNeuralNetwork(x.shape[1]-1, hidden_size, out_classes, n_hidden_layers, 3)

	error = deepNN.train(x, y, learning_rate, epochs, 0)

	yAprox = deepNN.feedForward(x)

	print(accuracyVector(y, yAprox))

	print("SQD ERROR:")
	
	plt.plot([i for i in range(len(error))], error, color='b', linestyle='-')
	plt.show()


	f = open("test1.txt", "w")

	for i in range(len(y)):
		f.write(str(y[i])+"->"+str(yAprox[i])+"\n")

	f.close()



if __name__ == '__main__':
	main()

