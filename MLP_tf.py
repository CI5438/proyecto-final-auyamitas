import tensorflow as tf
import numpy as np
import json
from deepNeuralLib import *
from dataLib import getData


# def getData(file):
#     print("Getting data from", file)
#     data = np.loadtxt(file, delimiter=";")
#     x = data[:, :-1]
#     y = data[:, -1]

#     print("Retrieved data. Sliced features in a matrix with",
#           x.shape[0], "rows and", x.shape[1], "columns")
#     return x, y


def main():
    file = 'Mark1 Data/Consumer Cyclical Sector/Consumer Cyclical Sector.test.30.txt'
    sectorName = file.split('/')[1]
    x, y = getData(file, True)

    n_hidden_layers = 2
    hidden_size = 2
    out_classes = 1

    learning_rate = 0.0005
    epochs = 12

    deepNeuralNetwork = DeepNeuralNetworkTF(x, y, n_hidden_layers, hidden_size, out_classes)
    deepNeuralNetwork.train(learning_rate, epochs)
    #deepNeuralNetwork.exportLayers(sectorName)
    prediction = deepNeuralNetwork.feedForward(x, True)

    prediction = prediction


    f = open("test.txt", "w")

    for trend in prediction:
        f.write(str(trend)+"\n")

    f.close()

if __name__ == '__main__':
    main()
