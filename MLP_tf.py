import tensorflow as tf
import numpy as np
import pandas as pd
import sys


def getData(file):
    data = np.loadtxt(file, delimiter=";")
    x = data[:, :-1]
    y = data[:, -1]

    return x, y


def main():
    # Se obtiene la data procesada en np.arrays
    x, y = getData(
        "./Mark1 Data/Consumer Cyclical Sector/Consumer Cyclical Sector.test.30.txt")

    # Se definen placeholders para las variables de datos en tensorflow
    x_ = tf.placeholder(tf.float32, [None, x.shape[1]])
    y_ = tf.placeholder(tf.int8, [None, y.shape[1]])

    # Parametros del funcionamiento de la red
    LEARNING_RATE = 0.0005          # Tasa de aprendizaje
    EPOCHS = 20                     # Numero de ciclos de feedforward - backpropagation
    # Numero de neuronas para input layer (tantas como filas en el archivo)
    N_INPUT = x.shape[0]
    N_HIDDEN_LAYERS = 2             # Numero de hidden layers
    # Por motivos de prueba, tendran la mitad de neuronas que input layer
    HIDDEN_LAYER_SIZE = N_INPUT/2
    OUTPUT_LAYER_SIZE = 1           # Output layer tendra una sola neurona
    N_CLASSES = 3                   # Los posibles valores que puede tomar la tendencia

    def DeepNeuralNetwork(n_hidden_layers):
        hidden_layers = []
        actLayerSize = x.shape[0]

        # Se definen las hidden layers que tendra la red
        for i in range(0, n_hidden_layers):
            hidden_layer_i = {
                "weights": tf.Variable(tf.random_normal([actLayerSize, HIDDEN_LAYER_SIZE])),
                "biases": tf.Variable(tf.random_normal(HIDDEN_LAYER_SIZE)),
                "name": "hidden_layer_" + str(i)
            }
            actLayerSize = HIDDEN_LAYER_SIZE
            hidden_layers.append(hidden_layer_i)

        # Se define la output layer
        output_layer = {
            "weights": tf.Variable(tf.random_normal([HIDDEN_LAYER_SIZE, N_CLASSES)),
            "biases": tf.Variable(tf.random_normal(N_CLASSES)),
            "name": "output_layer"
        }

        hidden_layers.append(output_layer)

        # Se calcula el output a traves del feed forward
        actLayer = x
        for layer in hidden_layers:
            if layer["name"] != "output_layer":
                li = tf.nn.relu_layer(actLayer, layer["weights"], layer["biases"])
                actLayer = li
            else:
                output = tf.matmul(li, layer["weights"]) + layer["biases"]

        return output

if __name__ == '__main__':
    main()
