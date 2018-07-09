import tensorflow as tf
import numpy as np
import pandas as pd
import sys


def getData(file):
    print("Getting data from", file)
    data = np.loadtxt(file, delimiter=";")
    x = data[:, :-1]
    y = data[:, -1]

    print("Retrieved data. Sliced features in a matrix with",
          x.shape[0], "rows and", x.shape[1], "columns")
    return x, y


def main():
    print("Starting execution")
    # Se obtiene la data procesada en np.arrays
    x, y = getData(
        "./Mark1 Data/Consumer Cyclical Sector/Consumer Cyclical Sector.test.30.txt")

    print("Defining tensorflow variables")
    # Se definen placeholders para las variables de datos en tensorflow
    x_ = tf.placeholder(tf.float32, [None, x.shape[1]])
    y_ = tf.placeholder(tf.int8, [None, 1])
    print("Defining parameters")
    # Parametros del funcionamiento de la red
    LEARNING_RATE = 0.0005          # Tasa de aprendizaje
    EPOCHS = 20                     # Numero de ciclos de feedforward - backpropagation
    # Numero de neuronas para input layer (tantas como filas en el archivo)
    N_INPUT = x.shape[0]
    N_HIDDEN_LAYERS = 2             # Numero de hidden layers
    # Por motivos de prueba, tendran la mitad de neuronas que input layer
    HIDDEN_LAYER_SIZE = int(N_INPUT/2)
    OUTPUT_LAYER_SIZE = 1           # Output layer tendra una sola neurona
    N_CLASSES = 1                   # Los posibles valores que puede tomar la tendencia
    print("Learning rate:", LEARNING_RATE, "\nEpochs:",
          EPOCHS, "\nInput layer neurons:", N_INPUT, "\nHidden layers:",
          N_HIDDEN_LAYERS, "\nHidden layers size:", HIDDEN_LAYER_SIZE,
          "\nNumber of classes:", N_CLASSES)

    def DeepNeuralNetwork(n_hidden_layers):
        print("Starting DNN model build..")
        hidden_layers = []
        actLayerSize = x.shape[1]

        print("Setting hidden layers weights and biases..")
        # Se definen las hidden layers que tendra la red
        for i in range(0, n_hidden_layers):
            hidden_layer_i = {
                "weights": tf.Variable(tf.random_normal([actLayerSize, HIDDEN_LAYER_SIZE]), dtype=tf.float32),
                "biases": tf.Variable(tf.random_normal([HIDDEN_LAYER_SIZE]), dtype=tf.float32),
                "name": "hidden_layer_" + str(i)
            }
            actLayerSize = HIDDEN_LAYER_SIZE
            hidden_layers.append(hidden_layer_i)
        print("Done.")
        print("Setting output layer weights and biases..")
        # Se define la output layer
        output_layer = {
            "weights": tf.Variable(tf.random_normal([HIDDEN_LAYER_SIZE, N_CLASSES]), dtype=tf.float32),
            "biases": tf.Variable(tf.random_normal([N_CLASSES]), dtype=tf.float32),
            "name": "output_layer"
        }
        print('Done.')
        hidden_layers.append(output_layer)

        # Se calcula el output a traves del feed forward
        x_tf = tf.convert_to_tensor(x, np.float32)
        actLayer = x_tf
        print("Preparing output for prediction..")
        for layer in hidden_layers:
            # En caso de que sean hidden layers tomamos como referencia la capa anterior
            # y aplicamos la funcion de activacion
            if layer["name"] != "output_layer":
                li = tf.nn.relu_layer(
                    actLayer, layer["weights"], layer["biases"])
                actLayer = li
            # En caso de que sea la capa oculta, solo hacemos la multiplicacion de los valores de la capa
            # anterior por el peso mas su bias
            else:
                output = tf.nn.xw_plus_b(li, layer["weights"], layer["biases"])

        print("Output ready.")
        print("Tensor DNN output:", output)
        print("DNN model built.")
        return output

    def train_neural_network(x, y):
        print("Starting training..")

        x_tf = tf.convert_to_tensor(x, np.float32)
        y = np.array([y], dtype=np.float32, ndmin=1)
        y = np.transpose(y)
        
        print(y.shape)
        
        y_tf = tf.convert_to_tensor(np.transpose(y), np.float32)
        
        print("Tensor x", x_tf, "Tensor y", y_tf)
        print("Getting the DNN model..")
        
        prediction = DeepNeuralNetwork(N_HIDDEN_LAYERS)

        print("Preparing cost function and optimizer")
        
        # cost = tf.reduce_mean(
            # tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y_tf))
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.transpose(prediction), logits=y_tf)
            )
        optimizer = tf.train.GradientDescentOptimizer(
            LEARNING_RATE).minimize(cost)
        
        print("Done.")

        print("Starting tensorflow session..")
        
        with tf.Session() as sess:
            print("Initializing variables..")
            sess.run(tf.global_variables_initializer())

            print("Running epochs..")
            
            for epoch in range(EPOCHS):
                epoch_loss = 0
                for _ in range(x.shape[0]):
                    _, c = sess.run([optimizer, cost],
                                    feed_dict={x_: x, y_: y})
                    epoch_loss += c
                
                print("Epoch", epoch, "completed out of", EPOCHS, "loss")

    train_neural_network(x, y)

    # DNN = DeepNeuralNetwork(N_HIDDEN_LAYERS)


if __name__ == '__main__':
    main()
