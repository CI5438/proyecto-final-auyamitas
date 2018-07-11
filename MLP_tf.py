import tensorflow as tf
import numpy as np
import json

class DeepNeuralNetwork():
    def __init__(self, file, n_hidden):
        self.file = file
        self.x, self.y = self.getData(self.file)

        self.LEARNING_RATE = 0.001
        self.EPOCHS = 10
        self.N_INPUT = self.x.shape[0]
        self.N_HIDDEN_LAYERS = 2
        self.HIDDEN_LAYER_SIZE = 15
        self.N_CLASSES = 1

        self.layers = []
        self.x_ = tf.placeholder(tf.float32, [None, self.x.shape[1]])
        self.y_ = tf.placeholder(tf.int8, [None, 1])

    def getData(self, file):
        print("Getting data from", file)
        data = np.loadtxt(file, delimiter=";")
        x = data[:, :-1]
        y = data[:, -1]

        print("Retrieved data. Sliced features in a matrix with",
              x.shape[0], "rows and", x.shape[1], "columns")
        return x, y

    def setDNNModel(self):
        actLayerSize = self.x.shape[1]

        # Se definen las hidden layers que tendra la red
        for i in range(0, self.N_HIDDEN_LAYERS):
            hidden_layer_i = {
                "weights": tf.Variable(tf.random_normal(
                        [actLayerSize, self.HIDDEN_LAYER_SIZE]), 
                        dtype=tf.float32
                    ),
                "biases": tf.Variable(
                    tf.random_normal([self.HIDDEN_LAYER_SIZE]),
                     dtype=tf.float32
                    ),
                "name": "hidden_layer_" + str(i)
            }
            actLayerSize = self.HIDDEN_LAYER_SIZE
            self.layers.append(hidden_layer_i)
        
        # Se define la output layer
        output_layer = {
            "weights": tf.Variable(
                    tf.random_normal([self.HIDDEN_LAYER_SIZE, self.N_CLASSES]),
                    dtype=tf.float32
                ),
            "biases": tf.Variable(
                tf.random_normal([self.N_CLASSES]),
                dtype=tf.float32
                ),
            "name": "output_layer"
        }
        self.layers.append(output_layer)

        # Se calcula el output a traves del feed forward
        x_tf = tf.convert_to_tensor(self.x, np.float32)
        actLayer = x_tf

        for layer in self.layers:
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

        return output

    def trainDNN(self):
        # Se convierten los arrays en tensors y se transpone *y* por consistencia con
        # la prediccion pues es un arreglo plano y debe ser una matriz columna
        x_tf = tf.convert_to_tensor(self.x, np.float32)
        self.y = np.array([self.y], dtype=np.float32, ndmin=1)
        self.y = np.transpose(self.y)
        y_tf = tf.convert_to_tensor(np.transpose(self.y), np.float32)

        # Se obtiene la prediccion
        prediction = self.setDNNModel()

        # Aca decidir entre usar como funcion de activacion cross entropy (hace backprop) a menos que
        # se indique lo contrario) o sigmoid

        # cost = tf.reduce_mean(
        #   tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y_tf)
        # )
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=tf.transpose(prediction), labels=y_tf)
        )
        optimizer = tf.train.GradientDescentOptimizer(
            self.LEARNING_RATE).minimize(cost)

        # Session es el entorno de ejecucion de tensorflow y aqui ocurre la ejecucion
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Una epoca es FeedForward + BackPropagation
            for epoch in range(self.EPOCHS):
                epoch_loss = 0

                # _ es una variable que no me interesa
                for _ in range(self.x.shape[0]):
                    _, c = sess.run([optimizer, cost],
                                    feed_dict={self.x_: self.x, self.y_: self.y})
                    # print(c)
                    epoch_loss += c

                print("Epoch", epoch, "completed out of",
                      self.EPOCHS, "loss:", epoch_loss)

        # Se muestra la precision del modelo
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        # Aqui interesa
        # print("Accuracy:", accuracy.eval(x_: matriz de prueba, y_: vector clase de prueba))

    def exportLayersAsJSON(self):
        sectorName = self.file.split('/')[1]
        json.dump(self.layers, sectorName + ".DNNweights.json")

    def importLayersFromJSON(self, fileName):
        with open(fileName) as f:
            self.layers = json.load(f)

def main():
    file = 'Mark1 Data/Consumer Cyclical Sector/Consumer Cyclical Sector.test.60.txt'
    n_hidden_layers = 2
    deepNeuralNetwork = DeepNeuralNetwork(file, n_hidden_layers)
    deepNeuralNetwork.trainDNN()

if __name__ == '__main__':
    main()
