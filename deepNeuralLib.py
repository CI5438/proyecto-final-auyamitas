# Universidad Simón Bolívar
# Abril-Julio 2018
# Inteligencia Artificial II
# Prof. Ivette Carolina Martínez 
# Proyecto 2
# Libreria de Redes Neuronales

# Autores:
    # Lautaro Villalón 12-10427
    # Yarima Luciani 13-10770
    # Benjamin Amos 12-10240

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import json
import sys

# Funcion sigmoidal
def sigmoid(x):
    sig = 1/(1+np.exp(-x))

    return sig

# Funcion sigmoidal derivada (recibe sigmoidal)
def deriveSigmoid(sig):
    # Solo funciona con entrada sigmoidal
    derive = sig*(1-sig)

    return derive

# Funcion de precision
def accuracy(y, yAprox):
	errors = (y == yAprox.round())
	errorTotal = 0 

	for element in errors:
		if sum(element) != len(element):
				errorTotal += 1

	return 100 - (errorTotal*100)/len(errors)

	# error = abs(y - yAprox)
	# errorTotal = 0

	# for row in error:
	# 	if round(row[0], 0) > 0:
	# 		errorTotal += 1

# Funcion que devuelve errores cuando y es un vector
def accuracyVector(y, yAprox):

	falsePositive = 0
	falseNegative = 0
	for i in range(len(y)):
		if y[i] < yAprox[i].round():
			falsePositive += 1

		if y[i] > yAprox[i].round():
			falseNegative += 1

	errorTotal = (falseNegative + falsePositive)*100/len(y)

	falseNegative = falseNegative*100/len(y)
	falsePositive = falsePositive*100/len(y)

	sqdError = sum((y-yAprox)**2)/(2*len(y))

	return errorTotal, falsePositive, falseNegative, sqdError

# Objeto Red Neuronal Profunda
class DeepNeuralNetwork(object):

    def __init__(self, inputSize, hiddenSize, outputSize, depth, seed):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize

        np.random.seed(seed)

        # Calculamos pesos iniciales
        self.w = []

        w0 = np.random.randn(inputSize+1, hiddenSize[0])
        self.w.append(w0)

        for i in range(1, depth):
            wi = np.random.randn(hiddenSize[i-1]+1, hiddenSize[i])
            self.w.append(wi)

        self.wf = np.random.randn(hiddenSize[-1]+1, outputSize)

    # Funcion de Feed Forward, recibe matriz de x
    def feedForward(self, x):

        self.hiddenValues = []
        values = x

        for i in range(len(self.w)):
            value_i = sigmoid(values.dot(self.w[i]))

            value_i = np.c_[
                np.ones(len(value_i)), value_i]

            values = value_i

            self.hiddenValues.append(value_i)

        y = sigmoid(self.hiddenValues[-1].dot(self.wf))

        return y

    # Funcion de Backward Propagation, recibe matriz de x, y, yAprox (resultado de feedForward) y alpha
    def backwardPropagation(self, x, y, yAprox, alpha):

        # yAprox ya es sigmoidal
        yAproxDelta = (y - yAprox) * deriveSigmoid(yAprox)

        hiddenDelta = yAproxDelta.dot(self.wf.T) * deriveSigmoid(self.hiddenValues[-1])

        hiddenDelta = hiddenDelta[:, 1:]

        # Actualiza pesos

        self.wf += alpha*self.hiddenValues[-1].T.dot(yAproxDelta)

        for i in reversed(range(1, len(self.w))):
            self.w[i] += alpha*self.hiddenValues[i-1].T.dot(hiddenDelta)

            hiddenDelta = hiddenDelta.dot(self.w[i].T) * deriveSigmoid(self.hiddenValues[i-1])
            hiddenDelta = hiddenDelta[:, 1:]


        self.w[0] += alpha*x.T.dot(hiddenDelta)

    # Funcion que hace Feed Forward y Backward Propagation para entrenar red, devuelve errores a peticion
    def train(self, x, y, alpha, iters, getError):

        errors = []

        for i in range(iters):
            # FEED-FORWARD
            yAprox = self.feedForward(x)

            # BACKWARD PROPAGATION
            self.backwardPropagation(x, y, yAprox, alpha)

            if getError:
                errorTotal, falsePositive, falseNegative, sqdError = accuracyVector(y, yAprox)
                errors.append(sqdError)
                #errors.append(errorTotal)
                
                #currentError = 100 - accuracy(y, yAprox)
                #errors.append(currentError)

            percent = (i * 100)/iters
            sys.stdout.flush()
            sys.stdout.write("\rProgress: "+str(percent)+"%")

        return errors

# Objeto Red Neuronal con Tensor Flow
class DeepNeuralNetworkTF(object):
    def __init__(self, x, y, n_hidden, hidden_size, out_classes):

        self.x = x
        self.y = y

        self.N_INPUT = self.x.shape[0]
        self.N_HIDDEN_LAYERS = n_hidden
        self.HIDDEN_LAYER_SIZE = hidden_size
        self.N_CLASSES = out_classes


        self.layers = []
        self.x_ = tf.placeholder(tf.float32, [None, self.x.shape[1]])
        self.y_ = tf.placeholder(tf.int8, [None, 1])


    def setModel(self):
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

        

    def feedForward(self, x, array):

        # Se calcula el output a traves del feed forward
        x_tf = tf.convert_to_tensor(x, np.float32)
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

        if array:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                return output.eval()
        
        return output

    def train(self, alpha, epochs):
        # Se convierten los arrays en tensors y se transpone *y* por consistencia con
        # la prediccion pues es un arreglo plano y debe ser una matriz columna
        x_tf = tf.convert_to_tensor(self.x, np.float32)
        self.y = np.array([self.y], dtype=np.float32, ndmin=1)
        self.y = np.transpose(self.y)
        # y_tf = tf.convert_to_tensor(np.transpose(self.y), np.float32)
        y_tf = tf.convert_to_tensor(self.y, np.float32)

        # Se inicializan las variables del modelo
        self.setModel()

        # Se obtiene la prediccion
        prediction = self.feedForward(self.x, False)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(prediction.eval(), y_tf.eval())

        # Aca decidir entre usar como funcion de activacion cross entropy (hace backprop) a menos que
        # se indique lo contrario) o sigmoid

        # cost = tf.reduce_mean(
        #   tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y_tf)
        # )
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=prediction, labels=y_tf)
        )
        optimizer = tf.train.GradientDescentOptimizer(
            alpha).minimize(cost)

        # Session es el entorno de ejecucion de tensorflow y aqui ocurre la ejecucion
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Una epoca es FeedForward + BackPropagation
            for epoch in range(epochs):
                epoch_loss = 0

                # _ es una variable que no me interesa
                for _ in range(self.x.shape[0]):
                    _, c = sess.run([optimizer, cost],
                                    feed_dict={self.x_: self.x, self.y_: self.y})
                    # print(c)
                    epoch_loss += c

                print("Epoch", epoch, "completed out of",
                      epochs, "loss:", epoch_loss)

            # Se muestra la precision del modelo
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

            # Aqui interesa
            print("Accuracy:", accuracy.eval( feed_dict={self.x_: self.x, self.y_: self.y}))

    def exportLayers(self, sectorName):

        saver = tf.train.Saver({"layers": self.layers})

        with tf.Session() as sess:
            saver.save(sess, sectorName+".DNNweights.tf")

        #fp = open(sectorName + ".DNNweights.json", "w")

        #json.dump(self.layers, fp)

    def importLayers(self, fileName):
        with tf.Session() as sess:
            saver.restore(sess, fileName)

        # with open(fileName) as f:
        #     self.layers = json.load(f)

# Objeto Red Neuronal con Keras
class KerasDeepNN(object):

    def __init__(self, hiddenSize, dataSize, nFeatures, nDays):
        self.dataSize = dataSize
        self.nFeatures = nFeatures
        self.nDays = nDays

        self.model = Sequential()

        #for size in hiddenSize:
        #    self.model.add(LSTM(size, input_shape=(nDays, nFeatures)))
        self.model.add(Dense(hiddenSize[0], input_dim=nFeatures*nDays, activation='relu'))

        for size in hiddenSize[1:]:
            self.model.add(Dense(8, activation='relu'))

        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')


    def train(self, x, y, xTest, yTest, maxIters):

        #x = x.reshape((x.shape[0], self.nDays, self.nFeatures))

        #xTest = xTest.reshape((xTest.shape[0], self.nDays, self.nFeatures))


        history = self.model.fit(x, y, epochs=maxIters, batch_size=x.shape[0],
                                 validation_data=(xTest, yTest), verbose=2, shuffle=False)

        return history


    def predict(self, x):

        #x = x.reshape(x.shape[0], self.nDays, self.nFeatures)

        return self.model.predict(x)



