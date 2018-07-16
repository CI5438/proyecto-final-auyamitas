# ProyectoFinal

## Inteligencia Artificial II

### Predicción de la Tendencia de un Índice de Mercado
#### Utilizando Redes Neuronales Recurrentes (LSTM)

#### Autores:
* Lautaro Villalón 12-10427
* Yarima Luciani 13-10770
* Benjamín Amos 12-10240

### Implementación

* Se implementaron tres módulos: superDataLib.py, superKerasRNNLib.py y superKerasPredictor.py

* El módulo superDataLib.py, contiene la librería con todas las funciones relacionadas a la manipulación de datos utilizado en el predictor. Se divide en cuatro funciones:

  * readData: Lee un archivo CSV y devuelve un data frame.
  * prepareData: Abre un archivo con readData, y escala los datos en un rango deseado con un MinMaxScaler de SkLearn.
  * createDataSet: Crea los casos de prueba, y los separa en entradas, salidas y salidas de clasificación.
  * splitDataSet: Divide los casos de prueba en entrenamiento y prueba.

* El módulo superKerasRNNLib.py, contiene la librería con todas las funciones necesarias para la creación del modelo de red y obtener resultados de clasificación. Se divide en dos funciones:

  * getLSTMmodel: Con las especificaciones dadas, devuelve un modelo de red neuronal recurrente con capas LSTM.
  * classifyFromPrediction: Dada una predicción de precio, devuelve la predicción de tendencia.

* El módulo superKerasPredictor.py, contiene el algoritmo de predicción de precio de un índice, a partir del cual se obtienen resultados de precio y tendencia. Estos resultados se grafican al terminar la corrida. Permite recibir el caso deseado a probar, a partir de argumentos.

### Uso

python3 superKerasPredictor.py SECTOR_DE_PRUEBA INTERVALO_DE_MUESTRA DIAS_A_PREDECIR PORCENTAJE_ENTRENAMIENTO ITERACIONES

### Requerimientos

* Python 3.x
  * Librerias: Numpy, Pandas, MatplotLib, SciKit-Learn, TensorFlow, Keras


##### Notas:

* Comentado en el main, se encuentra el mejor caso de prueba.
* Se pretende predecir las tendencias, sin embargo, las predicciones de precios que se obtienen son más interesantes.
* En la carpeta Experimentación, se encuentran diferentes scripts y librerías que utilizamos en el proceso de desarrollo, que ya no son necesarias.
* En las carpetas Mejor Caso y Peor Caso, se encuentran las gráficas y valores de erores del mejor caso y los peores casos de nuestro experimento.
* En la carpeta Caso Curioso, se encuentran las gráficas y valores de un caso que nos pareció curioso en torno a cómo predice el precio. Ya que solo tiene un día por cada muestra del caso de prueba.