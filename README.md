# ProyectoFinal

## Inteligencia Artificial II

### Predicción de la Tendencia de un Índice de Mercado
#### Utilizando Redes Neuronales Recurrentes (LSTM)

#### Autores:
* Lautaro Villalón 12-10427
* Yarima Luciani 13-10770
* Benjamín Amos 12-10240

### Implementación

* Se implementaron tres módulos: `superDataLib.py`, `superKerasRNNLib.py` y `superKerasPredictor.py`

* El módulo `superDataLib.py`, contiene la librería con todas las funciones relacionadas a la manipulación de datos utilizado en el predictor. Se divide en cuatro funciones:

  * **readData**: Lee un archivo CSV y devuelve un data frame.
  * **prepareData**: Abre un archivo con readData, y escala los datos en un rango deseado con un MinMaxScaler de __SkLearn__.
  * **createDataSet**: Crea los casos de prueba, y los separa en entradas, salidas y salidas de clasificación.
  * **splitDataSet**: Divide los casos de prueba en entrenamiento y prueba.
  * **getCorrelations**: Arroja dos datasets con todos los features en funcion de la tendencia/precio, y las matrices de correlación respectivas.
  * **saveCorrelations**: Guarda en el directorio local las imágenes de correlaciones entre features y tendencia/precio, así como las matrices de correlación graficadas y los resultados de las matrices de correlación en un archivo de texto.
  * **save_corrMatrix_plot**: Funcionalidad para almacenar las matrices de correlación con aspecto legible.

* El módulo `superKerasRNNLib.py`, contiene la librería con todas las funciones necesarias para la creación del modelo de red y obtener resultados de clasificación. Se divide en dos funciones:

  * **getLSTMmodel**: Con las especificaciones dadas, devuelve un modelo de red neuronal recurrente con capas LSTM.
  * **classifyFromPrediction**: Dada una predicción de precio, devuelve la predicción de tendencia.

* El módulo `superKerasPredictor.py`, contiene el algoritmo de predicción de precio de un índice, a partir del cual se obtienen resultados de precio y tendencia. Estos resultados se grafican al terminar la corrida. Permite recibir el caso deseado a probar, a partir de argumentos.

### Uso

`python3 superKerasPredictor.py SECTOR_DE_PRUEBA INTERVALO_DE_MUESTRA DIAS_A_PREDECIR PORCENTAJE_ENTRENAMIENTO ITERACIONES`

### Requerimientos

* **Python 3.x**
  * Librerias: Numpy, Pandas, MatplotLib, SciKit-Learn, TensorFlow, Keras


##### Notas:

* Comentado en el main, se encuentra el mejor caso de prueba.
* Se pretende predecir las tendencias, sin embargo, las predicciones de precios que se obtienen son más interesantes.
* En la carpeta Experimentación, se encuentran diferentes scripts y librerías que utilizamos en el proceso de desarrollo, que ya no son necesarias.
* En las carpetas Mejor Caso y Peor Caso, se encuentran las gráficas y valores de erores del mejor caso y los peores casos de nuestro experimento.
* En la carpeta Caso Curioso, se encuentran las gráficas y valores de un caso que nos pareció curioso en torno a cómo predice el precio. Ya que solo tiene un día por cada muestra del caso de prueba.
* En la carpeta Correlaciones se encuentran subdirectorios con la información referente al estudio de la disposición de la data y de las correlaciones obtenidas para cada sector en particular. Dentro de cada subdirectorio se encuentran Scatter Plots de cada feature vs la tendencia y el peso respectivamente, además de Matrix Plots de las matrices de correlación para cada caso, y un archivo .txt por cada matriz de correlación para ser leído en su aplicación de hojas de cálculo de preferencia.
* Todas las correlaciones fueron calculadas haciendo uso de un sólo día de intervalo y para una predicción de precio/tendencia 3 días después. Usted puede obtener otros datos y gráficas producto del estudio con otros intervalos o días de predicción haciendo: 
```python 
    $python3
    > import superDataLib as sdl
    > sdl.saveCorrelations(nuevoIntervalo, nuevosDiasDePrediccion)
```