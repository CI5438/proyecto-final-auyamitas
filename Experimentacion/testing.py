# Universidad Simón Bolívar
# Abril-Julio 2018
# Inteligencia Artificial II
# Prof. Ivette Carolina Martínez 
# Proyecto Final
# Pruebas sobre casos de prueba

# Autores:
    # Lautaro Villalón 12-10427
    # Yarima Luciani 13-10770
    # Benjamin Amos 12-10240

from superKerasRNNLib import *
from superDataLib import *
import pandas as pd
import numpy as np

# Preparamos el archivo del indice (sector, scale?, scaleRange)
df = prepareData('Communication Services Sector', False, (0, 1))

# Obtenemos DataSet
x, y = createDataSet(df, 30, 3)

f = open("DataSetTest.txt", "w")

for i in range(len(x)):
	for col in x[i]:
		f.write(str(round(float(col), 3))+";")

	f.write(str(round(float(y[i]),3))+"\n")


f.close()
