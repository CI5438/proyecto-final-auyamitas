import numpy as np
import pandas as pd


def getData(file, normalize=False):
    print("Getting data from", file)
    data = np.loadtxt(file, delimiter=";")
    x = data[:, :-1]
    y = data[:, -1]

    if normalize:
        x = x/np.amax(x, axis=0)

    print("Retrieved data. Sliced features in a matrix with",
          x.shape[0], "rows and", x.shape[1], "columns")
    return x, y

# Lectura de datos CSV


def readData(file, sep):
    dataSet = pd.read_csv(file, sep=sep, comment="#", header=None)
    return dataSet

# Division de datos en x normalizado, x, y


def prepareData(df):
    x = df.drop(df.columns[[2]], axis=1)
    x = x.values
    lenx = len(x)
    xNorm = x/np.amax(x, axis=0)

    xNorm = np.c_[np.ones(lenx), xNorm]

    y = df[[2]]
    y = y.values

    return xNorm, x, y
