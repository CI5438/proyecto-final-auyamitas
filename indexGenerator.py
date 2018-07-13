import os
import sys
import pandas as pd
import datetime


def getIndex(sector, addPrices):

    # Ingresamos a la carpeta del sector
    os.chdir("Mark1 Data/"+sector)

    # Guardamos los datasets
    sets = []
    fileList = os.listdir()
    maxSet = []
    for file in fileList:
        if "Index" in file:
            continue

        dataSet = pd.read_csv(file, sep=";", header=None)
        dataSet = dataSet.set_index([0])

        # Tomamos los indices del dataset mas grande
        rowIndexes = dataSet.index.values
        if len(maxSet) < len(rowIndexes):
            maxSet = rowIndexes

        sets.append(dataSet)

    # Abrimos archivo de indice
    fIndex = open(sector+" - Index.txt", "w")

    for date in maxSet:

        openPrice = 0
        closePrice = 0

        if addPrices:
            lowerPrice = 0
            maxPrice = 0

        numSets = 0
        for dataSet in sets:
            # Obtenemos la fila de la fecha actual
            try:
                row = dataSet.loc[date]
                numSets += 1
            # Pasamos al siguiente set si no existe la fila
            except:
                continue

            openPrice += row[1]
            closePrice += row[4]
            if addPrices:
                lowerPrice += row[3]
                maxPrice += row[2]

        # Promediamos
        openPrice = round(openPrice / numSets, 3)
        closePrice = round(closePrice / numSets, 3)

        if addPrices:
            lowerPrice = round(lowerPrice / numSets, 3)
            maxPrice = round(maxPrice / numSets, 3)

        # Escribimos en el archivo
        if not addPrices:
            fIndex.write(str(date)+";"+str(openPrice)+";"+str(closePrice)+"\n")
        else:
            fIndex.write(str(date)+";"+str(openPrice)+";" +
                         str(closePrice)+";"+str(lowerPrice)+";"+str(maxPrice)+"\n")

    fIndex.close()


def main():
    try:
        sector = sys.argv[1]
    except:
        print("Ingrese el Sector a Producir el Indice:")
        sector = input()

    try:
        print("Ingrese 1 si desea incluir en el indice el precio mas alto y el mas bajo diario, de lo contrario ingrese 0:")
        addMaxMinPrices = int(input())
        assert(addMaxMinPrices == 1 or addMaxMinPrices == 0)
    except:
        print("Error, ha ingresado un valor incorrecto.")
        sys.exit()

    getIndex(sector, addMaxMinPrices)


if __name__ == '__main__':
    main()
