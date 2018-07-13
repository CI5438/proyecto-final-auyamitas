import os
import sys
import pandas as pd
import datetime


def getIndex(sector, addPrices, addVolume):

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
            maxPrice = 0
            lowerPrice = 0

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
                maxPrice += row[2]
                lowerPrice += row[3]
            if addVolume:
                volume += row[4]

        # Promediamos
        openPrice = round(openPrice / numSets, 3)
        closePrice = round(closePrice / numSets, 3)

        if addPrices:
            maxPrice = round(maxPrice / numSets, 3)
            lowerPrice = round(lowerPrice / numSets, 3)

        if addVolume:
            volume = round(volume / numSets, 3)

        # Escribimos en el archivo

        string = str(date)+";"+str(openPrice)+";"+str(closePrice)

        if addPrices:
            string += ";"+str(maxPrice)+";"+str(lowerPrice)

        if addVolume:
            string += ";"+str(volume)

        fIndex.write(string+"\n")

    fIndex.close()


def main():
    try:
        sector = sys.argv[1]
    except:
        print("Ingrese el Sector a Producir el Indice:")
        sector = input()

    try:
        addMaxMinPrices = int(sys.argv[2])
        assert(addMaxMinPrices == 1 or addMaxMinPrices == 0)

    except:

        try:
            print("Ingrese 1 si desea incluir en el indice el precio mas alto y el mas bajo diario, de lo contrario ingrese 0:")
            addMaxMinPrices = int(input())
            assert(addMaxMinPrices == 1 or addMaxMinPrices == 0)
        except:
            print("Error, ha ingresado un valor incorrecto.")
            sys.exit()


    try:
        addVolume = int(sys.argv[3])
        assert(addVolume == 1 or addVolume == 0)

    except:

        try:
            print("Ingrese 1 si desea incluir en el volumen de transacciones diario, de lo contrario ingrese 0:")
            addVolume = int(input())
            assert(addVolume == 1 or addVolume == 0)
        except:
            print("Error, ha ingresado un valor incorrecto.")
            sys.exit()


    getIndex(sector, addMaxMinPrices, addVolume)


if __name__ == '__main__':
    main()
