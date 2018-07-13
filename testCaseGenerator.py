import os
import sys
import pandas as pd


def getCases(sector, interval, price, daysAfter, maxMinPrices):

    # Ingresamos a la carpeta del sector
    os.chdir("Mark1 Data/"+sector)

    # Creamos el archivo de caso de prueba
    if not price:
        testFile = open(sector+".test." + str(interval) + ".txt", "w")
    else:
        testFile = open(sector+".priceTest." + str(interval) + ".txt", "w")

    indexName = sector+" - Index.txt"

    indexDataSet = pd.read_csv(indexName, sep=";", header=None)

    dates = indexDataSet.index.values

    for date in dates[:-(interval+daysAfter)]:

        fromDateSet = indexDataSet.loc[date:]

        case = fromDateSet[:interval]

        # Calculamos la tendencia al tercer dia despues del caso de prueba
        result = fromDateSet[:interval+daysAfter].tail(1)

        caseClosure = float(case.tail(1)[2])
        resultClosure = float(result[2])
        # print(caseClosure, resultClosure)

        if not price:
            if caseClosure < resultClosure:
                trend = 1
            else:
                trend = 0

        else:
            trend = resultClosure

        # Guardamos el caso de prueba

        if maxMinPrices:
            for index, row in case.iterrows():
                testFile.write(
                    str(round(row[1], 3))+";"+str(round(row[2], 3))+";" +
                    str(round(row[3], 3))+";"+str(round(row[4], 3))+";"
                )
        else:
            for index, row in case.iterrows():
                testFile.write(
                    str(round(row[1], 3))+";"+str(round(row[2], 3))+";"
                )

        testFile.write(str(trend)+"\n")

    testFile.close()


def main():
    try:
        sector = sys.argv[1]
    except:
        print("Ingrese el Sector a Producir el Indice:")
        sector = input()

    try:
        interval = int(sys.argv[2])
    except:
        print("Ingrese el Intervalo del Caso de Prueba:")
        interval = int(input())

    try:
        daysAfter = int(sys.argv[3])
    except:
        print("Ingrese cuantos dias despues quiere predecir:")
        daysAfter = int(input())

    try:
        trend = int(sys.argv[4])
    except:
        print("Ingrese 1 si desea calcular la tendencia, 0 si desea el precio:")
        trend = int(input())

    try:
        maxMinPrices = int(sys.argv[5])
    except:
        print("Ingrese 1 si desea incluir el precio maximo y minimo, de lo contrario 0:")
        maxMinPrices = int(input())

    getCases(sector, interval, trend, daysAfter, maxMinPrices)


if __name__ == '__main__':
    main()
