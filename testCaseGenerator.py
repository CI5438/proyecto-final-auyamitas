import os, sys
import pandas as pd

def getCases(sector, interval):

	# Ingresamos a la carpeta del sector
	os.chdir("Mark1 Data/"+sector)

	# Creamos el archivo de caso de prueba
	testFile = open(sector+".test.txt", "w")

	indexName = sector+" - Index.txt"

	indexDataSet = pd.read_csv(indexName, sep=";", header=None)

	dates = indexDataSet.index.values

	for date in dates[:-(interval+3)]:

		fromDateSet = indexDataSet.loc[date:]

		case = fromDateSet[:interval]

		# Calculamos la tendencia al tercer dia despues del caso de prueba
		result = fromDateSet[:interval+3].tail(1)

		caseClosure = float(case.tail(1)[2])
		resultClosure = float(result[2])

		if caseClosure < resultClosure:
			trend = 1
		elif caseClosure > resultClosure:
			trend = -1
		else:
			trend = 0


		# Guardamos el caso de prueba

		for index, row in case.iterrows():
			testFile.write(str(round(row[1], 3))+";"+str(round(row[2], 3))+";")

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
		interval = input()
		interval = int(interval)

	getCases(sector, interval)

if __name__ == '__main__':
	main()