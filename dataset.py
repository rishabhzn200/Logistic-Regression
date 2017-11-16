
import math

class Dataset:
    def __init__(self):
        self.loadedDataSet = None
        self.train_Data = None
        self.test_Data = None


    def loadDataFile(self, filename):
        import csv
        dataFileObject = open(filename, "rb")
        data = csv.reader(dataFileObject)
        rows = list(data)
        #print(rows)
        #print("\n\n")
        dataFileObject.close()

        for rowIndex in range(rows.__len__()):
            rows[rowIndex] = [float(column) for column in rows[rowIndex]]

        self.loadedDataSet = rows

        return self.loadedDataSet


    def convertToFloat(self, data):
        numRows = data.__len__()
        numCols = data[0].__len__()

        for row in range(0,numRows):
            for col in range(0,numCols):
                data[row][col] = float(data[row][col])
                pass
        pass



    def splitDataset(self, datasetToBeSplit, splitRatio):
        train_Len = int(datasetToBeSplit.__len__() * splitRatio)
        test_Len = datasetToBeSplit.__len__() - train_Len
        
        self.train_Data = datasetToBeSplit[:train_Len]
        self.test_Data = datasetToBeSplit[train_Len:]

        return  [self.train_Data, self.test_Data]


    def normalize(self, data):
        summary = []
        #find max and min values for each col
        for col in range(0,data[0].__len__()):
            rowvals = [data[index][col] for index in range(0,data.__len__())]
            maxVal = max(rowvals)
            minVal = min(rowvals)
            summary.append([maxVal,minVal])


        for rowIndex in range(0, data.__len__()):
            for col in range(0, data[rowIndex].__len__()):
                data[rowIndex][col] = (data[rowIndex][col] - summary[col][1]) / ( summary[col][0] - summary[col][1] )
        pass
