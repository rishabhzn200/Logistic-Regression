import math

class LogisticRegression:
    def __init__(self):
        self.weight_Vector = None
        pass

    def logisticFunction(self, weight):
        y = 1.0 / (1.0 + math.exp(-weight))
        return y


    def fitModel(self, train_data, learning_rate = 0.01, numIterations = 150):

        #Step 1: Set initial weight vector to 0
        # w = w0 + w1x1 + w2x2 + w3x3 + .... + wNxN
        # set initial Wi's to 0
        weight_Vector = [0.0 for index in range(0, train_data[0].__len__())]  #w0 which is bias is added. No -1 in length

        #Step 2: Run till number of iterations
        for iteration in range(0, numIterations):
            # Step 3: Loop over all training data
            for row in train_data:
                #predict value of Y for each row using logistic function
                weight = weight_Vector[0]      #Set Y_Predicted to weight 0, which is the bias or intercept
                for colIndex in range(0, row.__len__() - 1):        # -1 to remove the Y Labels
                    weight += weight_Vector[colIndex + 1] * row[colIndex]

                Y_Predicted = self.logisticFunction(weight)
                Y_True = row[-1]
                
                #Step 4: Update Weight Vector
                weight_Vector[0] = weight_Vector[0] + learning_rate * (Y_True - Y_Predicted) * Y_Predicted * (1.0 - Y_Predicted)
                for colIndex in range(0, row.__len__() - 1):
                    weight_Vector[colIndex + 1] = weight_Vector[colIndex + 1] + learning_rate * (Y_True - Y_Predicted) * Y_Predicted * (1.0 - Y_Predicted) * row[colIndex]

        self.weight_Vector = weight_Vector



        pass


    def predict(self, test_data):
        predicted_test_values = []
        for row in test_data:
            weight = self.weight_Vector[0]
            for colIndex in range(0, row.__len__() - 1):
                weight = weight + self.weight_Vector[colIndex+1] * row[colIndex + 1]
            Y_Predicted = self.logisticFunction(weight)
            Y_Predicted = round(Y_Predicted)
            predicted_test_values.append(Y_Predicted)
        return predicted_test_values

    def accuracyScore(self, true_labels, predicted_labels):
        totalCount = true_labels.__len__()
        matchCount = 0
        for index in range(0, totalCount):
            if true_labels[index] == predicted_labels[index]:
                matchCount += 1

        accuracy = matchCount * 100.0 / totalCount
        return accuracy


