import dataset as ds
import LogisticRegression #user defined module


if __name__ == "__main__":

    dataset = ds.Dataset()
    lr = LogisticRegression.LogisticRegression()

    data = dataset.loadDataFile("pima-indians-diabetes.csv")
    dataset.convertToFloat(data)

    dataset.normalize(data)
    training_Data, test_Data = dataset.splitDataset(data, 0.70)

    print('Dataset has {0} rows. Split into training data with {1} rows and test data with {2} rows').format(data.__len__(), training_Data.__len__(), test_Data.__len__())

    lr.fitModel(training_Data)
    predicted_labels = lr.predict(test_Data)

    true_labels = [row[-1] for row in test_Data]

    accuracy = lr.accuracyScore(true_labels, predicted_labels)

    print("Accuracy of model is {0}%", accuracy)

    pass







