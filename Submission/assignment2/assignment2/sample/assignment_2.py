import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import svm


testFile = ".//SemEval-PIT2015/data//SemEval-PIT2015-github//SemEval-PIT2015-github//data//test.data"
devFile = ".//SemEval-PIT2015//data//SemEval-PIT2015-github//SemEval-PIT2015-github//data//dev.data"
trainFile = ".//SemEval-PIT2015//data//SemEval-PIT2015-github//SemEval-PIT2015-github//data//train.data"


label_encoder = LabelEncoder()


def readDataset(fileName):
    data = []
    fileData = open(fileName)

    for line in fileData:
        (topicId, topicName, originalSent, paraSent, judgement, originalSentTag,
         paraSentTag) = (None, None, None, None, None, None, None)
        line = line.strip()
        fieldList = line.split('\t')
        fieldListCount = len(fieldList)
        if fieldListCount == 7:
            (topicId, topicName, originalSent, paraSent,
             judgement, originalSentTag, paraSentTag) = fieldList
        elif fieldListCount == 6:
            (topicId, topicName, originalSent, paraSent,
             originalSentTag, paraSentTag) = fieldList
        else:
            continue

        if judgement == None:
            data.append((topicId, topicName, originalSent,
                        paraSent, None, originalSentTag, paraSentTag))

        if judgement[0] == '(':
            numberOfYes = int(judgement[1])
            if numberOfYes >= 3:
                data.append((topicId, topicName, originalSent,
                            paraSent, True, originalSentTag, paraSentTag))
            elif numberOfYes <= 1:
                data.append((topicId, topicName, originalSent,
                            paraSent, False, originalSentTag, paraSentTag))
        elif judgement[0].isdigit():
            numberOfYes = int(judgement[0])
            if numberOfYes >= 4:
                data.append((topicId, topicName, originalSent,
                            paraSent, True, originalSentTag, paraSentTag))
            elif numberOfYes <= 2:
                data.append((topicId, topicName, originalSent,
                            paraSent, False, originalSentTag, paraSentTag))
            else:
                data.append((topicId, topicName, originalSent,
                            paraSent, None, originalSentTag, paraSentTag))

    return data


def vectorize_dataset_X(datasetTrain, datasetTest, datasetDev):
    originalSentTrain = [x[2] for x in datasetTrain]
    paraSentTrain = [x[3] for x in datasetTrain]
    datasetDfTrain = pd.DataFrame()
    datasetDfTrain["Original Sentance"] = originalSentTrain
    datasetDfTrain["Paraphrase Sentance"] = paraSentTrain

    originalSentTest = [x[2] for x in datasetTest]
    paraSentTest = [x[3] for x in datasetTest]
    datasetDfTest = pd.DataFrame()
    datasetDfTest["Original Sentance"] = originalSentTest
    datasetDfTest["Paraphrase Sentance"] = paraSentTest

    originalSentDev = [x[2] for x in datasetDev]
    paraSentDev = [x[3] for x in datasetDev]
    datasetDfDev = pd.DataFrame()
    datasetDfDev["Original Sentance"] = originalSentDev
    datasetDfDev["Paraphrase Sentance"] = paraSentDev

    originalSentVectorizer = CountVectorizer().fit(pd.concat(
        [datasetDfTrain["Original Sentance"], datasetDfTest["Original Sentance"], datasetDfDev["Original Sentance"]]))
    paraSentVectorizer = CountVectorizer().fit(pd.concat(
        [datasetDfTrain["Paraphrase Sentance"], datasetDfTest["Paraphrase Sentance"], datasetDfDev["Paraphrase Sentance"]]))

    originalSentVecTrain = originalSentVectorizer.transform(
        datasetDfTrain["Original Sentance"])
    paraSentVecTrain = paraSentVectorizer.transform(
        datasetDfTrain["Paraphrase Sentance"])
    trainX = sp.hstack((originalSentVecTrain, paraSentVecTrain))

    originalSentVecTest = originalSentVectorizer.transform(
        datasetDfTest["Original Sentance"])
    paraSentVecTest = paraSentVectorizer.transform(
        datasetDfTest["Paraphrase Sentance"])
    testX = sp.hstack((originalSentVecTest, paraSentVecTest))

    originalSentVecDev = originalSentVectorizer.transform(
        datasetDfDev["Original Sentance"])
    paraSentVecDev = paraSentVectorizer.transform(
        datasetDfDev["Paraphrase Sentance"])
    devX = sp.hstack((originalSentVecDev, paraSentVecDev))

    return trainX, testX, devX


def simpleBaseLine(dataset):
    truePositive = 0
    falsePositive = 0
    trueNegative = 0
    falseNegative = 0

    for x in dataset:
        originalSent = x[2]
        paraSent = x[3]
        actualLabel = x[4]
        guessLabel = None

        if originalSent == paraSent:
            guessLabel = True
        else:
            guessLabel = False

        if guessLabel == True and actualLabel == False:
            falsePositive += 1
        elif guessLabel == False and actualLabel == True:
            falseNegative += 1
        elif guessLabel == True and actualLabel == True:
            truePositive += 1
        elif guessLabel == False and actualLabel == False:
            trueNegative += 1

    accuracy = (truePositive + trueNegative) / (trueNegative +
                                                truePositive + falseNegative + falsePositive)
    precision = truePositive / (truePositive + falsePositive)
    recall = truePositive / (truePositive + falseNegative)
    fOne = 1 * precision * recall / (precision + recall)

    print("Result of Baseline:-")
    print("-------------------")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", fOne)
    print("True Positive: ", truePositive)
    print("True Negative: ", trueNegative)
    print("False Positive: ", falsePositive)
    print("False Negative: ", falseNegative)


def stateOfTheArt(trainX, trainY, testX, testY, encoder):
    truePositive = 0
    falsePositive = 0
    trueNegative = 0
    falseNegative = 0

    model = LogisticRegression()
    model.fit(trainX, trainY)

    guessedLabels = model.predict(testX)

    for i in range(len(testY)):
        actualLabel = encoder.inverse_transform([testY[i]])[0]
        guessLabel = encoder.inverse_transform([guessedLabels[i]])[0]

        if guessLabel == True and actualLabel == False:
            falsePositive += 1
        elif guessLabel == False and actualLabel == True:
            falseNegative += 1
        elif guessLabel == True and actualLabel == True:
            truePositive += 1
        elif guessLabel == False and actualLabel == False:
            trueNegative += 1

    accuracy = (truePositive + trueNegative) / (trueNegative +
                                                truePositive + falseNegative + falsePositive)
    precision = truePositive / (truePositive + falsePositive)
    recall = truePositive / (truePositive + falseNegative)
    fOne = 1 * precision * recall / (precision + recall)

    print()
    print("Result of State of the Art:-")
    print("----------------------------")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", fOne)
    print("True Positive: ", truePositive)
    print("True Negative: ", trueNegative)
    print("False Positive: ", falsePositive)
    print("False Negative: ", falseNegative)


def stateOfTheArtEnhanced(trainX, trainY, testX, testY, encoder):
    truePositive = 0
    falsePositive = 0
    trueNegative = 0
    falseNegative = 0

    model = svm.SVC(decision_function_shape='ovo')
    model.fit(trainX, trainY)

    guessedLabels = model.predict(testX)

    for i in range(len(testY)):
        actualLabel = encoder.inverse_transform([testY[i]])[0]
        guessLabel = encoder.inverse_transform([guessedLabels[i]])[0]

        if guessLabel == True and actualLabel == False:
            falsePositive += 1
        elif guessLabel == False and actualLabel == True:
            falseNegative += 1
        elif guessLabel == True and actualLabel == True:
            truePositive += 1
        elif guessLabel == False and actualLabel == False:
            trueNegative += 1

    accuracy = (truePositive + trueNegative) / (trueNegative +
                                                truePositive + falseNegative + falsePositive)
    precision = truePositive / (truePositive + falsePositive)
    recall = truePositive / (truePositive + falseNegative)
    fOne = 1 * precision * recall / (precision + recall)

    print()
    print("Result of Enhanced State of the Art:-")
    print("----------------------------")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", fOne)
    print("True Positive: ", truePositive)
    print("True Negative: ", trueNegative)
    print("False Positive: ", falsePositive)
    print("False Negative: ", falseNegative)


def encode_dataset_Y(trainY, testY, devY):
    encoder = label_encoder.fit(trainY + testY + devY)
    encoded_trainY = encoder.transform(trainY)
    encoded_testY = encoder.transform(testY)
    encoded_devY = encoder.transform(devY)

    return encoded_trainY, encoded_testY, encoded_devY, encoder


if __name__ == "__main__":

    print("***********Assignment 2***********")
    print("==================================")

    trainDataset = readDataset(trainFile)
    testDataset = readDataset(testFile)
    devDataset = readDataset(devFile)

    trainDatasetX, testDatasetX, devDatasetX = vectorize_dataset_X(
        trainDataset, testDataset, devDataset)
    trainDatasetY, testDatasetY, devDatasetY, encoder = encode_dataset_Y(
        [x[4] for x in trainDataset], [x[4] for x in testDataset], [x[4] for x in devDataset])

    simpleBaseLine(devDataset)

    stateOfTheArt(trainDatasetX, trainDatasetY,
                  devDatasetX, devDatasetY, encoder)

    stateOfTheArtEnhanced(trainDatasetX, trainDatasetY,
                          devDatasetX, devDatasetY, encoder)
