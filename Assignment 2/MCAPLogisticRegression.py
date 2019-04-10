import glob
import math
import sys
import time
from itertools import chain

import numpy as np

if __name__ == '__main__':
    trainFolder = str(sys.argv[1])
    testFolder = str(sys.argv[2])
    hamTrainPath = trainFolder + '/ham/*.txt'
    spamTrainPath = trainFolder + '/spam/*.txt'
    hamTestPath = testFolder + '/ham/*.txt'
    spamTestPath = testFolder + '/spam/*.txt'

    regularizedParameter = 2
    learningRate = 0.01
    iteration = 100


    def readData(file):
        data = open(file, 'r', encoding="ISO-8859-1")
        data = data.read().split()
        return data


    def createReadMatrix(fileName):
        files = glob.glob(fileName)
        return [readData(file) for file in files], len(files)


    # flatten the 2D list into 1D list
    def flattenMatrix(matrix):
        return list(chain.from_iterable(matrix))


    def mutuallyExclusiveFeatures(spamCleanList, hamCleanList):
        result = list()
        result.extend(list(set(spamCleanList).difference(hamCleanList)))
        result.extend(list(set(hamCleanList).difference(result)))
        return result


    def matrixCreation(matrix, feature_matrix, mef):
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                feature_matrix[i][mef.index(matrix[i][j])] = matrix[i].count(matrix[i][j])


    def sigmoid(x):
        try:
            d = (1 + np.exp(-x))
            p = float(1) / d
            return p
        except Exception:
            return 0.0


    def sumOfThetaAndFeatureMultiplication(file_count):
        for i in range(file_count):
            default_sum = 1.0
            for j in range(len(trainMef)):
                default_sum += theta[j] * featuresMatrix[i][j]
            probability = sigmoid(default_sum)
            sigmoid_vector[i] = probability


    def updateTheta(theta_size):
        for i in range(theta_size):
            defaultDifference = theta0
            for doc_id in range(trainSetCount):
                y = trainLabel[doc_id]
                h = sigmoid_vector[doc_id]
                defaultDifference += featuresMatrix[doc_id][i] * (y - h)
            prevTheta = theta[i]
            theta[i] += learningRate * (defaultDifference - (regularizedParameter * prevTheta))


    # read data for wach file in train and test folder and return the matrix and count of files
    trainSpam, trainSpamCount = createReadMatrix(spamTrainPath)
    trainHam, trainHamCount = createReadMatrix(hamTrainPath)
    # Import variable for matrix manipulation
    trainSetCount = trainSpamCount + trainHamCount
    totalTrain = trainSpam + trainHam
    trainSpamList = flattenMatrix(trainSpam)
    trainHamList = flattenMatrix(trainHam)
    trainMef = mutuallyExclusiveFeatures(trainSpamList, trainHamList)
    featuresMatrix = np.zeros((trainSetCount, len(trainMef)))
    theta = [0.0] * len(trainMef)
    theta0 = 0
    spamValue = [1] * trainSpamCount
    hamValue = [0] * trainHamCount
    trainLabel = spamValue + hamValue

    matrixCreation(totalTrain, featuresMatrix, trainMef)
    sigmoid_vector = [0.0] * trainSetCount


    def train():
        for i in range(iteration):
            sumOfThetaAndFeatureMultiplication(trainSetCount)
            updateTheta(len(theta))


    testSpam, testSpamCount = createReadMatrix(spamTestPath)
    testHam, testHamCount = createReadMatrix(hamTestPath)
    testSetCount = testHamCount + testSpamCount
    totalTest = testSpam + testHam
    testSpamList = flattenMatrix(testSpam)
    testHamList = flattenMatrix(testHam)
    testSpamValue = [1] * testSpamCount
    testHamValue = [0] * testHamCount
    testLabel = testSpamValue + testHamValue

    testMef = mutuallyExclusiveFeatures(testSpamList, testHamList)
    test_features_matrix = np.zeros((testSetCount, len(testMef)))
    matrixCreation(totalTest, test_features_matrix, testMef)


    def test():
        hamRightPrediction = 0
        hamWrongPrediction = 0
        spamRightPrediction = 0
        spamWrongPediction = 0

        for doc_id in range(testSetCount):
            sum = 1.0
            for i in range(len(testMef)):
                word = testMef[i]
                if word in trainMef:
                    sum += theta[trainMef.index(word)] * test_features_matrix[doc_id][i]
            h = sigmoid(sum)
            if h > 0.5:
                if testLabel[doc_id] == 1:
                    spamRightPrediction += 1
                else:
                    spamWrongPediction += 1
            else:
                if testLabel[doc_id] == 0:
                    hamRightPrediction += 1
                else:
                    hamWrongPrediction += 1

        print("Accuracy ham file:" + str((hamRightPrediction / (hamRightPrediction + hamWrongPrediction)) * 100))
        print(
            "Accuracy spam file:" + str((spamRightPrediction / (spamRightPrediction + spamWrongPediction)) * 100))
        print("Accuracy :" + str(((spamRightPrediction + hamRightPrediction) / (
                spamRightPrediction + spamWrongPediction + hamRightPrediction + hamWrongPrediction)) * 100))


    train()
    test()
