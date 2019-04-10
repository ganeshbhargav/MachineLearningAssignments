import glob
import math
import sys
import time
from itertools import chain

import numpy as np

from stopwords import stopwords

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
        data = [d for d in data if d not in stopwords]
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


    def matrixCreation(matrix, featureMatrix, mef):
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                featureMatrix[i][mef.index(matrix[i][j])] = matrix[i].count(matrix[i][j])


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
            sigmoidVector[i] = probability


    def updateTheta(thetaSize):
        for i in range(thetaSize):
            defaultDifference = theta0
            for docId in range(trainSetCount):
                y = trainLabel[docId]
                h = sigmoidVector[docId]
                defaultDifference += featuresMatrix[docId][i] * (y - h)
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
    sigmoidVector = [0.0] * trainSetCount


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
    testFeaturesMatrix = np.zeros((testSetCount, len(testMef)))
    matrixCreation(totalTest, testFeaturesMatrix, testMef)


    def test():
        hamRightPrediction = 0
        hamWrongPrediction = 0
        spamRightPrediction = 0
        spamWrongPrediction = 0

        for docId in range(testSetCount):
            sum = 1.0
            for i in range(len(testMef)):
                word = testMef[i]
                if word in trainMef:
                    sum += theta[trainMef.index(word)] * testFeaturesMatrix[docId][i]
            h = sigmoid(sum)
            if h > 0.5:
                if testLabel[docId] == 1:
                    spamRightPrediction += 1
                else:
                    spamWrongPrediction += 1
            else:
                if testLabel[docId] == 0:
                    hamRightPrediction += 1
                else:
                    hamWrongPrediction += 1

        print("Accuracy ham file:" + str((hamRightPrediction / (hamRightPrediction + hamWrongPrediction)) * 100))
        print(
            "Accuracy spam file:" + str((spamRightPrediction / (spamRightPrediction + spamWrongPrediction)) * 100))
        print("Accuracy :" + str(((spamRightPrediction + hamRightPrediction) / (
                spamRightPrediction + spamWrongPrediction + hamRightPrediction + hamWrongPrediction)) * 100))

    train()
    test()
