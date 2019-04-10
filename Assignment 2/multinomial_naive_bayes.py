import glob
import math
import sys
from collections import Counter

LABEL = ['ham', 'spam']

if __name__ == '__main__':
    trainFolder = str(sys.argv[1])
    testFolder = str(sys.argv[2])
    hamTrainPath = trainFolder+'/ham/*.txt'
    spamTrainPath = trainFolder+'/spam/*.txt'
    hamTestPath = testFolder+'/ham/*.txt'
    spamTestPath = testFolder+'/spam/*.txt'
    prior = dict()
    likelihood = dict()


    def preProcessData(file_data, clas, train):
        counter = Counter(file_data)
        for word in train:
            likelihood[word + clas] = float(counter.get(word,0) + 1) / (len(file_data) + len(train))


    def readTrainData(path):
        files = glob.glob(path)
        words = list()
        documentCount = 0
        for file in files:
            f = open(file, 'r', encoding="ISO-8859-1")
            file_word = f.read().split()
            documentCount += 1
            words.extend(file_word)
        return words, documentCount


    # Spam and Ham Train Data
    trainHam, trainHamCount = readTrainData(hamTrainPath)
    trainSpam, trainSpamCount = readTrainData(spamTrainPath)

    # All words in training set
    train = trainHam + trainSpam

    # Distinct number of words in training set
    trainDistinct = list(set(train))

    # Prior Probability
    prior[LABEL[1]] = float(trainSpamCount) / (trainHamCount + trainSpamCount)
    prior[LABEL[0]] = float(trainHamCount) / (trainHamCount + trainSpamCount)

    preProcessData(trainHam, LABEL[0], trainDistinct)
    preProcessData(trainSpam, LABEL[1], trainDistinct)


    def checkTestdata(path, given_class):
        files = glob.glob(path)
        rightPrediction = 0
        wrongPrediction = 0
        for file in files:
            f = open(file, 'r', encoding="ISO-8859-1")
            words = f.read().split()
            wordsCounter = Counter(words)
            maxScore = -10000000.0
            predictedClass = None
            for clas in LABEL:
                score = math.log(prior[clas])
                for word, count in wordsCounter.items():
                    score += count * (math.log(likelihood.get(word + clas, 1)))
                if score > maxScore:
                    maxScore = score
                    predictedClass = clas
            if predictedClass == given_class:
                rightPrediction += 1
            else:
                wrongPrediction += 1
        return rightPrediction, wrongPrediction


    hamRightPredicted, hamWrongPredicted = checkTestdata(hamTestPath, LABEL[0])
    print("Accuracy ham file:" + str((hamRightPredicted / (hamRightPredicted + hamWrongPredicted)) * 100))

    spamRightPrediction, spamWrongPrediction = checkTestdata(spamTestPath, LABEL[1])
    print("Accuracy spam file:" + str((spamRightPrediction / (spamRightPrediction + spamWrongPrediction)) * 100))
    accuracy = float(hamRightPredicted + spamRightPrediction) / (
                hamRightPredicted + hamWrongPredicted + spamRightPrediction + spamWrongPrediction)
    print("Accuracy: "+str(accuracy*100))


