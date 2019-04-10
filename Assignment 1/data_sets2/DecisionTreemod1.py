import copy
import random
import sys

import numpy as np
import pandas as pd




"""

Calculate Variance Gain for a particular attribute

"""

def calculateVarianceGain(dataset, splitAttributeName, targetName="Class"):
    datasetVariance = calculateVarianceImpurity(dataset[targetName])
    typeOfValues, counts = np.unique(dataset[splitAttributeName], return_counts=True)
    sumOfCounts = np.sum(counts)
    splitVariance = [
        calculateVarianceImpurity(dataset.where(dataset[splitAttributeName] == typeOfValues[i]).dropna()[targetName])
        for i in range(len(typeOfValues))]
    splitVarianceSum = np.sum([(counts[i] / sumOfCounts) * splitVariance[i] for i in range(len(typeOfValues))])
    varianceGain = datasetVariance - splitVarianceSum
    return varianceGain

"""

Calculate the entropy of a particular column

"""

def calculateEntropy(targetColumn):
    typeOfValues, counts = np.unique(targetColumn, return_counts=True)
    sumOfCounts = np.sum(counts)
    countToTotals = [counts[i] / sumOfCounts for i in range(len(typeOfValues))]
    entropy = np.sum([-countToTotal * (np.log2(countToTotal)) for countToTotal in countToTotals])
    return entropy




"""
    
    Calculate the Variance Impurity according to Given Formula

"""

def calculateVarianceImpurity(targetColumn):
    typeOfValues, counts = np.unique(targetColumn, return_counts=True)
    sumOfCounts = np.sum(counts)
    countToTotals = [counts[i] / sumOfCounts for i in range(len(typeOfValues))]
    variance = np.prod(countToTotals)
    return variance

"""
    
    Update the Decision tree with new attributes

"""
def update(tree, targetKey, dataset):
    for key, value in tree.items():
        if key == targetKey:
            prominentIndex = np.argmax(np.unique(dataset[key], return_counts=True)[1])
            prominentClass = np.unique(dataset[key])[prominentIndex]
            tree[key] = prominentClass
            break
        elif isinstance(value, dict):
            update(value, targetKey, dataset)
    return tree



"""
    
    Predict the classification for a particular query
    
"""
def predictClassification(query, tree, default):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default
            result = tree[key][query[key]]
            if isinstance(result, dict):
                return predictClassification(query, result, default)
            else:
                return result


"""

Build the decision tree

"""
def decisionTree(partialDataset, originalDataset, attributes, type, targetName="Class", parentClass=None):
    if len(np.unique(partialDataset[targetName])) <= 1:
        return np.unique(partialDataset[targetName])[0], None
    elif len(partialDataset) == 0:
        maxElementIndex = np.argmax(np.unique(originalDataset[targetName], return_counts=True)[1])
        return np.unique(originalDataset[targetName])[maxElementIndex], None
    # Need to check
    elif len(attributes) == 0:
        return parentClass, None
    else:
        parentClassIndex = np.argmax(np.unique(partialDataset[targetName], return_counts=True)[1])
        parentClass = np.unique(partialDataset[targetName])[parentClassIndex]

        if type == 1:
            attributesInformationGain = [calculateInformationGain(partialDataset, atrribute, targetName) for atrribute
                                           in
                                           attributes]
        else:
            attributesInformationGain = [calculateVarianceGain(partialDataset, atrribute, targetName) for atrribute in
                                           attributes]
        most_gain_index = np.argmax(attributesInformationGain)
        bestAttribute = attributes[most_gain_index]

        root = dict()
        root[bestAttribute] = dict()
        printDecisionTreeTree = dict()
        attributes = [atrribute for atrribute in attributes if atrribute is not bestAttribute]

        for decision in np.unique(partialDataset[bestAttribute]):
            # print("Desicion : "+str(decision)+" "+str(bestAttribute))
            sub_data = partialDataset.where(partialDataset[bestAttribute] == decision).dropna()
            sub_tree = decisionTree(sub_data, originalDataset, attributes, type, targetName, parentClass)
            root[bestAttribute][decision] = sub_tree[0]
            printDecisionTreetreeTuple = (bestAttribute, decision)
            if isinstance(sub_tree[0], dict):
                printDecisionTreeTree[printDecisionTreetreeTuple] = sub_tree[1]
            else:
                printDecisionTreeTree[printDecisionTreetreeTuple] = sub_tree[0]
        return root, printDecisionTreeTree




"""

Get the Non-Leaf Node in a tree

"""

def getNonLeafNode(tree, nonLeafList):
    for key, value in tree.items():
        if isinstance(value, dict):
            if type(key) is str:
                nonLeafList.append(key)
            getNonLeafNode(value, nonLeafList)
    return nonLeafList

"""
    
    Post Prune the decision tree
    
"""

def runPostPruning(L, K, tree, dataset, default):
    treeBest = copy.deepcopy(tree)
    treeBestAccuracy = getAccuracyofTestDataset(dataset, treeBest, default)
    for i in range(L):
        dtree = copy.deepcopy(tree)
        m = random.randint(0, K)
        for j in range(m):
            nonLeafList = list()
            non_leaf = getNonLeafNode(dtree, nonLeafList)
            if len(non_leaf):
                p = random.randint(0, len(non_leaf) - 1)
                targetElement = non_leaf[p]
                dtree = update(dtree, targetElement, dataset)
        if dtree:
            dtreeAccuracy = getAccuracyofTestDataset(dataset, dtree, default)
            if dtreeAccuracy > treeBestAccuracy:
                treeBest = copy.deepcopy(dtree)
                treeBestAccuracy = dtreeAccuracy
    return treeBest

"""
    
    Print the decision tree
    
"""
def printDecisionTree(tree, depth=0):
    if depth == 0:
        print('TREE Visualization \n')
    
    for index, splitCriterion in enumerate(tree):
        subTrees = tree[splitCriterion]
        print('|\t' * depth, end='')
        variable = str(splitCriterion[0]).split('.')[0]
        value = str(splitCriterion[1]).split('.')[0]
        if type(subTrees) is dict:
            print('{0} = {1}:'.format(variable, value))
        else:
            subTrees = str(subTrees).split('.')[0]
            print('{0} = {1}:{2}'.format(variable, value, subTrees))
        if type(subTrees) is dict:
            printDecisionTree(subTrees, depth + 1)


"""
    
    Calculate the information Gain of a particular attribute
    
"""
def calculateInformationGain(dataset, splitAttributeName, targetName="Class"):
    datasetEntropy = calculateEntropy(dataset[targetName])
    typeOfValues, counts = np.unique(dataset[splitAttributeName], return_counts=True)
    sumOfCounts = np.sum(counts)
    splitEntropy = [
                    calculateEntropy(dataset.where(dataset[splitAttributeName] == typeOfValues[i]).dropna()[targetName]) for i in
                    range(len(typeOfValues))]
    splitEntropySum = np.sum([(counts[i] / sumOfCounts) * splitEntropy[i] for i in range(len(typeOfValues))])
    informationGain = datasetEntropy - splitEntropySum
    return informationGain






"""

Test the accuracyof the test set

"""
def getAccuracyofTestDataset(dataset, tree, default):
    testDict = dataset.iloc[:, :-1].to_dict(orient="records")
    predicted = pd.DataFrame(columns=["predict"])
    for i in range(len(dataset)):
        predicted.loc[i, "predict"] = predictClassification(testDict[i], tree, default)
    accuracy = (np.sum(predicted["predict"] == dataset["Class"]) / len(dataset)) * 100
    return accuracy




"""

Load all the data into a decision tree

"""
def loadData(file_name):
    df = pd.read_csv(file_name)
    return df


if __name__ == '__main__':
    L = int(int(sys.argv[1]))
    K = int(sys.argv[2])
    train_filename = str(sys.argv[3])
    validation_filename = str(sys.argv[4])
    test_filename = str(sys.argv[5])
    to_print = str(sys.argv[6])

    print("Information gain without pruning\n")
    print("Loading the training set for" + train_filename)
    df = loadData(train_filename)
    print("Completed Loading of Training Set\n")
    print("Creating Decision Tree for Training Set" + train_filename)
    tree, printDecisionTreeTree = decisionTree(df, df, df.columns[:-1], 1)
    print("Decision Tree creation completed \n")
    print("Loading Validation Set for" + validation_filename)
    df_validation = loadData(validation_filename)
    print(('Validation Set Loading done'))
    default_index = np.argmax(np.unique(df_validation['Class'], return_counts=True)[1])
    default = np.unique(df_validation['Class'])[default_index]
    accuracy = getAccuracyofTestDataset(df_validation, tree, default)
    print('The predictClassification accuracy on Validation is: ', accuracy, '%')
    print('\n')
    print("Loading Test Set for " + test_filename)
    dfTest = loadData(test_filename)
    print('Test Set Loading done')
    default_index = np.argmax(np.unique(dfTest['Class'], return_counts=True)[1])
    default = np.unique(dfTest['Class'])[default_index]
    accuracy = getAccuracyofTestDataset(dfTest, tree, default)
    print('The predictClassification accuracy on Test Set is: ', accuracy, '%')
    print("\n\n\n")

    if 'yes' in to_print or 'Yes' in to_print or 'YES' in to_print:
        printDecisionTree(printDecisionTreeTree)
    print("\n\n\n\n\n\n")

    print("Variance Gain Without Pruning\n")
    print("Loading the Training Set for " + train_filename)
    df = loadData(train_filename)
    print("Training Set loading done\n")
    print("Creating Decision Tree for Training Set" + train_filename)
    tree, printDecisionTreeTree = decisionTree(df, df, df.columns[:-1], 2)
    print("Decision Tree creation completed\n")
    print("Loading Validation Set for " + validation_filename)
    df_validation = loadData(validation_filename)
    print(('Validation Set Loading done'))
    default_index = np.argmax(np.unique(df_validation['Class'], return_counts=True)[1])
    default = np.unique(df_validation['Class'])[default_index]
    accuracy = getAccuracyofTestDataset(df_validation, tree, default)
    print('The predictClassification accuracy on Validation is: ', accuracy, '%')
    print('\n')
    print("Loading Test Set for " + test_filename)
    dfTest = loadData(test_filename)
    print('Test Set Loading done')
    default_index = np.argmax(np.unique(dfTest['Class'], return_counts=True)[1])
    default = np.unique(dfTest['Class'])[default_index]
    accuracy = getAccuracyofTestDataset(dfTest, tree, default)
    print('The predictClassification accuracy on Test Set is: ', accuracy, '%')
    print("\n\n\n")

    if 'yes' in to_print or 'Yes' in to_print or 'YES' in to_print:
        printDecisionTree(printDecisionTreeTree)
    print("\n\n\n\n\n\n")

    print("Information Gain with pruning\n")
    print('Using L=' + str(L) + " K=" + str(K))
    print("Loading Training Set for " + train_filename)
    df = loadData(train_filename)
    print("Training Set loading done\n")
    print("Creating Decision Tree for Training Set" + train_filename)
    tree, printDecisionTreeTree = decisionTree(df, df, df.columns[:-1], 1)
    print("Decision Tree creation completed \n")
    print("Loading Validation Set for " + validation_filename)
    df_validation = loadData(validation_filename)
    print(('Validation Set Loading done'))
    default_index = np.argmax(np.unique(df_validation['Class'], return_counts=True)[1])
    default = np.unique(df_validation['Class'])[default_index]
    print("Pruning Begins")
    tree = runPostPruning(L, K, tree, df_validation, default)
    print('Pruning Completed')
    dfTest = loadData(test_filename)
    default_index = np.argmax(np.unique(dfTest['Class'], return_counts=True)[1])
    default = np.unique(dfTest['Class'])[default_index]
    accuracy = getAccuracyofTestDataset(dfTest, tree, default)
    print('The predictClassification accuracy is: ', accuracy, '%')

    if 'yes' in to_print or 'Yes' in to_print or 'YES' in to_print:
        printDecisionTree(printDecisionTreeTree)
    print("\n\n\n\n\n\n")

    print("Variance Gain with pruning\n")
    print('Using L=' + str(L) + " K=" + str(K))
    print("Loading Training Set for " + train_filename)
    df = loadData(train_filename)
    print("Training Set loading done\n")
    print("Creating Decision Tree for Training Set" + train_filename)
    tree, printDecisionTreeTree = decisionTree(df, df, df.columns[:-1], 2)
    print("Decision Tree creation completed  \n")
    print("Loading Validation Set for " + validation_filename)
    df_validation = loadData(validation_filename)
    print(('Validation Set Loading done'))
    default_index = np.argmax(np.unique(df_validation['Class'], return_counts=True)[1])
    default = np.unique(df_validation['Class'])[default_index]
    print("Pruning Begins")
    tree = runPostPruning(L, K, tree, df_validation, default)
    print('Pruning Completed')
    dfTest = loadData(test_filename)
    default_index = np.argmax(np.unique(dfTest['Class'], return_counts=True)[1])
    default = np.unique(dfTest['Class'])[default_index]
    accuracy = getAccuracyofTestDataset(dfTest, tree, default)
    print('The predictClassification accuracy is: ', accuracy, '%')

    if 'yes' in to_print or 'Yes' in to_print or 'YES' in to_print:
        printDecisionTree(printDecisionTreeTree)
    print("\n\n\n\n\n\n")
