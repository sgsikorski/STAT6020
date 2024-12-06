import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from DPMM import DPMM
from config import Config
from evaluation import Evaluation
from util.dataStats import countRunTypes
from runModeling import runSkLearn, fitDPM, predictDPM, predictSkDPMM
from util.plot import plotClusters2d
from util.mapRun import mapLabelToRun

import sys


def reduceClusters(data, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return pd.DataFrame(reduced_data, columns=[f"PC{i+1}" for i in range(n_components)])


def getDataStats(data, labels, isSplit=""):
    plotClusters2d(reduceClusters(data).values, labels, f"{isSplit}_true")
    countRunTypes(mapLabelToRun(labels), isSplit)


def main():
    # Set random seed for reproducibility
    np.random.seed(0)
    Config.SetConfig(sys.argv[1:])
    dpmm = DPMM(alpha=3, iters=100)
    dpmm.initializeData()

    data = dpmm.preprocessData()
    getDataStats(data, dpmm.labels, isSplit="full")
    data = reduceClusters(data, n_components=2)

    trainSplit = 0.8
    trainIdx, testIdx = train_test_split(
        data.index, train_size=trainSplit, random_state=0, shuffle=False
    )
    trainData, testData = data.iloc[trainIdx], data.iloc[testIdx]
    trainLabels, testLabels = dpmm.labels[trainIdx.values], dpmm.labels[testIdx.values]

    fullTrain, fullTest = trainData, testData
    # trainData = reduceClusters(trainData, n_components=2)
    # testData = reduceClusters(testData, n_components=2)

    getDataStats(trainData, trainLabels, isSplit="train")
    getDataStats(testData, testLabels, isSplit="test")

    assignments = np.array([])
    if Config.USE_SKLEARN:
        assignments, skDPMM = runSkLearn(trainData, trainLabels, fullTrain)
    else:
        assignments = fitDPM(dpmm, trainData, trainLabels, fullTrain)

    eval = Evaluation()
    print("TRAINING...")
    eval.fullEvaluate(assignments, trainLabels, trainData.values, "train")

    # Test on unseen data
    testAssignments = np.array([])
    if Config.USE_SKLEARN:
        testAssignments = predictSkDPMM(skDPMM, testData, testLabels, fullTest)
    else:
        testAssignments = predictDPM(dpmm, testData, testLabels, trainData, fullTest)

    print("TESTING...")
    eval.fullEvaluate(testAssignments, testLabels, testData.values, "test")

    ######
    # dpmm.alpha = 1
    # completeData = pd.concat([trainData, testData])
    # completeLabels = np.concatenate((trainLabels, testLabels))
    # completeAssignments = fitDPM(
    #     dpmm, completeData, completeLabels, pd.concat([fullTrain, fullTest])
    # )
    # eval.fullEvaluate(
    #     completeAssignments, completeLabels, completeData.values, "complete"
    # )


if __name__ == "__main__":
    main()
