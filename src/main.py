import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import train_test_split

from plot import plotClusters2d, plotClusters3d, plotParallelCoordinates
from DPMM import DPMM
from config import Config
from evaluation import Evaluation
from util.mapRun import transformLabels, mapRunToTier, HungarianMatch

import sys


# Use sklearn's DPMM implementation
def getSklDPMM(alpha):
    model = BayesianGaussianMixture(
        n_components=50,  # Upper limit on number of components
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",  # Enable Dirichlet Process behavior
        weight_concentration_prior=alpha,  # Control sparsity of the mixture
    )
    return model


def reduceClusters(data, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return pd.DataFrame(reduced_data, columns=[f"PC{i+1}" for i in range(n_components)])


def printToFile(data, assignments):
    p = f"res/{'sk/' if Config.USE_SKLEARN else 'dpm/'}sklearn_cluster_assignments{Config.OUTPUT_SUFFIX}.txt"
    with open(
        p,
        "w+",
    ) as f:
        for i, (_, d) in enumerate(data.iterrows()):
            f.write(f"Point: {d} assigned to cluster {assignments[i]}\n")
            # print(f"Point: {d} assigned to cluster {assignments[i]}")


def main():
    # Set random seed for reproducibility
    np.random.seed(0)
    Config.SetConfig(sys.argv[1:])
    dpmm = DPMM(alpha=2, iters=100)
    dpmm.initializeData()

    data = dpmm.preprocessData()
    trainSplit = 0.8
    trainIdx, testIdx = train_test_split(
        data.index, train_size=trainSplit, random_state=0
    )
    trainData, testData = data.iloc[trainIdx], data.iloc[testIdx]
    trainLabels, testLabels = dpmm.labels[trainIdx.values], dpmm.labels[testIdx.values]

    fullTrain, fullTest = trainData, testData
    trainData = reduceClusters(trainData, n_components=2)
    testData = reduceClusters(testData, n_components=2)

    assignments = np.array([])
    if Config.USE_SKLEARN:
        skDPMM = getSklDPMM(100)
        skDPMM.fit(trainData)
        assignments = skDPMM.predict(trainData)
        assignments = transformLabels(assignments)
        assignments = HungarianMatch(assignments, trainLabels)
        plotClusters2d(trainData.values, assignments)
        # plotClusters3d(data.values, assignments)
        plotParallelCoordinates(
            pd.DataFrame(fullTrain, columns=fullTrain.columns), assignments
        )

        if Config.DEBUG:
            printToFile(fullTrain, assignments)
    else:
        dpmm.data = trainData
        assignments = dpmm.fit(trainData)
        assignments = transformLabels(assignments)
        assignments = HungarianMatch(assignments, trainLabels)
        plotClusters2d(trainData.values, assignments, "raw")
        # plotClusters3d(data.values, assignments)
        plotParallelCoordinates(
            pd.DataFrame(fullTrain, columns=fullTrain.columns), assignments, "raw"
        )

        if Config.DEBUG:
            printToFile(fullTrain, assignments)

    plotClusters2d(trainData.values, trainLabels)
    eval = Evaluation()
    results = eval.evaluateAll(assignments, trainLabels, trainData.values)
    with open(
        f"res/{'sk/' if Config.USE_SKLEARN else 'dpm/'}results{Config.OUTPUT_SUFFIX}.txt",
        "w+",
    ) as f:
        print(results, file=f)
        print(results)

    newClusters = mapRunToTier(assignments)
    newLabels = mapRunToTier(trainLabels)
    results2 = eval.evaluateAll(newClusters, newLabels, trainData.values)
    print(results2)
    plotClusters2d(trainData.values, newClusters, "tiered")

    # Test on unseen data
    testAssignments = np.array([])
    if Config.USE_SKLEARN:
        testAssignments = skDPMM.predict(testData)
        testAssignments = transformLabels(testAssignments)
        testAssignments = HungarianMatch(testAssignments, testLabels)

        if Config.DEBUG:
            printToFile(fullTest, testAssignments)
    else:
        testAssignments = dpmm.predictSamples(testData.values, trainData)
        testAssignments = transformLabels(testAssignments)
        testAssignments = HungarianMatch(testAssignments, testLabels)

        if Config.DEBUG:
            printToFile(fullTest, testAssignments)

    newClusters = mapRunToTier(testAssignments)
    newLabels = mapRunToTier(testLabels)
    results2 = eval.evaluateAll(newClusters, newLabels, testData.values)
    print(results2)
    plotClusters2d(testData.values, newClusters)

    ######
    # completeData = trainData + testData
    # completeLabels = np.concatenate((trainLabels, testLabels))
    # dpmm.predictFitSamples(testData.values, completeData.values)

    # assignments = dpmm.assignments
    # assignments = transformLabels(assignments)
    # assignments = HungarianMatch(assignments, completeLabels)
    # plotClusters2d(completeData.values, assignments)
    # newClusters = mapRunToTier(assignments)
    # newLabels = mapRunToTier(completeLabels)
    # results3 = eval.evaluateAll(newClusters, newLabels, completeData.values)


if __name__ == "__main__":
    main()
