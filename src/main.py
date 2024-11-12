import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture

from plot import plotClusters2d, plotClusters3d, plotParallelCoordinates
from DPMM import DPMM
from config import Config

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
            print(f"Point: {d} assigned to cluster {assignments[i]}")


def main():
    # Set random seed for reproducibility
    np.random.seed(0)
    Config.SetConfig(sys.argv[1:])
    dpmm = DPMM(alpha=25)
    dpmm.initializeData()

    data = dpmm.preprocessData()
    fullData = data
    data = reduceClusters(data, n_components=3)

    assignments = np.array([])
    if Config.USE_SKLEARN:
        skDPMM = getSklDPMM(10)
        skDPMM.fit(data)
        assignments = skDPMM.predict(data)
        plotClusters2d(reduceClusters(data, n_components=2).values, assignments)
        plotClusters3d(data.values, assignments)
        plotParallelCoordinates(
            pd.DataFrame(fullData, columns=fullData.columns), assignments
        )

        if Config.DEBUG:
            printToFile(fullData, assignments)
    else:
        dpmm.data = data
        assignments = dpmm.fit()
        plotClusters2d(data.values, assignments)
        plotClusters3d(data.values, assignments)
        plotParallelCoordinates(
            pd.DataFrame(fullData, columns=fullData.columns), assignments
        )

        if Config.DEBUG:
            printToFile(fullData, assignments)

    assignments = dpmm.transformLabels(assignments)

    silhouette = dpmm.evaluate(assignments, dpmm.labels, "Silhouette")
    ari = dpmm.evaluate(assignments, dpmm.labels, "ARI")
    accuracy = dpmm.evaluate(assignments, dpmm.labels, "Accuracy")
    f1 = dpmm.evaluate(assignments, dpmm.labels, "F1")
    print(f"Silhouette: {silhouette}")
    print(f"ARI: {ari}")
    print(f"Accuracy: {accuracy}")
    print(f"F1: {f1}")


if __name__ == "__main__":
    main()
