import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture

from plot import plotClusters2d, plotClusters3d, plotParallelCoordinates
from DPMM import DPMM
from config import Config

import sys

# Set random seed for reproducibility
np.random.seed(0)


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
    return reduced_data


def printToFile(data, assignments):
    p = f"res/{'sk/' if Config.USE_SKLEARN else ''}sklearn_cluster_assignments{Config.OUTPUT_SUFFIX}.txt"
    with open(
        p,
        "w+",
    ) as f:
        for i, (_, d) in enumerate(data.iterrows()):
            f.write(f"Point: {d} assigned to cluster {assignments[i]}\n")
            print(f"Point: {d} assigned to cluster {assignments[i]}")


def main():
    global DEBUG
    if "-d" in sys.argv:
        DEBUG = True
    dpmm = DPMM(alpha=0.001)
    dpmm.initializeData()

    data = dpmm.preprocessData()
    fullData = data
    data = reduceClusters(data, n_components=3)

    if Config.USE_SKLEARN:
        skDPMM = getSklDPMM(10)
        skDPMM.fit(data)
        skAssignments = skDPMM.predict(data)
        print(skAssignments)
        plotClusters2d(reduceClusters(data, n_components=2), skAssignments)
        plotClusters3d(data, skAssignments)
        plotParallelCoordinates(
            pd.DataFrame(fullData, columns=fullData.columns), skAssignments
        )

        if Config.DEBUG:
            printToFile(fullData, skAssignments)
    else:
        assignments = dpmm.fit()
        print(assignments)
        plotClusters2d(reduceClusters(data, n_components=2), assignments)
        plotClusters3d(data, assignments)
        plotParallelCoordinates(
            pd.DataFrame(fullData, columns=fullData.columns), assignments
        )

        if Config.DEBUG:
            printToFile(fullData, assignments)


if __name__ == "__main__":
    main()
