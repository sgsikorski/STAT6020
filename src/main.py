import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture

from plot import plotClusters2d, plotClusters3d, plotParallelCoordinates
from DPMM import DPMM
from config import Config
from evaluation import Evaluation
from util.mapRun import transformLabels

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
    dpmm = DPMM(alpha=0.5, iters=100)
    dpmm.initializeData()

    data = dpmm.preprocessData()
    fullData = data
    data = reduceClusters(data, n_components=2)

    assignments = np.array([])
    if Config.USE_SKLEARN:
        skDPMM = getSklDPMM(100)
        skDPMM.fit(data)
        assignments = skDPMM.predict(data)
        assignments = transformLabels(assignments)
        plotClusters2d(reduceClusters(data, n_components=2).values, assignments)
        # plotClusters3d(data.values, assignments)
        plotParallelCoordinates(
            pd.DataFrame(fullData, columns=fullData.columns), assignments
        )

        if Config.DEBUG:
            printToFile(fullData, assignments)
    else:
        dpmm.data = data
        assignments = dpmm.fit()
        assignments = transformLabels(assignments)
        plotClusters2d(data.values, assignments)
        # plotClusters3d(data.values, assignments)
        plotParallelCoordinates(
            pd.DataFrame(fullData, columns=fullData.columns), assignments
        )

        if Config.DEBUG:
            printToFile(fullData, assignments)

    assignments = transformLabels(assignments)
    eval = Evaluation()
    results = eval.evaluateAll(assignments, dpmm.labels, dpmm.data.values)
    with open(
        f"res/{'sk/' if Config.USE_SKLEARN else 'dpm/'}results{Config.OUTPUT_SUFFIX}.txt",
        "w+",
    ) as f:
        print(results, file=f)
        print(results)


if __name__ == "__main__":
    main()
