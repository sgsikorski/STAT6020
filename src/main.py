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
        n_components=10,  # Upper limit on number of components
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",  # Enable Dirichlet Process behavior
        weight_concentration_prior=alpha,  # Control sparsity of the mixture
    )
    return model


def reduceClusters(data, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data


def main():
    global DEBUG
    if "-d" in sys.argv:
        DEBUG = True
    dpmm = DPMM(alpha=1)
    dpmm.initializeData()

    skDPMM = getSklDPMM(10)
    data = dpmm.preprocessData()
    fullData = data
    data = reduceClusters(data, n_components=3)
    skDPMM.fit(data)

    assignments = skDPMM.predict(data)
    print(assignments)
    plotClusters2d(data, assignments)
    plotClusters3d(data, assignments)
    plotParallelCoordinates(
        pd.DataFrame(fullData, columns=fullData.columns), assignments
    )

    if Config.DEBUG:
        pointToAssign = defaultdict(list)
        with open(f"res/cluster_assignments{Config.OUTPUT_SUFFIX}.txt", "w+") as f:
            for i, (_, d) in enumerate(fullData.iterrows()):
                pointToAssign[assignments[i]].append(d)
                f.write(f"Point: {d} assigned to cluster {assignments[i]}\n")
                print(f"Point: {d} assigned to cluster {assignments[i]}")


if __name__ == "__main__":
    main()
