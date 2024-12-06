from util.mapRun import transformLabels, mapRunToTier, HungarianMatch
from sklearn.mixture import BayesianGaussianMixture
from util.plot import plotClusters2d, plotParallelCoordinates
from config import Config
import pandas as pd


# Use sklearn's DPMM implementation
def getSklDPMM(alpha):
    model = BayesianGaussianMixture(
        n_components=50,  # Upper limit on number of components
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",  # Enable Dirichlet Process behavior
        weight_concentration_prior=alpha,  # Control sparsity of the mixture
    )
    return model


def runSkLearn(data, labels, fullData):
    skDPMM = getSklDPMM(300)
    skDPMM.fit(data)
    assignments = skDPMM.predict(data)
    assignments = transformLabels(assignments)
    assignments = HungarianMatch(assignments, labels)
    print(f"{len(set(assignments))} clusters found")
    plotClusters2d(data.values, assignments)

    return assignments, skDPMM


def predictSkDPMM(skDPMM, data, labels, trainData):
    assignments = skDPMM.predict(data)
    assignments = transformLabels(assignments)
    assignments = HungarianMatch(assignments, labels)
    return assignments


def fitDPM(dpmm, data, labels, fullData):
    dpmm.data = data
    assignments = dpmm.fit(data)
    assignments = transformLabels(assignments)
    assignments = HungarianMatch(assignments, labels)
    # plotClusters2d(data.values, assignments, "raw")
    # plotParallelCoordinates(
    #    pd.DataFrame(fullData, columns=fullData.columns), assignments, "raw"
    # )

    return assignments


def predictDPM(dpmm, data, labels, trainData, fullData):
    assignments = dpmm.predictSamples(data.values, trainData)
    assignments = transformLabels(assignments)
    assignments = HungarianMatch(assignments, labels)
    return assignments
