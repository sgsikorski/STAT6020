import numpy as np


def mapRunToLabel(labels):
    labelMap = {
        "NonRun": -1,
        "Race": 0,
        "Workout": 1,
        "Long Run": 2,
        "Training Run": 3,
        "Recovery": 4,
        "WU/CD": 5,
        "Shakeout": 6,
    }
    return np.array([labelMap[label] for label in labels if label in labelMap])


def mapRunToTier(labels):
    tiers = {
        "NonRun": -1,
        "Race": 0,
        "Workout": 1,
        "Long Run": 1,
        "Training Run": 2,
        "Recovery": 3,
        "WU/CD": 3,
        "Shakeout": 3,
    }
    return np.array([tiers[label] for label in labels if label in tiers])


def transformLabels(labels):
    clusters = np.unique(labels)
    mapping = {val: idx for idx, val in enumerate(clusters)}
    newLabels = np.array([mapping[val] for val in labels])
    return newLabels
