import numpy as np
from scipy.optimize import linear_sum_assignment


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


def mapLabelToRun(labels):
    labelMap = {
        -1: "NonRun",
        0: "Race",
        1: "Workout",
        2: "Long Run",
        3: "Training Run",
        4: "Recovery",
        5: "WU/CD",
        6: "Shakeout",
    }
    return np.array([labelMap[label] for label in labels if label in labelMap])


def mapRunToTier(labels):
    tiers = {
        -1: -1,
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 4,
        6: 4,
    }
    mapping = []
    for label in labels:
        mapping.append(tiers[label]) if label in tiers else mapping.append(5)
    return mapping


def transformLabels(labels):
    clusters = np.unique(labels)
    mapping = {val: idx for idx, val in enumerate(clusters)}
    newLabels = np.array([mapping[val] for val in labels])
    return newLabels


def HungarianMatch(predLabels, trueLabels):
    numLabels = max(np.unique(predLabels).shape[0], np.unique(trueLabels).shape[0]) + 1
    # Run the Hungarian algorithm to find mapping between true and predicted labels
    cost = np.zeros((numLabels, numLabels))
    for true, pred in zip(trueLabels, predLabels):
        cost[true][pred] += 1

    rowIdx, colIdx = linear_sum_assignment(cost, maximize=True)
    mapping = {pred: true for true, pred in zip(rowIdx, colIdx)}
    predLabels = np.array([mapping[pred] for pred in predLabels])
    return predLabels
