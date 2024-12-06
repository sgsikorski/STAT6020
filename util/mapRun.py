import numpy as np
from scipy.optimize import linear_sum_assignment


def mapRunToLabel(labels):
    labelMap = {
        "NonRun": -1,
        "Shakeout": 0,
        "WU/CD": 1,
        "Recovery": 2,
        "Training Run": 3,
        "Long Run": 4,
        "Workout": 5,
        "Race": 6,
    }
    return np.array([labelMap[label] for label in labels if label in labelMap])


def mapLabelToRun(labels):
    labelMap = {
        -1: "NonRun",
        0: "Shakeout",
        1: "WU/CD",
        2: "Recovery",
        3: "Training Run",
        4: "Long Run",
        5: "Workout",
        6: "Race",
    }
    return np.array([labelMap[label] for label in labels if label in labelMap])


def mapLabelToPlotName(label, isTiered=False):
    labelMap = (
        {
            -1: "NR",
            0: "SO",
            1: "WU/CD",
            2: "Rec",
            3: "TR",
            4: "LR",
            5: "WO",
            6: "Race",
        }
        if not isTiered
        else {-1: "NR", 0: "SO/WU/CD/Rec", 1: "TR", 2: "LR", 3: "WO", 4: "Race"}
    )
    return labelMap[label] if label in labelMap else "Unknown"


def mapRunToTier(labels):
    tiers = {
        -1: -1,
        0: 0,
        1: 0,
        2: 0,
        3: 1,
        4: 2,
        5: 3,
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
