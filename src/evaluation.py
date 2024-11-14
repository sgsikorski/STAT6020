from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.optimize import linear_sum_assignment
import numpy as np


class Evaluation:
    def __init__(self):
        self.methods = ["ARI", "Silhouette", "Accuracy", "F1", "Precision", "Recall"]
        self.results = {}

    def evaluateAll(self, predLabels, trueLabels, dataValues=None):
        for method in self.methods:
            self.results[method] = self.evaluate(
                predLabels, trueLabels, method, dataValues
            )
        return self.results

    def evaluate(self, predLabels, trueLabels=None, method="ARI", dataValues=None):
        if trueLabels is not None:
            numLabels = (
                max(np.unique(predLabels).shape[0], np.unique(trueLabels).shape[0]) + 1
            )
            # Run the Hungarian algorithm to find mapping between true and predicted labels
            cost = np.zeros((numLabels, numLabels))
            for true, pred in zip(trueLabels, predLabels):
                cost[true][pred] += 1

            rowIdx, colIdx = linear_sum_assignment(cost, maximize=True)
            mapping = {pred: true for true, pred in zip(rowIdx, colIdx)}
            predLabels = np.array([mapping[pred] for pred in predLabels])
        if method == "ARI":
            ari = adjusted_rand_score(trueLabels, predLabels)
            return ari
        elif method == "Silhouette":
            silhouette = silhouette_score(dataValues, predLabels)
            return silhouette
        elif method == "Accuracy":
            if trueLabels is None:
                return None
            accuracy = accuracy_score(trueLabels, predLabels)
            return accuracy
        elif method == "F1":
            if trueLabels is None:
                return None
            f1 = f1_score(trueLabels, predLabels, average="weighted")
            return f1
        elif method == "Precision":
            if trueLabels is None:
                return None
            precision = precision_score(trueLabels, predLabels, average="weighted")
            return precision
        elif method == "Recall":
            if trueLabels is None:
                return None
            recall = recall_score(trueLabels, predLabels, average="weighted")
            return recall
        else:
            print("Invalid evaluation method")

    def hierarchicalEvaluate(self):
        pass
