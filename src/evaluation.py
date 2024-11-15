from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

from util.mapRun import HungarianMatch


class Evaluation:
    def __init__(self):
        self.methods = ["ARI", "Silhouette", "Accuracy", "F1", "Precision", "Recall"]

    def evaluateAll(self, predLabels, trueLabels, dataValues=None):
        results = {}
        for method in self.methods:
            results[method] = self.evaluate(predLabels, trueLabels, method, dataValues)
        return results

    def evaluate(self, predLabels, trueLabels=None, method="ARI", dataValues=None):
        if trueLabels is not None:
            predLabels = HungarianMatch(predLabels, trueLabels)

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
            recall = recall_score(
                trueLabels, predLabels, average="weighted", zero_division=1
            )
            return recall
        else:
            print("Invalid evaluation method")

    def hierarchicalEvaluate(self):
        pass
