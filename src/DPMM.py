from scipy.stats import invwishart, multivariate_normal, wishart
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Literal

from config import Config


class DPMM:
    def __init__(self, alpha, iters=100):
        self.alpha = alpha
        self._lambda = 1
        self.iters = iters
        self.cols = [
            "Time",
            "Distance",
            "Avg Pace",
            "Best Pace",
            "Avg HR",
            "Max HR",
            "Avg Power",
            "Max Power",
            # "Calories",
            # "Avg Run Cadence",
            # "Max Run Cadence",
            # "Total Ascent"
        ]
        self.clusters = {0: [0]}
        self.idx = 0

    def initializeData(self, dataset="src/Activities.csv"):
        self.data, self.labels = self.loadData(dataset)

    def loadData(self, dataset):
        df = pd.read_csv(dataset)
        return df, self.mapRunToLabel(df["Activity Label"])

    def preprocessData(self):
        df = self.data
        df["Date"] = pd.to_datetime(df["Date"])

        df.replace("--", np.nan, inplace=True)
        df["Best Pace"].fillna(df["Avg Pace"], inplace=True)

        one_year_ago = datetime.now() - timedelta(days=120)
        df = df[df["Date"] > one_year_ago]
        df = df[
            df["Activity Type"].isin(["Running", "Track Running", "Treadmill Running"])
        ]

        df = df[self.cols + ["Activity Label"]]
        df.dropna(inplace=True)
        self.labels = self.mapRunToLabel(df["Activity Label"])
        df = df[self.cols]

        def hmsToSeconds(x):
            h, m, s = map(float, x.split(":"))
            return h * 3600 + m * 60 + s

        def paceToSpeed(pace):
            minutes, seconds = map(int, pace.split(":"))
            pace_in_minutes = minutes + seconds / 60
            speed = 60 / pace_in_minutes
            return speed

        df["Time"] = df["Time"].apply(hmsToSeconds)
        df["Avg Pace"] = df["Avg Pace"].apply(paceToSpeed)
        df["Best Pace"] = df["Best Pace"].apply(paceToSpeed)

        if Config.DEBUG:
            print(df.columns.array.tolist())
            print(df)

        # TODO: See how PCA prior to MinMax changes results
        # data = reduceClusters(df, 3)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaledData = scaler.fit_transform(df)
        scaledDF = pd.DataFrame(scaledData, columns=df.columns)

        self.data = scaledDF
        return scaledDF

    def mapRunToLabel(self, labels):
        labelMap = {
            "NonRun": -1,
            "Race": 0,
            "Workout": 1,
            "Long Run": 2,
            "Training Run": 3,
            "Recovery": 4,
            "Double": 5,
            "WU/CD": 6,
            "Shakeout": 7,
        }
        return np.array([labelMap[label] for label in labels if label in labelMap])

    def fit(self):
        if self.data.empty:
            self.initializeData()
        self.mu = np.zeros(self.data.shape[1])
        self.psi = np.eye(self.data.shape[1])
        self.nu = self.data.shape[1] + 10
        self.N, self.D = self.data.shape
        self.assignments = np.zeros(self.N, dtype=int)

        # Start over if rerunning fit
        if self.idx > 0:
            self.idx = 0
            self.clusters = {0: [0]}
        for _ in range(self.iters):
            for n in range(self.N):
                currP = self.data.iloc[n].values
                currC = self.assignments[n]
                if n in self.clusters[currC]:
                    self.clusters[currC].remove(n)
                if len(self.clusters[currC]) == 0:
                    del self.clusters[currC]

                self.predict(currP)

            if Config.DEBUG:
                print(f"Iteration {_ + 1} complete")
                print(f"Number of Clusters: {len(self.clusters)}")
                print(f"Cluster assignments: {self.assignments}")
            # TODO: Add better convergence criterion
            if len(self.clusters) == 8:
                break
            self.idx = 0
        return self.assignments

    def addPoint(self, pointIdx, newCluster, new_cluster_possible):
        options = list(self.clusters.keys())
        if pointIdx >= len(self.assignments):
            self.assignments = self.assignments.append(-1)
        # Assign the data point to the chosen cluster (either existing or new)
        if new_cluster_possible and newCluster == len(options):  # New cluster
            maxId = max(options) + 1 if len(options) > 0 else 0
            self.assignments[pointIdx] = maxId
            self.clusters[maxId] = [pointIdx]
        else:  # Existing cluster
            self.assignments[pointIdx] = options[newCluster]
            self.clusters[options[newCluster]].append(pointIdx)

    def predict(self, sample):
        sizes = {k: len(self.clusters[k]) for k in self.clusters}
        totalP = sum(sizes.values())

        posteriorProbs = []
        options = list(self.clusters.keys())

        for k in options:
            cluster_data = np.array(self.data.iloc[self.clusters[k]].values)
            mu_k = np.mean(cluster_data, axis=0)
            cov_k = np.eye(self.D) * 5e-2
            if len(cluster_data) > 1:
                cov_k = np.cov(cluster_data, rowvar=False) + np.eye(self.D) * 1e-3
            likelihood = multivariate_normal.pdf(sample, mean=mu_k, cov=cov_k)

            # Prior: proportion of data already in cluster
            prior = sizes[k] / (totalP + self.alpha)

            posteriorProbs.append(prior * likelihood)

        if len(self.clusters) < 20:
            # Add the probability of forming a new cluster
            sigma = wishart.rvs(df=self.nu, scale=self.psi)
            newMu = np.random.multivariate_normal(self.mu, sigma / self._lambda)
            newLikelihood = multivariate_normal.pdf(sample, mean=newMu, cov=sigma)
            newClusterPrior = self.alpha / (totalP + self.alpha)
            posteriorProbs.append(newClusterPrior * newLikelihood)
            new_cluster_possible = True
        else:
            new_cluster_possible = False

        # Normalize posterior probabilities
        posteriorProbs = np.array(posteriorProbs)
        posteriorProbs /= posteriorProbs.sum()

        # Sample new cluster assignment
        newCluster = np.random.choice(len(posteriorProbs), p=posteriorProbs)
        self.addPoint(self.idx, newCluster, new_cluster_possible)
        self.idx += 1
        return newCluster, new_cluster_possible

    def predictSamples(self, samples):
        for sample in samples:
            self.predict(sample)

    def transformLabels(self, labels):
        clusters = np.unique(labels)
        mapping = {val: idx for idx, val in enumerate(clusters)}
        newLabels = np.array([mapping[val] for val in labels])
        return newLabels

    def evaluate(
        self,
        predLabels,
        trueLabels=None,
        method: Literal["ARI", "Silhouette", "Accuracy", "F1"] = "Silhouette",
    ):
        if trueLabels is not None:
            # Run the Hungarian algorithm to find mapping between true and predicted labels
            cost = np.zeros((8, 8))
            for true, pred in zip(trueLabels, predLabels):
                cost[true][pred] += 1

            rowIdx, colIdx = linear_sum_assignment(cost, maximize=True)
            mapping = {pred: true for true, pred in zip(rowIdx, colIdx)}
            predLabels = np.array([mapping[pred] for pred in predLabels])
        if method == "ARI":
            ari = adjusted_rand_score(trueLabels, predLabels)
            return ari
        elif method == "Silhouette":
            silhouette = silhouette_score(self.data.values, predLabels)
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
        else:
            print("Invalid evaluation method")
