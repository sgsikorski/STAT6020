from scipy.stats import wishart, multivariate_normal
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

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

    def initializeData(self, dataset="src/Activities.csv"):
        self.data, self.labels = self.loadData(dataset)
        self.mu = np.zeros(self.data.shape[1])
        self.psi = np.eye(self.data.shape[1])
        self.nu = self.data.shape[1] + 2

    def loadData(self, dataset):
        df = pd.read_csv(dataset)
        self.labels = df["Activity Label"]
        return df, df["Activity Label"]

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

        df = df[self.cols]
        df.dropna(inplace=True)

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

        scaler = MinMaxScaler()
        scaledData = scaler.fit_transform(df)
        scaledDF = pd.DataFrame(scaledData, columns=df.columns)

        self.data = scaledDF
        return scaledDF

    def fit(self):
        if not self.data:
            self.initializeData()
        N, D = self.data.shape
        assignments = np.zeros(N, dtype=int)
        clusters = {0: [self.data[0]]}

        samples = []

        for _ in range(self.iters):
            for n in range(N):
                currP = self.data[n]
                currC = assignments[n]
                clusters[currC].remove(currP)
                if len(clusters[currC]) == 0:
                    del clusters[currC]

                sizes = {k: len(clusters[k]) for k in clusters}
                totalP = sum(sizes.values())

                posteriorProbs = []
                options = list(clusters.keys())

                for k in options:
                    cluster_data = np.array(clusters[k])
                    mu_k = np.mean(cluster_data, axis=0)
                    cov_k = np.cov(cluster_data.T) + np.eye(D) * 1e-6
                    likelihood = multivariate_normal.pdf(currP, mean=mu_k, cov=cov_k)

                    # Prior: proportion of data already in cluster
                    prior = sizes[k] / (totalP + self.alpha)

                    posteriorProbs.append(prior * likelihood)

                # Adding the probability of forming a new cluster
                sigma = wishart.rvs(df=self.nu, scale=self.psi)
                newMu = np.random.multivariate_normal(self.mu, sigma / self._lambda)
                newLikelihood = multivariate_normal.pdf(currP, mean=newMu, cov=sigma)

                newClusterPrior = self.alpha / (totalP + self.alpha)
                posteriorProbs.append(newClusterPrior * newLikelihood)

                # Normalize posterior probabilities
                posteriorProbs = np.array(posteriorProbs)
                posteriorProbs /= posteriorProbs.sum()

                # Sample new cluster assignment
                newCluster = np.random.choice(len(posteriorProbs), p=posteriorProbs)

                # Assign the data point to the chosen cluster (either existing or new)
                if newCluster == len(options):  # New cluster
                    maxId = max(options) + 1
                    assignments[n] = maxId
                    clusters[maxId] = [currP]
                else:  # Existing cluster
                    assignments[n] = options[newCluster]
                    clusters[options[newCluster]].append(currP)

            # Store cluster assignments
            samples.append(np.copy(assignments))

            if Config.DEBUG:
                print(f"Iteration {_ + 1} complete")
                print(f"Number of Clusters: {len(clusters)}")
                print(f"Cluster assignments: {assignments}")

    def evaluate(self):
        for sample in self.data.iterrows():
            pass
