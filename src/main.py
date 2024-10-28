import numpy as np
import pandas as pd
from collections import defaultdict

from datetime import datetime, timedelta

from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import wishart, multivariate_normal
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns

import sys

# Set random seed for reproducibility
np.random.seed(0)
DEBUG = True


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
        ]

    def initializeData(self, dataset="src/Activities.csv"):
        self.data = self.loadData(dataset)
        self.mu = np.zeros(self.data.shape[1])
        self.psi = np.eye(self.data.shape[1])
        self.nu = self.data.shape[1] + 2

    def loadData(self, dataset):
        df = pd.read_csv(dataset)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.replace("--", np.nan)
        one_year_ago = datetime.now() - timedelta(days=120)
        df = df[df["Date"] > one_year_ago]
        df = df[df["Activity Type"] == "Running"]
        df = df[self.cols]
        df.dropna(inplace=True)

        def hmsToSeconds(x):
            h, m, s = map(float, x.split(":"))
            return h * 3600 + m * 60 + s

        def paceToSpeed(pace):
            print(pace)
            minutes, seconds = map(int, pace.split(":"))
            pace_in_minutes = minutes + seconds / 60
            speed = 60 / pace_in_minutes
            return speed

        df["Time"] = df["Time"].apply(hmsToSeconds)
        df["Avg Pace"] = df["Avg Pace"].apply(paceToSpeed)
        df["Best Pace"] = df["Best Pace"].apply(paceToSpeed)

        if DEBUG:
            print(df.columns.array.tolist())
            print(df)
        return df

    def preprocessData(self):
        scaler = MinMaxScaler()
        scaledData = scaler.fit_transform(self.data)
        scaledDF = pd.DataFrame(scaledData, columns=self.data.columns)
        # for col in scaledDF.columns:
        #     scaledDF[col] = scaledDF[col].apply(
        #         lambda x: (x - scaledDF[col].mean()) / scaledDF[col].std()
        #     )
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

            if DEBUG:
                print(f"Iteration {_ + 1} complete")
                print(f"Number of Clusters: {len(clusters)}")
                print(f"Cluster assignments: {assignments}")


# Use sklearn's DPMM implementation
def getSklDPMM(alpha):
    model = BayesianGaussianMixture(
        n_components=7,  # Upper limit on number of components
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",  # Enable Dirichlet Process behavior
        weight_concentration_prior=alpha,  # Control sparsity of the mixture
    )
    return model


def reduceClusters(data, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data


def plotParallelCoordinates(data, clusters):
    data["Cluster"] = clusters

    plt.figure(figsize=(12, 6))
    pd.plotting.parallel_coordinates(
        data,
        "Cluster",
        color=sns.color_palette("viridis", len(np.unique(clusters))),
        alpha=0.7,
    )
    plt.title("Parallel Coordinates Plot for Clusters")
    plt.xlabel("Features")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig("res/parallel_coordinates.png")
    if DEBUG:
        plt.show()


def plotClusters3d(data, clusters):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Scatter plot of the data points
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=clusters, cmap="viridis")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")

    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster Label")

    for cluster in np.unique(clusters):
        cluster_points = data[clusters == cluster]
        center = cluster_points.mean(axis=0)

        ax.text(
            center[0],
            center[1],
            center[2],
            str(cluster),
            color="black",
            fontsize=12,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
        )
    plt.savefig("res/cluster_plot_3d.png")
    if DEBUG:
        plt.show()


def plotClusters2d(data, clusters):
    fig = plt.figure()
    ax = fig.add_subplot()

    # Scatter plot of the data points
    scatter = ax.scatter(data[:, 0], data[:, 1], c=clusters, cmap="viridis")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster Label")

    for cluster in np.unique(clusters):
        cluster_points = data[clusters == cluster]
        center = cluster_points.mean(axis=0)
        radius_x = cluster_points[:, 0].std()
        radius_y = cluster_points[:, 1].std()

        ellipse = Ellipse(
            (center[0], center[1]),
            width=2 * radius_x,
            height=2 * radius_y,
            edgecolor="r",
            facecolor="none",
            lw=2,
        )
        ax.add_patch(ellipse)

        ax.text(
            center[0],
            center[1],
            str(cluster),
            color="black",
            fontsize=12,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
        )
    plt.savefig("res/cluster_plot.png")
    if DEBUG:
        plt.show()


def main():
    global DEBUG
    if "-d" in sys.argv:
        DEBUG = True
    dpmm = DPMM(alpha=1)
    dpmm.initializeData()

    skDPMM = getSklDPMM(0.001)
    data = dpmm.preprocessData()
    skDPMM.fit(data[dpmm.cols])

    assignments = skDPMM.predict(data[dpmm.cols])

    if DEBUG:
        pointToAssign = defaultdict(list)
        with open("res/cluster_assignments.txt", "w+") as f:
            for i, (_, d) in enumerate(data.iterrows()):
                pointToAssign[assignments[i]].append(d)
                f.write(f"Point: {d} assigned to cluster {assignments[i]}\n")

        for i, (_, d) in enumerate(data.iterrows()):
            pointToAssign[assignments[i]].append(d)
            print(f"Point: {d} assigned to cluster {assignments[i]}")

    plotClusters2d(reduceClusters(data[dpmm.cols], n_components=2), assignments)
    plotClusters3d(reduceClusters(data[dpmm.cols], n_components=3), assignments)
    plotParallelCoordinates(data, assignments)


if __name__ == "__main__":
    main()
