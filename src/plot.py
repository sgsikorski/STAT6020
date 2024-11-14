import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
import numpy as np
import pandas as pd

from config import Config


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
    plt.xticks()
    plt.grid(True)
    if Config.SAVE_FIGS:
        plt.savefig(
            f"res/{'sk/' if Config.USE_SKLEARN else 'dpm/'}parallel_coordinates{Config.OUTPUT_SUFFIX}.png"
        )
    if Config.DEBUG:
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
    if Config.SAVE_FIGS:
        plt.savefig(
            f"res/{'sk/' if Config.USE_SKLEARN else 'dpm/'}cluster_plot_3d{Config.OUTPUT_SUFFIX}.png"
        )
    if Config.DEBUG:
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
    if Config.SAVE_FIGS:
        plt.savefig(
            f"res/{'sk/' if Config.USE_SKLEARN else 'dpm/'}cluster_plot{Config.OUTPUT_SUFFIX}.png"
        )
    if Config.DEBUG:
        plt.show()
