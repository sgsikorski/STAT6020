import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def countRunTypes(labels, isSplit=""):
    uniques, counts = np.unique(labels, return_counts=True)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.pie(counts, labels=uniques, autopct="%1.1f%%")
    ax.set_title(f"{isSplit.capitalize()} Activity Label Distribution")
    plt.savefig(f"res/{isSplit.capitalize()}_activity_label_distribution.png")
