import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def countRunTypes(data, isSplit=""):
    counts = data["Activity Label"].value_counts()
    plt.pie(counts, labels=data["Activity Label"].index, autopct="%1.1f%%")
    plt.title(f"{isSplit} Activity Label Distribution")
    plt.savefig(f"res/{isSplit}activity_label_distribution.png")
