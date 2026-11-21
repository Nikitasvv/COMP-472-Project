# k-Means clustering experiment on the sentiment data (negative / neutral / positive).
# Run on its own:  python test_kmeans.py
import csv
import time

import numpy as np

from experiment_helpers import setup, load_data
import matplotlib.pyplot as plt  # imported after experiment_helpers, which switches matplotlib to file output
from kmeans import fit_kmeans, calculate_inertia, cluster_purity
from utils import SEED

K_VALUES = [2, 3, 4, 5, 6]
CLASS_NAMES = ["Negative", "Neutral", "Positive"]


def run_kmeans(data):
    features = data["sentiment"]["train_features"]
    # The true labels are never given to k-Means, they are only used to score the clusters
    true_labels = data["sentiment"]["train_labels"]
    print("\n++++ Clustering: k-Means on the sentiment data ++++")
    print(f"{len(features)} points, {features.shape[1]} features")

    results = []
    k3_assignments = None  # kept for the cluster-vs-class table at the end

    for k in K_VALUES:
        centers, assignments = fit_kmeans(features, k=k, seed=SEED)
        inertia = calculate_inertia(features, centers, assignments)
        purity = cluster_purity(assignments, true_labels)
        results.append((k, inertia, purity))
        if k == 3:
            k3_assignments = assignments

    with open("kmeans_results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["k", "Inertia", "Purity"])
        for k, inertia, purity in results:
            writer.writerow([k, f"{inertia:.2f}", f"{purity:.4f}"])
    print("Saved kmeans_results.csv")

    print(f"\n{'k':>3} {'inertia':>14} {'purity':>8}")
    for k, inertia, purity in results:
        print(f"{k:>3} {inertia:14.1f} {purity:8.3f}")
    print(
        "Note: purity tends to rise when k rises, so the highest purity is not automatically the best k."
    )

    # Elbow plot: the k where inertia stops dropping quickly is a reasonable choice
    plt.figure(figsize=(6, 4))
    plt.plot(K_VALUES, [r[1] for r in results], marker="o")
    plt.xlabel("number of clusters k")
    plt.ylabel("inertia")
    plt.xticks(K_VALUES)
    plt.title("k-Means: inertia vs k (sentiment)")
    plt.tight_layout()
    plt.savefig("figures/kmeans_inertia.png", dpi=120)
    plt.close()
    print("Saved figures/kmeans_inertia.png")

    # With k=3 the clusters could in theory match the 3 sentiment classes; count how well they do
    print("\nk=3: clusters (rows) vs true sentiment (columns)")
    print(f"{'':>10}" + "".join(f"{name:>10}" for name in CLASS_NAMES))
    for cluster in range(3):
        counts = np.bincount(true_labels[k3_assignments == cluster], minlength=3)
        print(
            f"{'cluster ' + str(cluster):>10}"
            + "".join(f"{count:>10}" for count in counts)
        )


if __name__ == "__main__":
    setup()
    data = load_data()
    start_time = time.time()
    run_kmeans(data)
    print(f"\nk-Means finished in {time.time() - start_time:.1f}s")
