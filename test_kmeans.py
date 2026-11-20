# k-Means clustering experiment on the sentiment data (negative / neutral / positive).
# Run on its own:  python test_kmeans.py
import csv
import time

import numpy as np

from experiment_helpers import setup, load_data
import matplotlib.pyplot as plt  # after experiment_helpers, which switches matplotlib to file output
from kmeans import fit_kmeans, calculate_inertia, cluster_purity
from utils import SEED

K_VALUES = [2, 3, 4, 5, 6]
PREPROCESSING = ["none", "standard"]
CLASS_NAMES = ["Negative", "Neutral", "Positive"]


def standardize_features(features):
    # k-Means uses Euclidean distance, so features with a large range would dominate it.
    # Standardizing gives every feature about the same weight.
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0  # constant feature: avoid dividing by 0
    return (features - mean) / std


def run_kmeans(data):
    features = data["sentiment"]["train_features"]
    true_labels = data["sentiment"][
        "train_labels"
    ]  # never given to k-Means, only used to score it
    print("\n++++ Clustering: k-Means on the sentiment data ++++")
    print(f"{len(features)} points, {features.shape[1]} features")

    results = []  # one row per (preprocessing, k)
    k3_assignments = {}  # k=3 clusters, compared with the true labels at the end

    for method in PREPROCESSING:
        method_features = (
            features if method == "none" else standardize_features(features)
        )

        for k in K_VALUES:
            centers, assignments = fit_kmeans(method_features, k=k, seed=SEED)
            inertia = calculate_inertia(method_features, centers, assignments)
            purity = cluster_purity(assignments, true_labels)
            results.append((method, k, inertia, purity))
            if k == 3:
                k3_assignments[method] = assignments

    # Save the results
    with open("kmeans_results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Preprocessing", "k", "Inertia", "Purity"])
        for method, k, inertia, purity in results:
            writer.writerow([method, k, f"{inertia:.2f}", f"{purity:.4f}"])
    print("Saved kmeans_results.csv")

    # Print a summary table
    print(f"\n{'preprocessing':>13} {'k':>3} {'inertia':>14} {'purity':>8}")
    for method, k, inertia, purity in results:
        print(f"{method:>13} {k:>3} {inertia:14.1f} {purity:8.3f}")
    print(
        "Note: inertia of 'none' and 'standard' cannot be compared, because the scales are different."
    )
    print(
        "Note: purity tends to rise when k rises, so the highest purity is not automatically the best k."
    )

    # Inertia vs k. The two methods have very different scales, so each gets its own y-axis:
    # this shows the trend of each line, but not their absolute values.
    inertia_by_method = {m: [r[2] for r in results if r[0] == m] for m in PREPROCESSING}

    figure, left_axis = plt.subplots(figsize=(6, 4))
    right_axis = left_axis.twinx()
    line_none = left_axis.plot(
        K_VALUES,
        inertia_by_method["none"],
        marker="o",
        color="tab:blue",
        label="none (left axis)",
    )
    line_standard = right_axis.plot(
        K_VALUES,
        inertia_by_method["standard"],
        marker="o",
        color="tab:orange",
        label="standard (right axis)",
    )
    left_axis.set_xlabel("number of clusters k")
    left_axis.set_ylabel("inertia (none)", color="tab:blue")
    right_axis.set_ylabel("inertia (standard)", color="tab:orange")
    left_axis.set_xticks(K_VALUES)
    left_axis.set_title(
        "k-Means: inertia vs k (sentiment)\nthe two axes have different scales"
    )
    both_lines = line_none + line_standard
    left_axis.legend(
        both_lines, [line.get_label() for line in both_lines], loc="upper right"
    )
    figure.tight_layout()
    figure.savefig("plots/kmeans_inertia.png", dpi=120)
    plt.close(figure)
    print("Saved plots/kmeans_inertia.png")

    # For k=3: how many samples of each true class ended up in each cluster
    for method in PREPROCESSING:
        assignments = k3_assignments[method]
        print(
            f"\nk=3, preprocessing={method}: clusters (rows) vs true sentiment (columns)"
        )
        print(f"{'':>10}" + "".join(f"{name:>10}" for name in CLASS_NAMES))
        for cluster in range(3):
            counts = np.bincount(true_labels[assignments == cluster], minlength=3)
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
