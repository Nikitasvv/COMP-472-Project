# k-NN experiment: try several k values, scalings, distance metrics and voting styles.
# Run on its own: python test_knn.py
import time
import matplotlib.pyplot as plt

from experiment_helpers import (
    TASKS,
    setup,
    load_data,
    load_state,
    save_state,
    clear_model_results,
    record,
)
from knn import normalize_rows, find_neighbors, vote
from utils import prep, accuracy

KS = (1, 3, 5, 10, 20, 50, 100)

# (scaling, distance metric) combinations to compare
SETTINGS = [
    ("none", "euclidean"),
    ("none", "cosine"),
    ("standardize", "euclidean"),
    ("standardize", "cosine"),
]


def run_knn(data, state):
    """Test every setting on both datasets and keep the most accurate one of each."""
    best = state["best"]
    clear_model_results(state, "knn")
    curves = {}  # (task, scaling, metric, weighted) -> accuracy for each k, used for the graph

    for task in TASKS:
        dataset = data[task]
        print(f"\n=== {task} ===")

        for scaling, metric in SETTINGS:
            train, val = prep(
                {"scaling": scaling}, dataset["train_features"], dataset["val_features"]
            )

            # Cosine distance is Euclidean distance on rows scaled to length 1
            if metric == "cosine":
                train = normalize_rows(train)
                val = normalize_rows(val)

            # Search once for the largest k, then reuse those neighbours for every smaller k
            indices, distances = find_neighbors(train, val, max(KS))

            for weighted in (False, True):
                scores = []

                for k in KS:
                    predictions = vote(
                        dataset["train_labels"],
                        indices,
                        distances,
                        k,
                        weighted,
                        dataset["n_classes"],
                    )
                    acc = accuracy(dataset["val_labels"], predictions)
                    scores.append(acc)

                    params = {
                        "k": k,
                        "metric": metric,
                        "weighted": weighted,
                        "scaling": scaling,
                    }
                    record(best, task, "knn", params, acc, predictions)

                curves[(task, scaling, metric, weighted)] = scores
                print(
                    f"{scaling} {metric} weighted={weighted}: "
                    + "  ".join(f"k={k}: {a:.3f}" for k, a in zip(KS, scores))
                )

        print("Best k-NN:", best[task]["knn"]["params"])
        print("Accuracy:", best[task]["knn"]["acc"])

    # One graph per dataset: weighted vs unweighted voting for the standardized + cosine setting
    fig, graphs = plt.subplots(1, 2, figsize=(10, 3.5))

    for graph, task in zip(graphs, TASKS):
        for weighted in (False, True):
            graph.plot(
                KS,
                curves[(task, "standardize", "cosine", weighted)],
                marker="o",
                label="weighted" if weighted else "unweighted",
            )
        graph.set_title(f"k-NN: {task}")
        graph.set_xlabel("k")
        graph.set_ylabel("validation accuracy")
        graph.legend()

    plt.tight_layout()
    plt.savefig("figures/knn_k.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    setup()
    data = load_data()
    state = load_state()  # keep results of the other models that were already trained

    start = time.time()
    run_knn(data, state)
    print(f"\nk-NN finished in {time.time() - start:.1f}s")
    save_state(state)
