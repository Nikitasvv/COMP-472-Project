
import time
import matplotlib.pyplot as plt

from experiment_helpers import (
    TASKS, setup, load_data, load_state, save_state,
    clear_model_results, record
)
from knn import normalize_rows, find_neighbors, vote
from utils import prep, accuracy


KS = (1, 3, 5, 10, 20)

# Scaling and distance combinations to test
SETTINGS = [
    ("none", "euclidean"),
    ("none", "cosine"),
    ("standardize", "euclidean"),
    ("standardize", "cosine")
]


def run_knn(data, state):
    best = state["best"]
    clear_model_results(state, "knn")
    curves = {}

    for task in TASKS:
        dataset = data[task]
        print(f"\n=== {task} ===")

        # Images also test sqrt_std with cosine
        settings = SETTINGS.copy()
        if task == "image":
            settings.append(("sqrt_std", "cosine"))

        for scaling, metric in settings:

            # Scale the training and validation features
            train, val = prep(
                {"scaling": scaling},
                dataset["train_features"],
                dataset["val_features"]
            )

            # Normalize for cosine distance
            if metric == "cosine":
                train = normalize_rows(train)
                val = normalize_rows(val)

            # Find 20 nearest neighbors for every validation row
            indices, distances = find_neighbors(train, val, max(KS))

            # Test weighted and unweighted voting
            for weighted in (False, True):
                scores = []

                # Test each k value
                for k in KS:
                    predictions = vote(
                        dataset["train_labels"],
                        indices, distances, k,
                        weighted, dataset["n_classes"]
                    )

                    acc = accuracy(dataset["val_labels"], predictions)
                    scores.append(acc)

                    params = {
                        "k": k,
                        "metric": metric,
                        "weighted": weighted,
                        "scaling": scaling
                    }

                    record(best, task, "knn", params, acc, predictions)

                curves[(task, scaling, metric, weighted)] = scores

                print(
                    f"{scaling} {metric} weighted={weighted}: "
                    + "  ".join(
                        f"k={k}: {a:.3f}"
                        for k, a in zip(KS, scores)
                    )
                )

        print("Best k-NN:", best[task]["knn"]["params"])
        print("Accuracy:", best[task]["knn"]["acc"])

    # Create accuracy graphs
    fig, graphs = plt.subplots(1, 2, figsize=(10, 3.5))

    for graph, task in zip(graphs, TASKS):
        for weighted in (False, True):
            graph.plot(
                KS,
                curves[(task, "standardize", "cosine", weighted)],
                marker="o",
                label="weighted" if weighted else "unweighted"
            )

        if task == "image":
            graph.plot(
                KS,
                curves[(task, "sqrt_std", "cosine", True)],
                marker="o",
                linestyle="--",
                label="sqrt-std scaling (weighted)"
            )

        graph.set_title(f"k-NN: {task}")
        graph.set_xlabel("k")
        graph.set_ylabel("validation accuracy")
        graph.legend()

    plt.tight_layout()
    plt.savefig("plots/knn_k.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    setup()
    data = load_data()
    state = load_state()

    start = time.time()
    run_knn(data, state)

    print(f"\nk-NN finished in {time.time() - start:.1f}s")
    save_state(state)
