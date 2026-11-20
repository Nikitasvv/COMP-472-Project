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

# Different numbers of neighbors to test for K-NN
KS = (1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)

# Test different scaling methods with Euclidean and cosine distance
SETTINGS = [
    ("none", "euclidean"),
    ("none", "cosine"),
    ("standardize", "euclidean"),
    ("standardize", "cosine"),
]


def run_knn(data, state):
    """Test different K-NN settings and save the best results."""

    # Get saved results, clear old K-NN results, and prepare to store graph data
    best = state["best"]
    clear_model_results(state, "knn")
    curves = {}

    # Run K-NN experiments on both datasets
    for task in TASKS:
        dataset = data[task]
        print(f"\n=== {task} ===")

        # Test an additional scaling method for image data
        settings = SETTINGS.copy()
        if task == "image":
            settings.append(("sqrt_std", "cosine"))

        # Try each combination of scaling and distance metric
        for scaling, metric in settings:

            # Apply the selected scaling to training and validation features
            train, val = prep(
                {"scaling": scaling}, dataset["train_features"], dataset["val_features"]
            )

            # Normalize rows when using cosine distance
            if metric == "cosine":
                train = normalize_rows(train)
                val = normalize_rows(val)

            # Find the 100 nearest neighbors for each validation sample
            # Reuse these neighbors instead of searching again for every k
            indices, distances = find_neighbors(train, val, max(KS))

            # Compare regular voting and distance-weighted voting
            for weighted in (False, True):
                scores = []

                # Test each k value to see how the number of neighbors affects accuracy
                for k in KS:

                    # Predict labels using the selected neighbors and voting method
                    predictions = vote(
                        dataset["train_labels"],
                        indices,
                        distances,
                        k,
                        weighted,
                        dataset["n_classes"],
                    )

                    # Calculate validation accuracy and store it for the graph
                    acc = accuracy(dataset["val_labels"], predictions)
                    scores.append(acc)

                    # Store the settings used for this experiment
                    params = {
                        "k": k,
                        "metric": metric,
                        "weighted": weighted,
                        "scaling": scaling,
                    }

                    # Keep this result if it has the best accuracy so far
                    record(best, task, "knn", params, acc, predictions)

                # Save all accuracy scores for this combination of settings
                curves[(task, scaling, metric, weighted)] = scores

                # Print the accuracy for each k value
                print(
                    f"{scaling} {metric} weighted={weighted}: "
                    + "  ".join(f"k={k}: {a:.3f}" for k, a in zip(KS, scores))
                )

        # Display the best K-NN settings and accuracy for this dataset
        print("Best k-NN:", best[task]["knn"]["params"])
        print("Accuracy:", best[task]["knn"]["acc"])

    # Create two graphs to compare K-NN accuracy on both datasets
    fig, graphs = plt.subplots(1, 2, figsize=(10, 3.5))

    for graph, task in zip(graphs, TASKS):

        # Compare weighted and unweighted voting using standardized features
        for weighted in (False, True):
            graph.plot(
                KS,
                curves[(task, "standardize", "cosine", weighted)],
                marker="o",
                label="weighted" if weighted else "unweighted",
            )

        # Add an extra curve for image data using sqrt-std scaling
        if task == "image":
            graph.plot(
                KS,
                curves[(task, "sqrt_std", "cosine", True)],
                marker="o",
                linestyle="--",
                label="sqrt-std scaling (weighted)",
            )

        # Add labels to make the graphs easier to understand
        graph.set_title(f"k-NN: {task}")
        graph.set_xlabel("k")
        graph.set_ylabel("validation accuracy")
        graph.legend()

    # Adjust spacing, save the graphs, and close the figure
    plt.tight_layout()
    plt.savefig("plots/knn_k.png", dpi=120)
    plt.close()


if __name__ == "__main__":

    # Set up the environment and load the datasets and saved experiment state
    setup()
    data = load_data()
    state = load_state()

    # Measure how long the K-NN experiments take
    start = time.time()
    run_knn(data, state)

    # Display the total runtime and save the best results
    print(f"\nk-NN finished in {time.time() - start:.1f}s")
    save_state(state)
