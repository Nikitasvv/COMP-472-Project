"""Random Forest experiment: how does the number of trees affect accuracy?

Run directly: python -u test_randomforest.py [--n-trees 100] [--max-depth 12]
Or via:       python -u run_all.py --only rf --rf-n-trees 10
The 80/20 validation split is provided by experiment_helpers.load_data().

One forest with the largest tree count is trained per task. Its first 10, 50, ... trees
are exactly the forest you would get by training that many trees with the same seed,
so every smaller forest is scored without retraining.
"""
import argparse
import time

import numpy as np

from experiment_helpers import (
    TASKS, PROJECT_DIR, setup, load_data, load_state, save_state, clear_model_results, record,
)
import matplotlib.pyplot as plt  # imported after experiment_helpers, which switches matplotlib to file output
from utils import SEED, accuracy, fit_predict, prep

RF_PARAMS = {
    "n_trees": 100,       # size of the largest forest that is trained
    "max_depth": None,    # None = grow until the leaves are pure
    "min_samples_split": 2,
    "seed": SEED,
    "scaling": "none",    # tree splits do not depend on feature scale
}

RF_TREE_COUNTS = (10, 50, 100)  # forest sizes that are scored (the largest is always included)


def run_rf(data, state, params=None):
    """Train the forest once and score its first k trees for every k in RF_TREE_COUNTS."""
    params = {**RF_PARAMS, **(params or {})}
    n_trees = params["n_trees"]
    counts = sorted({c for c in RF_TREE_COUNTS if c < n_trees} | {n_trees})
    clear_model_results(state, "rf")
    print("\n========== RANDOM FOREST EXPERIMENT ==========", flush=True)
    print(f"Parameters: {params}", flush=True)
    print(f"Forest sizes tested: {counts}", flush=True)

    curves = {}  # task -> validation accuracy for each forest size, used for the graph
    for task in TASKS:
        dataset = data[task]
        print(f"\n--- {task.upper()} Random Forest ---", flush=True)
        start = time.perf_counter()
        _, model = fit_predict(
            "rf", params,
            dataset["train_features"], dataset["train_labels"],
            dataset["val_features"], dataset["n_classes"],
        )
        print(f"Forest trained in {time.perf_counter() - start:.1f}s", flush=True)

        classes = np.unique(dataset["train_labels"])
        _, val_x = prep(params, dataset["train_features"], dataset["val_features"])
        rows = np.arange(len(val_x))

        # Add the trees' votes one at a time and score the forest at each requested size
        votes = np.zeros((len(val_x), len(classes)), dtype=int)
        scores = []
        for t, tree in enumerate(model.trees, start=1):
            votes[rows, tree.predict(val_x)] += 1
            if t in counts:
                val_pred = classes[votes.argmax(axis=1)]
                val_acc = accuracy(dataset["val_labels"], val_pred)
                record(state["best"], task, "rf", {**params, "n_trees": t}, val_acc, val_pred)
                scores.append(val_acc)
                print(f"Random Forest {task} with {t:>3} trees: "
                      f"validation accuracy {val_acc * 100:.2f}%", flush=True)
        curves[task] = scores

        # The forest is meant to beat a single tree, so show the best tree next to it
        dt = state["best"][task].get("dt")
        if dt is not None:
            print(f"Best single decision tree ({task}): {dt['acc'] * 100:.2f}%", flush=True)

    fig, graphs = plt.subplots(1, 2, figsize=(10, 3.5))
    for graph, task in zip(graphs, TASKS):
        graph.plot(counts, curves[task], marker="o", label="random forest")
        dt = state["best"][task].get("dt")
        if dt is not None:
            graph.axhline(dt["acc"], linestyle="--", color="gray", label="best decision tree")
        graph.set_title(f"Random forest: {task}")
        graph.set_xlabel("number of trees")
        graph.set_ylabel("validation accuracy")
        graph.legend()
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "figures" / "rf_trees.png", dpi=120)
    plt.close()
    print("Saved figures/rf_trees.png", flush=True)
    print("Random Forest results will be saved to results.json and experiment_state.pkl", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trees", type=int, default=RF_PARAMS["n_trees"])
    parser.add_argument("--max-depth", type=int, default=RF_PARAMS["max_depth"])
    parser.add_argument("--min-samples-split", type=int, default=RF_PARAMS["min_samples_split"])
    parser.add_argument("--seed", type=int, default=RF_PARAMS["seed"])
    args = parser.parse_args(argv)
    setup()
    data = load_data()
    state = load_state()  # keep results of the other models that were already trained
    run_rf(data, state, {
        "n_trees": args.n_trees,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "seed": args.seed,
    })
    save_state(state)


if __name__ == "__main__":
    main()
