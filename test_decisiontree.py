"""Decision tree experiment: how does the maximum depth affect accuracy?

Run directly: python -u test_decisiontree.py [--max-depth 12]
Or via:       python -u run_all.py --only dt --dt-max-depth 12
The 80/20 validation split is provided by experiment_helpers.load_data().

One fully grown tree is fitted per task and then cut at each depth in DT_DEPTHS. That gives
exactly the same trees as refitting with every max_depth, at the cost of a single fit.
"""
import argparse
import time

import numpy as np

from experiment_helpers import (
    TASKS, PROJECT_DIR, setup, load_data, load_state, save_state, clear_model_results, record,
)
import matplotlib.pyplot as plt  # imported after experiment_helpers, which switches matplotlib to file output
from utils import SEED, accuracy, fit_predict, prep

DT_PARAMS = {
    "max_depth": None,        # set by the depth sweep below
    "min_samples_split": 2,
    "max_features": None,     # None = every feature is considered at every split
    "seed": SEED,
    "scaling": "none",        # tree splits do not depend on feature scale
}

DT_DEPTHS = (5, 10, 20, None)  # None = no depth limit


def run_dt(data, state, params=None, depths=DT_DEPTHS):
    """Score the tree at every depth on the training and validation splits and keep the best depth."""
    params = {**DT_PARAMS, **(params or {})}
    clear_model_results(state, "dt")
    # Growing the deepest tree needed covers every shallower depth too
    fit_depth = None if None in depths else max(depths)
    print("\n========== DECISION TREE EXPERIMENT ==========", flush=True)
    print(f"Parameters: {params}", flush=True)
    print(f"Depths tested: {list(depths)} (one tree is fitted, then cut at each depth)", flush=True)

    curves = {}  # task -> (depth labels, train accuracies, validation accuracies) for the graph
    for task in TASKS:
        dataset = data[task]
        print(f"\n--- {task.upper()} Decision Tree ---", flush=True)
        start = time.perf_counter()
        _, model = fit_predict(
            "dt", {**params, "max_depth": fit_depth},
            dataset["train_features"], dataset["train_labels"],
            dataset["val_features"], dataset["n_classes"],
        )
        print(f"Tree fitted in {time.perf_counter() - start:.1f}s "
              f"(grew to depth {model.depth()})", flush=True)

        classes = np.unique(dataset["train_labels"])
        train_x, val_x = prep(params, dataset["train_features"], dataset["val_features"])

        labels, train_accs, val_accs = [], [], []
        print(f"{'max_depth':>10} {'train acc':>10} {'val acc':>10}", flush=True)
        for depth in depths:
            train_pred = classes[model.predict(train_x, max_depth=depth)]
            val_pred = classes[model.predict(val_x, max_depth=depth)]
            train_acc = accuracy(dataset["train_labels"], train_pred)
            val_acc = accuracy(dataset["val_labels"], val_pred)
            record(state["best"], task, "dt", {**params, "max_depth": depth}, val_acc, val_pred)

            labels.append("None" if depth is None else str(depth))
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            print(f"{labels[-1]:>10} {train_acc:10.3f} {val_acc:10.3f}", flush=True)
        curves[task] = (labels, train_accs, val_accs)

        print(f"Best decision tree ({task}): {state['best'][task]['dt']['params']['max_depth']} "
              f"-> validation accuracy {state['best'][task]['dt']['acc'] * 100:.2f}%", flush=True)

    # A widening gap between the two lines means the tree is memorizing the training data
    fig, graphs = plt.subplots(1, 2, figsize=(10, 3.5))
    for graph, task in zip(graphs, TASKS):
        labels, train_accs, val_accs = curves[task]
        graph.plot(labels, train_accs, marker="o", label="train")
        graph.plot(labels, val_accs, marker="o", label="validation")
        graph.set_title(f"Decision tree: {task}")
        graph.set_xlabel("max depth")
        graph.set_ylabel("accuracy")
        graph.legend()
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "figures" / "dt_depth.png", dpi=120)
    plt.close()
    print("Saved figures/dt_depth.png", flush=True)
    print("Decision Tree results will be saved to results.json and experiment_state.pkl", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-depth", type=int, default=None,
                        help="Test only this depth instead of the default sweep.")
    parser.add_argument("--min-samples-split", type=int, default=DT_PARAMS["min_samples_split"])
    parser.add_argument("--seed", type=int, default=DT_PARAMS["seed"])
    args = parser.parse_args(argv)
    setup()
    data = load_data()
    state = load_state()  # keep results of the other models that were already trained
    run_dt(
        data, state,
        {"min_samples_split": args.min_samples_split, "seed": args.seed},
        depths=DT_DEPTHS if args.max_depth is None else (args.max_depth,),
    )
    save_state(state)


if __name__ == "__main__":
    main()
