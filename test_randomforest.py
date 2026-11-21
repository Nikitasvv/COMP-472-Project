"""Random Forest validation experiment using the original random_forest.py implementation.

Run directly: python -u test_randomforest.py [--n-trees 300] [--max-depth 12]
Or run via:   python -u run_all.py --only rf --rf-n-trees 10
The 80/20 validation split is provided by experiment_helpers.load_data().
"""
import argparse
import time

from experiment_helpers import (
    TASKS, setup, load_data, load_state, save_state, clear_model_results, record,
)
from utils import SEED, accuracy, fit_predict

RF_PARAMS = {
    "n_trees": 300,        # Preserve original random_forest.py CLI default.
    "max_depth": None,    # None means unrestricted tree depth.
    "min_samples_split": 2,
    "seed": SEED,
    "scaling": "none",   # Trees do not need standardization.
}


def run_rf(data, state, params=None):
    """Validate the original forest on the same data split as the other models."""
    params = {**RF_PARAMS, **(params or {})}
    clear_model_results(state, "rf")
    print("\n========== RANDOM FOREST EXPERIMENT ==========", flush=True)
    print(f"Parameters: {params}", flush=True)
    for task in TASKS:
        dataset = data[task]
        print(f"\n--- {task.upper()} Random Forest ---", flush=True)
        start = time.perf_counter()
        val_pred, model = fit_predict(
            "rf", params,
            dataset["train_features"], dataset["train_labels"],
            dataset["val_features"], dataset["n_classes"],
        )
        val_accuracy = accuracy(dataset["val_labels"], val_pred)
        record(state["best"], task, "rf", params.copy(), val_accuracy, val_pred)
        print(
            f"Random Forest {task} validation accuracy: "
            f"{val_accuracy * 100:.2f}% ({len(model.trees)} trees) "
            f"| {time.perf_counter() - start:.1f}s", flush=True,
        )
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
    state = load_state()  # Keep other already trained model results.
    run_rf(data, state, {
        "n_trees": args.n_trees,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "seed": args.seed,
    })
    save_state(state)


if __name__ == "__main__":
    main()
