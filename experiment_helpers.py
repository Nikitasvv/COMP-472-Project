# Shared pieces used by the experiment scripts: data loading, remembering the
# best result of each model, and saving those results to disk.
import json
import os
import pickle
import random
import sys
from importlib.metadata import PackageNotFoundError, version

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # write graphs to files instead of opening windows

from pathlib import Path
from utils import SEED, load_train, train_val_split

PROJECT_DIR = Path(__file__).resolve().parent
# Use data/ if it exists, otherwise expect the CSVs next to the scripts
DATA_DIR = PROJECT_DIR / "data" if (PROJECT_DIR / "data").is_dir() else PROJECT_DIR
TASKS = ("image", "sentiment")
MODEL_NAMES = {
    "knn": "k-NN",
    "nn": "Neural Network",
    "rf": "Random Forest",
    "dt": "Decision Tree",
}

RESULTS_FILE = PROJECT_DIR / "results.json"  # best accuracy and parameters of every model
KNN_RESULTS_FILE = PROJECT_DIR / "results_knn.json"  # same, k-NN only (read by make_submission.py as a fallback)
# Pickle with the full results including predictions, so each experiment can be
# run on its own and combined with the others later
STATE_FILE = PROJECT_DIR / "experiment_state.pkl"

def _package_version(name):
    """Installed version of a package, read without importing it."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


# Library versions and settings, saved alongside the results for reproducibility
ENVIRONMENT = dict(
    python=sys.version.split()[0],
    numpy=np.__version__,
    pandas=pd.__version__,
    matplotlib=matplotlib.__version__,
    torch=_package_version("torch"),
    scikit_learn=_package_version("scikit-learn"),  # only used for evaluation metrics
    seed=SEED,
    train_val_split="80% / 20%",
)


def setup():
    """Seed the random generators and create the output folder for graphs."""
    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs(PROJECT_DIR / "figures", exist_ok=True)  # test_knn.py and test_kmeans.py save graphs here
    print("Environment:", ENVIRONMENT, flush=True)


def load_data():
    """Load both datasets and split each into 80% training / 20% validation (same seed every time)."""
    data = {}

    for task in TASKS:
        csv_path = DATA_DIR / f"{task}_train.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Missing training file: {csv_path}. Put both training CSVs "
                "in data/ or next to the Python scripts."
            )
        features, labels, _ = load_train(csv_path)
        train_features, val_features, train_labels, val_labels = train_val_split(
            features, labels
        )

        data[task] = {
            "train_features": train_features,
            "val_features": val_features,
            "train_labels": train_labels,
            "val_labels": val_labels,
            "n_classes": int(labels.max()) + 1,  # labels run from 0 to n_classes - 1
        }
        print(
            f"{task:9s} train {train_features.shape}  validation {val_features.shape}", flush=True
        )

    return data


# state["best"][task][model] = {"acc": ..., "params": ..., "val_pred": ...}
# holds the best setting found so far for each model on each task.
def new_state():
    return {"best": {task: {} for task in TASKS}}


def load_state():
    """Load the results of earlier experiments, or start empty if there are none."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "rb") as file:
            return pickle.load(file)
    return new_state()


def clear_model_results(state, model_key):
    """Forget the old results of one model before its experiment runs again."""
    for task in TASKS:
        state["best"][task].pop(model_key, None)


def record(best, task, model_key, params, val_accuracy, predictions):
    """Keep this result only if the model has none yet or this one is more accurate."""
    if model_key not in best[task] or val_accuracy > best[task][model_key]["acc"]:
        best[task][model_key] = {
            "acc": val_accuracy,
            "params": params,
            "val_pred": predictions,  # kept so error_analysis.py does not have to retrain
        }


def save_state(state):
    """Write the full state to the pickle and the accuracies/parameters to JSON."""
    with open(STATE_FILE, "wb") as file:
        pickle.dump(state, file)

    # The JSON leaves out predictions so it stays small and readable
    saved = {
        task: {
            key: {"acc": result["acc"], "params": result["params"]}
            for key, result in state["best"].get(task, {}).items()
            if key in MODEL_NAMES
        }
        for task in TASKS
    }
    saved["environment"] = ENVIRONMENT

    with open(RESULTS_FILE, "w") as file:
        json.dump(saved, file, indent=2)

    knn_only = {
        task: ({"knn": saved[task]["knn"]} if "knn" in saved[task] else {})
        for task in TASKS
    }
    knn_only["environment"] = ENVIRONMENT
    with open(KNN_RESULTS_FILE, "w") as file:
        json.dump(knn_only, file, indent=2)

    print(f"\nSaved {RESULTS_FILE.name} and {KNN_RESULTS_FILE.name}", flush=True)
