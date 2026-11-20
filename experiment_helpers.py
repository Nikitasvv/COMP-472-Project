
# Helpers for k-NN experiments:
# Load datasets, split data, track best results, and save them.

import json
import os
import pickle
import random
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from utils import SEED, load_train, train_val_split


DATA_DIR = "data" if os.path.isdir("data") else "."
TASKS = ("image", "sentiment")

RESULTS_FILE = "results_knn_dt.json"
STATE_FILE = "experiment_state.pkl"

ENVIRONMENT = {
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "matplotlib": matplotlib.__version__,
    "seed": SEED,
    "train_val_split": "80% / 20%"
}


def setup():
    """Set random seeds, create the plots folder, and print versions."""
    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs("plots", exist_ok=True)
    print("Environment:", ENVIRONMENT)


def load_data():
    """Load image and sentiment data and split them 80/20."""
    data = {}

    for task in TASKS:
        path = os.path.join(DATA_DIR, f"{task}_train.csv")
        features, labels, _ = load_train(path)

        train_X, val_X, train_y, val_y = train_val_split(
            features, labels
        )

        data[task] = {
            "train_features": train_X,
            "val_features": val_X,
            "train_labels": train_y,
            "val_labels": val_y,
            "n_classes": int(labels.max()) + 1
        }

        print(
            f"{task} train {train_X.shape} "
            f"validation {val_X.shape}"
        )

    return data


def new_state():
    """Create empty results for both datasets."""
    return {"best": {task: {} for task in TASKS}}


def load_state():
    """Load previous results, or start fresh."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "rb") as file:
            return pickle.load(file)

    return new_state()


def clear_model_results(state, model_key):
    """Clear previous results before running KNN again."""
    for task in TASKS:
        state["best"][task].pop(model_key, None)


def record(best, task, model_key, params, val_accuracy, predictions):
    """Keep the best accuracy and settings."""
    if model_key not in best[task] or val_accuracy > best[task][model_key]["acc"]:
        best[task][model_key] = {
            "acc": val_accuracy,
            "params": params,
            "val_pred": predictions
        }


def save_state(state):
    """Save results to pickle and JSON."""
    with open(STATE_FILE, "wb") as file:
        pickle.dump(state, file)

    saved = {}

    for task in TASKS:
        saved[task] = {}

        for model, result in state["best"][task].items():
            saved[task][model] = {
                "acc": result["acc"],
                "params": result["params"]
            }

    saved["environment"] = ENVIRONMENT

    with open(RESULTS_FILE, "w") as file:
        json.dump(saved, file, indent=2)

    print(f"\nSaved {RESULTS_FILE}")
