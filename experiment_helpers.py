# Shared pieces used by the k-NN, k-Means and error-analysis scripts:
# data loading, remembering the best result of each model, and saving it.
import json
import os
import pickle
import random
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # save graphs to files instead of opening windows

from utils import SEED, load_train, train_val_split

DATA_DIR = "data" if os.path.isdir("data") else "."
TASKS = ("image", "sentiment")
MODEL_NAMES = {"knn": "k-NN"}

RESULTS_FILE = "results_knn.json"  # best accuracy + settings of my model (k-NN)
STATE_FILE = "experiment_state.pkl"  # pickle of all results so far, so each experiment can be run on its own and the results combined later

# Store the library versions and experiment settings for reproducibility
ENVIRONMENT = dict(
    python=sys.version.split()[0],  # Python version
    numpy=np.__version__,  # NumPy version
    pandas=pd.__version__,  # Pandas version
    matplotlib=matplotlib.__version__,  # Matplotlib version
    seed=SEED,  # Random seed used in experiments
    train_val_split="80% / 20%",  # Training and validation split ratio
)


def setup():
    """Set up the environment before running the experiments."""

    # Set random seeds to get consistent results across runs
    random.seed(SEED)
    np.random.seed(SEED)

    # Create the figures folder if it does not already exist
    os.makedirs("figures", exist_ok=True)

    # Display the library versions and environment information
    print("Environment:", ENVIRONMENT)


def load_data():
    """Load both datasets and split each into 80% training / 20% validation (same seed every time)."""

    # Store the training and validation data for each task
    data = {}

    # Load and prepare each dataset
    for task in TASKS:
        # Load the features and labels from the training CSV file
        features, labels, _ = load_train(os.path.join(DATA_DIR, f"{task}_train.csv"))

        # Split the dataset into training and validation sets
        train_features, val_features, train_labels, val_labels = train_val_split(
            features, labels
        )

        # Save the split data and number of classes for this task
        data[task] = {
            "train_features": train_features,
            "val_features": val_features,
            "train_labels": train_labels,
            "val_labels": val_labels,
            "n_classes": int(labels.max()) + 1,
        }

        # Display the size of the training and validation sets
        print(
            f"{task:9s} train {train_features.shape}  validation {val_features.shape}"
        )

    # Return the prepared datasets
    return data


# state["best"][task][model] = {"acc": ..., "params": ..., "val_pred": ...} (best setting of each model)
def new_state():
    return {"best": {task: {} for task in TASKS}}


def load_state():
    """Results of experiments run earlier (so each experiment can be run on its own)."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "rb") as file:
            return pickle.load(file)
    return new_state()


def clear_model_results(state, model_key):
    """Forget old results of one model before its experiment runs again."""
    for task in TASKS:
        state["best"][task].pop(model_key, None)


def record(best, task, model_key, params, val_accuracy, predictions):
    """Remember the best result for each model and dataset."""

    # Update the result if the model has no saved result or gets a better accuracy
    if model_key not in best[task] or val_accuracy > best[task][model_key]["acc"]:
        best[task][model_key] = {
            "acc": val_accuracy,  # Best validation accuracy
            "params": params,  # Parameters used for this result
            "val_pred": predictions,  # Predictions from the best model
        }


def save_state(state):
    """Save progress (pickle) and the best accuracies / settings of my models (json)."""

    # Save the full experiment state so progress can be restored later
    with open(STATE_FILE, "wb") as file:
        pickle.dump(state, file)

    # Collect the best accuracy and parameters for each model in each task
    saved = {
        task: {
            key: {"acc": result["acc"], "params": result["params"]}
            for key, result in state["best"][task].items()
            if key in MODEL_NAMES  # Only include the models we are using
        }
        for task in TASKS
    }

    # Include the environment information in the saved results
    saved["environment"] = ENVIRONMENT

    # Save the results as a readable JSON file
    with open(RESULTS_FILE, "w") as file:
        json.dump(saved, file, indent=2)

    # Confirm that the results were saved
    print(f"\nSaved {RESULTS_FILE}")
