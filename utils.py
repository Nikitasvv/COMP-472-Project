
# Utilities needed for KNN: loading data, splitting, scaling, and accuracy.
import numpy as np
import pandas as pd

SEED = 42


def load_train(path):
    """Load the features and labels from a CSV file."""
    data = pd.read_csv(path)
    feature_columns = [col for col in data.columns if col.startswith("feature_")]
    features = data[feature_columns].values.astype(np.float32)
    labels = data["label"].values.astype(int)
    return features, labels, feature_columns


def train_val_split(features, labels, val_fraction=0.20, seed=SEED):
    """Shuffle and split into 80% training and 20% validation."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(features))
    n_val = int(len(features) * val_fraction)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    return (features[train_idx], features[val_idx],
            labels[train_idx], labels[val_idx])


def prep(params, train_features, eval_features):
    """Scale features using statistics from training data only."""
    scaling = params.get("scaling", "none")

    if scaling == "none":
        return train_features, eval_features

    if scaling == "standardize":
        mean = train_features.mean(axis=0)
        std = train_features.std(axis=0) + 1e-8

        train = (train_features - mean) / std
        val = (eval_features - mean) / std

        return train.astype(np.float32), val.astype(np.float32)

    if scaling == "sqrt_std":
        scale = np.sqrt(train_features.std(axis=0) + 1e-8)

        train = train_features / scale
        val = eval_features / scale

        return train.astype(np.float32), val.astype(np.float32)

    raise ValueError(f"Unknown scaling method: {scaling}")


def accuracy(true_labels, predicted_labels):
    """Return the proportion of correct predictions."""
    return float(np.mean(np.asarray(true_labels) == np.asarray(predicted_labels)))
