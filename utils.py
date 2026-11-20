# Helpers: reading a CSV, train/validation split, feature scaling, metrics, model building,and fit_predict
import numpy as np
import pandas as pd

# k-NN building blocks from knn.py
# normalize_rows: scales each row to unit length (for cosine distance)
# find_neighbors: finds the k closest training rows for each evaluation row

# vote: turns neighbours' labels into a predicted class
from knn import normalize_rows, find_neighbors, vote

SEED = 123


def load_train(path):
    """Read a training CSV. Returns (features, labels, feature_column_names)"""
    data = pd.read_csv(path)  # load the whole CSV into a DataFrame
    # Every column whose name starts with "feature_" is an input feature
    feature_columns = [
        column for column in data.columns if column.startswith("feature_")
    ]
    features = data[feature_columns].values.astype(np.float32)
    # The "label" column is the target class, stored as integers
    labels = data["label"].values.astype(int)
    return features, labels, feature_columns


def train_val_split(features, labels, val_fraction=0.20, seed=SEED):
    """Shuffle with a fixed seed, then split into training (80%) and validation (20%) sets"""
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(features))
    number_validation = int(len(features) * val_fraction)
    validation_indices = shuffled_indices[:number_validation]
    training_indices = shuffled_indices[number_validation:]
    return (
        features[training_indices],
        features[validation_indices],
        labels[training_indices],
        labels[validation_indices],
    )


def fit_standardizer(train_features):
    """Learn mean and std from the TRAINING data only; return a function that standardizes any array"""
    mean = train_features.mean(axis=0)  # per column mean
    std = (
        train_features.std(axis=0) + 1e-8
    )  # per column std, add small value to avoid division by zero

    def standardize(features):
        """Subtract the training mean and divide by training std, using the values learned from training data"""
        return ((features - mean) / std).astype(np.float32)

    return standardize  # remembers the mean and std from the training data, so we can use it on validation/test data


def prep(params, train_features, eval_features):
    """Scale both arrays, learning the scaling from the training data only.
    params["scaling"] is "none", "standardize" or "sqrt_std"."""
    scaling = params.get("scaling", "none")  # default to no scaling if not specified

    if scaling == "standardize":
        # Standardize both sets using the training data's mean and standard deviation
        standardize = fit_standardizer(train_features)
        return standardize(train_features), standardize(eval_features)

    if scaling == "sqrt_std":
        # Divide by sqrt(training std) without subtracting the mean.
        scale = np.sqrt(train_features.std(axis=0) + 1e-8)
        return (train_features / scale).astype(np.float32), (
            eval_features / scale
        ).astype(np.float32)

    if scaling == "none":
        return train_features, eval_features  # untouched, no scaling

    raise ValueError(f"Unknown scaling method: {scaling}")


def accuracy(true_labels, predicted_labels):
    # Fraction of samples where the predicted label matches the true label.
    return float(np.mean(np.asarray(true_labels) == np.asarray(predicted_labels)))


def fit_predict(name, params, train_features, train_labels, eval_features, n_classes):
    """Scale, train one model and predict. Used by make_submission.py
    Returns (predictions, model); k-NN has no trained model object, so it returns None there.
    """
    scaled_train, scaled_eval = prep(params, train_features, eval_features)

    if name == "knn":
        # Normalize the features if we are using cosine distance
        if params["metric"] == "cosine":
            scaled_train, scaled_eval = normalize_rows(scaled_train), normalize_rows(
                scaled_eval
            )
        # Find the k closest training samples for each evaluation sample
        indices, distances = find_neighbors(scaled_train, scaled_eval, params["k"])

        # Predict the class using neighbour votes
        # Weighted voting gives closer neighbours more influence
        predictions = vote(
            train_labels, indices, distances, params["k"], params["weighted"], n_classes
        )

        # Return the predictions
        return predictions, None

    raise ValueError(f"fit_predict does not know the model '{name}' yet")
