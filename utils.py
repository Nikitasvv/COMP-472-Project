# Shared helpers: CSV loading, train/validation split, feature scaling, metrics and fit_predict.
import numpy as np
import pandas as pd

from decision_tree import ScratchDecisionTree
from knn import normalize_rows, find_neighbors, vote
from random_forest import RandomForestModel

SEED = 123


def load_train(path):
    """Read a training CSV. Returns (features, labels, feature_column_names)."""
    data = pd.read_csv(path)
    # Every column named "feature_*" is an input; "label" is the target class
    feature_columns = [c for c in data.columns if c.startswith("feature_")]
    features = data[feature_columns].values.astype(np.float32)
    labels = data["label"].values.astype(int)
    return features, labels, feature_columns


def train_val_split(features, labels, val_fraction=0.20, seed=SEED):
    """Shuffle with a fixed seed, then split into training (80%) and validation (20%) sets."""
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
    """Learn per-column mean and std from the TRAINING data only.
    Returns a function that applies those same values to any array, so validation
    and test data are scaled without leaking information from them."""
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0) + 1e-8  # avoids dividing by zero for constant columns

    def standardize(features):
        return ((features - mean) / std).astype(np.float32)

    return standardize


def prep(params, train_features, eval_features):
    """Scale both arrays using statistics from the training data only.
    params["scaling"] is "none" or "standardize"."""
    scaling = params.get("scaling", "none")

    if scaling == "standardize":
        standardize = fit_standardizer(train_features)
        return standardize(train_features), standardize(eval_features)

    if scaling == "none":
        return train_features, eval_features

    raise ValueError(f"Unknown scaling method: {scaling}")


def accuracy(true_labels, predicted_labels):
    """Fraction of samples where the predicted label matches the true label."""
    return float(np.mean(np.asarray(true_labels) == np.asarray(predicted_labels)))


def fit_predict(name, params, train_features, train_labels, eval_features, n_classes):
    """Scale, train one model and predict. Used by the experiments and by make_submission.py.

    Returns (predictions, model). The model is None for k-NN (it has nothing to train).
    """
    scaled_train, scaled_eval = prep(params, train_features, eval_features)

    if name == "knn":
        # After row normalization, Euclidean distance ranks neighbours exactly like cosine distance
        if params["metric"] == "cosine":
            scaled_train = normalize_rows(scaled_train)
            scaled_eval = normalize_rows(scaled_eval)
        indices, distances = find_neighbors(scaled_train, scaled_eval, params["k"])
        predictions = vote(
            train_labels, indices, distances, params["k"], params["weighted"], n_classes
        )
        return predictions, None

    if name == "nn":
        # Imported here because neural_network.py itself imports from utils.py
        from neural_network import train_neural_network, predict

        model = train_neural_network(params, scaled_train, train_labels)
        predictions = predict(model, scaled_eval, model.classes_, model.batch_size_)
        return predictions.astype(int), model

    if name in ("rf", "dt"):
        # Gini impurity and bincount need labels encoded as 0..n_classes-1;
        # the predictions are decoded back to the original labels afterwards.
        classes = np.unique(train_labels)
        encoded_labels = np.searchsorted(classes, train_labels).astype(int)

        if name == "rf":
            model = RandomForestModel(
                input_size=scaled_train.shape[1],
                output_size=len(classes),
                n_trees=int(params.get("n_trees", 300)),
                max_depth=params.get("max_depth"),
                min_samples_split=int(params.get("min_samples_split", 2)),
                seed=int(params.get("seed", SEED)),
            )
        else:
            model = ScratchDecisionTree(
                max_depth=params.get("max_depth"),
                min_samples_split=int(params.get("min_samples_split", 2)),
                max_features=params.get("max_features"),
                seed=int(params.get("seed", SEED)),
            )

        model.fit(scaled_train, encoded_labels)
        predictions = classes[model.predict(scaled_eval)]
        return predictions.astype(int), model

    raise ValueError(f"fit_predict does not know the model '{name}'")
