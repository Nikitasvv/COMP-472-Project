import argparse
import csv
import secrets
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# One randomised decision tree (the building block of the forest)
# ----------------------------------------------------------------------

class DecisionTree:
    """A Gini decision tree that considers a random subset of
    attributes at each split - the 'random' part of a Random Forest."""

    def __init__(self, n_classes, max_features, max_depth=None,
                 min_samples_split=2, rng=None):
        self.n_classes = n_classes
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.tree_ = None

    # -- impurity -------------------------------------------------------
    @staticmethod
    def _gini_from_counts(counts):
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts / total
        return 1.0 - float(np.sum(p ** 2))

    # -- best split ------------------------------------------------------
    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        parent_counts = np.bincount(y, minlength=self.n_classes).astype(float)
        parent_gini = self._gini_from_counts(parent_counts)
        if parent_gini == 0.0:
            return None

        # Random subset of attributes for this node.
        if self.max_features < n_features:
            features = self.rng.choice(n_features, size=self.max_features,
                                       replace=False)
        else:
            features = np.arange(n_features)

        best = {"gain": 0.0, "feature": None, "threshold": None}
        for feature in features:
            order = np.argsort(X[:, feature])
            xs = X[order, feature]
            ys = y[order]
            # Cumulative class counts as the threshold moves down the
            # sorted values - every candidate split in one pass.
            onehot = np.zeros((n_samples, self.n_classes))
            onehot[np.arange(n_samples), ys] = 1.0
            cumulative = np.cumsum(onehot, axis=0)
            # A split is only possible where the value actually changes.
            valid = np.where(xs[:-1] != xs[1:])[0]
            if len(valid) == 0:
                continue
            left_counts = cumulative[valid]
            right_counts = parent_counts - left_counts
            n_left = valid + 1.0
            n_right = n_samples - n_left
            p_left = left_counts / n_left[:, None]
            p_right = right_counts / n_right[:, None]
            gini_left = 1.0 - np.sum(p_left ** 2, axis=1)
            gini_right = 1.0 - np.sum(p_right ** 2, axis=1)
            weighted = (n_left * gini_left + n_right * gini_right) / n_samples
            gains = parent_gini - weighted
            k = int(np.argmax(gains))
            if gains[k] > best["gain"]:
                i = valid[k]
                best = {"gain": float(gains[k]), "feature": int(feature),
                        "threshold": float((xs[i] + xs[i + 1]) / 2.0)}
        return best if best["feature"] is not None else None

    # -- recursive build ---------------------------------------------------
    def _build(self, X, y, depth):
        counts = np.bincount(y, minlength=self.n_classes)
        node = {"label": int(np.argmax(counts)), "n_samples": int(len(y))}
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or len(y) < self.min_samples_split
            or np.count_nonzero(counts) == 1
        ):
            return node
        split = self._best_split(X, y)
        if split is None:
            return node
        feature, threshold = split["feature"], split["threshold"]
        left = X[:, feature] <= threshold
        if left.sum() == 0 or left.sum() == len(y):
            return node
        node.update({
            "feature": feature,
            "threshold": threshold,
            "left": self._build(X[left], y[left], depth + 1),
            "right": self._build(X[~left], y[~left], depth + 1),
        })
        return node

    def fit(self, X, y):
        self.tree_ = self._build(np.asarray(X, dtype=float),
                                 np.asarray(y, dtype=int), depth=0)
        return self

    def _predict_one(self, x):
        node = self.tree_
        while "feature" in node:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node["label"]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X], dtype=int)


# ----------------------------------------------------------------------
# The Random Forest model 
# ----------------------------------------------------------------------

class RandomForestModel:
    def __init__(self, input_size, output_size, n_trees=100,
                 max_depth=None, min_samples_split=2, seed=42):
        self.input_size = input_size
        self.output_size = output_size
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.seed = seed
        # Standard classification choice: sqrt of the attributes.
        self.max_features = max(1, int(np.sqrt(input_size)))
        self.trees = []

    def fit(self, X, y):
        """Train every tree on a bootstrap sample of the training rows."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or X.shape[1] != self.input_size:
            raise ValueError(
                f"Expected training features with {self.input_size} columns"
            )
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("Training labels must match the number of rows")
        if len(X) == 0:
            raise ValueError("Training data must contain at least one row")
        if np.any(y < 0) or np.any(y >= self.output_size):
            raise ValueError("Training labels must be encoded as class indexes")
        if self.n_trees < 1:
            raise ValueError("n_trees must be at least 1")

        self.trees = []
        rng = np.random.default_rng(self.seed)
        for t in range(self.n_trees):
            bootstrap = rng.integers(0, len(X), size=len(X))
            tree = DecisionTree(
                n_classes=self.output_size,
                max_features=self.max_features,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                rng=np.random.default_rng(rng.integers(0, 2 ** 63 - 1)),
            )
            tree.fit(X[bootstrap], y[bootstrap])
            self.trees.append(tree)

            if t == 0 or (t + 1) % 10 == 0 or t + 1 == self.n_trees:
                print(f"Tree [{t + 1}/{self.n_trees}] trained", flush=True)
        return self

    def predict(self, X):
        """Majority vote over all trees."""
        X = np.asarray(X, dtype=float)
        if not self.trees:
            raise ValueError("The random forest must be fitted before prediction")
        if X.ndim != 2 or X.shape[1] != self.input_size:
            raise ValueError(
                f"Expected prediction features with {self.input_size} columns"
            )
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        votes = np.zeros((X.shape[0], self.output_size), dtype=int)
        for j in range(X.shape[0]):
            votes[j] = np.bincount(all_preds[:, j],
                                   minlength=self.output_size)
        return np.argmax(votes, axis=1)


# ----------------------------------------------------------------------
# Dataset 
# ----------------------------------------------------------------------

class DataSet:
    LABEL_NAMES = ("label", "diagnosis", "class", "target", "y")

    def __init__(self, data_path, has_label=True, feature_encoders=None,
                 feature_columns=None):
        self.ids = []
        self.attributes = []
        self.diagnosis = []
        self.has_label = has_label
        # feature_encoders: per attribute column, None if the column is
        # numeric, otherwise a {value: code} dict learned from the
        # training file, so the test file is encoded identically.
        self.feature_encoders = feature_encoders

        try:
            with open(data_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader, None)
                if not header:
                    raise ValueError("CSV file must include a header row")

                normalized = [name.strip().lower() for name in header]
                label_index = next(
                    (i for i, name in enumerate(normalized)
                     if name in self.LABEL_NAMES),
                    None,
                )
                if has_label and label_index is None:
                    raise ValueError("Labeled CSV is missing a label column")

                if feature_columns is None:
                    named_features = [
                        i for i, name in enumerate(normalized)
                        if name.startswith("feature_")
                    ]
                    if named_features:
                        feature_indices = named_features
                        self.feature_columns = [
                            header[i] for i in feature_indices
                        ]
                    else:
                        excluded = {label_index} if label_index is not None else set()
                        id_index = normalized.index("id") if "id" in normalized else None
                        if id_index is not None:
                            excluded.add(id_index)
                        feature_indices = [
                            i for i in range(len(header)) if i not in excluded
                        ]
                        # Preserve the older id,label,feature... CSV layout.
                        if label_index == 1 and id_index is None:
                            feature_indices = list(range(2, len(header)))
                        self.feature_columns = [
                            header[i] for i in feature_indices
                        ]
                else:
                    self.feature_columns = list(feature_columns)
                    feature_indices = []
                    for column in self.feature_columns:
                        try:
                            feature_indices.append(header.index(column))
                        except ValueError as e:
                            raise ValueError(
                                f"CSV is missing feature column {column!r}"
                            ) from e

                id_index = normalized.index("id") if "id" in normalized else None
                if id_index is None and feature_indices and feature_indices[0] > 0:
                    candidate = feature_indices[0] - 1
                    if candidate != label_index:
                        id_index = candidate

                rows = [row for row in reader if row]
                raw = []
                for row in rows:
                    if any(i >= len(row) for i in feature_indices):
                        raise ValueError("CSV row has fewer values than its header")
                    raw.append([row[i] for i in feature_indices])

                if self.feature_encoders is None:
                    self.feature_encoders = [
                        self._make_encoder([r[j] for r in raw])
                        for j in range(len(feature_indices))
                    ]
                if len(self.feature_encoders) != len(feature_indices):
                    raise ValueError("Training and testing feature counts differ")

                for row_number, (row, raw_attrs) in enumerate(zip(rows, raw)):
                    self.ids.append(row[id_index] if id_index is not None else str(row_number))
                    self.attributes.append([
                        self._encode_value(v, self.feature_encoders[j])
                        for j, v in enumerate(raw_attrs)
                    ])
                    if has_label:
                        if label_index is None or label_index >= len(row):
                            raise ValueError("CSV row is missing its label")
                        self.diagnosis.append(row[label_index])

        except (OSError, ValueError, IndexError) as e:
            raise RuntimeError(
                f"Error reading the dataset {data_path}"
            ) from e

    @staticmethod
    def _make_encoder(values):
        try:
            [float(v) for v in values]
            return None  # numeric column, no encoding needed
        except (ValueError, TypeError):
            return {v: float(k) for k, v in enumerate(sorted(set(values)))}

    @staticmethod
    def _encode_value(value, encoder):
        if encoder is None:
            return float(value)
        return encoder.get(value, -1.0)  # unseen category -> -1


def _test_file_has_label(test_path):
    """Detect labels by header name so unlabeled test CSVs are supported."""
    with open(test_path, mode='r', newline='') as file:
        header = next(csv.reader(file), None)
    return bool(header) and any(
        name.strip().lower() in DataSet.LABEL_NAMES for name in header
    )


# ----------------------------------------------------------------------
# Training 
# ----------------------------------------------------------------------

class RandomForestTraining:

    def __init__(self, model, training_set):
        model.fit(training_set.attributes, training_set.encoded_diagnosis)


# ----------------------------------------------------------------------
# Testing 
# ----------------------------------------------------------------------

class RandomForestTesting:

    def __init__(self, model, testing_set):
        x_test = np.array(testing_set.attributes, dtype=float)

        # One prediction (a class index) per test instance
        self.predictions = model.predict(x_test)

        self.accuracy = None
        if testing_set.has_label:
            y_test = np.array(testing_set.encoded_diagnosis, dtype=int)
            correct = int(np.sum(self.predictions == y_test))
            count = len(x_test)
            self.accuracy = (correct / count) * 100 if count > 0 else 0


# ----------------------------------------------------------------------
# Export predictions in sample-submission order and format
# ----------------------------------------------------------------------

def export_predictions(predictions_by_id, submit_path, sample_submission_path=None):
    header = ["id", "label"]
    item_ids = list(predictions_by_id)
    if sample_submission_path is not None:
        with open(sample_submission_path, mode="r", newline="") as sample_file:
            reader = csv.reader(sample_file)
            sample_header = next(reader, None)
            if sample_header != header:
                raise ValueError(
                    "Sample submission must have 'id,label' columns"
                )
            item_ids = [row[0] for row in reader if row]

        if len(item_ids) != len(set(item_ids)):
            raise ValueError("Sample submission contains duplicate IDs")
        if set(item_ids) != set(predictions_by_id):
            raise ValueError(
                "Predicted IDs do not match the sample submission IDs "
                f"({len(set(item_ids) - set(predictions_by_id))} missing, "
                f"{len(set(predictions_by_id) - set(item_ids))} unexpected)"
            )

    with open(submit_path, mode='w', newline='') as file:
        # lineterminator="\n" matches sample_submission.csv exactly:
        # Unix line endings, not the csv module's default \r\n.
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        for item_id in item_ids:
            writer.writerow([item_id, predictions_by_id[item_id]])


def _new_submission_path(output_dir):
    while True:
        submit_path = output_dir / (
            f"submission{secrets.randbelow(1_000_000):06d}.csv"
        )
        if not submit_path.exists():
            return submit_path


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train both bundled datasets, or one custom training/testing pair."
        )
    )
    parser.add_argument("--train", default=None,
                        help="Custom training CSV (runs only with --test)")
    parser.add_argument("--test", default=None,
                        help="Custom testing CSV (runs only with --train)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--n-trees", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None,
                        help="Maximum depth per tree (default: no limit)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if bool(args.train) != bool(args.test):
        parser.error("--train and --test must be provided together")

    project_dir = Path(__file__).resolve().parent
    # Also work with the same data/ directory used by run_all.py.
    csv_dir = next(
        (folder for folder in (project_dir / "data", project_dir / "CSVS", project_dir)
         if (folder / "image_train.csv").is_file()),
        project_dir / "data",
    )
    if args.train and args.test:
        datasets = [("custom", args.train, args.test)]
        sample_submission_path = None
    else:
        datasets = [
            (
                "image",
                csv_dir / "image_train.csv",
                csv_dir / "image_test.csv",
            ),
            (
                "sentiment",
                csv_dir / "sentiment_train.csv",
                csv_dir / "sentiment_test.csv",
            ),
        ]
        sample_submission_path = csv_dir / "sample_submission.csv"

    # Custom CSV pairs may live outside the project and need a valid output folder.
    output_dir = Path(args.test).resolve().parent if args.train else csv_dir
    submit_path = args.output or _new_submission_path(output_dir)
    predictions_by_id = {}

    for dataset_name, training_path, testing_path in datasets:
        print(f"\nLoading {dataset_name} dataset...")
        training_set = DataSet(training_path, has_label=True)
        testing_set = DataSet(
            testing_path,
            has_label=_test_file_has_label(testing_path),
            feature_encoders=training_set.feature_encoders,
            feature_columns=training_set.feature_columns,
        )

        classes = sorted(set(training_set.diagnosis))
        label_to_index = {
            label: index
            for index, label in enumerate(classes)
        }

        training_set.encoded_diagnosis = [
            label_to_index[label]
            for label in training_set.diagnosis
        ]

        if testing_set.has_label:
            try:
                testing_set.encoded_diagnosis = [
                    label_to_index[label]
                    for label in testing_set.diagnosis
                ]
            except KeyError as e:
                raise ValueError(
                    f"Unknown diagnosis in {dataset_name} testing dataset: {e}"
                ) from e

        print(f"{dataset_name.capitalize()} dataset loaded successfully.")

        input_size = len(training_set.feature_columns)
        if input_size == 0:
            raise ValueError(
                f"{dataset_name.capitalize()} training CSV must contain "
                "at least one feature column"
            )
        output_size = len(classes)

        model = RandomForestModel(
            input_size,
            output_size,
            n_trees=args.n_trees,
            max_depth=args.max_depth,
            seed=args.seed,
        )

        print(f"\nTraining {dataset_name} random forest...")
        RandomForestTraining(model, training_set)
        print(f"{dataset_name.capitalize()} training completed.")

        test = RandomForestTesting(model, testing_set)
        predicted_labels = [
            classes[prediction]
            for prediction in test.predictions
        ]
        for item_id, label in zip(testing_set.ids, predicted_labels):
            if item_id in predictions_by_id:
                raise ValueError(f"Duplicate test ID across datasets: {item_id}")
            predictions_by_id[item_id] = label

        print(f"\n{dataset_name.capitalize()} results")
        print(f"Training instances: {len(training_set.attributes)}")
        print(f"Testing instances: {len(testing_set.attributes)}")
        print(f"Number of input attributes: {input_size}")
        print(f"Number of trees: {model.n_trees}")
        print(f"Number of output classes: {output_size}")

        if test.accuracy is not None:
            print(f"Random Forest Accuracy: {test.accuracy:.2f}%")
        else:
            print("Random Forest Accuracy: n/a (test file has no labels)")

    export_predictions(
        predictions_by_id,
        submit_path,
        sample_submission_path=sample_submission_path,
    )

    print(f"\nCombined predictions written to: {submit_path}")


if __name__ == "__main__":
    main()
