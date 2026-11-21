"""Neural network classifier (PyTorch) for the image and sentiment datasets.

Network: one 64-unit hidden layer with ReLU, trained with Adam and cross-entropy
loss for 100 epochs in batches of 256. Uses the same 80/20 split and the same
training-only standardizer as the other models.

Run this file directly to train both tasks and write per-task prediction files
plus a combined nn_submission.csv.

The training helpers below are also used by test_neuralnetwork.py (validation
runs for run_all.py) and by utils.fit_predict() (retraining for make_submission.py).
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils import SEED, train_val_split, fit_standardizer


# Resolve paths from this file's location rather than the terminal's current directory
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" if (PROJECT_DIR / "data").is_dir() else PROJECT_DIR

HIDDEN_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 256


class NeuralNetworkModel(nn.Module):
    """Input -> hidden layer (ReLU) -> one raw score (logit) per class."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# This load_data(task) belongs to the standalone main() below. The shared
# experiment_helpers.load_data() prepares both tasks for run_all.py instead.
def load_data(task):
    """Read the train and test CSVs of one task. Returns (X, y, X_test, test_ids)."""
    train = pd.read_csv(DATA_DIR / f"{task}_train.csv")
    test = pd.read_csv(DATA_DIR / f"{task}_test.csv")

    features = [col for col in train.columns if col.startswith("feature_")]
    if not features:
        raise ValueError(f"No feature_ columns found in {task}_train.csv")

    X = train[features].to_numpy(dtype=np.float32)
    y = train["label"].to_numpy(dtype=np.int64)
    X_test = test[features].to_numpy(dtype=np.float32)

    return X, y, X_test, test["id"]


def create_dataloaders(X_train, y_train, X_val=None, y_val=None, batch_size=BATCH_SIZE):
    """Wrap the arrays in mini-batch loaders. The validation set is optional because
    the final submission model trains on all labelled rows and has nothing to validate on."""
    train_data = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_loader = None
    if X_val is not None and y_val is not None:
        val_data = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def evaluate(model, val_loader):
    """Classification accuracy of the model, as a percentage."""
    if val_loader is None:
        raise ValueError("Validation loader is required for accuracy evaluation")

    model.eval()  # switch off training-only behaviour
    correct = 0
    total = 0

    with torch.no_grad():  # no gradients needed when only predicting
        for X, y in val_loader:
            predictions = model(X).argmax(dim=1)
            correct += (predictions == y).sum().item()
            total += len(y)

    return 100 * correct / total


def train_and_evaluate(
    model, train_loader, val_loader, title,
    *, epochs=EPOCHS, learning_rate=LEARNING_RATE
):
    """Train the model and print progress. Returns (model, final accuracy in %).
    The accuracy is None when there is no validation loader."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\n--- Training {title} ---", flush=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)                  # forward pass
            loss = criterion(outputs, y)
            loss.backward()                     # backpropagation
            optimizer.step()                    # update the weights

            total_loss += loss.item() * len(X)  # weight by batch size so the last, smaller batch is averaged correctly

        avg_loss = total_loss / len(train_loader.dataset)

        # Print the first epoch, every 10th epoch and the last one
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            if val_loader is not None:
                accuracy = evaluate(model, val_loader)
                print(
                    f"Epoch [{epoch + 1}/{epochs}] - "
                    f"Loss: {avg_loss:.4f} | "
                    f"Validation Accuracy: {accuracy:.2f}%",
                    flush=True,
                )
            else:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] - "
                    f"Loss: {avg_loss:.4f} | training on all labeled rows",
                    flush=True,
                )

    if val_loader is not None:
        accuracy = evaluate(model, val_loader)
        print(
            f"Final Validation Accuracy for {title}: {accuracy:.2f}%",
            flush=True,
        )
    else:
        accuracy = None  # no labels to score against, so no accuracy is reported
        print(f"Finished training {title} on all labeled rows", flush=True)

    return model, accuracy


def predict(model, X_test, classes, batch_size=BATCH_SIZE):
    """Predict original class labels. Works in batches so the activations of a
    large dataset never have to be held in memory at once."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for start in range(0, len(X_test), batch_size):
            X = torch.tensor(X_test[start:start + batch_size], dtype=torch.float32)
            predictions.append(model(X).argmax(dim=1).cpu().numpy())

    if not predictions:
        return np.asarray([], dtype=np.asarray(classes).dtype)
    # The network outputs class positions; look them up in `classes` to get the real labels
    return np.asarray(classes)[np.concatenate(predictions)]


def seed_everything():
    """Use the same fixed seed as the other project models."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def encode_labels(y, classes):
    """Map the real label values to 0-based positions, as the network's outputs expect."""
    label_to_index = {int(label): i for i, label in enumerate(classes)}
    return np.asarray([label_to_index[int(v)] for v in y], dtype=np.int64)


def train_neural_network(params, train_features, train_labels, title="Neural Network"):
    """Train on ALL supplied rows (no validation split).

    Called from utils.fit_predict() when retraining for the final submission.
    The features have already been scaled by utils using the training rows.
    """
    seed_everything()
    classes = np.unique(train_labels)
    encoded_labels = encode_labels(train_labels, classes)
    batch_size = int(params.get("batch_size", BATCH_SIZE))
    hidden_size = int(params.get("hidden_size", HIDDEN_SIZE))
    epochs = int(params.get("epochs", EPOCHS))
    learning_rate = float(params.get("learning_rate", LEARNING_RATE))

    train_loader, _ = create_dataloaders(
        train_features, encoded_labels, batch_size=batch_size
    )
    model = NeuralNetworkModel(
        input_size=train_features.shape[1],
        hidden_size=hidden_size,
        output_size=len(classes),
    )
    model, _ = train_and_evaluate(
        model, train_loader, None, title,
        epochs=epochs, learning_rate=learning_rate,
    )
    # Stored on the model so predict() can map outputs back to the real labels
    model.classes_ = classes
    model.batch_size_ = batch_size
    return model


def train_validation_network(params, X_train, y_train, X_val, y_val, title):
    """Train on the training split and score on the validation split.

    Called from test_neuralnetwork.py so run_all.py can print epoch-level validation
    scores and save predictions for error analysis.
    Returns (model, accuracy between 0 and 1, validation predictions as original labels).
    """
    seed_everything()
    # Scaling statistics come from the training rows only, so nothing leaks from validation
    standardize = fit_standardizer(X_train)
    X_train = standardize(X_train)
    X_val = standardize(X_val)

    # Validation labels are included only to build the label mapping, never for training
    classes = np.unique(np.concatenate((y_train, y_val)))
    train_labels = encode_labels(y_train, classes)
    val_labels = encode_labels(y_val, classes)

    batch_size = int(params.get("batch_size", BATCH_SIZE))
    train_loader, val_loader = create_dataloaders(
        X_train, train_labels, X_val, val_labels, batch_size=batch_size
    )
    model = NeuralNetworkModel(
        input_size=X_train.shape[1],
        hidden_size=int(params.get("hidden_size", HIDDEN_SIZE)),
        output_size=len(classes),
    )
    model, accuracy_percent = train_and_evaluate(
        model, train_loader, val_loader, title,
        epochs=int(params.get("epochs", EPOCHS)),
        learning_rate=float(params.get("learning_rate", LEARNING_RATE)),
    )
    model.classes_ = classes
    model.batch_size_ = batch_size
    val_predictions = predict(model, X_val, classes, batch_size=batch_size)
    return model, accuracy_percent / 100, val_predictions


def main():
    """Standalone run: train and validate both tasks, then write nn_submission.csv."""
    seed_everything()

    all_predictions = []
    results = {}

    for task in ["image", "sentiment"]:
        X, y, X_test, test_ids = load_data(task)
        X_train, X_val, y_train, y_val = train_val_split(X, y)

        standardize = fit_standardizer(X_train)
        X_train = standardize(X_train)
        X_val = standardize(X_val)
        X_test = standardize(X_test)

        classes = np.unique(y)
        y_train = encode_labels(y_train, classes)
        y_val = encode_labels(y_val, classes)

        train_loader, val_loader = create_dataloaders(
            X_train, y_train, X_val, y_val
        )

        model = NeuralNetworkModel(
            input_size=X_train.shape[1],
            hidden_size=HIDDEN_SIZE,
            output_size=len(classes),
        )

        model, accuracy = train_and_evaluate(
            model, train_loader, val_loader,
            title=f"{task.capitalize()} Neural Network",
        )
        results[task] = accuracy

        predictions = predict(model, X_test, classes)

        task_results = pd.DataFrame({
            "id": test_ids,
            "label": predictions,
        })
        output_file = PROJECT_DIR / f"nn_{task}_predictions.csv"
        task_results.to_csv(output_file, index=False)
        print(f"Saved {output_file.name}", flush=True)
        all_predictions.append(task_results)

    # Combine both tasks in the row order of sample_submission.csv
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    predictions = pd.concat(all_predictions, ignore_index=True)

    # Check the ids before merging so a mismatch cannot silently drop or duplicate rows
    assert predictions["id"].is_unique
    assert set(predictions["id"]) == set(sample["id"])

    submission = sample[["id"]].merge(
        predictions, on="id", how="left", validate="one_to_one"
    )

    assert len(submission) == 10000
    assert submission["label"].notna().all()
    assert submission["id"].is_unique

    submission["label"] = submission["label"].astype(int)

    image_mask = submission["id"].str.startswith("img_")
    sentiment_mask = submission["id"].str.startswith("text_")

    assert image_mask.sum() == 5000
    assert sentiment_mask.sum() == 5000
    assert submission.loc[image_mask, "label"].between(0, 49).all()
    assert submission.loc[sentiment_mask, "label"].between(0, 2).all()

    submission.to_csv(PROJECT_DIR / "nn_submission.csv", index=False)

    print("\n================================", flush=True)
    print("NEURAL NETWORK RESULTS", flush=True)
    print("================================", flush=True)
    print(f"Image Accuracy: {results['image']:.2f}%", flush=True)
    print(f"Sentiment Accuracy: {results['sentiment']:.2f}%", flush=True)
    print(f"Average Accuracy: {np.mean(list(results.values())):.2f}%", flush=True)
    print("Saved predictions to nn_submission.csv", flush=True)


if __name__ == "__main__":
    main()
