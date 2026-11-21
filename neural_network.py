"""Neural network classifier (PyTorch) for image and sentiment datasets.

This keeps the original full neural-network workflow:
- 64-unit hidden layer with ReLU
- Adam optimizer, cross-entropy loss, 100 epochs, and batches of 256
- same 80/20 split and training-only standardizer as the k-NN code
- printed epoch losses and validation accuracy
- per-task test predictions and a Kaggle-formatted nn_submission.csv

It also exposes training helpers to test_nn.py and utils.fit_predict(), so the
same network can participate in run_all.py, saved results, error analysis, and
make_submission.py. Run this file directly for a neural-network-only submission.
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


# Use the Python file's location rather than the terminal's current directory.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" if (PROJECT_DIR / "data").is_dir() else PROJECT_DIR

# Hyperparameters (kept from the original network)
HIDDEN_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 256


# One hidden layer with ReLU activation
class NeuralNetworkModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# Read the features, labels and test IDs
# This load_data(task) belongs to this standalone script. The shared
# experiment_helpers.load_data() prepares both tasks for run_all.py.
def load_data(task):
    train = pd.read_csv(DATA_DIR / f"{task}_train.csv")
    test = pd.read_csv(DATA_DIR / f"{task}_test.csv")

    features = [col for col in train.columns if col.startswith("feature_")]
    if not features:
        raise ValueError(f"No feature_ columns found in {task}_train.csv")

    X = train[features].to_numpy(dtype=np.float32)
    y = train["label"].to_numpy(dtype=np.int64)
    X_test = test[features].to_numpy(dtype=np.float32)

    return X, y, X_test, test["id"]


# Convert features and labels into mini-batches
# Validation data are optional when training on ALL data for submission.
def create_dataloaders(
    X_train, y_train, X_val=None, y_val=None, batch_size=BATCH_SIZE
):
    train_data = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )

    val_loader = None
    if X_val is not None and y_val is not None:
        val_data = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=False
        )

    return train_loader, val_loader


# Evaluate the model using classification accuracy
# Return percentage accuracy as in the original script.
def evaluate(model, val_loader):
    if val_loader is None:
        raise ValueError("Validation loader is required for accuracy evaluation")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in val_loader:
            predictions = model(X).argmax(dim=1)
            correct += (predictions == y).sum().item()
            total += len(y)

    return 100 * correct / total


# Train using forward propagation and backpropagation
# Optional keyword arguments allow experiment settings to be reused for
# submission retraining; defaults stay exactly as in the original script.
def train_and_evaluate(
    model, train_loader, val_loader, title,
    *, epochs=EPOCHS, learning_rate=LEARNING_RATE
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\n--- Training {title} ---", flush=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(X)

        avg_loss = total_loss / len(train_loader.dataset)

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
        accuracy = None  # Test labels are hidden; don't invent accuracy.
        print(f"Finished training {title} on all labeled rows", flush=True)

    return model, accuracy


# Predict the labels of the hidden test dataset
# Predictions are computed in batches to avoid storing the full activation
# matrix when evaluating large datasets.
def predict(model, X_test, classes, batch_size=BATCH_SIZE):
    model.eval()
    predictions = []

    with torch.no_grad():
        for start in range(0, len(X_test), batch_size):
            X = torch.tensor(X_test[start:start + batch_size], dtype=torch.float32)
            predictions.append(model(X).argmax(dim=1).cpu().numpy())

    if not predictions:
        return np.asarray([], dtype=np.asarray(classes).dtype)
    return np.asarray(classes)[np.concatenate(predictions)]


def seed_everything():
    """Use the same fixed seed as the other project models."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def encode_labels(y, classes):
    """Map true label values to zero-based network output indices."""
    label_to_index = {int(label): i for i, label in enumerate(classes)}
    return np.asarray([label_to_index[int(v)] for v in y], dtype=np.int64)


def train_neural_network(params, train_features, train_labels, title="Neural Network"):
    """Train PyTorch on ALL supplied training rows (no validation split).

    Called from utils.fit_predict() during final submission generation.
    All preprocessing has already been fitted on the training rows by utils.
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
    model.classes_ = classes  # needed to map neural-network outputs to labels
    model.batch_size_ = batch_size
    return model


def train_validation_network(params, X_train, y_train, X_val, y_val, title):
    """Validate the SAME PyTorch network, using training-only standardization.

    Called from test_nn.py so run_all.py can print epoch-level validation
    scores and save predictions for the shared error-analysis files.
    Returns predictions using the original class labels, plus accuracy (0-1).
    """
    seed_everything()
    standardize = fit_standardizer(X_train)
    X_train = standardize(X_train)
    X_val = standardize(X_val)

    # Include validation labels in the output mapping, but never in training.
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
    """Run the original standalone neural network and write nn_submission.csv."""
    seed_everything()

    all_predictions = []
    results = {}

    for task in ["image", "sentiment"]:
        X, y, X_test, test_ids = load_data(task)

        # Same 80/20 split and standardizer as the k-NN code
        X_train, X_val, y_train, y_val = train_val_split(X, y)

        standardize = fit_standardizer(X_train)
        X_train = standardize(X_train)
        X_val = standardize(X_val)
        X_test = standardize(X_test)

        # Encode original labels as class indices
        classes = np.unique(y)
        label_to_index = {
            label: i for i, label in enumerate(classes)
        }
        y_train = np.array([label_to_index[v] for v in y_train])
        y_val = np.array([label_to_index[v] for v in y_val])

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

    # Combine both datasets using the required Kaggle ID order
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    predictions = pd.concat(all_predictions, ignore_index=True)

    # Validate IDs before merging, to avoid a silent row mismatch.
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
    print("Saved Kaggle predictions to nn_submission.csv", flush=True)


if __name__ == "__main__":
    main()
