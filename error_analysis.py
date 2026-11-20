"""Error analysis for our best K-NN models on image and sentiment data.

Run test_knn.py first to save the predictions, then run:
    python error_analysis.py

This file uses the saved predictions. It does not train K-NN again.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from experiment_helpers import TASKS, setup, load_data, load_state

# Folder where all K-NN error analysis results will be stored
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "erroranalysis_knn"


def analyze_task(task, data, best):
    """Evaluate the best saved K-NN predictions for one dataset."""

    # Get the correct validation labels and predictions from our best K-NN model
    actual = np.asarray(data[task]["val_labels"])
    predicted = np.asarray(best[task]["knn"]["val_pred"])

    # Make sure every validation sample has a prediction
    if len(actual) != len(predicted):
        raise ValueError(
            f"{task}: actual labels and predictions have different lengths"
        )

    # Get all possible classes, including classes the model never predicted
    labels = list(range(data[task]["n_classes"]))

    # Use sentiment names for text data and numeric labels for images
    names = (
        ["Negative", "Neutral", "Positive"]
        if task == "sentiment"
        else [str(label) for label in labels]
    )

    # Create a separate folder to save the results for each dataset
    folder = OUTPUT_DIR / task
    folder.mkdir(parents=True, exist_ok=True)

    # Calculate accuracy and macro F1 to measure overall model performance
    accuracy = accuracy_score(actual, predicted)

    # Macro F1 gives equal importance to each class
    macro_f1 = f1_score(
        actual, predicted, labels=labels, average="macro", zero_division=0
    )

    # Calculate precision, recall, F1 score, and support for every class
    # Support is the number of actual validation samples in each class
    report = classification_report(
        actual,
        predicted,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )

    # Convert the classification results into a table and save them
    class_metrics = pd.DataFrame(report).T.loc[
        names, ["precision", "recall", "f1-score", "support"]
    ]
    class_metrics.index.name = "class"
    class_metrics.to_csv(folder / "class_metrics.csv")

    # Create a confusion matrix to see which classes the model confuses
    # Rows show actual labels and columns show predicted labels
    matrix = confusion_matrix(actual, predicted, labels=labels)
    confusion = pd.DataFrame(matrix, index=names, columns=names)
    confusion.index.name = "actual_class"
    confusion.columns.name = "predicted_class"
    confusion.to_csv(folder / "confusion_matrix.csv")

    # Find the validation samples where the prediction was incorrect
    wrong = np.flatnonzero(actual != predicted)
    mistakes = pd.DataFrame(
        {"actual_label": actual[wrong], "predicted_label": predicted[wrong]}
    )

    # Group mistakes by actual and predicted class
    # Count each type of mistake and sort from most to least common
    confused_pairs = (
        mistakes.groupby(["actual_label", "predicted_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    confused_pairs.to_csv(folder / "confused_pairs.csv", index=False)

    # Print the model's accuracy, F1 score, and total number of mistakes
    print(f"\n--- {task.upper()}: k-NN error analysis ---")
    print("Best parameters:", best[task]["knn"]["params"])
    print(f"Accuracy: {accuracy:.3f} | Macro F1: {macro_f1:.3f}")
    print(f"Incorrect predictions: {len(wrong)} / {len(actual)}")

    # For sentiment, display the performance of all three classes
    if task == "sentiment":
        print("\nPrecision, recall and F1 for each sentiment class:")
        print(class_metrics.round(3).to_string())

    # For images, find the five classes with the lowest recall
    # Lower recall means more actual samples from that class were missed
    else:
        print("\nFive hardest image classes (lowest recall):")
        print(class_metrics.nsmallest(5, "recall").round(3).to_string())

    # Display the five most common class prediction mistakes
    print("\nFive most common mistakes (actual -> predicted):")
    if confused_pairs.empty:
        print("No mistakes found.")
    else:
        print(confused_pairs.head(5).to_string(index=False))

    # Show where all CSV results were saved
    print("Files saved in:", folder)


def run_error_analysis(data, best):
    """Analyze the best K-NN predictions on both datasets."""

    # Go through both image and sentiment datasets
    for task in TASKS:

        # Check that K-NN predictions were saved before running analysis
        if "knn" not in best.get(task, {}):
            raise SystemExit(
                f"No saved K-NN results for {task}. Run test_knn.py first."
            )

        # Run the error analysis for the current dataset
        analyze_task(task, data, best)


def main():
    """Load saved experiments and run K-NN error analysis."""

    # Set up the experiment environment and load both datasets
    setup()
    data = load_data()

    # Load the saved model results instead of training again
    state = load_state()

    # Analyze the best K-NN models for both datasets
    run_error_analysis(data, state["best"])


if __name__ == "__main__":
    main()
