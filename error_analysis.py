"""Error analysis for the best k-NN models on the image and sentiment data.

Run test_knn.py first so the predictions are saved, then run:
    python error_analysis.py

This file reads the saved predictions and does not train k-NN again.
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

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "erroranalysis_knn"


def analyze_task(task, data, best):
    """Evaluate the best saved k-NN predictions for one dataset and save the tables as CSVs."""
    actual = np.asarray(data[task]["val_labels"])
    predicted = np.asarray(best[task]["knn"]["val_pred"])

    if len(actual) != len(predicted):
        raise ValueError(
            f"{task}: actual labels and predictions have different lengths"
        )

    # Pass every class explicitly so classes the model never predicted still appear in the report
    labels = list(range(data[task]["n_classes"]))
    names = (
        ["Negative", "Neutral", "Positive"]
        if task == "sentiment"
        else [str(label) for label in labels]
    )

    folder = OUTPUT_DIR / task
    folder.mkdir(parents=True, exist_ok=True)

    accuracy = accuracy_score(actual, predicted)
    # Macro F1 averages the per-class F1 scores, so every class counts equally
    macro_f1 = f1_score(
        actual, predicted, labels=labels, average="macro", zero_division=0
    )

    # Precision, recall, F1 and support (number of true samples) for every class
    report = classification_report(
        actual,
        predicted,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    class_metrics = pd.DataFrame(report).T.loc[
        names, ["precision", "recall", "f1-score", "support"]
    ]
    class_metrics.index.name = "class"
    class_metrics.to_csv(folder / "class_metrics.csv")

    # Rows are the true classes, columns are the predicted classes
    matrix = confusion_matrix(actual, predicted, labels=labels)
    confusion = pd.DataFrame(matrix, index=names, columns=names)
    confusion.index.name = "actual_class"
    confusion.columns.name = "predicted_class"
    confusion.to_csv(folder / "confusion_matrix.csv")

    # Count each (true class, predicted class) mistake, most common first
    wrong = np.flatnonzero(actual != predicted)
    mistakes = pd.DataFrame(
        {"actual_label": actual[wrong], "predicted_label": predicted[wrong]}
    )
    confused_pairs = (
        mistakes.groupby(["actual_label", "predicted_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    confused_pairs.to_csv(folder / "confused_pairs.csv", index=False)

    print(f"\n--- {task.upper()}: k-NN error analysis ---")
    print("Best parameters:", best[task]["knn"]["params"])
    print(f"Accuracy: {accuracy:.3f} | Macro F1: {macro_f1:.3f}")
    print(f"Incorrect predictions: {len(wrong)} / {len(actual)}")

    if task == "sentiment":
        print("\nPrecision, recall and F1 for each sentiment class:")
        print(class_metrics.round(3).to_string())
    else:
        # 50 image classes are too many to print, so show only the five with the lowest recall
        print("\nFive hardest image classes (lowest recall):")
        print(class_metrics.nsmallest(5, "recall").round(3).to_string())

    print("\nFive most common mistakes (actual -> predicted):")
    if confused_pairs.empty:
        print("No mistakes found.")
    else:
        print(confused_pairs.head(5).to_string(index=False))

    print("Files saved in:", folder)


def run_error_analysis(data, best):
    """Analyze the best k-NN predictions on both datasets."""
    for task in TASKS:
        if "knn" not in best.get(task, {}):
            raise SystemExit(
                f"No saved k-NN results for {task}. Run test_knn.py first."
            )
        analyze_task(task, data, best)


def main():
    setup()
    data = load_data()
    state = load_state()  # reads the saved predictions instead of training again
    run_error_analysis(data, state["best"])


if __name__ == "__main__":
    main()
