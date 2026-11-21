# Picks the best model of each task (by validation accuracy), retrains it on ALL the
# training data and writes submission.csv.
import json
import os
import random
import time

import numpy as np
import pandas as pd

from utils import SEED, load_train, fit_predict

DATA_DIR = "data" if os.path.isdir("data") else "."
KNN_RESULTS_FILE = "results_knn.json"  # k-NN only
ALL_RESULTS_FILE = "results.json"      # every model, written by run_all.py

start_time = time.time()
random.seed(SEED)
np.random.seed(SEED)

names = {
    "knn": "k-NN",
    "dt": "Decision Tree",
    "rf": "Random Forest",
    "nn": "Neural Network",
}

# Prefer the combined results; fall back to the k-NN-only file if they do not exist
results_file = ALL_RESULTS_FILE if os.path.exists(ALL_RESULTS_FILE) else KNN_RESULTS_FILE
print("using results from", results_file)
with open(results_file) as f:
    results = json.load(f)

for task in ("image", "sentiment"):
    if not any(m in results.get(task, {}) for m in names):
        raise ValueError(
            f"{results_file} has no model result for {task}. Run run_all.py first."
        )

parts = []
for task, train_file, test_file in [
    ("image", "image_train.csv", "image_test.csv"),
    ("sentiment", "sentiment_train.csv", "sentiment_test.csv"),
]:
    # Retrain on every labelled row, not just the 80% used during validation
    X, y, feature_cols = load_train(os.path.join(DATA_DIR, train_file))
    test = pd.read_csv(os.path.join(DATA_DIR, test_file))
    ids = test["id"].values
    X_test = test[feature_cols].values.astype(np.float32)

    candidates = [m for m in results[task] if m in names]
    best_name = max(candidates, key=lambda n: results[task][n]["acc"])
    params = results[task][best_name]["params"]
    print(
        f"{task}: using {names[best_name]} {params} (validation acc {results[task][best_name]['acc']:.3f})"
    )

    preds, _ = fit_predict(best_name, params, X, y, X_test, int(y.max()) + 1)
    print(
        f"   retrained on all {len(X)} training rows and predicted {len(X_test)} test rows"
    )
    parts.append(pd.DataFrame({"id": ids, "label": preds.astype(int)}))

# The template rows are in a different order, so match predictions to it by id
template = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
predictions = pd.concat(parts, ignore_index=True)
assert set(predictions["id"]) == set(
    template["id"]
), "test IDs must match sample_submission.csv exactly"
sub = template[["id"]].merge(predictions, on="id", how="left", validate="one_to_one")

# Check the file against the required format before saving it
assert list(sub.columns) == [
    "id",
    "label",
], "the file must have exactly the columns id, label"
assert (
    len(sub) == 10000 and sub["id"].is_unique
), "all 10,000 ids must appear exactly once"
assert sub["label"].notna().all(), "no missing predictions allowed"
sub["label"] = sub["label"].astype(int)
assert (
    sub.loc[sub["id"].str.startswith("img_"), "label"].between(0, 49).all()
), "image labels must be 0-49"
assert (
    sub.loc[sub["id"].str.startswith("text_"), "label"].between(0, 2).all()
), "sentiment labels must be 0-2"

sub.to_csv("submission.csv", index=False)
print("saved submission.csv with", len(sub), "rows")
print(f"[time] make_submission total: {time.time() - start_time:.1f}s")
print(sub.head())
