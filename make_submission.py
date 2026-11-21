# Picks the best model of each task (by validation accuracy), retrains it on ALL the training data and writes submission.csv.
import json
import os
import random
import time

import numpy as np
import pandas as pd

from utils import SEED, load_train, fit_predict

DATA_DIR = "data" if os.path.isdir("data") else "."
MY_RESULTS_FILE = "results_knn.json"  # k-NN only
TEAM_RESULTS_FILE = "results.json"  # all four models (when the team has combined them)

start_time = time.time()
random.seed(SEED)
np.random.seed(SEED)
names = {
    "knn": "k-NN",
    "dt": "Decision Tree",
    "rf": "Random Forest",
    "nn": "Neural Network",
}

SUPPORTED_MODELS = ["knn"]

results_file = (
    TEAM_RESULTS_FILE if os.path.exists(TEAM_RESULTS_FILE) else MY_RESULTS_FILE
)
print("using results from", results_file)
with open(results_file) as f:
    results = json.load(f)

# Only models that fit_predict can run are allowed; results of other models are skipped.
for task in ("image", "sentiment"):
    if not any(m in results.get(task, {}) for m in SUPPORTED_MODELS):
        raise ValueError(
            f"{results_file} has no result for {task} from a supported model. Run run_all.py first."
        )
skipped = sorted(
    {
        m
        for task in ("image", "sentiment")
        for m in results[task]
        if m not in SUPPORTED_MODELS
    }
)
if skipped:
    print(
        "WARNING: skipping",
        ", ".join(names.get(m, m) for m in skipped),
        "- fit_predict in utils.py cannot run them yet.",
    )

parts = []
for task, train_file, test_file in [
    ("image", "image_train.csv", "image_test.csv"),
    ("sentiment", "sentiment_train.csv", "sentiment_test.csv"),
]:
    X, y, feature_cols = load_train(
        os.path.join(DATA_DIR, train_file)
    )  # ALL the training rows
    test = pd.read_csv(os.path.join(DATA_DIR, test_file))
    ids = test["id"].values
    X_test = test[feature_cols].values.astype(np.float32)

    candidates = [m for m in results[task] if m in SUPPORTED_MODELS]
    best_name = max(
        candidates, key=lambda n: results[task][n]["acc"]
    )  # best on validation
    params = results[task][best_name]["params"]
    print(
        f"{task}: using {names[best_name]} {params} (validation acc {results[task][best_name]['acc']:.3f})"
    )

    preds, _ = fit_predict(best_name, params, X, y, X_test, int(y.max()) + 1)
    print(
        f"   retrained on all {len(X)} training rows and predicted {len(X_test)} test rows"
    )
    parts.append(pd.DataFrame({"id": ids, "label": preds.astype(int)}))

# match predictions to the template by ID (the template rows are in a different order)
template = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
predictions = pd.concat(parts, ignore_index=True)
assert set(predictions["id"]) == set(
    template["id"]
), "test IDs must match sample_submission.csv exactly"
sub = template[["id"]].merge(predictions, on="id", how="left", validate="one_to_one")

# validate exactly what the assignment requires
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
