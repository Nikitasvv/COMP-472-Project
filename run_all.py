"""Run PyTorch NN, Random Forest, k-NN, error analysis and k-Means.

Run everything:       python -u run_all.py
Just neural networks: python -u run_all.py --only nn
Just random forest:   python -u run_all.py --only rf --rf-n-trees 10
Results are printed immediately and saved to run_all_output.txt.
"""
import argparse
import contextlib
import os
from pathlib import Path
import sys
import time

# Save plots to PNGs even if test_knn imports pyplot in other environments.
os.environ.setdefault("MPLBACKEND", "Agg")

from experiment_helpers import (
    TASKS, PROJECT_DIR, setup, load_data, load_state, new_state, save_state,
)
from test_knn import run_knn
from test_nn import run_nn, NN_PARAMS
from test_rf import run_rf, RF_PARAMS
from test_kmeans import run_kmeans
from error_analysis import run_error_analysis


class Tee:
    """Write every printed result both to the terminal and a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", nargs="+", choices=("knn", "nn", "rf", "kmeans"),
        help="Run only the selected models (example: --only nn).",
    )
    parser.add_argument("--rf-n-trees", type=int, default=RF_PARAMS["n_trees"],
                        help="Number of Random Forest trees (default: 300).")
    parser.add_argument("--rf-max-depth", type=int, default=RF_PARAMS["max_depth"],
                        help="Cap forest tree depth for faster experiments.")
    parser.add_argument("--rf-min-samples-split", type=int, default=RF_PARAMS["min_samples_split"])
    parser.add_argument("--nn-epochs", type=int, default=NN_PARAMS["epochs"],
                        help="Override epochs for testing (default: 100).")
    args = parser.parse_args(argv)
    if args.rf_n_trees < 1 or args.nn_epochs < 1 or args.rf_min_samples_split < 2:
        parser.error("--rf-n-trees and --nn-epochs must be positive and --rf-min-samples-split >= 2")
    if args.rf_max_depth is not None and args.rf_max_depth < 1:
        parser.error("--rf-max-depth must be positive")
    selections = args.only or ["nn", "rf", "knn", "kmeans"]

    # Standalone execution works even if launched from another directory.
    os.chdir(PROJECT_DIR)
    log_path = Path(PROJECT_DIR) / "run_all_output.txt"
    with log_path.open("w", encoding="utf-8", buffering=1) as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout, logfile)):
            program_start = time.perf_counter()
            durations = {}
            try:
                print("Starting experiments:", ", ".join(selections), flush=True)
                setup()
                step_start = time.perf_counter()
                data = load_data()
                durations["Load data + split"] = time.perf_counter() - step_start

                # A full run is fresh. Partial runs keep other previously saved models.
                state = new_state() if args.only is None else load_state()
                for model_key, label, run_function in [
                    ("nn", "Neural Network", run_nn),
                    ("rf", "Random Forest", run_rf),
                    ("knn", "k-NN", run_knn),
                    ("kmeans", "Clustering (k-Means)", run_kmeans),
                ]:
                    if model_key not in selections:
                        continue
                    print(f"\n{'=' * 12} STARTING {label.upper()} {'=' * 12}", flush=True)
                    step_start = time.perf_counter()
                    if model_key == "kmeans":
                        run_function(data)
                    else:
                        if model_key == "rf":
                            run_function(data, state, {
                                "n_trees": args.rf_n_trees,
                                "max_depth": args.rf_max_depth,
                                "min_samples_split": args.rf_min_samples_split,
                            })
                        elif model_key == "nn":
                            run_function(data, state, {"epochs": args.nn_epochs})
                        else:
                            run_function(data, state)
                        save_state(state)  # Preserve completed experiments even if a later step fails.
                    durations[label] = time.perf_counter() - step_start
                    print(f"COMPLETED {label} in {durations[label]:.1f}s", flush=True)

                if any(key in selections for key in ("knn", "nn", "rf")):
                    print("\n=========== ERROR ANALYSIS ===========", flush=True)
                    step_start = time.perf_counter()
                    run_error_analysis(data, state["best"])
                    durations["Error analysis"] = time.perf_counter() - step_start

                save_state(state)
                print("\n=========== FINAL VALIDATION RESULTS ===========", flush=True)
                for task in TASKS:
                    print(f"\n{task.upper()}", flush=True)
                    available = state["best"].get(task, {})
                    for model_key in ("nn", "rf", "knn"):
                        result = available.get(model_key)
                        if result is not None:
                            print(
                                f"  {model_key:4s} | accuracy={result['acc']:.4f} "
                                f"| parameters={result['params']}", flush=True,
                            )
                    candidates = [key for key in ("knn", "nn", "rf") if key in available]
                    if candidates:
                        winner = max(candidates, key=lambda key: available[key]["acc"])
                        print(f"  BEST FOR SUBMISSION: {winner} "
                              f"({available[winner]['acc']:.4f})", flush=True)
                print("\n=========== RUNNING TIMES ===========", flush=True)
                for name, seconds in durations.items():
                    print(f"{name:24s} {seconds:7.1f}s", flush=True)
                print(f"{'TOTAL':24s} {time.perf_counter() - program_start:7.1f}s", flush=True)
                print(f"Results saved in results.json; output log: {log_path.name}", flush=True)
            except Exception as exc:
                print(f"\nFAILED: {type(exc).__name__}: {exc}", flush=True)
                print(f"See output above and {log_path.name} for the last completed step.", flush=True)
                raise


if __name__ == "__main__":
    main()
