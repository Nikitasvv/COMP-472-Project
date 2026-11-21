"""Run k-NN, the PyTorch neural network, the decision tree, the random forest, k-Means
and the k-NN error analysis, in that order.

Run everything:        python -u run_all.py
Just k-NN:             python -u run_all.py --only knn
Just neural networks:  python -u run_all.py --only nn
Just decision tree:    python -u run_all.py --only dt --dt-max-depth 12
Just random forest:    python -u run_all.py --only rf --rf-n-trees 10
Results are printed immediately and also written to run_all_output.txt.
"""
import argparse
import contextlib
import os
from pathlib import Path
import sys
import time

from experiment_helpers import (
    TASKS, MODEL_NAMES, PROJECT_DIR, setup, load_data, load_state, new_state, save_state,
)
from test_knn import run_knn
from test_neuralnetwork import run_nn, NN_PARAMS, NN_LEARNING_RATES
from test_decisiontree import run_dt, DT_PARAMS, DT_DEPTHS
from test_randomforest import run_rf, RF_PARAMS
from test_kmeans import run_kmeans
from error_analysis import run_error_analysis

# Experiments run in this order
RUN_ORDER = ("knn", "nn", "dt", "rf", "kmeans")


class Tee:
    """File-like object that sends everything printed to several streams (terminal and log file)."""
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


def print_results_table(best):
    """Validation accuracy of every model on both tasks, in the layout the project report asks for."""
    print(f"\n{'Method':<16}{'Image Val Acc':>16}{'Sentiment Val Acc':>20}", flush=True)
    for model_key in ("knn", "dt", "rf", "nn"):
        cells = []
        for task in TASKS:
            result = best.get(task, {}).get(model_key)
            cells.append(f"{result['acc'] * 100:.2f}%" if result else "-")
        print(f"{MODEL_NAMES[model_key]:<16}{cells[0]:>16}{cells[1]:>20}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", nargs="+", choices=RUN_ORDER,
        help="Run only the selected models (example: --only nn).",
    )
    parser.add_argument("--nn-epochs", type=int, default=NN_PARAMS["epochs"],
                        help="Override epochs for testing (default: 100).")
    parser.add_argument("--nn-lr", type=float, default=None,
                        help=f"Train with only this learning rate (default: sweep {NN_LEARNING_RATES}).")
    parser.add_argument("--dt-max-depth", type=int, default=None,
                        help=f"Test only this tree depth (default: sweep {DT_DEPTHS}).")
    parser.add_argument("--dt-min-samples-split", type=int, default=DT_PARAMS["min_samples_split"])
    parser.add_argument("--rf-n-trees", type=int, default=RF_PARAMS["n_trees"],
                        help="Size of the largest forest; smaller sizes (10, 50, ...) are scored too (default: 100).")
    parser.add_argument("--rf-max-depth", type=int, default=RF_PARAMS["max_depth"],
                        help="Cap forest tree depth for faster experiments.")
    parser.add_argument("--rf-min-samples-split", type=int, default=RF_PARAMS["min_samples_split"])
    args = parser.parse_args(argv)
    if args.rf_n_trees < 1 or args.nn_epochs < 1:
        parser.error("--rf-n-trees and --nn-epochs must be positive")
    if args.nn_lr is not None and args.nn_lr <= 0:
        parser.error("--nn-lr must be positive")
    if args.rf_min_samples_split < 2 or args.dt_min_samples_split < 2:
        parser.error("--rf-min-samples-split and --dt-min-samples-split must be >= 2")
    if args.rf_max_depth is not None and args.rf_max_depth < 1:
        parser.error("--rf-max-depth must be positive")
    if args.dt_max_depth is not None and args.dt_max_depth < 1:
        parser.error("--dt-max-depth must be positive")
    selections = args.only or list(RUN_ORDER)

    # Run from the project folder so relative paths (figures/, kmeans_results.csv) work from anywhere
    os.chdir(PROJECT_DIR)
    log_path = Path(PROJECT_DIR) / "run_all_output.txt"
    with log_path.open("w", encoding="utf-8", buffering=1) as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout, logfile)):
            program_start = time.perf_counter()
            durations = {}
            try:
                print("Starting experiments:", ", ".join(k for k in RUN_ORDER if k in selections), flush=True)
                setup()
                step_start = time.perf_counter()
                data = load_data()
                durations["Load data + split"] = time.perf_counter() - step_start

                # A full run starts fresh; a partial run keeps the results of models it does not rerun
                state = new_state() if args.only is None else load_state()
                for model_key, label, run_function in [
                    ("knn", "k-NN", run_knn),
                    ("nn", "Neural Network", run_nn),
                    ("dt", "Decision Tree", run_dt),
                    ("rf", "Random Forest", run_rf),
                    ("kmeans", "Clustering (k-Means)", run_kmeans),
                ]:
                    if model_key not in selections:
                        continue
                    print(f"\n{'=' * 12} STARTING {label.upper()} {'=' * 12}", flush=True)
                    step_start = time.perf_counter()
                    if model_key == "kmeans":
                        run_function(data)  # clustering is not a validated model, so nothing goes into the state
                    else:
                        if model_key == "knn":
                            run_function(data, state)
                        elif model_key == "nn":
                            run_function(
                                data, state, {"epochs": args.nn_epochs},
                                learning_rates=NN_LEARNING_RATES if args.nn_lr is None else (args.nn_lr,),
                            )
                        elif model_key == "dt":
                            run_function(
                                data, state, {"min_samples_split": args.dt_min_samples_split},
                                depths=DT_DEPTHS if args.dt_max_depth is None else (args.dt_max_depth,),
                            )
                        else:
                            run_function(data, state, {
                                "n_trees": args.rf_n_trees,
                                "max_depth": args.rf_max_depth,
                                "min_samples_split": args.rf_min_samples_split,
                            })
                        save_state(state)  # keep finished experiments even if a later step fails
                    durations[label] = time.perf_counter() - step_start
                    print(f"COMPLETED {label} in {durations[label]:.1f}s", flush=True)

                # The error analysis reads saved k-NN predictions, so run it whenever they exist
                if all("knn" in state["best"].get(task, {}) for task in TASKS):
                    print("\n=========== ERROR ANALYSIS ===========", flush=True)
                    step_start = time.perf_counter()
                    run_error_analysis(data, state["best"])
                    durations["Error analysis"] = time.perf_counter() - step_start

                save_state(state)
                print("\n=========== FINAL VALIDATION RESULTS ===========", flush=True)
                for task in TASKS:
                    print(f"\n{task.upper()}", flush=True)
                    available = state["best"].get(task, {})
                    for model_key in ("knn", "nn", "dt", "rf"):
                        result = available.get(model_key)
                        if result is not None:
                            print(
                                f"  {model_key:4s} | accuracy={result['acc']:.4f} "
                                f"| parameters={result['params']}", flush=True,
                            )
                    candidates = [key for key in ("knn", "nn", "dt", "rf") if key in available]
                    if candidates:
                        winner = max(candidates, key=lambda key: available[key]["acc"])
                        print(f"  BEST FOR SUBMISSION: {winner} "
                              f"({available[winner]['acc']:.4f})", flush=True)
                print_results_table(state["best"])
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
