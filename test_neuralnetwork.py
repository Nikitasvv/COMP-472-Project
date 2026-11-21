"""Run the original 100-epoch PyTorch neural network on both datasets.

Usage: python -u test_neuralnetwork.py
This module is imported by run_all.py as well as run independently.
"""
import time

from experiment_helpers import (
    TASKS, setup, load_data, load_state, save_state,
    clear_model_results, record,
)
from neural_network import (
    HIDDEN_SIZE, LEARNING_RATE, EPOCHS, BATCH_SIZE,
    train_validation_network,
)

# Exact same network as the original standalone neural_network.py.
NN_PARAMS = {
    "hidden_size": HIDDEN_SIZE,
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "scaling": "standardize",
}


def run_nn(data, state, params=None):
    """Train and score PyTorch using the project's shared validation split."""
    params = {**NN_PARAMS, **(params or {})}
    clear_model_results(state, "nn")
    task_accuracies = {}

    for task in TASKS:
        dataset = data[task]
        print(f"\n===== NEURAL NETWORK: {task.upper()} =====", flush=True)
        print(f"Hyperparameters: {params}", flush=True)
        start = time.perf_counter()

        model, score, predictions = train_validation_network(
            params,
            dataset["train_features"], dataset["train_labels"],
            dataset["val_features"], dataset["val_labels"],
            title=f"{task.capitalize()} Neural Network",
        )
        record(state["best"], task, "nn", params.copy(), score, predictions)
        task_accuracies[task] = score * 100

        print(
            f"BEST neural network ({task}): "
            f"Accuracy: {score * 100:.2f}% | "
            f"Time: {time.perf_counter() - start:.1f}s",
            flush=True,
        )

    print("\n================================", flush=True)
    print("NEURAL NETWORK RESULTS", flush=True)
    print("================================", flush=True)
    for task in TASKS:
        print(f"{task.capitalize()} Accuracy: {task_accuracies[task]:.2f}%", flush=True)
    average_accuracy = sum(task_accuracies.values()) / len(task_accuracies)
    print(f"Average Accuracy: {average_accuracy:.2f}%", flush=True)
    print("Results will be saved in results.json and experiment_state.pkl", flush=True)


if __name__ == "__main__":
    setup()
    data = load_data()
    state = load_state()
    start = time.perf_counter()
    run_nn(data, state)
    save_state(state)
    print(f"\nNeural network finished in {time.perf_counter() - start:.1f}s", flush=True)
