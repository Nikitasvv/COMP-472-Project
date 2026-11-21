"""Run the PyTorch neural network on both datasets and record its validation accuracy.

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

# Same settings as the standalone neural_network.py
NN_PARAMS = {
    "hidden_size": HIDDEN_SIZE,
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "scaling": "standardize",
}


NN_LEARNING_RATES = (0.01, 0.001, 0.0001)  # learning rates compared for each task


def run_nn(data, state, params=None, learning_rates=NN_LEARNING_RATES):
    """Train the network once per learning rate on the shared training split and score it
    on the validation split. The most accurate run of each task is kept."""
    params = {**NN_PARAMS, **(params or {})}
    clear_model_results(state, "nn")
    task_accuracies = {}

    for task in TASKS:
        dataset = data[task]
        for learning_rate in learning_rates:
            run_params = {**params, "learning_rate": learning_rate}
            print(f"\n===== NEURAL NETWORK: {task.upper()} (learning rate {learning_rate}) =====", flush=True)
            print(f"Hyperparameters: {run_params}", flush=True)
            start = time.perf_counter()

            model, score, predictions = train_validation_network(
                run_params,
                dataset["train_features"], dataset["train_labels"],
                dataset["val_features"], dataset["val_labels"],
                title=f"{task.capitalize()} Neural Network",
            )
            record(state["best"], task, "nn", run_params, score, predictions)

            print(
                f"neural network ({task}, lr={learning_rate}): "
                f"Accuracy: {score * 100:.2f}% | "
                f"Time: {time.perf_counter() - start:.1f}s",
                flush=True,
            )
        task_accuracies[task] = state["best"][task]["nn"]["acc"] * 100

    print("\n================================", flush=True)
    print("NEURAL NETWORK RESULTS (best learning rate per task)", flush=True)
    print("================================", flush=True)
    for task in TASKS:
        best_lr = state["best"][task]["nn"]["params"]["learning_rate"]
        print(f"{task.capitalize()} Accuracy: {task_accuracies[task]:.2f}% (learning rate {best_lr})", flush=True)
    average_accuracy = sum(task_accuracies.values()) / len(task_accuracies)
    print(f"Average Accuracy: {average_accuracy:.2f}%", flush=True)
    print("Results will be saved in results.json and experiment_state.pkl", flush=True)


if __name__ == "__main__":
    setup()
    data = load_data()
    state = load_state()  # keep results of the other models that were already trained
    start = time.perf_counter()
    run_nn(data, state)
    save_state(state)
    print(f"\nNeural network finished in {time.perf_counter() - start:.1f}s", flush=True)
