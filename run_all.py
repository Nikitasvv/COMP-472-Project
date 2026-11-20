# Runs my experiments: k-NN, error analysis, k-Means. python run_all.py
import time

from experiment_helpers import setup, load_data, new_state, save_state
from test_knn import run_knn
from test_kmeans import run_kmeans
from error_analysis import run_error_analysis

program_start_time = time.time()
section_times = {}

setup()
start_time = time.time()
data = load_data()
section_times["load data + split"] = time.time() - start_time

state = new_state()  # fresh results

start_time = time.time()
run_knn(data, state)
section_times["k-NN"] = time.time() - start_time

start_time = time.time()
run_error_analysis(data, state["best"])
section_times["Error analysis"] = time.time() - start_time

start_time = time.time()
run_kmeans(data)
section_times["Clustering (k-Means)"] = time.time() - start_time

save_state(state)

print("\n##### Running time #####")
for section_name, section_seconds in section_times.items():
    print(f"{section_name:24s} {section_seconds:7.1f}s")
print(f"{'TOTAL':24s} {time.time() - program_start_time:7.1f}s")
