"""
Noise-characterization experiment for human-driven laps.

Repeatedly drives the same set of ``warp_factor`` values (shuffled per round)
and records every lap time to a CSV. The resulting spread captures how much
run-to-run variance comes from human inconsistency, which informs the noisy
acquisition function used in :mod:`optimize`. Run :mod:`scripts.analyze_noise`
afterwards to visualize the collected data.
"""

import time
import random
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
import pandas as pd
import os
import time
import random
from game.app import RacingApp

# --- Your existing RacingApp/run_app logic ---
def run_app_wrapper(result_queue, kwargs_dict):
    """Standalone function to run the app and put results in a queue."""
    try:
        app = RacingApp(**kwargs_dict)
        start_time = time.perf_counter()
        app.run()
        duration = time.perf_counter() - start_time
        result_queue.put(duration)
    except Exception as e:
        result_queue.put(e)

# 2. Update evaluate_single_run
def evaluate_single_run(timeout_seconds=300, **kwargs):
    """Run one simulation lap in a subprocess and return its duration.

    Spawns the app in a separate process (so a CARLA/PyGame crash can't take
    down the experiment), waits up to ``timeout_seconds``, and returns the
    measured wall-clock lap time. A timed-out run returns a fixed penalty of
    ``timeout_seconds + 60``; a crashed run re-raises the exception.

    Parameters
    ----------
    timeout_seconds : int, optional
        Maximum time to allow the lap before terminating it.
    **kwargs
        Parameters forwarded to :class:`~game.app.RacingApp` (e.g.
        ``warp_factor``).

    Returns
    -------
    float
        Lap duration in seconds (or the timeout penalty).
    """
    result_queue = multiprocessing.Queue()
    
    # Now we pass the global function 'run_app_wrapper'
    p = multiprocessing.Process(
        target=run_app_wrapper, 
        args=(result_queue, kwargs)
    )
    p.start()
    p.join(timeout_seconds)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return timeout_seconds + 60 
    
    result = result_queue.get()
    if isinstance(result, Exception):
        raise result
    return result

# --- The Execution Script ---=
def save_to_csv(data_row, filename="racing_results.csv"):
    """Appends a single result row to a CSV file."""
    df = pd.DataFrame([data_row])
    # Create file with header if it doesn't exist, otherwise append without header
    header = not os.path.exists(filename)
    df.to_csv(filename, mode='a', index=False, header=header)

def run_full_experiment():
    """Drive every warp factor across multiple shuffled rounds, logging to CSV.

    For each of ``n_replays`` rounds the warp factors are shuffled and driven
    in turn (the shuffling reduces order/learning bias). Every result is
    appended immediately to ``racing_results.csv`` so partial progress is
    never lost. This data quantifies the human-driving noise that the
    Bayesian optimizer must contend with.
    """
    warp_factors = [0.5, 1, 3, 7]
    n_replays = 8
    
    print("Starting experiment. Results will be saved to 'racing_results.csv'")
    
    for round_num in range(1, n_replays + 1):
        print(f"\n--- Starting Round {round_num}/{n_replays} ---")
        
        # Shuffle settings each round
        current_order = list(warp_factors)
        random.shuffle(current_order)
        
        for wf in current_order:
            print(f"\n>>> Testing warp_factor: {wf}")
            
            # Execute run
            duration = evaluate_single_run(warp_factor=wf)
            print(f"Result: {duration:.2f}s")
            
            # Save the row immediately
            row = {
                "iteration": round_num,
                "warp_factor": wf,
                "time_seconds": duration
            }
            save_to_csv(row)
            
            # Pause for user
            input("Press ENTER to continue to the next run...")
            
    print("\nExperiment complete. All results saved.")

# --- Run it ---
if __name__ == "__main__":
    # 1. Run the experiment
    run_full_experiment()
    
    # 2. Analyze from the CSV (which is now the source of truth)
    print("\n--- Summary ---")
    df = pd.read_csv("results/noise/racing_results.csv")
    
    # Group by warp_factor and calculate stats for the summary printout
    summary = df.groupby('warp_factor')['time_seconds'].agg(['mean', 'count'])
    for wf, row in summary.iterrows():
        print(f"Warp Factor {wf}: Average {row['mean']:.2f}s (Runs: {int(row['count'])})")
