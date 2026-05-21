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
