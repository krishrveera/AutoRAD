"""
Human-in-the-loop Bayesian Optimization driver for the AutoRAD simulator.

This script tunes the audio ``warp_factor`` (and any other parameters listed
in the YAML config) by repeatedly asking a human to drive a lap in the CARLA
racing app and treating the measured lap time as the objective to minimize.

The optimization proceeds in phases:

1. **Exploration** – one center point plus Sobol-sampled points.
2. **Baseline** – the config's default parameter values.
3. **Bayesian Optimization** – a :class:`~botorch.models.SingleTaskGP` surrogate
   with a ``qNoisyExpectedImprovement`` acquisition function proposes the next
   parameter set each iteration.
4. **Reporting** – best parameters, JSON metrics, a step-by-step CSV history,
   and per-iteration surrogate plots are written to ``results/optimization``.

Lap time is log-transformed and negated so the optimizer can *maximize* a
well-behaved score while we conceptually *minimize* time.
"""

import torch
import math
import multiprocessing
import time
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from torch.quasirandom import SobolEngine

from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from game.app import RacingApp
from opt.extract_params import read_config_file, build_parameter_space

# ==========================================
# 1. PARAMETER EXTRACTION
# ==========================================
def extract_parameters_from_config(config_file):
    """Read a YAML config file and build the optimizable parameter space.

    Parameters
    ----------
    config_file : str | pathlib.Path
        Path to the YAML config (e.g. ``config/minimal.yml``) describing each
        parameter's default value and bound interval.

    Returns
    -------
    opt.params.ParameterSpace
        The parameter space, exposing BoTorch-compatible bounds and
        tensor/kwargs conversion helpers.
    """
    config_dict = read_config_file(config_file)
    parameter_space = build_parameter_space(config_dict)
    return parameter_space

# ==========================================
# 2. TOP-LEVEL WRAPPER FOR MULTIPROCESSING
# ==========================================
def run_app_wrapper(result_queue, kwargs_dict):
    """
    Top-level function to run the app and put results in a queue.
    This avoids the multiprocessing 'pickling' errors on Windows.
    """
    try:
        app = RacingApp(**kwargs_dict)
        start_time = time.perf_counter()
        app.run()
        duration = time.perf_counter() - start_time
        result_queue.put(duration)
    except Exception as e:
        result_queue.put(e)

def evaluate_my_program(timeout_seconds=300, **kwargs):
    """Evaluates the simulation and returns a Log-Transformed, Negated score."""
    print(f"    Evaluating with parameters: {kwargs}")
    result_queue = multiprocessing.Queue()
    
    p = multiprocessing.Process(
        target=run_app_wrapper, 
        args=(result_queue, kwargs)
    )
    p.start()
    p.join(timeout_seconds)
    
    # Check for timeout
    if p.is_alive():
        print(f"    [WARNING] Timeout reached ({timeout_seconds}s). Terminating process!")
        p.terminate() 
        p.join() 
        duration = timeout_seconds + 60.0 # Standardized penalty
    else:
        # Retrieve successful result
        result = result_queue.get()
        if isinstance(result, Exception):
            print(f"    [ERROR] Simulation crashed: {result}")
            duration = timeout_seconds + 60.0
        else:
            duration = result
            print(f"    [Success] Ran in {duration:.4f} seconds")

    # Apply the Log-Transform and Negate for minimization
    log_duration = math.log(duration)
    score = torch.tensor([[-log_duration]], dtype=torch.double)
    return score

# ==========================================
# 3. SAMPLING & BO LOGIC
# ==========================================
def initial_sampling(parameter_space, botorch_bounds, N_init_samples):
    """Samples 1 Center Point, then (N-1) Sobol points."""
    dim = botorch_bounds.shape[1]
    lower = botorch_bounds[0]
    range_sizes = botorch_bounds[1] - botorch_bounds[0]
    
    X_list = []
    
    # 1. The Center Point
    center_point = lower + (range_sizes / 2.0)
    X_list.append(center_point.unsqueeze(0))
    
    # 2. Sobol Sequence for the remainder
    if N_init_samples > 1:
        sobol = SobolEngine(dimension=dim, scramble=True)
        sobol_raw = sobol.draw(N_init_samples - 1).to(dtype=torch.double)
        sobol_scaled = lower + range_sizes * sobol_raw
        X_list.append(sobol_scaled)
        
    train_X = torch.cat(X_list)
    train_Y_list = []
    
    for i in range(N_init_samples):
        kwargs = parameter_space.tensor_to_kwargs(train_X[i])
        print(f"\n[Random Run {i+1}/{N_init_samples}]")
        input("🟢 Press ENTER when ready to drive...")

        score = evaluate_my_program(**kwargs)
        train_Y_list.append(score)
    
    train_Y = torch.cat(train_Y_list) 
    return train_X, train_Y

def plot_bo_step(model, acq_func, train_X, train_Y, bounds, step_name, output_dir, next_X=None):
    """
    Creates a 1D plot of the Surrogate Model and Acquisition Function.
    Saves directly to the versioned output directory.
    """
    X_test = torch.linspace(bounds[0, 0].item(), bounds[1, 0].item(), 200, dtype=torch.double).view(-1, 1)
    
    with torch.no_grad():
        posterior = model.posterior(X_test)
        mean = posterior.mean.squeeze(-1)
        lower, upper = posterior.mvn.confidence_region()
        
        X_test_q = X_test.unsqueeze(1)
        acq_vals = acq_func(X_test_q)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Top Plot: Gaussian Process
    ax[0].plot(X_test.numpy(), mean.numpy(), color='blue', label="GP Mean (Predicted Score)")
    ax[0].fill_between(X_test.numpy().flatten(), lower.numpy(), upper.numpy(), alpha=0.2, color='blue', label="95% Confidence Interval")
    ax[0].scatter(train_X.numpy(), train_Y.numpy(), color='black', s=50, zorder=5, label="Evaluated Points")
    
    if next_X is not None:
        ax[0].axvline(next_X.item(), color='red', linestyle='--', linewidth=2, label="Next Suggestion")
        
    ax[0].set_ylabel("Objective Score (-Log(Time))")
    ax[0].set_title(f"Step: {step_name} | Gaussian Process Surrogate")
    ax[0].legend(loc="upper left")
    ax[0].grid(True, linestyle='--', alpha=0.5)

    # Bottom Plot: Acquisition Function
    ax[1].plot(X_test.numpy(), acq_vals.numpy(), color='green', linewidth=2, label="qNoisyExpectedImprovement")
    if next_X is not None:
        ax[1].axvline(next_X.item(), color='red', linestyle='--', linewidth=2)
        
    ax[1].set_xlabel("Warp Factor")
    ax[1].set_ylabel("Acquisition Value")
    ax[1].legend(loc="upper left")
    ax[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"bo_step_{step_name}.png"))
    plt.close()

def bayesian_optimization_loop(train_X, train_Y, parameter_space, botorch_bounds, num_iterations, output_dir):
    """Run the main BO loop: fit a GP, propose a point, evaluate it, repeat.

    Each iteration fits a :class:`SingleTaskGP` to the data gathered so far,
    builds a ``qNoisyExpectedImprovement`` acquisition function, optimizes it
    to suggest the next parameter set, saves a diagnostic plot, prompts the
    human to drive that configuration, and appends the result to the training
    data.

    Parameters
    ----------
    train_X : torch.Tensor
        Already-evaluated parameter points, shape ``(n, d)``.
    train_Y : torch.Tensor
        Corresponding objective scores, shape ``(n, 1)``.
    parameter_space : opt.params.ParameterSpace
        Used to convert candidate tensors back into ``RacingApp`` kwargs.
    botorch_bounds : torch.Tensor
        ``(2, d)`` lower/upper bounds for the search space.
    num_iterations : int
        Number of BO iterations to run.
    output_dir : str
        Directory where per-iteration surrogate plots are saved.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The augmented ``(train_X, train_Y)`` including every BO evaluation.
    """
    for iteration in range(num_iterations):
        print(f"\n[Bayesian Opt Run {iteration + 1}/{num_iterations}]")
        
        model = SingleTaskGP(
            train_X, 
            train_Y,
            input_transform=Normalize(d=train_X.shape[-1], bounds=botorch_bounds),
            outcome_transform=Standardize(m=train_Y.shape[-1])
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        
        acq_func = qNoisyExpectedImprovement(
            model=model, 
            X_baseline=train_X
        )
        
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=botorch_bounds, 
            q=1, 
            num_restarts=5,
            raw_samples=20
        )
        
        # The surrogate/acquisition plot is 1-D only; posterior evaluation on a
        # linspace grid crashes for multi-dimensional search spaces.
        if train_X.shape[-1] == 1:
            plot_bo_step(model, acq_func, train_X, train_Y, botorch_bounds, f"Iteration_{iteration+1}", output_dir, next_X=candidates[0])
        
        next_kwargs = parameter_space.tensor_to_kwargs(candidates)
        print(f"BoTorch Suggests: {next_kwargs}")
        
        input("🟢 Program is ready. Press ENTER when you are ready to drive...")
        new_result = evaluate_my_program(**next_kwargs)
        print(f"    [Result] Score: {new_result.item():.4f}")
        
        train_X = torch.cat([train_X, candidates])
        train_Y = torch.cat([train_Y, new_result])
        
    return train_X, train_Y

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main(config_file, version="v0", n_init_samples=3, num_bo_iterations=8):
    """Orchestrate a full optimization run end to end.

    Runs all four phases (exploration, default baseline, BO loop, reporting),
    then writes ``optimization_results.json`` and ``optimization_history.csv``
    plus per-iteration plots under ``results/optimization/<version>/``.

    Parameters
    ----------
    config_file : str
        Path to the YAML parameter config.
    version : str, optional
        Label for the output sub-directory, letting multiple BO setups
        coexist without overwriting each other.
    n_init_samples : int, optional
        Number of initial exploration samples (1 center point + Sobol).
    num_bo_iterations : int, optional
        Number of Bayesian-optimization iterations after exploration.
    """
    print(f"\n🏎️  RACING GAME OPTIMIZER INITIALIZED (Version: {version})  🏎️")
    
    # Setup Output Directory
    output_dir = f"results/optimization/{version}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"All plots and data will be saved to: {output_dir}/")
    
    parameter_space = extract_parameters_from_config(config_file)
    botorch_bounds = parameter_space.bounds 
    
    default_values = [p._current for p in parameter_space.params.values()]
    default_tensor = torch.tensor([default_values], dtype=torch.double)
    default_kwargs = parameter_space.tensor_to_kwargs(default_tensor)

    # ---------------- Phase 1: Exploration ----------------
    print(f"\n--- Phase 1: Center + Sobol Exploration ({n_init_samples} Runs) ---")
    train_X, train_Y = initial_sampling(parameter_space, botorch_bounds, n_init_samples)
    
    # ---------------- Phase 2: Default Baseline -----------
    print("\n--- Phase 2: Baseline Default Run ---")
    print(f"Default Setup: {default_kwargs}")
    input("🟢 Press ENTER when you are ready to drive the Default setup...")
    
    default_score = evaluate_my_program(**default_kwargs)
    train_X = torch.cat([train_X, default_tensor])
    train_Y = torch.cat([train_Y, default_score])

    # ---------------- Phase 3: BO Loop --------------------
    print(f"\n--- Phase 3: Bayesian Optimization ({num_bo_iterations} Runs) ---")
    train_X, train_Y = bayesian_optimization_loop(train_X, train_Y, parameter_space, botorch_bounds, num_bo_iterations, output_dir)

    # ---------------- Phase 4: Reporting & Saving ---------
    print("\n🏁 OPTIMIZATION COMPLETE 🏁")

    # Reverse the Log-Transform: Time = exp(-Score)
    default_time = math.exp(-default_score.item())
    best_time = math.exp(-train_Y.max().item())
    
    best_index = train_Y.argmax().item()
    best_tensor = train_X[best_index].unsqueeze(0)
    best_kwargs = parameter_space.tensor_to_kwargs(best_tensor)
    
    improvement_sec = default_time - best_time
    improvement_pct = (improvement_sec / default_time) * 100

    print("\n--- FINAL REPORT ---")
    print(f"Default Parameters Time: {default_time:.3f} seconds")
    print(f"Best Parameters Time:    {best_time:.3f} seconds")
    print(f"Total Improvement:       {improvement_sec:.3f} seconds ({improvement_pct:.1f}% faster!)")
    print(f"\nThe Ultimate Setup:")
    for key, val in best_kwargs.items():
        print(f"  - {key}: {val:.4f}")

    # Export to JSON
    results_data = {
        "version": version,
        "config_file": config_file,
        "metrics": {
            "default_time_seconds": round(default_time, 4),
            "best_time_seconds": round(best_time, 4),
            "improvement_seconds": round(improvement_sec, 4),
            "improvement_percentage": round(improvement_pct, 4)
        },
        "best_parameters": best_kwargs
    }
    
    results_path = os.path.join(output_dir, "optimization_results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=4)
        
    print(f"\n📁 Metrics successfully saved to: {results_path}")

    # ==========================================
    # Phase 5: Export History to CSV
    # ==========================================
    history_data = []
    
    for i in range(len(train_X)):
        # Extract parameters and score
        run_kwargs = parameter_space.tensor_to_kwargs(train_X[i])
        score = train_Y[i].item()
        actual_time = math.exp(-score)
        
        # Build row
        row = {
            "iteration": i + 1,
            "score_log": round(score, 4),
            "time_seconds": round(actual_time, 4)
        }
        
        # Merge dynamic kwargs into the row
        for key, val in run_kwargs.items():
            row[key] = round(val, 4)
            
        history_data.append(row)

    df_history = pd.DataFrame(history_data)
    csv_path = os.path.join(output_dir, "optimization_history.csv")
    df_history.to_csv(csv_path, index=False)
    
    print(f"📁 Full step-by-step history saved to: {csv_path}")

if __name__ == "__main__":
    # Change version here when testing different BO setups or parameters.
    # Phase 2 (6-D: warp + steering/brake + attentional): config/phase2.yml.
    # Phase 1 (1-D warp_factor only): config/minimal.yml.
    VERSION = "v2"
    CONFIG_FILE = "config/phase2.yml"

    main(CONFIG_FILE, version=VERSION, n_init_samples=5, num_bo_iterations=10)