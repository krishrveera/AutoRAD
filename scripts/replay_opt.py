"""
Offline replay of a Bayesian-optimization run from logged history.

Re-fits the GP surrogate step by step over a hard-coded historical dataset
and renders a 3-panel diagnostic plot per iteration (noisy surrogate,
idealized noiseless interpolation, and acquisition function). Lets you
visualize *why* the optimizer made each choice without re-running the
expensive human-in-the-loop driving. Output goes to ``results/replays``.
"""

import os
import io
import torch
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

# ==========================================
# 1. THE 3-PANEL PLOTTING FUNCTION
# ==========================================
def plot_bo_step(model, acq_func, train_X, train_Y, bounds, step_name, output_dir, next_X=None):
    """
    Creates a 3-panel 1D plot:
    1. Actual Noisy Surrogate
    2. Forced Noiseless Surrogate (Oval interpolation)
    3. Acquisition Function
    """
    X_test = torch.linspace(bounds[0, 0].item(), bounds[1, 0].item(), 200, dtype=torch.double).view(-1, 1)
    
    # --- 1. Predictions from the ACTUAL (noisy) model ---
    with torch.no_grad():
        posterior = model.posterior(X_test)
        mean = posterior.mean.squeeze(-1)
        lower, upper = posterior.mvn.confidence_region()
        
        X_test_q = X_test.unsqueeze(1)
        acq_vals = acq_func(X_test_q)

    # --- 2. Train a NOISELESS model purely for the "Oval" visualization ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vis_model = SingleTaskGP(
            train_X, 
            train_Y,
            input_transform=Normalize(d=train_X.shape[-1], bounds=bounds),
            outcome_transform=Standardize(m=train_Y.shape[-1])
        )
        vis_model.likelihood.noise_covar.noise = 1e-4
        vis_model.likelihood.noise_covar.raw_noise.requires_grad_(False)
        
        try:
            mll_vis = ExactMarginalLogLikelihood(vis_model.likelihood, vis_model)
            fit_gpytorch_mll(mll_vis)
        except Exception:
            pass 
            
        with torch.no_grad():
            posterior_vis = vis_model.posterior(X_test)
            mean_vis = posterior_vis.mean.squeeze(-1)
            lower_vis, upper_vis = posterior_vis.mvn.confidence_region()

    # --- 3. Create the 3-Panel Figure ---
    fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Panel 1: Actual GP
    ax[0].plot(X_test.numpy(), mean.numpy(), color='blue', label="GP Mean (Actual Model)")
    ax[0].fill_between(X_test.numpy().flatten(), lower.numpy(), upper.numpy(), alpha=0.2, color='blue', label="95% CI (With Noise)")
    ax[0].scatter(train_X.numpy(), train_Y.numpy(), color='black', s=50, zorder=5, label="Evaluated Points")
    if next_X is not None:
        ax[0].axvline(next_X.item(), color='red', linestyle='--', linewidth=2, label="Next Suggestion")
    ax[0].set_ylabel("Score (-Log(Time))")
    ax[0].set_title(f"Step: {step_name} | Actual Surrogate (Handles Noise)")
    ax[0].legend(loc="upper left")
    ax[0].grid(True, linestyle='--', alpha=0.5)

    # Panel 2: Interpolated GP
    ax[1].plot(X_test.numpy(), mean_vis.numpy(), color='purple', label="GP Mean (Interpolated)")
    ax[1].fill_between(X_test.numpy().flatten(), lower_vis.numpy(), upper_vis.numpy(), alpha=0.2, color='purple', label="95% CI (Zero Noise)")
    ax[1].scatter(train_X.numpy(), train_Y.numpy(), color='black', s=50, zorder=5)
    if next_X is not None:
        ax[1].axvline(next_X.item(), color='red', linestyle='--', linewidth=2)
    ax[1].set_ylabel("Score (-Log(Time))")
    ax[1].set_title(f"Step: {step_name} | Idealized Interpolation (Uncertainty = 0 at points)")
    ax[1].legend(loc="upper left")
    ax[1].grid(True, linestyle='--', alpha=0.5)

    # Panel 3: Acquisition Function
    ax[2].plot(X_test.numpy(), acq_vals.numpy(), color='green', linewidth=2, label="qNoisyExpectedImprovement")
    if next_X is not None:
        ax[2].axvline(next_X.item(), color='red', linestyle='--', linewidth=2)
    ax[2].set_xlabel("Warp Factor")
    ax[2].set_ylabel("Acquisition Value")
    ax[2].legend(loc="upper left")
    ax[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"replay_{step_name}.png"))
    plt.close()

# ==========================================
# 2. THE REPLAY LOGIC
# ==========================================
def replay_optimization_history():
    """Re-run the BO surrogate over logged history, plotting each step.

    Loads the embedded historical CSV, seeds the training set with the
    exploration + baseline rows, then walks through the optimization rows one
    at a time — re-fitting the GP and saving a 3-panel plot for each — to
    reconstruct the optimizer's decision process post-hoc.
    """
    print("🎬 Starting BO Replay Generator...")
    output_dir = "results/replays/v1"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load the Historical Data
    csv_data = """phase,iteration,warp_factor,time_seconds,score_log
Exploration,1,3.0000005,138.3916,-4.9301
Exploration,2,4.6022181,160.8277,-5.0803
Exploration,3,2.5430852,104.2480,-4.6468
Exploration,4,0.7979392,76.8029,-4.3412
Exploration,5,3.2540268,128.6362,-4.8570
Baseline,6,1.0,195.6117,-5.2761
Optimization,7,0.000001,129.3611,-4.8626
Optimization,8,6.0,360.0000,-5.8861
Optimization,9,0.7072119,80.8368,-4.3924
Optimization,10,1.7190784,84.2908,-4.4343
Optimization,11,2.0734835,194.6505,-5.2712
Optimization,12,1.3816790,106.9929,-4.6728
Optimization,13,0.3841557,88.3473,-4.4813
Optimization,14,1.5841174,181.5393,-5.2015
Optimization,15,1.1671852,73.3543,-4.2953
Optimization,16,0.1898575,91.2131,-4.5132"""

    df = pd.read_csv(io.StringIO(csv_data))
    
    # 2. Define standard bounds for Warp Factor (0.0 to 6.0)
    botorch_bounds = torch.tensor([[1e-6], [6.0]], dtype=torch.double)
    
    # 3. Extract Initial Setup (Exploration + Baseline)
    init_df = df[df['phase'].isin(['Exploration', 'Baseline'])]
    train_X = torch.tensor(init_df['warp_factor'].values, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.tensor(init_df['score_log'].values, dtype=torch.double).unsqueeze(-1)
    
    # 4. Step through the Optimization Phase
    opt_df = df[df['phase'] == 'Optimization']
    
    for i, row in enumerate(opt_df.itertuples(), start=1):
        print(f"Generating Plot for Step {i}/10 (Historical Row {row.iteration})...")
        
        # Step A: Train the Surrogate Model on the CURRENT historical data state
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SingleTaskGP(
                train_X, 
                train_Y,
                input_transform=Normalize(d=train_X.shape[-1], bounds=botorch_bounds),
                outcome_transform=Standardize(m=train_Y.shape[-1])
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            # Step B: Define Acquisition Function
            acq_func = qNoisyExpectedImprovement(model=model, X_baseline=train_X)
        
        # NOTE: Instead of running optimize_acqf() to find a new candidate, 
        # we extract the ACTUAL candidate the optimizer picked historically.
        historical_next_X = torch.tensor([[row.warp_factor]], dtype=torch.double)
        historical_score = torch.tensor([[row.score_log]], dtype=torch.double)
        
        # Step C: Plot the state BEFORE evaluating the candidate
        plot_bo_step(model, acq_func, train_X, train_Y, botorch_bounds, f"Iteration_{i}", output_dir, next_X=historical_next_X[0])
        
        # Step D: Update the training data to include this row, prepping for the next loop
        train_X = torch.cat([train_X, historical_next_X])
        train_Y = torch.cat([train_Y, historical_score])
        
    print(f"✅ Replay complete! All 10 plots saved to '{output_dir}/'")

if __name__ == "__main__":
    replay_optimization_history()