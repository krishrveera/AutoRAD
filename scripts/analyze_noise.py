import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def analyze_and_plot_all(filename="racing_results.csv", output_dir="results/noise"):
    """
    Reads experiment data and generates both linear and log-scale plots.
    Saves all output to the specified directory.
    """
    if not os.path.exists(filename):
        print(f"Error: Could not find '{filename}'. Make sure the experiment has run.")
        return

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating plots and saving to '{output_dir}/' ...")

    # Load and prepare data
    df = pd.read_csv(filename)
    df = df.rename(columns={'warp_factor': 'Warp Factor', 'time_seconds': 'Time (s)'})
    
    # Apply Log10 transformation
    df['Log Time'] = np.log10(df['Time (s)'])
    
    # Sort Warp Factors to guarantee perfect centering on Categorical X-axes
    sorted_wfs = sorted(df['Warp Factor'].unique())

    # ==========================================
    # 1. NORMAL SCALE PLOTS
    # ==========================================

    # --- Normal Distribution Plot ---
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='Warp Factor', y='Time (s)', order=sorted_wfs, inner=None, color=".9")
    sns.boxplot(data=df, x='Warp Factor', y='Time (s)', order=sorted_wfs, width=0.2, boxprops={'zorder': 2})
    sns.stripplot(data=df, x='Warp Factor', y='Time (s)', order=sorted_wfs, color="red", size=5, alpha=0.6, zorder=3)
    
    plt.title("Execution Time Distribution (Linear Scale)")
    plt.xlabel("Warp Factor")
    plt.ylabel("Time (s)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dist_plot_normal.png"))
    plt.close()

    # --- Normal Trend Plot ---
    plt.figure(figsize=(10, 6))
    stats_norm = df.groupby('Warp Factor')['Time (s)'].agg(['mean', 'std']).reset_index()
    
    plt.plot(stats_norm['Warp Factor'], stats_norm['mean'], marker='o', linewidth=2, label='Mean Time')
    plt.fill_between(stats_norm['Warp Factor'], 
                     stats_norm['mean'] - stats_norm['std'], 
                     stats_norm['mean'] + stats_norm['std'], 
                     alpha=0.2, label='Std Dev (Noise)')
    
    plt.title("Performance Trend: Mean Time with Noise Boundary")
    plt.xlabel("Warp Factor")
    plt.ylabel("Time (s)")
    plt.xlim(0, 8) 
    plt.ylim(0, 420) # Caps just above the 360s penalty
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trend_plot_normal.png"))
    plt.close()

    # ==========================================
    # 2. LOG SCALE PLOTS
    # ==========================================

    # --- Log Distribution Plot ---
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='Warp Factor', y='Log Time', order=sorted_wfs, inner=None, color=".9")
    sns.boxplot(data=df, x='Warp Factor', y='Log Time', order=sorted_wfs, width=0.2, boxprops={'zorder': 2})
    sns.stripplot(data=df, x='Warp Factor', y='Log Time', order=sorted_wfs, color="darkblue", size=5, alpha=0.6, zorder=3)
    
    plt.title("Execution Time Distribution (Log10 Scale)")
    plt.xlabel("Warp Factor")
    plt.ylabel("Log10( Time in Seconds )")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dist_plot_log.png"))
    plt.close()

    # --- Log Trend Plot ---
    plt.figure(figsize=(10, 6))
    stats_log = df.groupby('Warp Factor')['Log Time'].agg(['mean', 'std']).reset_index()
    
    plt.plot(stats_log['Warp Factor'], stats_log['mean'], marker='o', linewidth=2, color="darkblue", label='Mean Log Time')
    plt.fill_between(stats_log['Warp Factor'], 
                     stats_log['mean'] - stats_log['std'], 
                     stats_log['mean'] + stats_log['std'], 
                     alpha=0.2, color="darkblue", label='Std Dev (Noise)')
    
    plt.title("Performance Trend: Log-Transformed with Noise Boundary")
    plt.xlabel("Warp Factor")
    plt.ylabel("Log10( Time in Seconds )")
    plt.xlim(0, 8)
    
    # Dynamically pad the Y-axis based on the data
    min_log = df['Log Time'].min()
    max_log = df['Log Time'].max()
    plt.ylim(min_log - 0.1, max_log + 0.1) 
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trend_plot_log.png"))
    plt.close()

    print("✅ All plots successfully generated!")

if __name__ == "__main__":
    analyze_and_plot_all("results/noise/racing_results.csv")