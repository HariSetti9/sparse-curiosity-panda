"""
Visualization script for generating training curves and trajectory plots.

Generates:
1. 4-panel training curves comparing baseline vs sparse+curiosity
2. 3D trajectory comparison plot
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_training_history(history_path):
    """Load training history from JSON file."""
    with open(history_path, 'r') as f:
        return json.load(f)


def smooth_data(data, window_size=100):
    """Apply moving average smoothing to data."""
    if len(data) < window_size:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data), i + window_size // 2)
        smoothed.append(np.mean(data[start_idx:end_idx]))
    
    return smoothed


def plot_training_curves(baseline_history, sparse_history, output_path='results/training_curves.png'):
    """
    Generate 4-panel training curves figure.
    
    Panels:
    1. Episode rewards over timesteps
    2. Success rate (rolling average)
    3. Intrinsic rewards (sparse+curiosity only)
    4. Cumulative success rate
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    baseline_timesteps = baseline_history['timesteps']
    baseline_rewards = baseline_history['rewards']
    baseline_successes = baseline_history['successes']
    
    sparse_timesteps = sparse_history['timesteps']
    sparse_rewards = sparse_history['rewards']
    sparse_successes = sparse_history['successes']
    sparse_intrinsic = sparse_history['intrinsic_rewards']
    
    # Compute rolling metrics
    window = 5000  # Rolling window in timesteps
    
    # Panel 1: Rewards over timesteps
    ax1 = axes[0, 0]
    baseline_rewards_smooth = smooth_data(baseline_rewards, window_size=100)
    sparse_rewards_smooth = smooth_data(sparse_rewards, window_size=100)
    
    ax1.plot(baseline_timesteps[::100], baseline_rewards_smooth[::100], 
             'b-', label='Baseline (Dense)', alpha=0.7, linewidth=1.5)
    ax1.plot(sparse_timesteps[::100], sparse_rewards_smooth[::100], 
             'r-', label='Sparse + Curiosity', alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Reward (smoothed)')
    ax1.set_title('Episode Rewards Over Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Success rate over training
    ax2 = axes[0, 1]
    
    # Compute rolling success rate
    def compute_rolling_success(successes, window=1000):
        rolling_success = []
        for i in range(len(successes)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(successes), i + window // 2)
            rolling_success.append(np.mean(successes[start_idx:end_idx]))
        return rolling_success
    
    baseline_rolling_success = compute_rolling_success(baseline_successes)
    sparse_rolling_success = compute_rolling_success(sparse_successes)
    
    ax2.plot(baseline_timesteps[::100], baseline_rolling_success[::100], 
             'b-', label='Baseline (Dense)', alpha=0.7, linewidth=1.5)
    ax2.plot(sparse_timesteps[::100], sparse_rolling_success[::100], 
             'r-', label='Sparse + Curiosity', alpha=0.7, linewidth=1.5)
    ax2.axhline(y=0.9, color='g', linestyle='--', label='90% Target', alpha=0.5)
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Success Rate (rolling)')
    ax2.set_title('Success Rate Over Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    # Panel 3: Intrinsic rewards (sparse+curiosity)
    ax3 = axes[1, 0]
    sparse_intrinsic_smooth = smooth_data(sparse_intrinsic, window_size=100)
    
    ax3.plot(sparse_timesteps[::100], sparse_intrinsic_smooth[::100], 
             'orange', label='Intrinsic Reward', alpha=0.7, linewidth=1.5)
    ax3.set_xlabel('Timesteps')
    ax3.set_ylabel('Intrinsic Reward')
    ax3.set_title('Curiosity-Driven Exploration Bonus')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Cumulative success rate
    ax4 = axes[1, 1]
    
    def compute_cumulative_success(successes):
        cumulative = []
        for i in range(1, len(successes) + 1):
            cumulative.append(np.mean(successes[:i]))
        return cumulative
    
    baseline_cumulative = compute_cumulative_success(baseline_successes)
    sparse_cumulative = compute_cumulative_success(sparse_successes)
    
    # Subsample for plotting
    step = max(1, len(baseline_timesteps) // 500)
    ax4.plot(baseline_timesteps[::step], baseline_cumulative[::step], 
             'b-', label='Baseline (Dense)', alpha=0.7, linewidth=1.5)
    ax4.plot(sparse_timesteps[::step], sparse_cumulative[::step], 
             'r-', label='Sparse + Curiosity', alpha=0.7, linewidth=1.5)
    ax4.axhline(y=0.9, color='g', linestyle='--', label='90% Target', alpha=0.5)
    ax4.set_xlabel('Timesteps')
    ax4.set_ylabel('Cumulative Success Rate')
    ax4.set_title('Learning Progress (Cumulative)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {output_path}")


def plot_trajectory_comparison(output_path='results/path_comparison.png'):
    """
    Generate 3D trajectory comparison plot.
    
    Shows sample trajectories from both methods in 3D space.
    """
    # Load trajectories
    traj_path = 'results/trajectories.npz'
    
    if not os.path.exists(traj_path):
        print(f"Warning: Trajectory file not found at {traj_path}")
        print("Please run 'python evaluate.py' first.")
        return
    
    data = np.load(traj_path, allow_pickle=True)
    baseline_trajs = data['baseline_trajectories']
    sparse_trajs = data['sparse_trajectories']
    
    # Select a few representative trajectories
    num_samples = min(10, len(baseline_trajs), len(sparse_trajs))
    
    fig = plt.figure(figsize=(14, 6))
    
    # Plot baseline trajectories
    ax1 = fig.add_subplot(121, projection='3d')
    
    for i in range(num_samples):
        traj = baseline_trajs[i]
        if isinstance(traj, np.ndarray) and len(traj) > 1:
            ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                    alpha=0.5, linewidth=1.5, color='blue')
            
            # Mark start and end points
            ax1.scatter(traj[0, 0], traj[0, 1], traj[0, 2], 
                       c='green', s=50, marker='o', label='Start' if i == 0 else '')
            ax1.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], 
                       c='red', s=50, marker='x', label='End' if i == 0 else '')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Baseline (Dense Rewards)\nSample Trajectories')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot sparse+curiosity trajectories
    ax2 = fig.add_subplot(122, projection='3d')
    
    for i in range(num_samples):
        traj = sparse_trajs[i]
        if isinstance(traj, np.ndarray) and len(traj) > 1:
            ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                    alpha=0.5, linewidth=1.5, color='red')
            
            # Mark start and end points
            ax2.scatter(traj[0, 0], traj[0, 1], traj[0, 2], 
                       c='green', s=50, marker='o', label='Start' if i == 0 else '')
            ax2.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], 
                       c='blue', s=50, marker='x', label='End' if i == 0 else '')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('Sparse + Curiosity\nSample Trajectories')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Trajectory comparison saved to {output_path}")


def plot_path_efficiency_distribution(output_path='results/efficiency_distribution.png'):
    """Plot histogram of path efficiency for both methods."""
    
    # Try to load evaluation results
    baseline_results_path = 'results/baseline_results.json'
    sparse_results_path = 'results/sparse_results.json'
    
    if not os.path.exists(baseline_results_path):
        print(f"Warning: Results files not found. Skipping efficiency distribution plot.")
        return
    
    with open(baseline_results_path, 'r') as f:
        baseline_results = json.load(f)
    
    with open(sparse_results_path, 'r') as f:
        sparse_results = json.load(f)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart with error bars
    methods = ['Baseline\n(Dense)', 'Sparse +\nCuriosity']
    efficiencies = [
        baseline_results['mean_path_efficiency'],
        sparse_results['mean_path_efficiency']
    ]
    errors = [
        baseline_results['efficiency_std'],
        sparse_results['efficiency_std']
    ]
    
    colors = ['steelblue', 'coral']
    
    bars = ax.bar(methods, efficiencies, yerr=errors, capsize=8, 
                  color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Path Efficiency\n(actual path / Euclidean distance)')
    ax.set_title('Path Efficiency Comparison\n(Lower is Better)', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, eff, err in zip(bars, efficiencies, errors):
        height = bar.get_height()
        ax.annotate(f'{eff:.3f} ± {err:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height + err),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Efficiency distribution saved to {output_path}")


def plot_success_rate_comparison(output_path='results/success_rate_comparison.png'):
    """Plot success rate comparison."""
    
    baseline_results_path = 'results/baseline_results.json'
    sparse_results_path = 'results/sparse_results.json'
    
    if not os.path.exists(baseline_results_path):
        print(f"Warning: Results files not found. Skipping success rate plot.")
        return
    
    with open(baseline_results_path, 'r') as f:
        baseline_results = json.load(f)
    
    with open(sparse_results_path, 'r') as f:
        sparse_results = json.load(f)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Baseline\n(Dense)', 'Sparse +\nCuriosity']
    success_rates = [
        baseline_results['success_rate'] * 100,
        sparse_results['success_rate'] * 100
    ]
    errors = [
        baseline_results['success_std'] * 100,
        sparse_results['success_std'] * 100
    ]
    
    colors = ['steelblue', 'coral']
    
    bars = ax.bar(methods, success_rates, yerr=errors, capsize=8, 
                  color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate Comparison\n(Higher is Better)', fontsize=14)
    ax.axhline(y=90, color='green', linestyle='--', label='90% Target', alpha=0.5)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 105)
    
    # Add value labels
    for bar, rate, err in zip(bars, success_rates, errors):
        height = bar.get_height()
        ax.annotate(f'{rate:.1f}% ± {err:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height + err),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Success rate comparison saved to {output_path}")


def main():
    """Main visualization function."""
    
    print("="*60)
    print("Sparse Rewards with Intrinsic Curiosity - Visualization")
    print("="*60)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Check if training histories exist
    baseline_history_path = 'logs/baseline_history.json'
    sparse_history_path = 'logs/sparse_history.json'
    
    if os.path.exists(baseline_history_path) and os.path.exists(sparse_history_path):
        print("\nLoading training histories...")
        baseline_history = load_training_history(baseline_history_path)
        sparse_history = load_training_history(sparse_history_path)
        
        print("\nGenerating training curves...")
        plot_training_curves(baseline_history, sparse_history)
    else:
        print("\nWarning: Training histories not found.")
        print("Skipping training curves plot.")
    
    # Generate trajectory and efficiency plots
    print("\nGenerating trajectory comparison...")
    plot_trajectory_comparison()
    
    print("\nGenerating efficiency distribution...")
    plot_path_efficiency_distribution()
    
    print("\nGenerating success rate comparison...")
    plot_success_rate_comparison()
    
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)
    print("\nGenerated figures:")
    print("  - results/training_curves.png")
    print("  - results/path_comparison.png")
    print("  - results/efficiency_distribution.png")
    print("  - results/success_rate_comparison.png")


if __name__ == '__main__':
    main()
