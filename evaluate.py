"""
Evaluation script for comparing Dense vs Sparse+Curiosity methods.

Evaluates both trained models on 100 test episodes and computes:
- Success rate (% of episodes reaching goal within threshold)
- Path efficiency (actual path length / Euclidean distance)
- Average reward
- Episode length
"""

import os
import json
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import panda_gym

from environment import DenseRewardWrapper, SparseRewardWrapper


class EvaluationEnv(DenseRewardWrapper):
    """Environment wrapper for evaluation with trajectory tracking."""
    
    def __init__(self, env, track_trajectory=True):
        super(EvaluationEnv, self).__init__(env)
        self.track_trajectory = track_trajectory
        self.trajectory = []
        self.episode_start_pos = None
        self.goal_pos = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.trajectory = [obs[:3].copy()]  # Track end-effector position
        self.episode_start_pos = obs[:3].copy()
        self.goal_pos = info.get('desired_goal', obs[3:6]).copy()
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.track_trajectory:
            self.trajectory.append(obs[:3].copy())
            
        return obs, reward, terminated, truncated, info
    
    def get_trajectory(self):
        return np.array(self.trajectory)
    
    def compute_path_efficiency(self):
        """
        Compute path efficiency metric.
        
        Path efficiency = actual path length / Euclidean distance (start to goal)
        Lower is better (more direct path).
        """
        trajectory = np.array(self.trajectory)
        
        if len(trajectory) < 2:
            return float('inf')
        
        # Compute actual path length
        path_length = 0.0
        for i in range(1, len(trajectory)):
            path_length += np.linalg.norm(trajectory[i] - trajectory[i-1])
        
        # Compute Euclidean distance from start to goal
        euclidean_distance = np.linalg.norm(self.goal_pos - self.episode_start_pos)
        
        if euclidean_distance < 1e-6:
            return 1.0  # Already at goal
        
        return path_length / euclidean_distance


def evaluate_model(model_path, env_type='dense', num_episodes=100, seed=42, device='cpu'):
    """
    Evaluate a trained model on multiple episodes.
    
    Args:
        model_path: Path to saved model zip file
        env_type: 'dense' or 'sparse'
        num_episodes: Number of evaluation episodes
        seed: Random seed
        device: Device for evaluation
        
    Returns:
        results: Dictionary with evaluation metrics
        trajectories: List of trajectories for visualization
    """
    print(f"\nEvaluating {env_type} model from {model_path}")
    
    # Load model
    model = PPO.load(model_path, device=device)
    
    # Create evaluation environment
    def make_eval_env():
        env = panda_gym.make('PandaReach-v3')
        if env_type == 'dense':
            env = EvaluationEnv(DenseRewardWrapper(env))
        else:
            env = EvaluationEnv(SparseRewardWrapper(env, device=device))
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_eval_env])
    
    # Evaluation metrics
    successes = []
    rewards = []
    episode_lengths = []
    path_efficiencies = []
    final_distances = []
    trajectories = []
    
    # Run evaluation episodes
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            step_count += 1
        
        # Get metrics from environment
        unwrapped_env = env.envs[0]
        success = info[0].get('success', False)
        final_distance = info[0].get('distance', float('inf'))
        path_efficiency = unwrapped_env.compute_path_efficiency()
        trajectory = unwrapped_env.get_trajectory()
        
        successes.append(success)
        rewards.append(episode_reward)
        episode_lengths.append(step_count)
        path_efficiencies.append(path_efficiency)
        final_distances.append(final_distance)
        trajectories.append(trajectory)
        
        if (episode + 1) % 20 == 0:
            print(f"  Episode {episode + 1}/{num_episodes}: "
                  f"Success={success}, Reward={episode_reward:.2f}, "
                  f"Length={step_count}, Efficiency={path_efficiency:.2f}")
    
    # Compute aggregate statistics
    results = {
        'success_rate': np.mean(successes),
        'success_std': np.std(successes),
        'mean_reward': np.mean(rewards),
        'reward_std': np.std(rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'length_std': np.std(episode_lengths),
        'mean_path_efficiency': np.mean(path_efficiencies),
        'efficiency_std': np.std(path_efficiencies),
        'mean_final_distance': np.mean(final_distances),
        'distance_std': np.std(final_distances),
        'num_episodes': num_episodes
    }
    
    print(f"\n{env_type} Results:")
    print(f"  Success Rate: {results['success_rate']*100:.1f}% ± {results['success_std']*100:.1f}%")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['reward_std']:.2f}")
    print(f"  Mean Episode Length: {results['mean_episode_length']:.1f} ± {results['length_std']:.1f}")
    print(f"  Mean Path Efficiency: {results['mean_path_efficiency']:.2f} ± {results['efficiency_std']:.2f}")
    print(f"  Mean Final Distance: {results['mean_final_distance']:.4f} ± {results['distance_std']:.4f}")
    
    return results, trajectories


def compare_methods(baseline_results, sparse_results):
    """Compare the two methods and compute improvement metrics."""
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    # Success rate comparison
    success_diff = sparse_results['success_rate'] - baseline_results['success_rate']
    print(f"\nSuccess Rate:")
    print(f"  Baseline (Dense):     {baseline_results['success_rate']*100:.1f}%")
    print(f"  Sparse+Curiosity:     {sparse_results['success_rate']*100:.1f}%")
    print(f"  Difference:           {success_diff*100:+.1f}%")
    
    # Path efficiency comparison
    efficiency_improvement = (baseline_results['mean_path_efficiency'] - sparse_results['mean_path_efficiency']) / baseline_results['mean_path_efficiency']
    print(f"\nPath Efficiency (lower is better):")
    print(f"  Baseline (Dense):     {baseline_results['mean_path_efficiency']:.3f}")
    print(f"  Sparse+Curiosity:     {sparse_results['mean_path_efficiency']:.3f}")
    print(f"  Improvement:          {efficiency_improvement*100:+.1f}%")
    
    # Episode length comparison
    length_diff = sparse_results['mean_episode_length'] - baseline_results['mean_episode_length']
    print(f"\nEpisode Length:")
    print(f"  Baseline (Dense):     {baseline_results['mean_episode_length']:.1f}")
    print(f"  Sparse+Curiosity:     {sparse_results['mean_episode_length']:.1f}")
    print(f"  Difference:           {length_diff:+.1f}")
    
    return {
        'success_rate_diff': success_diff,
        'efficiency_improvement': efficiency_improvement,
        'episode_length_diff': length_diff
    }


def save_evaluation_report(baseline_results, sparse_results, comparison, output_path='results/evaluation_report.txt'):
    """Save detailed evaluation report to file."""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("EVALUATION REPORT: Sparse Rewards with Intrinsic Curiosity\n")
        f.write("="*60 + "\n\n")
        
        f.write("BASELINE (DENSE REWARDS)\n")
        f.write("-"*40 + "\n")
        f.write(f"Success Rate:       {baseline_results['success_rate']*100:.1f}%\n")
        f.write(f"Mean Reward:        {baseline_results['mean_reward']:.2f}\n")
        f.write(f"Episode Length:     {baseline_results['mean_episode_length']:.1f}\n")
        f.write(f"Path efficiency:    {baseline_results['mean_path_efficiency']:.3f}\n")
        f.write(f"Final Distance:     {baseline_results['mean_final_distance']:.4f}\n\n")
        
        f.write("SPARSE + CURIOSITY\n")
        f.write("-"*40 + "\n")
        f.write(f"Success Rate:       {sparse_results['success_rate']*100:.1f}%\n")
        f.write(f"Mean Reward:        {sparse_results['mean_reward']:.2f}\n")
        f.write(f"Episode Length:     {sparse_results['mean_episode_length']:.1f}\n")
        f.write(f"Path efficiency:    {sparse_results['mean_path_efficiency']:.3f}\n")
        f.write(f"Final Distance:     {sparse_results['mean_final_distance']:.4f}\n\n")
        
        f.write("COMPARISON\n")
        f.write("-"*40 + "\n")
        f.write(f"Success Rate Diff:      {comparison['success_rate_diff']*100:+.1f}%\n")
        f.write(f"Path efficiency Improvement: {comparison['efficiency_improvement']*100:+.1f}%\n")
        f.write(f"Episode Length Diff:    {comparison['episode_length_diff']:+.1f}\n\n")
        
        f.write("CONCLUSION\n")
        f.write("-"*40 + "\n")
        if sparse_results['success_rate'] >= 0.9:
            f.write("✓ Sparse+Curiosity achieves >90% success rate\n")
        else:
            f.write("✗ Sparse+Curiosity success rate below 90%\n")
            
        if comparison['efficiency_improvement'] > 0.15:
            f.write("✓ Path efficiency improved by >15%\n")
        else:
            f.write(f"  Path efficiency change: {comparison['efficiency_improvement']*100:.1f}%\n")
    
    print(f"\nEvaluation report saved to {output_path}")


def main():
    """Main evaluation function."""
    
    print("="*60)
    print("Sparse Rewards with Intrinsic Curiosity - Evaluation")
    print("="*60)
    
    # Check if models exist
    baseline_path = 'models/baseline_dense.zip'
    sparse_path = 'models/sparse_curiosity.zip'
    
    if not os.path.exists(baseline_path):
        print(f"Error: Baseline model not found at {baseline_path}")
        print("Please run 'python train.py' first.")
        return
    
    if not os.path.exists(sparse_path):
        print(f"Error: Sparse model not found at {sparse_path}")
        print("Please run 'python train.py' first.")
        return
    
    # Evaluation parameters
    NUM_EPISODES = 100
    SEED = 42
    DEVICE = 'cpu'
    
    print(f"\nEvaluation Parameters:")
    print(f"  Episodes: {NUM_EPISODES}")
    print(f"  Seed: {SEED}")
    print(f"  Device: {DEVICE}")
    
    # Set random seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Evaluate baseline
    print("\n" + "="*60)
    print("PHASE 1: Evaluating Baseline (Dense Rewards)")
    print("="*60)
    
    baseline_results, baseline_trajectories = evaluate_model(
        model_path=baseline_path,
        env_type='dense',
        num_episodes=NUM_EPISODES,
        seed=SEED,
        device=DEVICE
    )
    
    # Evaluate sparse + curiosity
    print("\n" + "="*60)
    print("PHASE 2: Evaluating Sparse + Curiosity")
    print("="*60)
    
    sparse_results, sparse_trajectories = evaluate_model(
        model_path=sparse_path,
        env_type='sparse',
        num_episodes=NUM_EPISODES,
        seed=SEED,
        device=DEVICE
    )
    
    # Compare methods
    comparison = compare_methods(baseline_results, sparse_results)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    # Save raw results
    with open('results/baseline_results.json', 'w') as f:
        json.dump(baseline_results, f, indent=2)
    
    with open('results/sparse_results.json', 'w') as f:
        json.dump(sparse_results, f, indent=2)
    
    # Save trajectories for visualization
    np.savez('results/trajectories.npz',
             baseline_trajectories=np.array(baseline_trajectories, dtype=object),
             sparse_trajectories=np.array(sparse_trajectories, dtype=object))
    
    # Save evaluation report
    save_evaluation_report(baseline_results, sparse_results, comparison)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print("\nSaved files:")
    print("  - results/baseline_results.json")
    print("  - results/sparse_results.json")
    print("  - results/trajectories.npz")
    print("  - results/evaluation_report.txt")
    print("\nNext step:")
    print("  Run 'python visualize.py' to generate plots")


if __name__ == '__main__':
    main()
