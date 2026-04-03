"""
Training script for Sparse Rewards with Intrinsic Curiosity.

Trains two PPO agents:
1. Baseline: Dense reward (negative distance to goal)
2. Sparse+Curiosity: Sparse binary reward + ICM intrinsic reward

Saves models and training histories to disk.
"""

import os
import json
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import time

from environment import make_env, DenseRewardWrapper, SparseRewardWrapper


class HistoryCallback(BaseCallback):
    """Callback to record training history."""
    
    def __init__(self, verbose=0):
        super(HistoryCallback, self).__init__(verbose)
        self.history = {
            'timesteps': [],
            'rewards': [],
            'successes': [],
            'intrinsic_rewards': []  # For sparse+curiosity method
        }
        
    def _on_step(self) -> bool:
        # Record metrics at each step
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            
            self.history['timesteps'].append(self.num_timesteps)
            self.history['rewards'].append(self.locals.get('rewards', [0])[0])
            self.history['successes'].append(info.get('success', False))
            self.history['intrinsic_rewards'].append(
                info.get('intrinsic_reward', 0.0)
            )
        
        return True


def create_eval_env(env_type='dense'):
    """Create evaluation environment for callbacks."""
    import panda_gym
    
    env = panda_gym.make('PandaReach-v3')
    
    if env_type == 'dense':
        env = DenseRewardWrapper(env)
    else:
        env = SparseRewardWrapper(env, device='cpu')
    
    env = Monitor(env)
    return env


def train_ppo(
    env_type='dense',
    total_timesteps=500000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    seed=42,
    device='cpu'
):
    """
    Train PPO agent with specified environment type.
    
    Args:
        env_type: 'dense' or 'sparse'
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for PPO
        n_steps: Number of steps per rollout
        batch_size: Mini-batch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        seed: Random seed
        device: Device for training
        
    Returns:
        model: Trained PPO model
        history: Training history dictionary
    """
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Training PPO with {env_type} rewards")
    print(f"{'='*60}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Device: {device}")
    
    # Create environment
    import panda_gym
    
    def make_monitored_env():
        env = panda_gym.make('PandaReach-v3')
        if env_type == 'dense':
            env = DenseRewardWrapper(env)
        else:
            env = SparseRewardWrapper(env, device=device)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_monitored_env])
    
    # Define policy network architecture
    policy_kwargs = dict(
        net_arch=[256, 256],  # Standard MLP architecture
        activation_fn=torch.nn.ReLU
    )
    
    # Create PPO model
    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device=device,
        tensorboard_log=None
    )
    
    # Setup callbacks
    history_callback = HistoryCallback()
    
    # Train the model
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=history_callback,
        progress_bar=True
    )
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    return model, history_callback.history


def save_training_results(model, history, env_type, output_dir='models'):
    """Save trained model and training history."""
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f'{env_type}.zip')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save training history
    history_path = os.path.join('logs', f'{env_type}_history.json')
    
    # Convert numpy types to Python types for JSON serialization
    serializable_history = {
        'timesteps': [int(x) for x in history['timesteps']],
        'rewards': [float(x) if x is not None else 0.0 for x in history['rewards']],
        'successes': [bool(x) for x in history['successes']],
        'intrinsic_rewards': [float(x) if x is not None else 0.0 for x in history['intrinsic_rewards']]
    }
    
    with open(history_path, 'w') as f:
        json.dump(serializable_history, f, indent=2)
    print(f"Training history saved to {history_path}")


def main():
    """Main training function."""
    
    print("="*60)
    print("Sparse Rewards with Intrinsic Curiosity - Training")
    print("="*60)
    
    # Training parameters
    TOTAL_TIMESTEPS = 500000  # Adjust based on available time
    DEVICE = 'cpu'  # Use CPU for reproducibility
    SEED = 42
    
    # Hyperparameters (identical for both methods)
    HYPERPARAMS = {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01
    }
    
    print("\nHyperparameters:")
    for key, value in HYPERPARAMS.items():
        print(f"  {key}: {value}")
    
    # Train baseline (dense rewards)
    print("\n\n" + "="*60)
    print("PHASE 1: Training Baseline (Dense Rewards)")
    print("="*60)
    
    baseline_model, baseline_history = train_ppo(
        env_type='baseline_dense',
        total_timesteps=TOTAL_TIMESTEPS,
        device=DEVICE,
        seed=SEED,
        **HYPERPARAMS
    )
    
    save_training_results(baseline_model, baseline_history, 'baseline_dense')
    
    # Train sparse + curiosity method
    print("\n\n" + "="*60)
    print("PHASE 2: Training Sparse + Curiosity")
    print("="*60)
    
    sparse_model, sparse_history = train_ppo(
        env_type='sparse_curiosity',
        total_timesteps=TOTAL_TIMESTEPS,
        device=DEVICE,
        seed=SEED,
        **HYPERPARAMS
    )
    
    save_training_results(sparse_model, sparse_history, 'sparse_curiosity')
    
    print("\n\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nSaved files:")
    print("  - models/baseline_dense.zip")
    print("  - models/sparse_curiosity.zip")
    print("  - logs/baseline_history.json")
    print("  - logs/sparse_history.json")
    print("\nNext steps:")
    print("  1. Run 'python evaluate.py' to compare methods")
    print("  2. Run 'python visualize.py' to generate plots")


if __name__ == '__main__':
    main()
