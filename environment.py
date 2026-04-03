"""
Environment wrappers for Sparse Rewards with Intrinsic Curiosity.

This module provides:
- DenseRewardWrapper: Standard dense reward (negative distance to goal)
- SparseRewardWrapper: Sparse binary reward with Intrinsic Curiosity Module (ICM)
- ICM: Intrinsic Curiosity Module for exploration bonus
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import Wrapper


class ICM(nn.Module):
    """
    Intrinsic Curiosity Module (ICM).
    
    Architecture:
    - Feature encoder: state (19-dim) -> Linear(128) -> ReLU -> Linear(64) -> feature
    - Inverse model: concat[feature, feature] -> Linear(64) -> ReLU -> Linear(7) -> action
    - Forward model: concat[feature, action] -> Linear(64) -> ReLU -> Linear(64) -> predicted feature
    
    Intrinsic reward: 0.1 * forward_loss
    """
    
    def __init__(self, state_dim=19, action_dim=7, device='cpu'):
        super(ICM, self).__init__()
        self.device = device
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Inverse model: predicts action from current and next features
        self.inverse_model = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # Forward model: predicts next feature from current feature and action
        self.forward_model = nn.Sequential(
            nn.Linear(64 + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        self.to(device)
        
    def encode(self, state):
        """Encode state to feature representation."""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        return self.feature_encoder(state)
    
    def compute_intrinsic_reward(self, state, action, next_state):
        """
        Compute intrinsic curiosity reward.
        
        Args:
            state: Current state (numpy array or tensor)
            action: Action taken (numpy array or tensor)
            next_state: Next state after action (numpy array or tensor)
            
        Returns:
            intrinsic_reward: Scalar intrinsic reward
            loss: Training loss for the ICM
        """
        # Convert to tensors if needed
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        
        # Encode states
        feature = self.encode(state)
        next_feature = self.encode(next_state)
        
        # Forward model prediction
        forward_input = torch.cat([feature, action], dim=-1)
        predicted_next_feature = self.forward_model(forward_input)
        
        # Inverse model prediction
        inverse_input = torch.cat([feature, next_feature], dim=-1)
        predicted_action = self.inverse_model(inverse_input)
        
        # Compute losses
        forward_loss = nn.MSELoss()(predicted_next_feature, next_feature.detach())
        inverse_loss = nn.MSELoss()(predicted_action, action.detach())
        
        total_loss = forward_loss + inverse_loss
        
        # Intrinsic reward is proportional to forward prediction error
        intrinsic_reward = 0.1 * forward_loss.item()
        
        return intrinsic_reward, total_loss
    
    def train_step(self, state, action, next_state):
        """
        Perform one training step for the ICM.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            loss: Training loss
        """
        self.train()
        
        # Convert to tensors
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        
        # Encode states
        feature = self.encode(state)
        next_feature = self.encode(next_state)
        
        # Forward model prediction
        forward_input = torch.cat([feature, action], dim=-1)
        predicted_next_feature = self.forward_model(forward_input)
        
        # Inverse model prediction
        inverse_input = torch.cat([feature, next_feature], dim=-1)
        predicted_action = self.inverse_model(inverse_input)
        
        # Compute losses
        forward_loss = nn.MSELoss()(predicted_next_feature, next_feature.detach())
        inverse_loss = nn.MSELoss()(predicted_action, action.detach())
        
        total_loss = forward_loss + inverse_loss
        
        # Backpropagate
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()


class DenseRewardWrapper(Wrapper):
    """
    Dense reward wrapper for PandaReach-v3.
    
    Reward: Negative Euclidean distance between end-effector and goal.
    """
    
    def __init__(self, env):
        super(DenseRewardWrapper, self).__init__(env)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Dense reward: negative distance to goal
        # The environment already provides this, but we ensure consistency
        achieved_goal = info.get('achieved_goal', obs[:3])
        desired_goal = info.get('desired_goal', obs[3:6])
        distance = np.linalg.norm(achieved_goal - desired_goal)
        
        # Return negative distance as reward
        dense_reward = -distance
        
        return obs, dense_reward, terminated, truncated, info


class SparseRewardWrapper(Wrapper):
    """
    Sparse reward wrapper with Intrinsic Curiosity Module.
    
    Sparse Reward: 0 if distance < 0.05m, -1 otherwise
    Intrinsic Reward: 0.1 * forward_prediction_error from ICM
    Total Reward: sparse_reward + intrinsic_reward
    """
    
    def __init__(self, env, icm=None, device='cpu'):
        super(SparseRewardWrapper, self).__init__(env)
        self.device = device
        self.success_threshold = 0.05  # 5cm threshold for success
        
        # Initialize ICM if not provided
        if icm is None:
            self.icm = ICM(state_dim=19, action_dim=7, device=device)
        else:
            self.icm = icm
            
        # Store last state for computing intrinsic reward
        self.last_obs = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs.copy()
        return obs, info
    
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # Compute sparse reward
        achieved_goal = info.get('achieved_goal', obs[:3])
        desired_goal = info.get('desired_goal', obs[3:6])
        distance = np.linalg.norm(achieved_goal - desired_goal)
        
        # Sparse binary reward: 0 if close enough, -1 otherwise
        if distance < self.success_threshold:
            sparse_reward = 0.0
            info['success'] = True
        else:
            sparse_reward = -1.0
            info['success'] = False
        
        # Compute intrinsic curiosity reward
        intrinsic_reward, icm_loss = self.icm.compute_intrinsic_reward(
            self.last_obs, action, obs
        )
        
        # Train ICM
        self.icm.train_step(self.last_obs, action, obs)
        
        # Total reward combines sparse and intrinsic rewards
        total_reward = sparse_reward + intrinsic_reward
        
        # Update last observation
        self.last_obs = obs.copy()
        
        # Store additional info
        info['sparse_reward'] = sparse_reward
        info['intrinsic_reward'] = intrinsic_reward
        info['distance'] = distance
        info['icm_loss'] = icm_loss.item() if hasattr(icm_loss, 'item') else icm_loss
        
        return obs, total_reward, terminated, truncated, info


def make_env(env_name='PandaReach-v3', reward_type='dense', device='cpu'):
    """
    Factory function to create environments with specified reward type.
    
    Args:
        env_name: Name of the environment
        reward_type: 'dense' or 'sparse'
        device: Device for ICM computation
        
    Returns:
        Wrapped environment
    """
    import panda_gym
    
    env = panda_gym.make(env_name)
    
    if reward_type == 'dense':
        env = DenseRewardWrapper(env)
    elif reward_type == 'sparse':
        env = SparseRewardWrapper(env, device=device)
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")
    
    return env
