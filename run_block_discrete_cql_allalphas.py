#!/usr/bin/env python3
"""
Block Discrete CQL Training Script
Discretizes VP2 continuous action space into bins
- VP1: Binary (0 or 1)
- VP2: Discretized into N bins (0 to 0.5 mcg/kg/min)
- Uses consistent hyperparameters:
  - tau = 0.8
  - lr = 1e-3 
  - batch_size = 128
  - alpha = [0.0, 0.001, 0.01]
  - epochs = 100
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import sys

# Import our unified pipeline
from integrated_data_pipeline_v2 import IntegratedDataPipelineV2
from integrated_data_pipeline_v3 import IntegratedDataPipelineV3

# Force unbuffered output
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__


# ============================================================================
# Dual Block Discrete CQL Implementation
# ============================================================================

class DualBlockDiscreteQNetwork(nn.Module):
    """
    Q-network for block discrete actions using Q(s,a) -> R architecture
    Takes state and discrete action index as input, outputs single Q-value
    VP1: 2 actions (binary)
    VP2: vp2_bins actions (discretized continuous)
    Total: 2 * vp2_bins possible actions
    """
    
    def __init__(self, state_dim: int, vp2_bins: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.vp2_bins = vp2_bins
        self.total_actions = 2 * vp2_bins  # VP1 (2) x VP2 (bins)
        
        # Network takes state + one-hot encoded action
        input_dim = state_dim + self.total_actions
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 64)
        self.fc4 = nn.Linear(64, 1)  # Output single Q-value
        
        # Better initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
    
    def forward(self, state: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]
            action_idx: [batch_size] - discrete action indices (0 to total_actions-1)
        Returns:
            q_value: [batch_size, 1] - Q-value for each (state, action) pair
        """
        batch_size = state.shape[0]
        
        # Convert action indices to one-hot encoding
        action_one_hot = F.one_hot(action_idx.long(), num_classes=self.total_actions).float()
        
        # Concatenate state and action
        x = torch.cat([state, action_one_hot], dim=-1)
        
        # Forward through network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DualBlockDiscreteCQL:
    """
    Block Discrete CQL agent with discretized VP2 action space
    VP1: Binary (0 or 1) 
    VP2: Discretized into bins (0 to 0.5 mcg/kg/min)
    """
    
    def __init__(self, state_dim: int, vp2_bins: int = 5, alpha: float = 1.0, 
                 gamma: float = 0.95, tau: float = 0.8, lr: float = 1e-3, 
                 grad_clip: float = 1.0):
        self.state_dim = state_dim
        self.vp2_bins = vp2_bins
        self.total_actions = 2 * vp2_bins  # VP1 (2) x VP2 (bins)
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip
        
        # Define VP2 bin edges (0 to 0.5)
        self.vp2_bin_edges = np.linspace(0, 0.5, vp2_bins + 1)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Q-networks
        self.q1 = DualBlockDiscreteQNetwork(state_dim, vp2_bins).to(self.device)
        self.q2 = DualBlockDiscreteQNetwork(state_dim, vp2_bins).to(self.device)
        
        # Initialize target networks
        self.q1_target = DualBlockDiscreteQNetwork(state_dim, vp2_bins).to(self.device)
        self.q2_target = DualBlockDiscreteQNetwork(state_dim, vp2_bins).to(self.device)
        
        # Copy parameters to target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Initialize optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
    
    def continuous_to_discrete_action(self, continuous_action: np.ndarray) -> int:
        """
        Convert continuous action [vp1, vp2] to discrete action index
        Args:
            continuous_action: [vp1, vp2] where vp1 is binary, vp2 is continuous
        Returns:
            action_idx: discrete action index (0 to total_actions-1)
        """
        vp1, vp2 = continuous_action
        vp1_idx = int(vp1)  # 0 or 1
        
        # Find which bin vp2 falls into
        vp2_bin = np.digitize(vp2, self.vp2_bin_edges) - 1
        vp2_bin = np.clip(vp2_bin, 0, self.vp2_bins - 1)
        
        # Combine into single action index
        action_idx = vp1_idx * self.vp2_bins + vp2_bin
        return action_idx
    
    def discrete_to_continuous_action(self, action_idx: int) -> np.ndarray:
        """
        Convert discrete action index to continuous action [vp1, vp2]
        Args:
            action_idx: discrete action index (0 to total_actions-1)
        Returns:
            continuous_action: [vp1, vp2] where vp1 is binary, vp2 is continuous
        """
        vp1_idx = action_idx // self.vp2_bins
        vp2_bin = action_idx % self.vp2_bins
        
        # Convert bin to continuous value (use bin center)
        vp2_value = (self.vp2_bin_edges[vp2_bin] + self.vp2_bin_edges[vp2_bin + 1]) / 2
        
        return np.array([float(vp1_idx), vp2_value])
    
    def select_action(self, state: np.ndarray, num_samples: int = 50) -> np.ndarray:
        """
        Select best action using Q-values over all discrete actions
        Optimized with batch processing - no for-loops
        """
        with torch.no_grad():
            if state.ndim == 1:
                state = state.reshape(1, -1)
            
            batch_size = state.shape[0]
            state_tensor = torch.FloatTensor(state).to(self.device)
            
            # Create all possible discrete actions for each state in batch
            # Shape: [batch_size, total_actions]
            all_actions = torch.arange(self.total_actions).to(self.device)
            all_actions = all_actions.unsqueeze(0).expand(batch_size, -1)
            
            # Expand states to match actions
            # Shape: [batch_size * total_actions, state_dim]
            state_expanded = state_tensor.unsqueeze(1).expand(-1, self.total_actions, -1)
            state_expanded = state_expanded.reshape(-1, self.state_dim)
            
            # Flatten actions for network input
            # Shape: [batch_size * total_actions]
            actions_flat = all_actions.reshape(-1)
            
            # Compute Q-values for all actions
            q1_values = self.q1(state_expanded, actions_flat).reshape(batch_size, self.total_actions)
            q2_values = self.q2(state_expanded, actions_flat).reshape(batch_size, self.total_actions)
            q_values = torch.min(q1_values, q2_values)
            
            # Get best action for each batch element
            best_action_indices = q_values.argmax(dim=1).cpu().numpy()
            
            # Vectorized conversion from discrete indices to continuous actions
            # Extract VP1 (binary) and VP2 bin indices
            vp1_actions = (best_action_indices // self.vp2_bins).astype(float)
            vp2_bin_indices = best_action_indices % self.vp2_bins
            
            # Convert VP2 bin indices to continuous values using bin centers
            vp2_bin_centers = (self.vp2_bin_edges[:-1] + self.vp2_bin_edges[1:]) / 2
            vp2_actions = vp2_bin_centers[vp2_bin_indices]
            
            # Stack VP1 and VP2 into action array
            actions = np.stack([vp1_actions, vp2_actions], axis=1)
            
            return actions if batch_size > 1 else actions[0]
    
    def update(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
               next_states: torch.Tensor, dones: torch.Tensor) -> dict:
        """
        Update Q-networks with TD loss and CQL penalty
        """
        batch_size = states.shape[0]
        
        # Convert continuous actions to discrete indices
        action_indices = []
        for i in range(batch_size):
            action_np = actions[i].cpu().numpy()
            action_idx = self.continuous_to_discrete_action(action_np)
            action_indices.append(action_idx)
        action_indices = torch.LongTensor(action_indices).to(self.device)
        
        # Compute target Q-values
        with torch.no_grad():
            # Get next actions using current Q-networks
            all_next_actions = torch.arange(self.total_actions).to(self.device)
            all_next_actions = all_next_actions.unsqueeze(0).expand(batch_size, -1)
            
            next_states_expanded = next_states.unsqueeze(1).expand(-1, self.total_actions, -1)
            next_states_expanded = next_states_expanded.reshape(-1, self.state_dim)
            next_actions_flat = all_next_actions.reshape(-1)
            
            next_q1 = self.q1_target(next_states_expanded, next_actions_flat).reshape(batch_size, self.total_actions)
            next_q2 = self.q2_target(next_states_expanded, next_actions_flat).reshape(batch_size, self.total_actions)
            next_q = torch.min(next_q1, next_q2)
            
            next_best_actions = next_q.argmax(dim=1)
            
            # Compute target values
            next_q1_target = self.q1_target(next_states, next_best_actions).squeeze()
            next_q2_target = self.q2_target(next_states, next_best_actions).squeeze()
            next_q_target = torch.min(next_q1_target, next_q2_target)
            
            target_q = rewards + self.gamma * next_q_target * (1 - dones)
        
        # Update Q1
        current_q1 = self.q1(states, action_indices).squeeze()
        q1_loss = F.mse_loss(current_q1, target_q)
        
        # CQL penalty for Q1
        if self.alpha > 0:
            # Compute logsumexp over all actions for CQL
            all_actions = torch.arange(self.total_actions).to(self.device)
            all_actions = all_actions.unsqueeze(0).expand(batch_size, -1)
            
            states_expanded = states.unsqueeze(1).expand(-1, self.total_actions, -1)
            states_expanded = states_expanded.reshape(-1, self.state_dim)
            actions_flat = all_actions.reshape(-1)
            
            q1_all = self.q1(states_expanded, actions_flat).reshape(batch_size, self.total_actions)
            cql1_loss = torch.logsumexp(q1_all, dim=1).mean() - current_q1.mean()
        else:
            cql1_loss = torch.tensor(0.0).to(self.device)
        
        total_q1_loss = q1_loss + self.alpha * cql1_loss
        
        self.q1_optimizer.zero_grad()
        total_q1_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.grad_clip)
        self.q1_optimizer.step()
        
        # Update Q2
        current_q2 = self.q2(states, action_indices).squeeze()
        q2_loss = F.mse_loss(current_q2, target_q)
        
        # CQL penalty for Q2
        if self.alpha > 0:
            q2_all = self.q2(states_expanded, actions_flat).reshape(batch_size, self.total_actions)
            cql2_loss = torch.logsumexp(q2_all, dim=1).mean() - current_q2.mean()
        else:
            cql2_loss = torch.tensor(0.0).to(self.device)
        
        total_q2_loss = q2_loss + self.alpha * cql2_loss
        
        self.q2_optimizer.zero_grad()
        total_q2_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.grad_clip)
        self.q2_optimizer.step()
        
        # Soft update target networks
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'cql1_loss': cql1_loss.item() if self.alpha > 0 else 0,
            'cql2_loss': cql2_loss.item() if self.alpha > 0 else 0,
            'total_q1_loss': total_q1_loss.item(),
            'total_q2_loss': total_q2_loss.item()
        }
    
    def save(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'state_dim': self.state_dim,
            'vp2_bins': self.vp2_bins,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'tau': self.tau
        }, filepath)


def train_block_discrete_cql(
    alpha: float = 0.001,
    vp2_bins: int = 5,
    epochs: int = 100,
    reward_model_path: str = None,
    suffix: str = "",
    save_dir: str = "experiment/ql"
):
    """Train Block Discrete CQL with specified alpha and optional learned reward"""

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Initialize data pipeline and infer reward type
    print("\nInitializing Block Discrete CQL data pipeline...", flush=True)
    if reward_model_path is None:
        reward_type = "manual"
        pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
    elif 'gcl' in reward_model_path:
        reward_type = "gcl"
        pipeline = IntegratedDataPipelineV3(model_type='dual', reward_source='learned', random_seed=42)
        pipeline.load_gcl_reward_model(reward_model_path)
    elif 'iq_learn' in reward_model_path:
        reward_type = "iq_learn"
        pipeline = IntegratedDataPipelineV3(model_type='dual', reward_source='learned', random_seed=42)
        pipeline.load_iq_learn_reward_model(reward_model_path)
    elif 'maxent' in reward_model_path:
        reward_type = "maxent"
        pipeline = IntegratedDataPipelineV3(model_type='dual', reward_source='learned', random_seed=42)
        pipeline.load_maxent_reward_model(reward_model_path)
    else:
        raise ValueError(f"Cannot infer reward model type from path: {reward_model_path}")

    experiment_prefix = f"{reward_type}{suffix}"

    print("="*70, flush=True)
    print(f" BLOCK DISCRETE CQL TRAINING WITH ALPHA={alpha}", flush=True)
    print(f" Reward: {reward_type} | Prefix: {experiment_prefix}", flush=True)
    print("="*70, flush=True)

    train_data, val_data, test_data = pipeline.prepare_data()
    
    # Get state dimension
    state_dim = train_data['states'].shape[1]

    # Print settings
    print("\n" + "="*70, flush=True)
    print("SETTINGS:", flush=True)
    print(f"  State dimension: {state_dim}", flush=True)
    print(f"  Action dimension: 2 (VP1: binary, VP2: {vp2_bins} bins)", flush=True)
    print(f"  Total discrete actions: {2 * vp2_bins}", flush=True)
    print(f"  ALPHA = {alpha}", flush=True)
    print(f"  TAU = 0.8 (target network update)", flush=True)
    print("  LR = 0.001 (learning rate)", flush=True)
    print("  BATCH_SIZE = 128", flush=True)
    print("  EPOCHS = 100", flush=True)
    print("="*70, flush=True)
    
    # Initialize agent with specified parameters
    agent = DualBlockDiscreteCQL(
        state_dim=state_dim,
        vp2_bins=vp2_bins,
        alpha=alpha,
        gamma=0.95,
        tau=0.8,      # As specified
        lr=1e-3,      # As specified  
        grad_clip=1.0
    )
    
    # Training loop
    batch_size = 128
    print(f"\nTraining for {epochs} epochs with batch size {batch_size}...", flush=True)
    start_time = time.time()
    
    best_val_loss = float('inf')
    os.makedirs('experiment', exist_ok=True)
    
    for epoch in range(epochs):
        # Training phase
        agent.q1.train()
        agent.q2.train()
        
        train_metrics = {
            'q1_loss': 0, 'q2_loss': 0,
            'cql1_loss': 0, 'cql2_loss': 0,
            'total_q1_loss': 0, 'total_q2_loss': 0
        }
        
        # Sample random batches for training
        n_batches = len(train_data['states']) // batch_size
        
        for _ in range(n_batches):
            # Get batch
            batch = pipeline.get_batch(batch_size=batch_size, split='train')
            
            # Convert to tensors
            states = torch.FloatTensor(batch['states']).to(agent.device)
            actions = torch.FloatTensor(batch['actions']).to(agent.device)
            rewards = torch.FloatTensor(batch['rewards']).to(agent.device)
            next_states = torch.FloatTensor(batch['next_states']).to(agent.device)
            dones = torch.FloatTensor(batch['dones']).to(agent.device)
            
            # Update agent
            metrics = agent.update(states, actions, rewards, next_states, dones)
            
            # Accumulate metrics
            for key in train_metrics:
                train_metrics[key] += metrics.get(key, 0)
        
        # Average metrics
        for key in train_metrics:
            train_metrics[key] /= n_batches
        
        # Validation phase
        agent.q1.eval()
        agent.q2.eval()
        
        val_q_values = []
        with torch.no_grad():
            # Sample validation batches
            for _ in range(10):  # Use 10 validation batches
                batch = pipeline.get_batch(batch_size=batch_size, split='val')
                
                states = torch.FloatTensor(batch['states']).to(agent.device)
                actions = torch.FloatTensor(batch['actions']).to(agent.device)
                
                # Convert continuous actions to discrete for validation
                action_indices = []
                for i in range(batch_size):
                    action_np = actions[i].cpu().numpy()
                    action_idx = agent.continuous_to_discrete_action(action_np)
                    action_indices.append(action_idx)
                action_indices = torch.LongTensor(action_indices).to(agent.device)
                
                q1_val = agent.q1(states, action_indices).squeeze()
                q2_val = agent.q2(states, action_indices).squeeze()
                q_val = torch.min(q1_val, q2_val)
                
                val_q_values.append(q_val.mean().item())
        
        val_loss = -np.mean(val_q_values)  # Negative because we want higher Q-values
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            agent.save(f'{save_dir}/{experiment_prefix}_alpha{alpha:.4f}_bins{vp2_bins}_best.pt')
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_td_loss = (train_metrics['q1_loss'] + train_metrics['q2_loss']) / 2
            print(f"Epoch {epoch+1}: "
                  f"TD Loss={avg_td_loss:.4f} (Q1={train_metrics['q1_loss']:.4f}, Q2={train_metrics['q2_loss']:.4f}), "
                  f"CQL Loss (Q1={train_metrics['cql1_loss']:.4f}, Q2={train_metrics['cql2_loss']:.4f}), "
                  f"Val Q={-val_loss:.4f}, "
                  f"Time={elapsed/60:.1f}min", flush=True)
    
    # Save final model
    agent.save(f'{save_dir}/{experiment_prefix}_alpha{alpha:.4f}_bins{vp2_bins}_final.pt')

    total_time = time.time() - start_time
    print(f"\nBlock Discrete CQL ({experiment_prefix}, alpha={alpha}, bins={vp2_bins}) completed in {total_time/60:.1f} minutes!", flush=True)
    print("Models saved:", flush=True)
    print(f"  - {save_dir}/{experiment_prefix}_alpha{alpha:.4f}_bins{vp2_bins}_best.pt", flush=True)
    print(f"  - {save_dir}/{experiment_prefix}_alpha{alpha:.4f}_bins{vp2_bins}_final.pt", flush=True)

    return agent, pipeline, experiment_prefix




def main():
    """Train Block Discrete CQL with multiple alpha values"""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Block Discrete CQL with different configurations')
    parser.add_argument('--vp2_bins', type=int, default=5,
                       help='Number of bins for VP2 discretization (default: 5)')
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.0],
                       help='Alpha values for CQL penalty (default: 0.0)')
    parser.add_argument('--single_alpha', type=float, default=None,
                       help='Train only with a single alpha value (overrides --alphas)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--reward_model_path', type=str, default=None,
                       help='Path to learned reward model (gcl/iq_learn/maxent). None=manual reward')
    parser.add_argument('--suffix', type=str, default='',
                       help='Suffix to add to experiment prefix (e.g., "_irl100")')
    parser.add_argument('--save_dir', type=str, default='experiment/ql',
                       help='Directory to save models (default: experiment/ql)')
    args = parser.parse_args()

    # Determine alpha values to use
    if args.single_alpha is not None:
        alphas = [args.single_alpha]
    else:
        alphas = args.alphas

    vp2_bins = args.vp2_bins

    print("="*70, flush=True)
    print(" BLOCK DISCRETE CQL TRAINING", flush=True)
    print("="*70, flush=True)
    print("\nConfiguration:", flush=True)
    print("  - VP1: Binary (0 or 1)", flush=True)
    print(f"  - VP2: Discretized into {vp2_bins} bins (0 to 0.5 mcg/kg/min)", flush=True)
    print(f"  - Alpha values: {alphas}", flush=True)
    print(f"  - Epochs: {args.epochs}", flush=True)
    print(f"  - Reward model: {args.reward_model_path or 'manual'}", flush=True)
    print(f"  - Suffix: {args.suffix}", flush=True)
    print("  - Consistent hyperparameters (tau=0.8, lr=1e-3)", flush=True)
    print(f"  - Total discrete actions: {2 * vp2_bins}", flush=True)

    # Train with different alpha values
    experiment_prefixes = []
    for alpha in alphas:
        print("\n" + "="*70, flush=True)
        agent, pipeline, exp_prefix = train_block_discrete_cql(
            alpha=alpha,
            vp2_bins=vp2_bins,
            epochs=args.epochs,
            reward_model_path=args.reward_model_path,
            suffix=args.suffix,
            save_dir=args.save_dir
        )
        experiment_prefixes.append(exp_prefix)

    print("\n" + "="*70, flush=True)
    print(" ALL TRAINING COMPLETE", flush=True)
    print("="*70, flush=True)
    print(f"\nModels saved in {args.save_dir}/ directory:", flush=True)
    for alpha, exp_prefix in zip(alphas, experiment_prefixes):
        print(f"  {exp_prefix}_alpha{alpha:.4f}_bins{vp2_bins}_*.pt", flush=True)


if __name__ == "__main__":
    main()
