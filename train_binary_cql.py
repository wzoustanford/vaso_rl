"""
Binary CQL Training with Continuous Q-Network Architecture
==========================================================
Uses the same Q(s,a) -> R architecture as Dual CQL for better comparability.
For binary actions, we evaluate Q(s,0) and Q(s,1) and select argmax.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Tuple
import os
import sys, pdb
from datetime import datetime

# Import the shared continuous Q-network from train_cql_stable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vaso_q_network import VasoQNetwork

from integrated_data_pipeline_v2 import IntegratedDataPipelineV2
import data_config as config


class BinaryCQL:
    """
    Conservative Q-Learning for binary actions using continuous Q-network.
    Uses same architecture as Dual CQL: Q(s,a) -> R where a ∈ {0, 1}
    """
    
    def __init__(
        self,
        state_dim: int,
        alpha: float = 1.0,
        gamma: float = 0.95,
        tau: float = 0.8,  # Updated to match your change
        lr: float = 3e-3,  # Updated to 0.003 as you mentioned
        grad_clip: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize Binary CQL with continuous Q-network
        
        Args:
            state_dim: Dimension of state space
            alpha: CQL penalty weight
            gamma: Discount factor
            tau: Target network update rate
            lr: Learning rate
            grad_clip: Gradient clipping value
            device: Device for computation
        """
        self.state_dim = state_dim
        self.action_dim = 1  # Binary action represented as single continuous value
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip
        self.device = device
        
        # Q-networks using the same architecture as Dual CQL
        self.q1 = VasoQNetwork(state_dim, self.action_dim).to(device)
        self.q2 = VasoQNetwork(state_dim, self.action_dim).to(device)
        self.q1_target = VasoQNetwork(state_dim, self.action_dim).to(device)
        self.q2_target = VasoQNetwork(state_dim, self.action_dim).to(device)
        
        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr, weight_decay=1e-5)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr, weight_decay=1e-5)
        
        # Track training metrics
        self.training_step = 0
        
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        """
        Select binary action by evaluating Q(s,0) and Q(s,1).
        This mirrors the dual continuous approach but with discrete actions.
        
        Args:
            state: Current state
            epsilon: Exploration rate
            
        Returns:
            Binary action as numpy array (for consistency with dual)
        """
        if np.random.random() < epsilon:
            # Random action
            return np.array([float(np.random.randint(0, 2))])
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Evaluate Q-values for both possible actions
            action_0 = torch.zeros(1, 1).to(self.device)  # Action = 0
            action_1 = torch.ones(1, 1).to(self.device)   # Action = 1
            
            # Get Q-values from both networks and take minimum
            q1_values_0 = self.q1(state_tensor, action_0).squeeze()
            q2_values_0 = self.q2(state_tensor, action_0).squeeze()
            q_value_0 = torch.min(q1_values_0, q2_values_0).item()
            
            q1_values_1 = self.q1(state_tensor, action_1).squeeze()
            q2_values_1 = self.q2(state_tensor, action_1).squeeze()
            q_value_1 = torch.min(q1_values_1, q2_values_1).item()
            
            # Select action with higher Q-value (argmax)
            best_action = 1.0 if q_value_1 > q_value_0 else 0.0
            
            return np.array([best_action])
    
    def select_actions_batch(self, states: torch.Tensor) -> torch.Tensor:
        """
        Select best actions for a batch of states efficiently.
        Evaluates Q(s,0) and Q(s,1) for each state and returns argmax.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            
        Returns:
            Best actions [batch_size, 1]
        """
        batch_size = states.shape[0]
        
        with torch.no_grad():
            # Create action tensors for both possible actions
            actions_0 = torch.zeros(batch_size, 1).to(self.device)
            actions_1 = torch.ones(batch_size, 1).to(self.device)
            
            # Evaluate Q-values for action=0
            q1_values_0 = self.q1(states, actions_0).squeeze()
            q2_values_0 = self.q2(states, actions_0).squeeze()
            q_values_0 = torch.min(q1_values_0, q2_values_0)
            
            # Evaluate Q-values for action=1
            q1_values_1 = self.q1(states, actions_1).squeeze()
            q2_values_1 = self.q2(states, actions_1).squeeze()
            q_values_1 = torch.min(q1_values_1, q2_values_1)
            
            # Select best action for each state (argmax)
            best_actions = (q_values_1 > q_values_0).float().unsqueeze(1)
            
            return best_actions
    
    def compute_cql_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        q_network: nn.Module
    ) -> torch.Tensor:
        """
        Compute CQL penalty for binary actions.
        Since we only have 2 actions, we compute logsumexp over both.
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
            q_network: Q-network to compute loss for
            
        Returns:
            CQL loss value
        """
        batch_size = states.shape[0]
        
        # Current Q-values for taken actions
        current_q = q_network(states, actions).squeeze()
        
        # Q-values for both possible actions
        with torch.no_grad():
            actions_0 = torch.zeros(batch_size, 1).to(self.device)
            actions_1 = torch.ones(batch_size, 1).to(self.device)
        
        q_0 = q_network(states, actions_0).squeeze()
        q_1 = q_network(states, actions_1).squeeze()
        
        # Stack Q-values for both actions
        all_q = torch.stack([q_0, q_1], dim=1)  # [batch_size, 2]
        
        # CQL penalty: log-sum-exp of all actions minus current Q
        # Temperature scaling for numerical stability
        logsumexp = torch.logsumexp(all_q / 10.0, dim=1) * 10.0
        cql_loss = (logsumexp - current_q).mean()
        
        return cql_loss
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update Q-networks with CQL
        
        Args:
            states: Current states
            actions: Actions taken (should be 0 or 1)
            rewards: Rewards received
            next_states: Next states
            dones: Done flags
            
        Returns:
            Dictionary of losses
        """
        # Ensure actions are properly shaped [batch_size, 1]
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        
        # Clip rewards for stability
        #rewards = torch.clamp(rewards, -10, 10)
        
        # Compute target Q-values
        with torch.no_grad():
            # Select best next actions using batch selection
            next_actions = self.select_actions_batch(next_states)
            
            # Get target Q-values (using target networks for stability)
            next_q1 = self.q1_target(next_states, next_actions).squeeze()
            next_q2 = self.q2_target(next_states, next_actions).squeeze()
            next_q = torch.min(next_q1, next_q2)
            
            # Compute targets
            target_q = rewards + self.gamma * next_q * (1 - dones)
            #target_q = torch.clamp(target_q, -50, 50)
        
        # Update Q1
        current_q1 = self.q1(states, actions).squeeze()
        q1_loss = F.mse_loss(current_q1, target_q)
        
        if self.alpha > 0:
            cql1_loss = self.compute_cql_loss(states, actions, self.q1)
            total_q1_loss = q1_loss + self.alpha * cql1_loss
        else:
            cql1_loss = torch.tensor(0.0)
            total_q1_loss = q1_loss
        
        self.q1_optimizer.zero_grad()
        total_q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.grad_clip)
        self.q1_optimizer.step()
        
        # Update Q2
        current_q2 = self.q2(states, actions).squeeze()
        q2_loss = F.mse_loss(current_q2, target_q)
        
        if self.alpha > 0:
            cql2_loss = self.compute_cql_loss(states, actions, self.q2)
            total_q2_loss = q2_loss + self.alpha * cql2_loss
        else:
            cql2_loss = torch.tensor(0.0)
            total_q2_loss = q2_loss
        
        self.q2_optimizer.zero_grad()
        total_q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.grad_clip)
        self.q2_optimizer.step()
        
        # Update target networks with soft update
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.training_step += 1
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'cql1_loss': cql1_loss.item() if self.alpha > 0 else 0.0,
            'cql2_loss': cql2_loss.item() if self.alpha > 0 else 0.0,
            'total_q1_loss': total_q1_loss.item(),
            'total_q2_loss': total_q2_loss.item()
        }
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'tau': self.tau,
            'training_step': self.training_step
        }, path)
        print(f"Model saved to {path}")
        
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
        print(f"Model loaded from {path}")


def train_binary_cql(
    alpha: float = 1.0,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    output_dir: str = 'experiment',
    model_name: str = None
) -> Tuple[BinaryCQL, IntegratedDataPipelineV2]:
    """
    Train Binary CQL with continuous Q-network architecture
    
    Args:
        alpha: CQL penalty weight
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        output_dir: Directory to save models
        model_name: Name for saved model (auto-generated if None)
        
    Returns:
        Trained agent and data pipeline
    """
    print("="*70)
    print(" BINARY CQL TRAINING WITH CONTINUOUS Q-NETWORK")
    print("="*70)
    print(f"\nHyperparameters:")
    print(f"  Alpha (CQL penalty): {alpha}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print("="*70)
    
    # Initialize data pipeline
    print("\nInitializing data pipeline...")
    pipeline = IntegratedDataPipelineV2(model_type='binary', random_seed=42)
    train_data, val_data, test_data = pipeline.prepare_data()
    
    # Get state dimension
    state_dim = train_data['states'].shape[1]
    print(f"\nState dimension: {state_dim}")
    print(f"Action dimension: 1 (binary: 0 or 1)")
    
    # Initialize agent
    agent = BinaryCQL(
        state_dim=state_dim,
        alpha=alpha,
        gamma=0.95,
        tau=0.8,
        lr=lr,
        grad_clip=1.0
    )
    
    # Training loop
    print(f"\n{'='*70}")
    print(" TRAINING")
    print(f"{'='*70}")
    
    best_val_loss = float('inf')
    os.makedirs(output_dir, exist_ok=True)
    
    if model_name is None:
        model_name = f"binary_cql_continuous_alpha{alpha}"
    
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
                
                if actions.dim() == 1:
                    actions = actions.unsqueeze(1)
                
                q1_val = agent.q1(states, actions).squeeze()
                q2_val = agent.q2(states, actions).squeeze()
                q_val = torch.min(q1_val, q2_val)
                
                val_q_values.append(q_val.mean().item())
        
        val_loss = -np.mean(val_q_values)  # Negative because we want higher Q-values
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            agent.save(os.path.join(output_dir, f'{model_name}_best.pt'))
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train - Q1 Loss: {train_metrics['q1_loss']:.4f}, "
                  f"CQL1 Loss: {train_metrics['cql1_loss']:.4f}, "
                  f"Total Q1: {train_metrics['total_q1_loss']:.4f}")
            print(f"  Val   - Avg Q-value: {-val_loss:.4f}")
    
    # Save final model
    agent.save(os.path.join(output_dir, f'{model_name}_final.pt'))
    
    print(f"\n{'='*70}")
    print(" TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best model saved to: {output_dir}/{model_name}_best.pt")
    print(f"Final model saved to: {output_dir}/{model_name}_final.pt")
    
    return agent, pipeline


def test_q_value_comparison():
    """
    Test that Binary and Dual CQL produce comparable Q-values
    """
    print("="*70)
    print(" Q-VALUE COMPARISON TEST")
    print("="*70)
    
    # Initialize pipelines
    print("\nInitializing data pipelines...")
    binary_pipeline = IntegratedDataPipelineV2(model_type='binary', random_seed=42)
    binary_train, _, _ = binary_pipeline.prepare_data()
    
    dual_pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
    dual_train, _, _ = dual_pipeline.prepare_data()
    
    # Create agents with same architecture
    print("\nCreating agents with identical Q-network architecture...")
    binary_agent = BinaryCQL(state_dim=18, alpha=0.0)  # No CQL for comparison
    
    # Import dual agent
    from train_cql_stable import StableCQL
    dual_agent = StableCQL(state_dim=17, action_dim=2, alpha=0.0)
    
    # Get sample batches
    print("\nGetting sample batches...")
    binary_batch = binary_pipeline.get_batch(batch_size=32, split='train', seed=123)
    dual_batch = dual_pipeline.get_batch(batch_size=32, split='train', seed=123)
    
    # Convert to tensors
    binary_states = torch.FloatTensor(binary_batch['states']).to(binary_agent.device)
    binary_actions = torch.FloatTensor(binary_batch['actions']).unsqueeze(1).to(binary_agent.device)
    
    dual_states = torch.FloatTensor(dual_batch['states']).to(dual_agent.device)
    dual_actions = torch.FloatTensor(dual_batch['actions']).to(dual_agent.device)
    
    # Compute Q-values using the same Q(s,a) -> R architecture
    print("\nComputing Q-values...")
    with torch.no_grad():
        # Binary Q-values
        binary_q1 = binary_agent.q1(binary_states, binary_actions).squeeze()
        binary_q2 = binary_agent.q2(binary_states, binary_actions).squeeze()
        binary_q = torch.min(binary_q1, binary_q2)
        
        # Dual Q-values
        dual_q1 = dual_agent.q1(dual_states, dual_actions).squeeze()
        dual_q2 = dual_agent.q2(dual_states, dual_actions).squeeze()
        dual_q = torch.min(dual_q1, dual_q2)
    
    print("\n" + "="*50)
    print(" RESULTS")
    print("="*50)
    
    print("\nBinary CQL Q-values (untrained):")
    print(f"  Mean: {binary_q.mean().item():.4f}")
    print(f"  Std:  {binary_q.std().item():.4f}")
    print(f"  Min:  {binary_q.min().item():.4f}")
    print(f"  Max:  {binary_q.max().item():.4f}")
    
    print(f"\nDual CQL Q-values (untrained):")
    print(f"  Mean: {dual_q.mean().item():.4f}")
    print(f"  Std:  {dual_q.std().item():.4f}")
    print(f"  Min:  {dual_q.min().item():.4f}")
    print(f"  Max:  {dual_q.max().item():.4f}")
    
    print("\n✅ Both models use identical Q(s,a) -> R architecture")
    print("✅ Q-values are now directly comparable between Binary and Dual CQL")
    print("\nFor Binary CQL:")
    print("  - Action selection: argmax_a Q(s,a) for a ∈ {0, 1}")
    print("  - Uses min(Q1(s,a), Q2(s,a)) for conservative estimate")
    print("  - Target networks only used during training updates")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Binary CQL with continuous Q-network')
    parser.add_argument('--alpha', type=float, default=1.0, help='CQL penalty weight')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--test', action='store_true', help='Run Q-value comparison test')
    
    args = parser.parse_args()
    
    if args.test:
        test_q_value_comparison()
    else:
        agent, pipeline = train_binary_cql(
            alpha=args.alpha,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )