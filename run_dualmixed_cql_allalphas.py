#!/usr/bin/env python3
"""
Training script for Dual Mixed CQL with multiple alpha values
VP1: Binary (0 or 1)
VP2: Continuous (0 to 0.5 mcg/kg/min)
Uses consistent hyperparameters:
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

# Force unbuffered output
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__


# ============================================================================
# Dual Mixed CQL Implementation (VP1 Binary, VP2 Continuous)
# ============================================================================

class DualMixedQNetwork(nn.Module):
    """
    Q-network for mixed actions: VP1 binary, VP2 continuous
    Q(s, vp1, vp2) where vp1 ∈ {0,1} and vp2 ∈ [0, 0.5]
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Network takes state + vp1 (binary) + vp2 (continuous) as input
        input_dim = state_dim + 2
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 64)
        self.fc4 = nn.Linear(64, 1)
        
        # Better initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]
            action: [batch_size, 2] - [vp1_binary, vp2_continuous]
        Returns:
            q_value: [batch_size, 1]
        """
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DualMixedCQL:
    """Dual CQL with VP1 binary and VP2 continuous"""
    
    def __init__(
        self,
        state_dim: int,
        alpha: float = 1.0,
        gamma: float = 0.95,
        tau: float = 0.8,
        lr: float = 1e-3,
        grad_clip: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.state_dim = state_dim
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip
        self.device = device
        
        # Q-networks
        self.q1 = DualMixedQNetwork(state_dim).to(device)
        self.q2 = DualMixedQNetwork(state_dim).to(device)
        self.q1_target = DualMixedQNetwork(state_dim).to(device)
        self.q2_target = DualMixedQNetwork(state_dim).to(device)
        
        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
    
    def select_action(self, state: np.ndarray, num_samples: int = 50) -> np.ndarray:
        """
        Select best action for a given state by sampling.
        VP1 is binary, VP2 is continuous [0, 0.5]
        """
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            elif state.dim() == 1:
                state = state.unsqueeze(0)
            
            batch_size = state.shape[0]
            
            # Sample actions: VP1 binary, VP2 continuous
            vp1_samples = torch.randint(0, 2, (batch_size, num_samples, 1)).float().to(self.device)
            vp2_samples = torch.rand(batch_size, num_samples, 1).to(self.device) * 0.5
            action_samples = torch.cat([vp1_samples, vp2_samples], dim=-1)
            
            # Evaluate Q-values for all samples (batched)
            state_expanded = state.unsqueeze(1).expand(-1, num_samples, -1)
            state_flat = state_expanded.reshape(-1, state.shape[-1])
            action_flat = action_samples.reshape(-1, 2)
            
            q1_vals = self.q1(state_flat, action_flat).reshape(batch_size, num_samples)
            q2_vals = self.q2(state_flat, action_flat).reshape(batch_size, num_samples)
            q_values = torch.min(q1_vals, q2_vals)
            best_idx = q_values.argmax(dim=1)
            
            batch_idx = torch.arange(batch_size).to(self.device)
            best_actions = action_samples[batch_idx, best_idx]
            
            return best_actions.squeeze(0).cpu().numpy() if batch_size == 1 else best_actions.cpu().numpy()
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> dict:
        """Update Q-networks with CQL loss"""
        
        # Compute target Q-value
        with torch.no_grad():
            # Sample multiple next actions for better Q estimation
            num_samples = 10
            batch_size = next_states.shape[0]
            
            vp1_samples = torch.randint(0, 2, (batch_size, num_samples, 1)).float().to(self.device)
            vp2_samples = torch.rand(batch_size, num_samples, 1).to(self.device) * 0.5
            next_action_samples = torch.cat([vp1_samples, vp2_samples], dim=-1)
            
            # Evaluate next Q-values
            next_states_exp = next_states.unsqueeze(1).expand(-1, num_samples, -1)
            next_states_flat = next_states_exp.reshape(-1, next_states.shape[-1])
            next_actions_flat = next_action_samples.reshape(-1, 2)
            
            next_q1 = self.q1_target(next_states_flat, next_actions_flat).reshape(batch_size, num_samples)
            next_q2 = self.q2_target(next_states_flat, next_actions_flat).reshape(batch_size, num_samples)
            next_q = torch.min(next_q1, next_q2)
            next_q = next_q.max(dim=1)[0].squeeze()
            
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Update Q1
        current_q1 = self.q1(states, actions).squeeze()
        q1_loss = F.mse_loss(current_q1, target_q)
        
        # CQL penalty for Q1
        if self.alpha > 0:
            # Sample random actions
            num_samples = 10
            batch_size = states.shape[0]
            
            random_vp1 = torch.randint(0, 2, (batch_size, num_samples, 1)).float().to(self.device)
            random_vp2 = torch.rand(batch_size, num_samples, 1).to(self.device) * 0.5
            random_actions = torch.cat([random_vp1, random_vp2], dim=-1)
            
            states_expanded = states.unsqueeze(1).expand(-1, num_samples, -1)
            states_flat = states_expanded.reshape(-1, states.shape[-1])
            actions_flat = random_actions.reshape(-1, 2)
            
            random_q1 = self.q1(states_flat, actions_flat).reshape(batch_size, num_samples)
            logsumexp_q1 = torch.logsumexp(random_q1 / 10.0, dim=1) * 10.0
            cql1_loss = (logsumexp_q1 - current_q1).mean()
        else:
            cql1_loss = torch.tensor(0.0)
        
        total_q1_loss = q1_loss + self.alpha * cql1_loss
        
        self.q1_optimizer.zero_grad()
        total_q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.grad_clip)
        self.q1_optimizer.step()
        
        # Update Q2 (similar process)
        current_q2 = self.q2(states, actions).squeeze()
        q2_loss = F.mse_loss(current_q2, target_q)
        
        if self.alpha > 0:
            # Reuse the same random actions
            random_q2 = self.q2(states_flat, actions_flat).reshape(batch_size, num_samples)
            logsumexp_q2 = torch.logsumexp(random_q2 / 10.0, dim=1) * 10.0
            cql2_loss = (logsumexp_q2 - current_q2).mean()
        else:
            cql2_loss = torch.tensor(0.0)
        
        total_q2_loss = q2_loss + self.alpha * cql2_loss
        
        self.q2_optimizer.zero_grad()
        total_q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.grad_clip)
        self.q2_optimizer.step()
        
        # Soft update targets
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'cql1_loss': cql1_loss.item() if isinstance(cql1_loss, torch.Tensor) else cql1_loss,
            'cql2_loss': cql2_loss.item() if isinstance(cql2_loss, torch.Tensor) else cql2_loss
        }


def train_dual_mixed_cql(alpha=0.001):
    """Train Dual Mixed CQL with VP1 binary, VP2 continuous"""
    print("\n" + "="*70, flush=True)
    print(f" DUAL MIXED CQL TRAINING WITH ALPHA={alpha}", flush=True)
    print("="*70, flush=True)
    
    # Initialize data pipeline
    print("\nInitializing Dual CQL data pipeline...", flush=True)
    pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
    train_data, val_data, test_data = pipeline.prepare_data()
    
    # Get state dimension
    state_dim = train_data['states'].shape[1]
    
    # Print settings
    print("\n" + "="*70, flush=True)
    print("SETTINGS:", flush=True)
    print(f"  State dimension: {state_dim}", flush=True)
    print(f"  Action dimension: 2 (VP1: binary 0/1, VP2: continuous 0-0.5)", flush=True)
    print(f"  ALPHA = {alpha} ({'no penalty' if alpha == 0 else 'conservative penalty'})", flush=True)
    print("  TAU = 0.8 (target network update)", flush=True)
    print("  LR = 0.001 (learning rate)", flush=True)
    print("  BATCH_SIZE = 128", flush=True)
    print("  EPOCHS = 100", flush=True)
    print("="*70, flush=True)
    
    # Initialize agent with specified parameters
    agent = DualMixedCQL(
        state_dim=state_dim,
        alpha=alpha,  # CQL penalty parameter
        gamma=0.95,
        tau=0.8,      # As specified
        lr=1e-3,      # As specified
        grad_clip=1.0
    )
    
    # Training loop
    epochs = 100
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
            'cql1_loss': 0, 'cql2_loss': 0
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
                
                q1_val = agent.q1(states, actions).squeeze()
                q2_val = agent.q2(states, actions).squeeze()
                q_val = torch.min(q1_val, q2_val)
                
                val_q_values.append(q_val.mean().item())
        
        val_loss = -np.mean(val_q_values)  # Negative because we want higher Q-values
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'q1_state_dict': agent.q1.state_dict(),
                'q2_state_dict': agent.q2.state_dict(),
                'q1_target_state_dict': agent.q1_target.state_dict(),
                'q2_target_state_dict': agent.q2_target.state_dict(),
                'state_dim': state_dim,
                'alpha': alpha,
                'gamma': 0.95,
                'tau': 0.8
            }, f'experiment/dual_rev_cql_alpha{alpha:.4f}_best.pt')
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}: "
                  f"Q1 Loss={train_metrics['q1_loss']:.4f}, "
                  f"CQL1 Loss={train_metrics['cql1_loss']:.4f}, "
                  f"Val Q-value={-val_loss:.4f}, "
                  f"Time={elapsed/60:.1f}min", flush=True)
    
    # Save final model
    torch.save({
        'q1_state_dict': agent.q1.state_dict(),
        'q2_state_dict': agent.q2.state_dict(),
        'q1_target_state_dict': agent.q1_target.state_dict(),
        'q2_target_state_dict': agent.q2_target.state_dict(),
        'state_dim': state_dim,
        'alpha': alpha,
        'gamma': 0.95,
        'tau': 0.8
    }, f'experiment/dual_rev_cql_alpha{alpha:.4f}_final.pt')
    
    total_time = time.time() - start_time
    print(f"\n✅ Dual Mixed CQL (alpha={alpha}) training completed in {total_time/60:.1f} minutes!", flush=True)
    print("Models saved:", flush=True)
    print(f"  - experiment/dual_rev_cql_alpha{alpha:.4f}_best.pt", flush=True)
    print(f"  - experiment/dual_rev_cql_alpha{alpha:.4f}_final.pt", flush=True)
    
    return agent, pipeline


def main():
    """Train Dual Mixed CQL with multiple alpha values"""
    print("="*70, flush=True)
    print(" DUAL MIXED CQL TRAINING - ALL ALPHAS", flush=True)
    print("="*70, flush=True)
    print("\nThis script trains Dual Mixed CQL models with:", flush=True)
    print("  - VP1: Binary (0 or 1)", flush=True)
    print("  - VP2: Continuous (0 to 0.5 mcg/kg/min)", flush=True)
    print("  - Alpha values: 0.0, 0.001, 0.01", flush=True)
    print("  - Consistent hyperparameters (tau=0.8, lr=1e-3)", flush=True)
    
    # Alpha values to train
    alphas = [0.0, 0.001, 0.01]
    
    # Train models for each alpha
    for alpha in alphas:
        agent, pipeline = train_dual_mixed_cql(alpha=alpha)
    
    print("\n" + "="*70, flush=True)
    print(" ALL TRAINING COMPLETE", flush=True)
    print("="*70, flush=True)
    print("\nModels saved in experiment/ directory:", flush=True)
    for alpha in alphas:
        print(f"  Alpha {alpha}: dual_rev_cql_alpha{alpha:.4f}_*.pt", flush=True)


if __name__ == "__main__":
    main()