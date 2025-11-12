#!/usr/bin/env python3
"""
Training script for Unified Stepwise CQL with multiple alpha values
VP1: Binary (0 or 1)
VP2: Stepwise changes from current dose (-0.1, -0.05, 0, +0.05, +0.1 mcg/kg/min)
Total: 10 discrete actions (2 VP1 × 5 VP2 changes)

Uses IntegratedDataPipelineV2 for consistent data handling
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
import pdb

# Import our unified pipeline
from integrated_data_pipeline_v2_simple_reward import IntegratedDataPipelineV2

# Force unbuffered output
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__


# ============================================================================
# Stepwise Action Space Definition
# ============================================================================

class StepwiseActionSpace:
    """Manages stepwise action space for VP2 with customizable range"""
    
    # VP2 dose bounds
    MIN_STEP = 0.05  # Fixed minimum step size (mcg/kg/min)
    VP2_MIN = MIN_STEP  # Minimum VP2 dose (0.05 to ensure always > 0)
    VP2_MAX = 0.5  # Standard therapeutic range for norepinephrine (mcg/kg/min)
    
    def __init__(self, max_step: float = 0.1):
        """
        Initialize stepwise action space
        
        Args:
            max_step: Maximum step size in mcg/kg/min (default 0.1)
                     Creates steps from -max_step to +max_step in increments of 0.05
                     E.g., max_step=0.2 gives: [-0.2, -0.15, -0.1, -0.05, 0, +0.05, +0.1, +0.15, +0.2]
        """
        self.max_step = max_step
        
        # Generate VP2 change actions from -max_step to +max_step in MIN_STEP increments
        # Use round() to handle floating point precision issues
        n_steps = round(max_step / self.MIN_STEP)
        self.VP2_CHANGES = np.arange(-n_steps, n_steps + 1) * self.MIN_STEP
        
        self.n_vp1_actions = 2  # Binary: 0 or 1
        self.n_vp2_actions = len(self.VP2_CHANGES)
        self.n_actions = self.n_vp1_actions * self.n_vp2_actions
        
        # VP2 discretization for state representation
        # Total bins = (0.5 - 0.05) / 0.05 + 1 = 10 bins
        # Bins: [0.05, 0.10, 0.15, ..., 0.50]
        self.n_vp2_bins = int((self.VP2_MAX - self.VP2_MIN) / self.MIN_STEP) + 1
        # Bin centers: [0.05, 0.10, 0.15, ..., 0.50]
        self.vp2_bin_centers = np.arange(self.n_vp2_bins) * self.MIN_STEP + self.VP2_MIN
        
        # Create action mapping for easy lookup
        self.action_map = []
        for vp1 in range(self.n_vp1_actions):
            for vp2_idx in range(self.n_vp2_actions):
                self.action_map.append((vp1, vp2_idx))
        
        # Precompute VP2 changes tensor for batched operations
        # This maps each action index to its corresponding VP2 change
        self.vp2_changes_per_action = np.zeros(self.n_actions)
        for action_idx in range(self.n_actions):
            _, vp2_change_idx = self.decode_action(action_idx)
            self.vp2_changes_per_action[action_idx] = self.VP2_CHANGES[vp2_change_idx]
    
    def decode_action(self, action_idx: int) -> tuple:
        """Decode discrete action to VP1 and VP2 change index"""
        return self.action_map[action_idx]
    
    def apply_action(self, action_idx: int, current_vp2: float) -> tuple:
        """
        Apply discrete action and return continuous VP1 and new VP2 values
        
        Args:
            action_idx: Discrete action index
            current_vp2: Current VP2 dose
            
        Returns:
            (vp1_value, new_vp2_value)
        """
        vp1, vp2_change_idx = self.decode_action(action_idx)
        vp2_change = self.VP2_CHANGES[vp2_change_idx]
        
        # Apply change with bounds checking
        new_vp2 = current_vp2 + vp2_change
        new_vp2 = np.clip(new_vp2, self.VP2_MIN, self.VP2_MAX)
        
        return float(vp1), new_vp2
    
    def vp2_to_bin(self, vp2_dose: float) -> int:
        """
        Convert continuous VP2 dose to discrete bin index
        
        Args:
            vp2_dose: VP2 dose in mcg/kg/min
            
        Returns:
            Bin index (0 to n_vp2_bins-1)
        """
        # Clip to valid range
        vp2_dose = np.clip(vp2_dose, self.VP2_MIN, self.VP2_MAX)
        # Find bin index (0-indexed from VP2_MIN)
        bin_idx = int(round((vp2_dose - self.VP2_MIN) / self.MIN_STEP))
        return min(bin_idx, self.n_vp2_bins - 1)
    
    def vp2_to_one_hot(self, vp2_dose: float) -> np.ndarray:
        """
        Convert VP2 dose to one-hot encoded vector
        
        Args:
            vp2_dose: VP2 dose in mcg/kg/min
            
        Returns:
            One-hot vector of shape (n_vp2_bins,)
        """
        bin_idx = self.vp2_to_bin(vp2_dose)
        one_hot = np.zeros(self.n_vp2_bins)
        one_hot[bin_idx] = 1.0
        return one_hot
    
    def get_valid_actions(self, current_vp2: float) -> np.ndarray:
        """
        Get mask of valid actions given current VP2 dose.
        Used to prevent actions that would push VP2 out of bounds.
        
        Args:
            current_vp2: Current VP2 dose
            
        Returns:
            Boolean mask of shape (n_actions,) where True indicates valid action
        """
        valid = np.ones(self.n_actions, dtype=bool)
        
        for action_idx in range(self.n_actions):
            vp1, vp2_change_idx = self.decode_action(action_idx)
            vp2_change = self.VP2_CHANGES[vp2_change_idx]
            new_vp2 = current_vp2 + vp2_change
            
            # Mark invalid if resulting dose would be out of bounds
            eps = 1e-6
            if new_vp2 < (self.VP2_MIN - eps) or new_vp2 > (self.VP2_MAX + eps):
                valid[action_idx] = False
        
        return valid
    
    def get_valid_actions_batch(self, current_vp2_doses: torch.Tensor) -> torch.Tensor:
        """
        Get mask of valid actions for a batch of VP2 doses.
        Batched version for efficient computation.
        
        Args:
            current_vp2_doses: Tensor of current VP2 doses, shape (batch_size,)
            
        Returns:
            Boolean mask tensor of shape (batch_size, n_actions) where True indicates valid action
        """
        device = current_vp2_doses.device
        
        # Use precomputed VP2 changes tensor
        vp2_changes = torch.tensor(self.vp2_changes_per_action, device=device, dtype=current_vp2_doses.dtype)
        
        # Broadcast to compute all resulting VP2 values
        # current_vp2_doses: (batch_size, 1)
        # vp2_changes: (1, n_actions)
        # new_vp2_values: (batch_size, n_actions)
        current_vp2_doses = current_vp2_doses.unsqueeze(1)  # (batch_size, 1)
        vp2_changes = vp2_changes.unsqueeze(0)  # (1, n_actions)
        new_vp2_values = current_vp2_doses + vp2_changes  # (batch_size, n_actions)
        
        # Check bounds
        eps = 1e-6
        valid_mask = (new_vp2_values >= (self.VP2_MIN - eps)) & (new_vp2_values <= (self.VP2_MAX + eps))
        
        return valid_mask


# ============================================================================
# Stepwise Q-Network Implementation  
# ============================================================================

class StepwiseQNetwork(nn.Module):
    """
    Q-network for stepwise actions
    Q(s, action) -> Q-value
    
    The action is a one-hot encoded discrete action (or action index to be embedded)
    State includes:
    - Base features: 17 dims from IntegratedDataPipelineV2
    - VP2 state: n_vp2_bins dims (one-hot encoded current VP2 dose)
    - Total state_dim: 17 + n_vp2_bins dims (e.g., 27 for default)
    """
    
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        # Network takes state (already includes VP2 one-hot) + action as input
        # state_dim should already be 17 + n_vp2_bins
        input_dim = state_dim + n_actions
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 64)
        self.fc4 = nn.Linear(64, 1)
        
        # Better initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        
        self.n_actions = n_actions
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim] - base features + VP2 one-hot
            action: [batch_size] - discrete action indices, or
                    [batch_size, n_actions] - one-hot encoded actions
        Returns:
            q_value: [batch_size, 1]
        """
        # Convert action indices to one-hot if needed
        if action.dim() == 1 or (action.dim() == 2 and action.shape[1] == 1):
            # Action is indices, convert to one-hot
            if action.dim() == 2:
                action = action.squeeze(1)
            action_one_hot = F.one_hot(action.long(), num_classes=self.n_actions).float()
        else:
            # Action is already one-hot
            action_one_hot = action
        
        # Concatenate state (already includes VP2 one-hot) with action
        x = torch.cat([state, action_one_hot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# ============================================================================
# Unified Stepwise CQL Implementation
# ============================================================================

class UnifiedStepwiseCQL:
    """Stepwise CQL with discrete actions using IntegratedDataPipelineV2"""
    
    def __init__(
        self,
        state_dim: int,  # Should be 17 (base features) + 10 (VP2 bins) = 27
        max_step: float = 0.1,
        alpha: float = 1.0,
        gamma: float = 0.95,
        tau: float = 0.8,
        lr: float = 1e-3,
        grad_clip: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.state_dim = state_dim  # Total state dim including VP2 bins
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip
        self.device = device
        
        # Initialize action space
        self.action_space = StepwiseActionSpace(max_step=max_step)
        self.n_actions = self.action_space.n_actions
        self.n_vp2_bins = self.action_space.n_vp2_bins
        
        # Q-networks - state_dim already includes VP2 bins
        self.q1 = StepwiseQNetwork(state_dim, self.n_actions).to(device)
        self.q2 = StepwiseQNetwork(state_dim, self.n_actions).to(device)
        self.q1_target = StepwiseQNetwork(state_dim, self.n_actions).to(device)
        self.q2_target = StepwiseQNetwork(state_dim, self.n_actions).to(device)
        
        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
    
    def select_action(self, state: torch.Tensor, current_vp2: float) -> int:
        """
        Select best valid discrete action for given state.
        State should already include VP2 one-hot encoding.
        
        Args:
            state: State tensor including VP2 one-hot [batch_size, state_dim]
            current_vp2: Current VP2 dose for determining valid actions
            
        Returns:
            Best discrete action index
        """
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            batch_size = state.shape[0]
            
            # Get valid action mask
            valid_actions = self.action_space.get_valid_actions(current_vp2)
            valid_actions_tensor = torch.tensor(valid_actions, device=self.device)
            
            # Batch evaluate Q-values for all discrete actions
            # Expand state to [batch_size * n_actions, state_dim]
            state_expanded = state.unsqueeze(1).expand(-1, self.n_actions, -1)
            state_flat = state_expanded.reshape(-1, state.shape[-1])
            
            # Create action indices for all actions
            action_indices = torch.arange(self.n_actions, device=self.device)
            action_indices = action_indices.unsqueeze(0).expand(batch_size, -1)
            action_flat = action_indices.reshape(-1)
            
            # Compute Q-values for all actions in one forward pass
            q1_vals = self.q1(state_flat, action_flat).reshape(batch_size, self.n_actions)
            q2_vals = self.q2(state_flat, action_flat).reshape(batch_size, self.n_actions)
            q_values = torch.min(q1_vals, q2_vals)
            
            # Mask invalid actions
            q_values[~valid_actions_tensor.unsqueeze(0).expand(batch_size, -1)] = -float('inf')
            
            # Select best valid action
            best_action = q_values.argmax(dim=1)
            
            return best_action.item() if batch_size == 1 else best_action.cpu().numpy()
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,  # Discrete action indices
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        current_vp2_doses: torch.Tensor,  # Current VP2 doses for each sample
        next_vp2_doses: torch.Tensor  # Next VP2 doses for each sample
    ) -> dict:
        """Update Q-networks with CQL loss for discrete stepwise actions"""
        
        batch_size = states.shape[0]
        
        # Compute target Q-value
        with torch.no_grad():
            # For next states, evaluate all valid actions and take max
            # Expand next_states for all actions
            next_states_exp = next_states.unsqueeze(1).expand(-1, self.n_actions, -1)
            next_states_flat = next_states_exp.reshape(-1, next_states.shape[-1])
            
            # Create action indices for all actions
            action_indices = torch.arange(self.n_actions, device=self.device)
            action_indices = action_indices.unsqueeze(0).expand(batch_size, -1)
            action_flat = action_indices.reshape(-1)
            
            # Compute Q-values for all actions
            next_q1 = self.q1_target(next_states_flat, action_flat).reshape(batch_size, self.n_actions)
            next_q2 = self.q2_target(next_states_flat, action_flat).reshape(batch_size, self.n_actions)
            next_q = torch.min(next_q1, next_q2)
            
            # Apply valid action masks for next states (batched)
            valid_masks = self.action_space.get_valid_actions_batch(next_vp2_doses)
            next_q[~valid_masks] = -float('inf')
            
            next_q_max = next_q.max(dim=1)[0]
            target_q = rewards + self.gamma * next_q_max * (1 - dones)
        
        # Update Q1
        current_q1 = self.q1(states, actions).squeeze()
        q1_loss = F.mse_loss(current_q1, target_q)
        
        # CQL penalty for Q1
        if self.alpha > 0:
            # Compute Q-values for all discrete actions (for CQL penalty)
            states_exp = states.unsqueeze(1).expand(-1, self.n_actions, -1)
            states_flat = states_exp.reshape(-1, states.shape[-1])
            
            action_indices = torch.arange(self.n_actions, device=self.device)
            action_indices = action_indices.unsqueeze(0).expand(batch_size, -1)
            action_flat = action_indices.reshape(-1)
            
            all_q1 = self.q1(states_flat, action_flat).reshape(batch_size, self.n_actions)
            
            # Apply valid action masks for current states (batched)
            valid_masks = self.action_space.get_valid_actions_batch(current_vp2_doses)
            
            # Only compute logsumexp over valid actions
            # Set invalid actions to -inf for masking
            masked_q1 = all_q1.clone()
            masked_q1[~valid_masks] = -float('inf')
            
            # Conservative penalty: penalize high Q-values for out-of-distribution actions
            logsumexp_q1 = torch.logsumexp(masked_q1, dim=1)
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
            # Reuse the same expanded states and actions from Q1
            all_q2 = self.q2(states_flat, action_flat).reshape(batch_size, self.n_actions)
            
            # Apply same valid action masks (reuse from Q1)
            masked_q2 = all_q2.clone()
            masked_q2[~valid_masks] = -float('inf')
            
            logsumexp_q2 = torch.logsumexp(masked_q2, dim=1)
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


def prepare_stepwise_batch(batch, action_space, device):
    """
    Prepare batch from get_stepwise_batch for training.
    
    Args:
        batch: Batch from pipeline.get_stepwise_batch() with VP2 change info
        action_space: StepwiseActionSpace instance
        device: Device to place tensors on
    
    Returns:
        Dictionary with prepared batch data
    """
    batch_size = len(batch['states'])
    
    # Create one-hot encodings for VP2 bins (vectorized)
    vp2_current_one_hot = np.zeros((batch_size, action_space.n_vp2_bins))
    vp2_current_one_hot[np.arange(batch_size), batch['vp2_current_bins']] = 1.0
    
    vp2_next_one_hot = np.zeros((batch_size, action_space.n_vp2_bins))
    vp2_next_one_hot[np.arange(batch_size), batch['vp2_next_bins']] = 1.0
    
    # Augment states with VP2 one-hot encodings
    extended_states = np.concatenate([batch['states'], vp2_current_one_hot], axis=1)
    extended_next_states = np.concatenate([batch['next_states'], vp2_next_one_hot], axis=1)
    
    # Create binary VP1 array: 1 if VP1 > 0.5, 0 otherwise
    vp1_binary = (batch['vp1_actions'] > 0.5).astype(np.int64)  # Shape: (batch_size,)
    
    # VP2 changes are discrete bin differences from get_stepwise_batch
    # Map to action space VP2_CHANGES indices
    # VP2_CHANGES goes from -max_step to +max_step (e.g., [-0.1, -0.05, 0, +0.05, +0.1])
    # vp2_changes are in bins (e.g., -2, -1, 0, +1, +2)
    # Since each bin = 0.05 and VP2_CHANGES are also in 0.05 increments, 
    # we can directly map: bin_change -> index in VP2_CHANGES
    middle_idx = len(action_space.VP2_CHANGES) // 2  # Index for 0 change
    vp2_change_idx = np.clip(batch['vp2_changes'] + middle_idx, 
                              0, len(action_space.VP2_CHANGES) - 1)  # Shape: (batch_size,)
    
    # Combine VP1 and VP2 into single discrete action array
    # Formula: vp1 * n_vp2_actions + vp2_change_idx (vectorized)
    discrete_actions = vp1_binary * action_space.n_vp2_actions + vp2_change_idx  # Shape: (batch_size,)
    
    # Convert to tensors (Q-network will handle one-hot conversion internally)
    return {
        'states': torch.FloatTensor(extended_states).to(device),
        'next_states': torch.FloatTensor(extended_next_states).to(device),
        'actions': torch.LongTensor(discrete_actions).to(device),
        'rewards': torch.FloatTensor(batch['rewards']).to(device),
        'dones': torch.FloatTensor(batch['dones']).to(device),
        'current_vp2_doses': torch.FloatTensor(batch['vp2_current']).to(device),
        'next_vp2_doses': torch.FloatTensor(batch['vp2_next']).to(device),
        'patient_ids': batch['patient_ids']
    }


def train_unified_stepwise_cql(alpha=0.001, max_step=0.1):
    """Train Unified Stepwise CQL with discrete actions"""
    print("\n" + "="*70, flush=True)
    print(f" UNIFIED STEPWISE CQL TRAINING WITH ALPHA={alpha}", flush=True)
    print("="*70, flush=True)
    
    # Initialize data pipeline
    print("\nInitializing data pipeline...", flush=True)
    pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
    train_data, val_data, test_data = pipeline.prepare_data()
    
    # Initialize action space
    action_space = StepwiseActionSpace(max_step=max_step)
    
    # Get state dimensions
    base_state_dim = train_data['states'].shape[1]  # Should be 17
    total_state_dim = base_state_dim + action_space.n_vp2_bins  # 17 + 10 = 27
    
    # Print settings
    print("\n" + "="*70, flush=True)
    print("SETTINGS:", flush=True)
    print(f"  Base state dimension: {base_state_dim}", flush=True)
    print(f"  VP2 bins: {action_space.n_vp2_bins}", flush=True)
    print(f"  Total state dimension: {total_state_dim}", flush=True)
    print(f"  Max step size: {max_step} mcg/kg/min", flush=True)
    print(f"  VP2 changes: {action_space.VP2_CHANGES}", flush=True)
    print(f"  Total discrete actions: {action_space.n_actions}", flush=True)
    print(f"  ALPHA = {alpha} ({'no penalty' if alpha == 0 else 'conservative penalty'})", flush=True)
    print("  TAU = 0.8 (target network update)", flush=True)
    print("  LR = 0.001 (learning rate)", flush=True)
    print("  BATCH_SIZE = 128", flush=True)
    print("  EPOCHS = 100", flush=True)
    print("="*70, flush=True)
    
    # Initialize agent
    agent = UnifiedStepwiseCQL(
        state_dim=total_state_dim,
        max_step=max_step,
        alpha=alpha,
        gamma=0.95,
        tau=0.8,
        lr=1e-3,
        grad_clip=1.0
    )
    
    # Training loop
    epochs = 100  # Reduced for testing
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
        #min(1500, len(train_data['states']) // batch_size)  # Limit batches for testing
        
        for _ in range(n_batches):
            # Get batch from pipeline with VP2 change information
            batch = pipeline.get_stepwise_batch(batch_size=batch_size, split='train', 
                                                vp2_bins=action_space.n_vp2_bins)
            
            # Prepare batch for training (add one-hot encodings and convert to tensors)
            stepwise_batch = prepare_stepwise_batch(batch, action_space, agent.device)
            
            # Update agent
            metrics = agent.update(
                stepwise_batch['states'],
                stepwise_batch['actions'],
                stepwise_batch['rewards'],
                stepwise_batch['next_states'],
                stepwise_batch['dones'],
                stepwise_batch['current_vp2_doses'],
                stepwise_batch['next_vp2_doses']
            )
            
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
            for _ in range(5):  # Use 5 validation batches for testing
                batch = pipeline.get_stepwise_batch(batch_size=batch_size, split='val',
                                                    vp2_bins=action_space.n_vp2_bins)
                
                # Prepare batch for training (add one-hot encodings and convert to tensors)
                stepwise_batch = prepare_stepwise_batch(batch, action_space, agent.device)
                
                # Evaluate Q-values for taken actions
                q1_val = agent.q1(stepwise_batch['states'], stepwise_batch['actions']).squeeze()
                q2_val = agent.q2(stepwise_batch['states'], stepwise_batch['actions']).squeeze()
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
                'state_dim': total_state_dim,
                'max_step': max_step,
                'alpha': alpha,
                'gamma': 0.95,
                'tau': 0.8
            }, f'experiment/stepwise_cql_alpha{alpha:.6f}_maxstep{max_step:.1f}_best.pt')
        
        # Print progress
        if (epoch + 1) % 1 == 0:
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
        'state_dim': total_state_dim,
        'max_step': max_step,
        'alpha': alpha,
        'gamma': 0.95,
        'tau': 0.8
    }, f'experiment/stepwise_cql_alpha{alpha:.6f}_maxstep{max_step:.1f}_final.pt')
    
    total_time = time.time() - start_time
    print(f"\n✅ Stepwise CQL (alpha={alpha}) training completed in {total_time/60:.1f} minutes!", flush=True)
    print("Models saved:", flush=True)
    print(f"  - experiment/stepwise_cql_alpha{alpha:.6f}_maxstep{max_step:.1f}_best.pt", flush=True)
    print(f"  - experiment/stepwise_cql_alpha{alpha:.6f}_maxstep{max_step:.1f}_final.pt", flush=True)
    
    return agent, pipeline


def main():
    """Train Stepwise CQL with command-line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Stepwise CQL')
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='CQL regularization parameter')
    parser.add_argument('--max_step', type=float, default=0.2,
                        help='Maximum step size for VP2 changes')
    args = parser.parse_args()
    
    print("="*70, flush=True)
    print(f" STEPWISE CQL TRAINING", flush=True)
    print("="*70, flush=True)
    print(f"\nTraining configuration:", flush=True)
    print(f"  - Alpha: {args.alpha}", flush=True)
    print(f"  - Max step: {args.max_step}", flush=True)
    print(f"  - VP1: Binary (0 or 1)", flush=True)
    print(f"  - VP2: Stepwise changes with max_step={args.max_step}", flush=True)
    print(f"  - Hyperparameters: tau=0.8, lr=1e-3", flush=True)
    
    # Train model
    agent, pipeline = train_unified_stepwise_cql(alpha=args.alpha, max_step=args.max_step)
    
    print("\n" + "="*70, flush=True)
    print(" TRAINING COMPLETE", flush=True)
    print("="*70, flush=True)
    print(f"\nModel saved: stepwise_cql_alpha{args.alpha:.6f}_maxstep{args.max_step:.1f}_*.pt", flush=True)


if __name__ == "__main__":
    main()