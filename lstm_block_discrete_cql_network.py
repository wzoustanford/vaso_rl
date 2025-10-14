"""
LSTM Q-Network for Block Discrete CQL
======================================
Implements LSTM-based Q-networks for discrete block dosing actions
(e.g., Norepinephrine dosing levels) with sequential processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class LSTMDiscreteQNetwork(nn.Module):
    """
    Q-Network with LSTM for discrete block dosing actions.
    
    Architecture:
    - Feature extraction from states
    - LSTM for temporal modeling
    - Output layer for Q-values for each discrete action
    
    For discrete actions: outputs Q(s,a) for all possible actions
    """
    
    def __init__(
        self,
        state_dim: int,
        num_actions: int = 5,  # Number of discrete dosing levels
        hidden_dim: int = 256,
        lstm_hidden: int = 128,
        num_lstm_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize LSTM Discrete Q-Network.
        
        Args:
            state_dim: Dimension of state space (17 without norepinephrine)
            num_actions: Number of discrete actions (e.g., 5 dosing levels)
            hidden_dim: Hidden dimension for feature extraction
            lstm_hidden: LSTM hidden state dimension
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.lstm_hidden = lstm_hidden
        self.num_lstm_layers = num_lstm_layers
        
        # Feature extraction layers
        # Input: state only (actions are handled in output layer)
        input_dim = state_dim
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Output layer - Q-values for all discrete actions
        self.q_head = nn.Linear(lstm_hidden, num_actions)
        
    def forward(
        self,
        states: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM Q-network.
        
        Args:
            states: [batch_size, sequence_length, state_dim]
            hidden_state: Optional LSTM hidden state (h, c)
        
        Returns:
            q_values: [batch_size, sequence_length, num_actions] - Q(s,a) for all actions
            new_hidden: Updated LSTM hidden state
        """
        batch_size, seq_len, _ = states.shape
        
        # Reshape for feature extraction
        states_flat = states.reshape(-1, self.state_dim)
        features_flat = self.feature_extractor(states_flat)
        features = features_flat.reshape(batch_size, seq_len, -1)
        
        # Process through LSTM
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size, states.device)
        
        lstm_out, new_hidden = self.lstm(features, hidden_state)
        
        # Compute Q-values for all actions
        q_values = self.q_head(lstm_out)  # [batch_size, seq_len, num_actions]
        
        return q_values, new_hidden
    
    def forward_single_step(
        self,
        state: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for a single timestep (useful for inference).
        
        Args:
            state: [batch_size, state_dim]
            hidden_state: Optional LSTM hidden state
        
        Returns:
            q_values: [batch_size, num_actions] - Q(s,a) for all actions
            new_hidden: Updated LSTM hidden state
        """
        # Add sequence dimension
        state_seq = state.unsqueeze(1)  # [batch_size, 1, state_dim]
        
        # Forward through network
        q_values_seq, new_hidden = self.forward(state_seq, hidden_state)
        
        # Remove sequence dimension
        q_values = q_values_seq.squeeze(1)  # [batch_size, num_actions]
        
        return q_values, new_hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        h0 = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden).to(device)
        c0 = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_hidden).to(device)
        return (h0, c0)
    
    def get_q_values(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get Q-values for specific actions or all actions.
        
        Args:
            states: [batch_size, sequence_length, state_dim]
            actions: Optional [batch_size, sequence_length] - action indices
            hidden_state: Optional LSTM hidden state
        
        Returns:
            q_values: If actions provided: [batch_size, sequence_length, 1]
                     Otherwise: [batch_size, sequence_length, num_actions]
            new_hidden: Updated LSTM hidden state
        """
        all_q_values, new_hidden = self.forward(states, hidden_state)
        
        if actions is not None:
            # Gather Q-values for specific actions
            batch_size, seq_len = actions.shape
            actions_expanded = actions.unsqueeze(-1)  # [batch_size, seq_len, 1]
            q_values = torch.gather(all_q_values, dim=-1, index=actions_expanded)
            return q_values, new_hidden
        else:
            return all_q_values, new_hidden


class LSTMBlockDiscreteCQL:
    """
    Block Discrete CQL with LSTM Q-networks for sequential learning.
    Handles discrete dosing levels (e.g., Norepinephrine: 0, 0.05, 0.1, 0.2, 0.5 mcg/kg/min)
    """
    
    def __init__(
        self,
        state_dim: int,
        num_actions: int = 5,  # Number of discrete dosing levels
        action_bins: np.ndarray = None,  # Dosing level boundaries
        hidden_dim: int = 256,
        lstm_hidden: int = 128,
        num_lstm_layers: int = 2,
        alpha: float = 1.0,
        gamma: float = 0.95,
        tau: float = 0.005,
        lr: float = 3e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize LSTM Block Discrete CQL."""
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.action_bins = action_bins if action_bins is not None else np.array([0, 0.05, 0.1, 0.2, 0.5])
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)
        
        # Create LSTM Q-networks
        self.q1 = LSTMDiscreteQNetwork(
            state_dim=state_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            lstm_hidden=lstm_hidden,
            num_lstm_layers=num_lstm_layers
        ).to(self.device)
        
        self.q2 = LSTMDiscreteQNetwork(
            state_dim=state_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            lstm_hidden=lstm_hidden,
            num_lstm_layers=num_lstm_layers
        ).to(self.device)
        
        # Target networks
        self.q1_target = LSTMDiscreteQNetwork(
            state_dim=state_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            lstm_hidden=lstm_hidden,
            num_lstm_layers=num_lstm_layers
        ).to(self.device)
        
        self.q2_target = LSTMDiscreteQNetwork(
            state_dim=state_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            lstm_hidden=lstm_hidden,
            num_lstm_layers=num_lstm_layers
        ).to(self.device)
        
        # Initialize targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)
    
    def compute_cql_loss_sequences(
        self,
        q_network: LSTMDiscreteQNetwork,
        q_target_1: LSTMDiscreteQNetwork,
        q_target_2: LSTMDiscreteQNetwork,
        burn_in_states: torch.Tensor,
        burn_in_actions: torch.Tensor,  # Action indices
        training_states: torch.Tensor,
        training_actions: torch.Tensor,  # Action indices
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute CQL loss for discrete action sequences.
        
        Args:
            q_network: Q-network to update
            q_target_1, q_target_2: Target networks
            burn_in_states: [batch_size, burn_in_len, state_dim]
            burn_in_actions: [batch_size, burn_in_len] - action indices
            training_states: [batch_size, training_len, state_dim]
            training_actions: [batch_size, training_len] - action indices
            rewards: [batch_size, training_len]
            next_states: [batch_size, training_len, state_dim]
            dones: [batch_size, training_len]
            weights: [batch_size] - importance sampling weights
        """
        batch_size = burn_in_states.shape[0]
        
        # Burn-in phase: warm up LSTM hidden state (no gradients)
        with torch.no_grad():
            hidden = q_network.init_hidden(batch_size, self.device)
            if burn_in_states.shape[1] > 0:  # If burn-in sequence exists
                _, hidden = q_network.forward(burn_in_states, hidden)
        
        # Forward pass through training sequence - get Q-values for taken actions
        q_values, _ = q_network.get_q_values(training_states, training_actions, hidden)
        q_values = q_values.squeeze(-1)  # [batch_size, training_len]
        
        # Compute target Q-values
        with torch.no_grad():
            # Get Q-values for all actions at next states
            next_q1_all, _ = q_target_1.forward(next_states, hidden)
            next_q2_all, _ = q_target_2.forward(next_states, hidden)
            
            # Take minimum of two Q-networks
            next_q_all = torch.min(next_q1_all, next_q2_all)
            
            # Take maximum over actions
            next_q, _ = torch.max(next_q_all, dim=-1)
            
            # Compute targets
            targets = rewards + self.gamma * next_q * (1 - dones)
        
        # TD loss with importance sampling weights
        td_loss = (q_values - targets) ** 2
        td_loss = td_loss.mean(dim=1)  # Average over sequence
        td_loss = (td_loss * weights).mean()
        
        # CQL regularization for discrete actions
        # Get Q-values for all actions
        all_q_values, _ = q_network.forward(training_states, hidden)
        
        # Compute logsumexp over all actions for CQL penalty
        logsumexp_q = torch.logsumexp(all_q_values, dim=-1)  # [batch_size, training_len]
        
        # Conservative penalty: minimize Q-values for all actions while maintaining data Q-values
        cql_loss = (logsumexp_q - q_values).mean()
        
        # Total loss
        total_loss = td_loss + self.alpha * cql_loss
        
        metrics = {
            'td_loss': td_loss.item(),
            'cql_loss': cql_loss.item(),
            'q_values': q_values.mean().item(),
            'targets': targets.mean().item()
        }
        
        return total_loss, metrics
    
    def update_sequences(
        self,
        burn_in_batch: Dict[str, torch.Tensor],
        training_batch: Dict[str, torch.Tensor],
        weights: torch.Tensor
    ) -> Dict[str, float]:
        """Update Q-networks using sequence batch."""
        
        # Update Q1
        q1_loss, q1_metrics = self.compute_cql_loss_sequences(
            self.q1, self.q1_target, self.q2_target,
            burn_in_batch['states'],
            burn_in_batch['actions'],
            training_batch['states'],
            training_batch['actions'],
            training_batch['rewards'],
            training_batch['next_states'],
            training_batch['dones'],
            weights
        )
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.q1_optimizer.step()
        
        # Update Q2
        q2_loss, q2_metrics = self.compute_cql_loss_sequences(
            self.q2, self.q1_target, self.q2_target,
            burn_in_batch['states'],
            burn_in_batch['actions'],
            training_batch['states'],
            training_batch['actions'],
            training_batch['rewards'],
            training_batch['next_states'],
            training_batch['dones'],
            weights
        )
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q2_optimizer.step()
        
        # Soft update target networks
        self.soft_update_targets()
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'q1_td_loss': q1_metrics['td_loss'],
            'q1_cql_loss': q1_metrics['cql_loss'],
            'q_values': q1_metrics['q_values']
        }
    
    def soft_update_targets(self):
        """Soft update of target networks."""
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def continuous_to_discrete(self, continuous_doses: np.ndarray) -> np.ndarray:
        """Convert continuous doses to discrete action indices.
        
        Args:
            continuous_doses: Continuous dose values
        
        Returns:
            action_indices: Discrete action indices
        """
        # Find closest bin for each dose
        action_indices = np.digitize(continuous_doses, self.action_bins[1:])  # [1:] because digitize uses upper bounds
        return np.clip(action_indices, 0, self.num_actions - 1)
    
    def discrete_to_continuous(self, action_indices: torch.Tensor) -> torch.Tensor:
        """Convert discrete action indices to continuous doses.
        
        Args:
            action_indices: Discrete action indices
        
        Returns:
            continuous_doses: Continuous dose values (midpoints of bins)
        """
        device = action_indices.device
        action_bins_tensor = torch.tensor(self.action_bins, device=device, dtype=torch.float32)
        return action_bins_tensor[action_indices]


# Test function
if __name__ == "__main__":
    print("Testing LSTM Discrete Q-Network...")
    
    # Test parameters
    batch_size = 32
    burn_in_len = 8
    training_len = 12
    state_dim = 17  # Without norepinephrine
    num_actions = 5  # 5 discrete dosing levels
    
    # Create network
    net = LSTMDiscreteQNetwork(state_dim=state_dim, num_actions=num_actions)
    
    # Create dummy data
    burn_in_states = torch.randn(batch_size, burn_in_len, state_dim)
    burn_in_actions = torch.randint(0, num_actions, (batch_size, burn_in_len))  # Action indices
    training_states = torch.randn(batch_size, training_len, state_dim)
    training_actions = torch.randint(0, num_actions, (batch_size, training_len))  # Action indices
    
    # Test burn-in
    with torch.no_grad():
        hidden = net.init_hidden(batch_size, burn_in_states.device)
        _, hidden = net.forward(burn_in_states, hidden)
        print(f"✓ Burn-in complete. Hidden shape: {hidden[0].shape}")
    
    # Test forward pass for all actions
    all_q_values, new_hidden = net.forward(training_states, hidden)
    print(f"✓ Forward pass complete. All Q-values shape: {all_q_values.shape}")
    
    # Test getting Q-values for specific actions
    specific_q_values, _ = net.get_q_values(training_states, training_actions, hidden)
    print(f"✓ Specific action Q-values computed. Shape: {specific_q_values.shape}")
    
    # Test CQL
    print("\nTesting LSTM Block Discrete CQL...")
    cql = LSTMBlockDiscreteCQL(state_dim=state_dim, num_actions=num_actions, device='cpu')
    
    # Create full batch
    rewards = torch.randn(batch_size, training_len)
    next_states = torch.randn(batch_size, training_len, state_dim)
    dones = torch.zeros(batch_size, training_len)
    weights = torch.ones(batch_size)
    
    # Test update
    burn_in_batch = {'states': burn_in_states, 'actions': burn_in_actions}
    training_batch = {
        'states': training_states,
        'actions': training_actions,
        'rewards': rewards,
        'next_states': next_states,
        'dones': dones
    }
    
    metrics = cql.update_sequences(burn_in_batch, training_batch, weights)
    print(f"✓ CQL update complete. Metrics: {metrics}")
    
    print("\n✓ All tests passed!")