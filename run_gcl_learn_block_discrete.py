#!/usr/bin/env python3
"""
Guided Cost Learning (GCL) Block Discrete Training Script
Learn cost function from clinician demonstrations via maximum entropy IRL.

Based on: "Guided Cost Learning" (Finn et al., 2016)
https://arxiv.org/abs/1603.00448

Key idea: Learn cost function where expert demonstrations have low cost.
GCL Loss: L = E_demo[c(s,a)] + log(E_samp[exp(-c(s,a)) / q(a|s)])
where q(a|s) is the sampling distribution (policy probability).

Action space:
- VP1: Binary (0 or 1)
- VP2: Discretized into N bins (0 to 0.5 mcg/kg/min)

Also includes Q-network for deriving sampling policy via softmax(Q).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import sys
from typing import Dict, Tuple, Optional

# Import our unified pipeline
from integrated_data_pipeline_v2 import IntegratedDataPipelineV2

# Force unbuffered output
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__


# ============================================================================
# Q-Network for Block Discrete Actions
# ============================================================================

class BlockDiscreteQNetwork(nn.Module):
    """
    Q-network for block discrete actions using Q(s,a) -> R architecture.
    Takes state and discrete action index as input, outputs single Q-value.

    VP1: 2 actions (binary)
    VP2: vp2_bins actions (discretized continuous)
    Total: 2 * vp2_bins possible actions
    """

    def __init__(self, state_dim: int, vp2_bins: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.vp2_bins = vp2_bins
        self.total_actions = 2 * vp2_bins

        # Network takes state + one-hot encoded action
        input_dim = state_dim + self.total_actions

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 64)
        self.fc4 = nn.Linear(64, 1)

        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, state: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]
            action_idx: [batch_size] discrete action indices
        Returns:
            q_value: [batch_size, 1]
        """
        action_one_hot = F.one_hot(action_idx.long(), num_classes=self.total_actions).float()
        x = torch.cat([state, action_one_hot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

    def get_all_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values for all actions given states.

        Args:
            state: [batch_size, state_dim]
        Returns:
            q_values: [batch_size, total_actions]
        """
        batch_size = state.shape[0]

        all_actions = torch.arange(self.total_actions, device=state.device)
        all_actions = all_actions.unsqueeze(0).expand(batch_size, -1)

        state_expanded = state.unsqueeze(1).expand(-1, self.total_actions, -1)
        state_expanded = state_expanded.reshape(-1, self.state_dim)
        actions_flat = all_actions.reshape(-1)

        q_values = self.forward(state_expanded, actions_flat)
        q_values = q_values.reshape(batch_size, self.total_actions)

        return q_values


# ============================================================================
# Cost Network for GCL
# ============================================================================

class CostNN(nn.Module):
    """
    Cost network for Guided Cost Learning.
    Takes state and action, outputs scalar cost c(s, a).

    From the GCL paper, the cost function is learned such that
    expert trajectories have low cost.
    """

    def __init__(
        self,
        state_dim: int,
        total_actions: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.total_actions = total_actions

        # Input: state + one-hot encoded action
        input_dim = state_dim + total_actions

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]
            action_idx: [batch_size] discrete action indices
        Returns:
            cost: [batch_size, 1]
        """
        action_one_hot = F.one_hot(action_idx.long(), num_classes=self.total_actions).float()
        x = torch.cat([state, action_one_hot], dim=-1)
        return self.net(x)


# ============================================================================
# GCL Agent for Block Discrete Actions
# ============================================================================

class GCLBlockDiscrete:
    """
    Guided Cost Learning agent for inverse reinforcement learning with block discrete actions.

    Learns cost function from expert demonstrations.
    Uses Q-networks to derive sampling policy: π(a|s) = softmax(Q(s,·))

    GCL Loss: L = E_demo[c(s,a)] + log(E_samp[exp(-c(s,a)) / q(a|s)])
    """

    def __init__(
        self,
        state_dim: int,
        vp2_bins: int = 5,
        gamma: float = 0.99,
        init_temp: float = 0.001,
        tau: float = 0.005,
        lr: float = 1e-4,
        cost_lr: float = 1e-2,
        grad_clip: float = 1.0,
        use_target: bool = True,
    ):
        """
        Args:
            state_dim: Dimension of state space
            vp2_bins: Number of bins for VP2 discretization
            gamma: Discount factor
            init_temp: Soft-value temperature for policy (small for discrete actions)
            tau: Target network soft update rate
            lr: Learning rate for Q-networks
            cost_lr: Learning rate for cost network (default 1e-2 as in main.py)
            grad_clip: Gradient clipping value
            use_target: Whether to use target network for next_V
        """
        self.state_dim = state_dim
        self.vp2_bins = vp2_bins
        self.total_actions = 2 * vp2_bins
        self.gamma = gamma
        self.init_temp = init_temp
        self.tau = tau
        self.grad_clip = grad_clip
        self.use_target = use_target

        # VP2 bin edges (0 to 0.5)
        self.vp2_bin_edges = np.linspace(0, 0.5, vp2_bins + 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Q-networks (double Q for stability)
        self.q1 = BlockDiscreteQNetwork(state_dim, vp2_bins).to(self.device)
        self.q2 = BlockDiscreteQNetwork(state_dim, vp2_bins).to(self.device)

        # Initialize target networks
        self.q1_target = BlockDiscreteQNetwork(state_dim, vp2_bins).to(self.device)
        self.q2_target = BlockDiscreteQNetwork(state_dim, vp2_bins).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Initialize cost network
        self.cost_f = CostNN(state_dim, self.total_actions).to(self.device)

        # Single optimizer for Q-networks and cost network
        self.optimizer = optim.Adam([
            {'params': list(self.q1.parameters()) + list(self.q2.parameters()), 'lr': lr},
            {'params': self.cost_f.parameters(), 'lr': cost_lr, 'weight_decay': 1e-4}
        ])

    def continuous_to_discrete_action(self, continuous_action: np.ndarray) -> int:
        """Convert continuous action [vp1, vp2] to discrete action index."""
        vp1, vp2 = continuous_action
        vp1_idx = int(vp1)
        vp2_bin = np.digitize(vp2, self.vp2_bin_edges) - 1
        vp2_bin = np.clip(vp2_bin, 0, self.vp2_bins - 1)
        action_idx = vp1_idx * self.vp2_bins + vp2_bin
        return action_idx

    def continuous_to_discrete_batch(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert batch of continuous actions to discrete indices."""
        batch_size = actions.shape[0]
        action_indices = []
        for i in range(batch_size):
            action_np = actions[i].cpu().numpy()
            action_idx = self.continuous_to_discrete_action(action_np)
            action_indices.append(action_idx)
        return torch.LongTensor(action_indices).to(self.device)

    def discrete_to_continuous_action(self, action_idx: int) -> np.ndarray:
        """Convert discrete action index to continuous action [vp1, vp2]."""
        vp1_idx = action_idx // self.vp2_bins
        vp2_bin = action_idx % self.vp2_bins
        vp2_value = (self.vp2_bin_edges[vp2_bin] + self.vp2_bin_edges[vp2_bin + 1]) / 2
        return np.array([float(vp1_idx), vp2_value])

    def getV(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute soft-value function from Q-values.
        V = alpha * logsumexp(Q / alpha) where alpha = init_temp

        Args:
            obs: [batch_size, state_dim]
        Returns:
            v: [batch_size, 1]
        """
        q1_all = self.q1.get_all_q_values(obs)
        q2_all = self.q2.get_all_q_values(obs)
        q_all = torch.min(q1_all, q2_all)
        v = self.init_temp * torch.logsumexp(q_all / self.init_temp, dim=1, keepdim=True)
        return v

    def get_targetV(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute soft-value function using target networks.

        Args:
            obs: [batch_size, state_dim]
        Returns:
            v: [batch_size, 1]
        """
        q1_all = self.q1_target.get_all_q_values(obs)
        q2_all = self.q2_target.get_all_q_values(obs)
        q_all = torch.min(q1_all, q2_all)
        v = self.init_temp * torch.logsumexp(q_all / self.init_temp, dim=1, keepdim=True)
        return v

    def select_action(self, state, return_discrete: bool = False):
        """
        Select best action using argmax over Q-values.

        Args:
            state: np.ndarray [batch, state_dim] or [state_dim],
                   or torch.Tensor [batch, state_dim]
            return_discrete: If True, return discrete tensor indices

        Returns:
            If return_discrete: torch.Tensor [batch] discrete action indices
            Else: np.ndarray [batch, 2] or [2] continuous actions
        """
        with torch.no_grad():
            # Handle input type
            if isinstance(state, np.ndarray):
                if state.ndim == 1:
                    state = state.reshape(1, -1)
                state_tensor = torch.FloatTensor(state).to(self.device)
                single_input = (state.shape[0] == 1)
            else:
                state_tensor = state
                single_input = False

            batch_size = state_tensor.shape[0]

            # Get Q-values for all actions
            q1_all = self.q1.get_all_q_values(state_tensor)
            q2_all = self.q2.get_all_q_values(state_tensor)
            q_all = torch.min(q1_all, q2_all)

            # Get best action (argmax)
            best_action_indices = q_all.argmax(dim=1)

            if return_discrete:
                return best_action_indices

            # Convert to continuous actions
            indices_np = best_action_indices.cpu().numpy()
            vp1_actions = (indices_np // self.vp2_bins).astype(float)
            vp2_bin_indices = indices_np % self.vp2_bins
            vp2_bin_centers = (self.vp2_bin_edges[:-1] + self.vp2_bin_edges[1:]) / 2
            vp2_actions = vp2_bin_centers[vp2_bin_indices]

            actions = np.stack([vp1_actions, vp2_actions], axis=1)
            return actions[0] if single_input else actions
    
    def get_action_probabilities(
        self,
        states: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute action probabilities using softmax over Q-values.
        This implements q(τ) in the GCL paper - the probability of actions
        under the current Q-function induced policy.

        π(a|s) = softmax(Q(s,·) / temperature)

        Args:
            states: [batch_size, state_dim] - states to compute probabilities for
            temperature: Softmax temperature (default 1.0)

        Returns:
            action_probs: [batch_size, total_actions] - probabilities for all actions
            selected_actions: [batch_size] - argmax action indices
            selected_q_values: [batch_size] - Q-values for selected actions
        """
        batch_size = states.shape[0]

        # Get Q-values for all actions
        all_actions = torch.arange(self.total_actions).to(self.device)
        all_actions = all_actions.unsqueeze(0).expand(batch_size, -1)

        state_expanded = states.unsqueeze(1).expand(-1, self.total_actions, -1)
        state_expanded = state_expanded.reshape(-1, self.state_dim)
        actions_flat = all_actions.reshape(-1)

        # Compute Q-values using both networks and take min (conservative)
        q1_values = self.q1(state_expanded, actions_flat).reshape(batch_size, self.total_actions)
        q2_values = self.q2(state_expanded, actions_flat).reshape(batch_size, self.total_actions)
        q_values = torch.min(q1_values, q2_values)

        # Apply softmax to get action probabilities
        # π(a|s) = exp(Q(s,a)/τ) / Σ_a' exp(Q(s,a')/τ)
        action_probs = F.softmax(q_values / temperature, dim=-1)

        # Get selected actions (argmax)
        selected_actions = action_probs.argmax(dim=-1)

        # Get Q-value for selected actions
        selected_q_values = q_values.gather(1, selected_actions.unsqueeze(1)).squeeze(1)

        return action_probs, selected_actions, selected_q_values

    def gcl_update(
        self,
        states: torch.Tensor,
        expert_actions: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """
        GCL update step (following main.py lines 92-101).

        Computes GCL/IOC loss:
        L = E_demo[c(s,a)] + log(E_samp[exp(-c(s,a)) / q(a|s)])

        Args:
            states: [batch, state_dim] expert states
            expert_actions: [batch, 2] expert continuous actions
            next_states: [batch, state_dim] next states (unused but kept for API)
            dones: [batch] terminal flags (unused but kept for API)

        Returns:
            Dictionary of loss metrics
        """
        batch_size = states.shape[0]

        # ==============================================================
        # Create policy batch and expert batch
        # ==============================================================

        # Expert actions (from clinician demonstrations)
        expert_action_indices = self.continuous_to_discrete_batch(expert_actions)

        # Policy actions and probabilities (gradients flow through Q-networks)
        # This implements q(τ) in the GCL paper
        policy_action_probs, policy_action_indices, policy_q_values = self.get_action_probabilities(
            states, temperature=self.init_temp
        )

        # Get probability of the selected action for each sample: q(a|s)
        policy_probs_for_actions = policy_action_probs.gather(
            1, policy_action_indices.unsqueeze(1)
        ).squeeze(1)

        # ==============================================================
        # Compute GCL loss (following main.py lines 92-101)
        # ==============================================================

        # Compute costs for demo (expert) batch
        costs_demo = self.cost_f(states, expert_action_indices)

        # Compute costs for sample batch (demo ∪ policy, as per main.py line 79-80)
        # Combine demo and policy for importance-weighted partition function estimate
        combined_actions = torch.cat([expert_action_indices, policy_action_indices], dim=0)
        combined_states = torch.cat([states, states], dim=0)
        expert_probs = torch.ones(batch_size, device=self.device)  # demo probs = 1.0 (main.py line 55)
        combined_probs = torch.cat([
            expert_probs,              # demo probs = 1.0
            policy_probs_for_actions   # policy probs = q(a|s)
        ], dim=0)

        costs_samp = self.cost_f(combined_states, combined_actions)

        # GCL/IOC Loss (main.py lines 96-97)
        # L = E_demo[c(s,a)] + log(E_samp[exp(-c(s,a)) / q(a|s)])
        loss_IOC = torch.mean(costs_demo) + \
            torch.log(torch.mean(torch.exp(-costs_samp.squeeze()) / (combined_probs + 1e-7)))

        loss_dict = {'loss_IOC': loss_IOC.item(), 'cost_demo_mean': costs_demo.mean().item()}

        # ==============================================================
        # Optimization step
        # ==============================================================

        self.optimizer.zero_grad()
        loss_IOC.backward()

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.cost_f.parameters(), self.grad_clip)

        self.optimizer.step()

        # ==============================================================
        # Soft update target networks
        # ==============================================================

        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # ==============================================================
        # Additional metrics
        # ==============================================================

        with torch.no_grad():
            loss_dict['cost_samp_mean'] = costs_samp.mean().item()
            loss_dict['q_mean'] = policy_q_values.mean().item()

        return loss_dict

    def get_recovered_reward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the recovered/implicit reward: r(s,a) = Q(s,a) - gamma*V(s')

        Args:
            states: [batch, state_dim]
            actions: [batch, 2] continuous actions
            next_states: [batch, state_dim]
            dones: [batch]

        Returns:
            rewards: [batch] implicit rewards
        """
        with torch.no_grad():
            action_indices = self.continuous_to_discrete_batch(actions)

            q1 = self.q1(states, action_indices)
            q2 = self.q2(states, action_indices)
            q = torch.min(q1, q2)

            next_v = self.get_targetV(next_states)
            y = (1 - dones.unsqueeze(1)) * self.gamma * next_v

            reward = (q - y).squeeze()

        return reward

    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'cost_f_state_dict': self.cost_f.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_dim': self.state_dim,
            'vp2_bins': self.vp2_bins,
            'gamma': self.gamma,
            'init_temp': self.init_temp,
            'tau': self.tau,
        }, filepath)
        print(f"Model saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str, device: Optional[torch.device] = None) -> 'GCLBlockDiscrete':
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=device)

        agent = cls(
            state_dim=checkpoint['state_dim'],
            vp2_bins=checkpoint['vp2_bins'],
            gamma=checkpoint['gamma'],
            init_temp=checkpoint['init_temp'],
            tau=checkpoint['tau'],
        )

        agent.q1.load_state_dict(checkpoint['q1_state_dict'])
        agent.q2.load_state_dict(checkpoint['q2_state_dict'])
        agent.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        agent.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        agent.cost_f.load_state_dict(checkpoint['cost_f_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Model loaded from: {filepath}")
        return agent


# ============================================================================
# Training Function
# ============================================================================

def train_gcl(
    vp2_bins: int = 5,
    epochs: int = 500,
    batch_size: int = 128,
    gamma: float = 0.99,
    init_temp: float = 0.001,
    tau: float = 0.005,
    lr: float = 1e-4,
    cost_lr: float = 1e-2,
    experiment_prefix: str = "gcl"
):
    """Train GCL agent for inverse RL cost/reward recovery."""

    print("=" * 70, flush=True)
    print(" GCL BLOCK DISCRETE TRAINING", flush=True)
    print(" Guided Cost Learning for Reward Recovery", flush=True)
    print("=" * 70, flush=True)

    # Initialize data pipeline
    print("\nInitializing data pipeline...", flush=True)
    pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
    train_data, val_data, test_data = pipeline.prepare_data()

    state_dim = train_data['states'].shape[1]

    # Print settings
    print("\n" + "=" * 70, flush=True)
    print("GCL SETTINGS:", flush=True)
    print(f"  State dimension: {state_dim}", flush=True)
    print(f"  Action: VP1 (binary) x VP2 ({vp2_bins} bins) = {2 * vp2_bins} actions", flush=True)
    print(f"  gamma = {gamma}", flush=True)
    print(f"  init_temp = {init_temp} (policy temperature)", flush=True)
    print(f"  tau = {tau} (target network update)", flush=True)
    print(f"  lr = {lr} (Q-network)", flush=True)
    print(f"  cost_lr = {cost_lr} (cost network)", flush=True)
    print(f"  batch_size = {batch_size}", flush=True)
    print(f"  epochs = {epochs}", flush=True)
    print("=" * 70, flush=True)

    # Initialize agent
    agent = GCLBlockDiscrete(
        state_dim=state_dim,
        vp2_bins=vp2_bins,
        gamma=gamma,
        init_temp=init_temp,
        tau=tau,
        lr=lr,
        cost_lr=cost_lr,
    )

    print(f"\nDevice: {agent.device}", flush=True)

    # Training loop
    print(f"\nTraining for {epochs} epochs...", flush=True)
    start_time = time.time()

    os.makedirs('experiment/gcl', exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training phase
        agent.q1.train()
        agent.q2.train()
        agent.cost_f.train()

        train_metrics = {
            'loss_IOC': 0, 'cost_demo_mean': 0, 'cost_samp_mean': 0, 'q_mean': 0
        }

        n_batches = len(train_data['states']) // batch_size

        for _ in range(n_batches):
            batch = pipeline.get_batch(batch_size=batch_size, split='train')

            states = torch.FloatTensor(batch['states']).to(agent.device)
            actions = torch.FloatTensor(batch['actions']).to(agent.device)
            next_states = torch.FloatTensor(batch['next_states']).to(agent.device)
            dones = torch.FloatTensor(batch['dones']).to(agent.device)
            # Note: We do NOT use batch['rewards'] - GCL learns cost/rewards implicitly

            metrics = agent.gcl_update(states, actions, next_states, dones)

            for key in train_metrics:
                if key in metrics:
                    train_metrics[key] += metrics[key]

        for key in train_metrics:
            train_metrics[key] /= n_batches

        # Validation phase
        agent.q1.eval()
        agent.q2.eval()
        agent.cost_f.eval()

        val_metrics = {
            'loss_IOC': 0, 'cost_demo_mean': 0, 'cost_samp_mean': 0, 'q_mean': 0
        }
        n_val_batches = 10

        with torch.no_grad():
            for _ in range(n_val_batches):
                batch = pipeline.get_batch(batch_size=batch_size, split='val')

                states = torch.FloatTensor(batch['states']).to(agent.device)
                actions = torch.FloatTensor(batch['actions']).to(agent.device)

                # Compute validation metrics (same as training but no backward)
                expert_action_indices = agent.continuous_to_discrete_batch(actions)
                policy_action_probs, policy_action_indices, policy_q_values = agent.get_action_probabilities(
                    states, temperature=agent.init_temp
                )
                policy_probs_for_actions = policy_action_probs.gather(
                    1, policy_action_indices.unsqueeze(1)
                ).squeeze(1)

                # Compute costs
                costs_demo = agent.cost_f(states, expert_action_indices)

                combined_actions = torch.cat([expert_action_indices, policy_action_indices], dim=0)
                combined_states = torch.cat([states, states], dim=0)
                expert_probs = torch.ones(batch_size, device=agent.device)
                combined_probs = torch.cat([expert_probs, policy_probs_for_actions], dim=0)

                costs_samp = agent.cost_f(combined_states, combined_actions)

                # GCL loss
                loss_IOC = torch.mean(costs_demo) + \
                    torch.log(torch.mean(torch.exp(-costs_samp.squeeze()) / (combined_probs + 1e-7)))

                val_metrics['loss_IOC'] += loss_IOC.item()
                val_metrics['cost_demo_mean'] += costs_demo.mean().item()
                val_metrics['cost_samp_mean'] += costs_samp.mean().item()
                val_metrics['q_mean'] += policy_q_values.mean().item()

        for key in val_metrics:
            val_metrics[key] /= n_val_batches

        # Save best model (based on validation loss, lower is better)
        val_loss = val_metrics['loss_IOC']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            agent.save(f'experiment/gcl/{experiment_prefix}_bins{vp2_bins}_best.pt')

        # Print progress
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:4d} | "
                  f"Tra [L:{train_metrics['loss_IOC']:.3f} "
                  f"C_demo:{train_metrics['cost_demo_mean']:.3f} "
                  f"C_samp:{train_metrics['cost_samp_mean']:.3f} "
                  f"Q:{train_metrics['q_mean']:.3f}] | "
                  f"Val [L:{val_metrics['loss_IOC']:.3f} "
                  f"C_demo:{val_metrics['cost_demo_mean']:.3f} "
                  f"C_samp:{val_metrics['cost_samp_mean']:.3f} "
                  f"Q:{val_metrics['q_mean']:.3f}] | "
                  f"{elapsed/60:.1f}min", flush=True)

    # Save final model
    agent.save(f'experiment/gcl/{experiment_prefix}_bins{vp2_bins}_final.pt')

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes!", flush=True)
    print(f"Best validation loss: {best_val_loss:.4f}", flush=True)

    return agent, pipeline


# ============================================================================
# Main
# ============================================================================

def main():
    """Train GCL with specified hyperparameters."""
    import argparse

    parser = argparse.ArgumentParser(description='Train GCL for Inverse RL')
    parser.add_argument('--vp2_bins', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--init_temp', type=float, default=0.001)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--cost_lr', type=float, default=1e-2)
    parser.add_argument('--prefix', type=str, default='gcl')

    args = parser.parse_args()

    print("=" * 70, flush=True)
    print(" GCL: GUIDED COST LEARNING", flush=True)
    print("=" * 70, flush=True)
    print(f"\nConfiguration:", flush=True)
    print(f"  VP1: Binary (0 or 1)", flush=True)
    print(f"  VP2: {args.vp2_bins} bins", flush=True)
    print(f"  init_temp: {args.init_temp}", flush=True)
    print(f"  lr: {args.lr}", flush=True)
    print(f"  cost_lr: {args.cost_lr}", flush=True)

    agent, pipeline = train_gcl(
        vp2_bins=args.vp2_bins,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gamma=args.gamma,
        init_temp=args.init_temp,
        tau=args.tau,
        lr=args.lr,
        cost_lr=args.cost_lr,
        experiment_prefix=args.prefix
    )

    print("\n" + "=" * 70, flush=True)
    print(" TRAINING COMPLETE", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
