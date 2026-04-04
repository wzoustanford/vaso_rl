#!/usr/bin/env python3
"""
Block Discrete SQIL (Soft Q Imitation Learning) Training Script
Offline SQIL baseline for imitation learning comparison.
Discretizes VP2 continuous action space into bins:
- VP1: Binary (0 or 1)
- VP2: Discretized into N bins (0 to 0.5 mcg/kg/min)
- Total discrete actions: 2 * vp2_bins

Reference: Reddy et al., "SQIL: Imitation Learning via Reinforcement
           Learning with Sparse Rewards", ICLR 2020
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import sys

from integrated_data_pipeline_v3 import IntegratedDataPipelineV3

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__


# ============================================================================
# Q-Network (same architecture as CQL version)
# ============================================================================

class DualBlockDiscreteQNetwork(nn.Module):
    """
    Q-network for block discrete actions using Q(s,a) -> R architecture.
    Same architecture as CQL version.
    VP1: 2 actions (binary), VP2: vp2_bins actions (discretized)
    Total: 2 * vp2_bins possible actions
    """

    def __init__(self, state_dim: int, vp2_bins: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.vp2_bins = vp2_bins
        self.total_actions = 2 * vp2_bins

        input_dim = state_dim + self.total_actions

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 64)
        self.fc4 = nn.Linear(64, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, state: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        action_one_hot = F.one_hot(action_idx.long(), num_classes=self.total_actions).float()
        x = torch.cat([state, action_one_hot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# ============================================================================
# Offline SQIL Agent
# ============================================================================

class BlockDiscreteSQIL:
    """
    Offline SQIL agent with block discrete action space.
    - Expert transitions: reward = 1
    - Random/agent transitions: reward = 0
    - Soft Bellman backup (logsumexp instead of max)
    - 50/50 balanced sampling of expert vs agent data
    """

    def __init__(self, state_dim: int, vp2_bins: int = 5,
                 gamma: float = 0.95, tau: float = 0.8, lr: float = 1e-3,
                 grad_clip: float = 1.0):
        self.state_dim = state_dim
        self.vp2_bins = vp2_bins
        self.total_actions = 2 * vp2_bins
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip

        self.vp2_bin_edges = np.linspace(0, 0.5, vp2_bins + 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q1 = DualBlockDiscreteQNetwork(state_dim, vp2_bins).to(self.device)
        self.q2 = DualBlockDiscreteQNetwork(state_dim, vp2_bins).to(self.device)
        self.q1_target = DualBlockDiscreteQNetwork(state_dim, vp2_bins).to(self.device)
        self.q2_target = DualBlockDiscreteQNetwork(state_dim, vp2_bins).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

    def continuous_to_discrete_batch(self, actions: np.ndarray) -> np.ndarray:
        """Vectorized conversion of continuous actions to discrete indices."""
        vp1 = actions[:, 0].astype(int)
        vp2 = np.clip(actions[:, 1], 0, 0.5)
        vp2_bin = np.digitize(vp2, self.vp2_bin_edges) - 1
        vp2_bin = np.clip(vp2_bin, 0, self.vp2_bins - 1)
        return vp1 * self.vp2_bins + vp2_bin

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select best action (argmax over min(Q1,Q2)) for evaluation."""
        with torch.no_grad():
            if state.ndim == 1:
                state = state.reshape(1, -1)
            batch_size = state.shape[0]
            state_tensor = torch.FloatTensor(state).to(self.device)

            all_actions = torch.arange(self.total_actions).to(self.device)
            all_actions = all_actions.unsqueeze(0).expand(batch_size, -1)

            state_expanded = state_tensor.unsqueeze(1).expand(-1, self.total_actions, -1)
            state_expanded = state_expanded.reshape(-1, self.state_dim)
            actions_flat = all_actions.reshape(-1)

            q1_values = self.q1(state_expanded, actions_flat).reshape(batch_size, self.total_actions)
            q2_values = self.q2(state_expanded, actions_flat).reshape(batch_size, self.total_actions)
            q_values = torch.min(q1_values, q2_values)

            best_indices = q_values.argmax(dim=1).cpu().numpy()

            vp1_actions = (best_indices // self.vp2_bins).astype(float)
            vp2_bin_indices = best_indices % self.vp2_bins
            vp2_bin_centers = (self.vp2_bin_edges[:-1] + self.vp2_bin_edges[1:]) / 2
            vp2_actions = vp2_bin_centers[vp2_bin_indices]
            actions = np.stack([vp1_actions, vp2_actions], axis=1)
            return actions if batch_size > 1 else actions[0]

    def update(self, states: torch.Tensor, actions: torch.Tensor,
               rewards: torch.Tensor, next_states: torch.Tensor,
               dones: torch.Tensor) -> dict:
        """
        One SQIL update step with soft Bellman backup.
        Rewards should already be set (1 for expert, 0 for agent).
        """
        batch_size = states.shape[0]

        # Soft Bellman target: r + γ * log(Σ_a' exp(Q(s', a')))
        with torch.no_grad():
            all_next_actions = torch.arange(self.total_actions).to(self.device)
            all_next_actions = all_next_actions.unsqueeze(0).expand(batch_size, -1)
            next_expanded = next_states.unsqueeze(1).expand(-1, self.total_actions, -1)
            next_expanded = next_expanded.reshape(-1, self.state_dim)
            next_flat = all_next_actions.reshape(-1)

            next_q1 = self.q1_target(next_expanded, next_flat).reshape(batch_size, self.total_actions)
            next_q2 = self.q2_target(next_expanded, next_flat).reshape(batch_size, self.total_actions)
            next_q = torch.min(next_q1, next_q2)

            # Soft value: logsumexp
            soft_v = torch.logsumexp(next_q, dim=1)
            target_q = rewards + self.gamma * soft_v * (1 - dones)

        # Update Q1
        current_q1 = self.q1(states, actions).squeeze()
        q1_loss = F.mse_loss(current_q1, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.grad_clip)
        self.q1_optimizer.step()

        # Update Q2
        current_q2 = self.q2(states, actions).squeeze()
        q2_loss = F.mse_loss(current_q2, target_q)

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
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
        }

    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'state_dim': self.state_dim,
            'vp2_bins': self.vp2_bins,
            'gamma': self.gamma,
            'tau': self.tau,
        }, filepath)


# ============================================================================
# Training Function
# ============================================================================

def train_block_discrete_sqil(
    vp2_bins: int = 5,
    epochs: int = 100,
    suffix: str = "",
    save_dir: str = "experiment/sqil",
    combined_or_train_data_path: str = None,
    eval_data_path: str = None,
):
    """Train Block Discrete SQIL.

    Args:
        vp2_bins: Number of bins for VP2 discretization
        epochs: Number of training epochs
        suffix: Suffix for experiment naming
        save_dir: Directory to save models
        combined_or_train_data_path: Path to training dataset
        eval_data_path: Path to evaluation dataset
    """
    os.makedirs(save_dir, exist_ok=True)

    # SQIL doesn't use rewards from the pipeline — it assigns its own
    print("\nInitializing Block Discrete SQIL data pipeline...", flush=True)
    pipeline = IntegratedDataPipelineV3(
        model_type='dual', reward_source='manual', random_seed=42,
        combined_or_train_data_path=combined_or_train_data_path,
        eval_data_path=eval_data_path
    )

    experiment_prefix = f"sqil_{suffix}" if suffix else "sqil"

    print("=" * 70, flush=True)
    print(f" BLOCK DISCRETE SQIL TRAINING", flush=True)
    print(f" Prefix: {experiment_prefix}", flush=True)
    print("=" * 70, flush=True)

    train_data, val_data, test_data = pipeline.prepare_data()
    state_dim = train_data['states'].shape[1]
    total_actions = 2 * vp2_bins

    print("\n" + "=" * 70, flush=True)
    print("SETTINGS:", flush=True)
    print(f"  State dimension: {state_dim}", flush=True)
    print(f"  Action dimension: 2 (VP1: binary, VP2: {vp2_bins} bins)", flush=True)
    print(f"  Total discrete actions: {total_actions}", flush=True)
    print(f"  Gamma: 0.95", flush=True)
    print(f"  Tau: 0.8", flush=True)
    print(f"  LR: 1e-3", flush=True)
    print(f"  Batch size: 128 (64 expert + 64 random)", flush=True)
    print(f"  Epochs: {epochs}", flush=True)
    print(f"  Soft Bellman backup (logsumexp)", flush=True)
    print("=" * 70, flush=True)

    agent = BlockDiscreteSQIL(
        state_dim=state_dim, vp2_bins=vp2_bins,
        gamma=0.95, tau=0.8, lr=1e-3, grad_clip=1.0,
    )

    # Training loop
    batch_size = 128
    half_batch = batch_size // 2  # 50/50 split per SQIL paper

    print(f"\nTraining for {epochs} epochs with batch size {batch_size} "
          f"(50/50 expert/random)...", flush=True)
    start_time = time.time()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        agent.q1.train()
        agent.q2.train()

        epoch_metrics = {'q1_loss': 0, 'q2_loss': 0}
        n_batches = len(train_data['states']) // half_batch

        for _ in range(n_batches):
            # === Expert half: real (s, a, s', done) from dataset ===
            batch = pipeline.get_batch(batch_size=half_batch, split='train')
            expert_states = torch.FloatTensor(batch['states']).to(agent.device)
            expert_actions_cont = torch.FloatTensor(batch['actions'])
            expert_actions = torch.LongTensor(
                agent.continuous_to_discrete_batch(expert_actions_cont.numpy())
            ).to(agent.device)
            expert_next_states = torch.FloatTensor(batch['next_states']).to(agent.device)
            expert_dones = torch.FloatTensor(batch['dones']).to(agent.device)
            expert_rewards = torch.ones(half_batch).to(agent.device)   # reward = 1

            # === Agent half: real states but RANDOM actions ===
            agent_batch = pipeline.get_batch(batch_size=half_batch, split='train')
            agent_states = torch.FloatTensor(agent_batch['states']).to(agent.device)
            agent_actions = torch.randint(0, total_actions, (half_batch,)).to(agent.device)
            agent_next_states = torch.FloatTensor(agent_batch['next_states']).to(agent.device)
            agent_dones = torch.FloatTensor(agent_batch['dones']).to(agent.device)
            agent_rewards = torch.zeros(half_batch).to(agent.device)   # reward = 0

            # === Combine and update ===
            all_states = torch.cat([expert_states, agent_states], dim=0)
            all_actions = torch.cat([expert_actions, agent_actions], dim=0)
            all_rewards = torch.cat([expert_rewards, agent_rewards], dim=0)
            all_next_states = torch.cat([expert_next_states, agent_next_states], dim=0)
            all_dones = torch.cat([expert_dones, agent_dones], dim=0)

            metrics = agent.update(all_states, all_actions, all_rewards,
                                   all_next_states, all_dones)

            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]

        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches

        # Validation phase
        agent.q1.eval()
        agent.q2.eval()

        val_q_values = []
        with torch.no_grad():
            for _ in range(10):
                batch = pipeline.get_batch(batch_size=batch_size, split='val')
                states = torch.FloatTensor(batch['states']).to(agent.device)
                actions_cont = torch.FloatTensor(batch['actions'])
                action_indices = torch.LongTensor(
                    agent.continuous_to_discrete_batch(actions_cont.numpy())
                ).to(agent.device)

                q1_val = agent.q1(states, action_indices).squeeze()
                q2_val = agent.q2(states, action_indices).squeeze()
                q_val = torch.min(q1_val, q2_val)
                val_q_values.append(q_val.mean().item())

        val_loss = -np.mean(val_q_values)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f'{save_dir}/{experiment_prefix}_bins{vp2_bins}_best.pt'
            print(f'  Saving best model at {save_path}')
            agent.save(save_path)

        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_td_loss = (epoch_metrics['q1_loss'] + epoch_metrics['q2_loss']) / 2
            print(f"Epoch {epoch+1}: "
                  f"TD Loss={avg_td_loss:.4f} "
                  f"(Q1={epoch_metrics['q1_loss']:.4f}, Q2={epoch_metrics['q2_loss']:.4f}), "
                  f"Val Q={-val_loss:.4f}, "
                  f"Time={elapsed/60:.1f}min", flush=True)

    # Save final model
    final_path = f'{save_dir}/{experiment_prefix}_bins{vp2_bins}_final.pt'
    agent.save(final_path)

    total_time = time.time() - start_time
    print(f"\nSQIL training completed in {total_time/60:.1f} minutes!", flush=True)
    print("Models saved:", flush=True)
    print(f"  - {save_dir}/{experiment_prefix}_bins{vp2_bins}_best.pt", flush=True)
    print(f"  - {final_path}", flush=True)

    return agent, pipeline, experiment_prefix


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train Block Discrete SQIL')
    parser.add_argument('--vp2_bins', type=int, default=5,
                        help='Number of bins for VP2 discretization (default: 5)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix for experiment naming')
    parser.add_argument('--save_dir', type=str, default='experiment/sqil',
                        help='Directory to save models (default: experiment/sqil)')
    parser.add_argument('--combined_or_train_data_path', type=str, default=None,
                        help='Path to training dataset')
    parser.add_argument('--eval_data_path', type=str, default=None,
                        help='Path to evaluation dataset')
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print(" BLOCK DISCRETE SQIL TRAINING", flush=True)
    print("=" * 70, flush=True)

    print("\nConfiguration:", flush=True)
    print("  - VP1: Binary (0 or 1)", flush=True)
    print(f"  - VP2: Discretized into {args.vp2_bins} bins (0 to 0.5 mcg/kg/min)", flush=True)
    print(f"  - Total discrete actions: {2 * args.vp2_bins}", flush=True)
    print(f"  - Epochs: {args.epochs}", flush=True)
    print(f"  - Hyperparameters: gamma=0.95, tau=0.8, lr=1e-3", flush=True)
    print(f"  - Batch: 128 (64 expert r=1, 64 random r=0)", flush=True)
    print(f"  - Soft Bellman backup (logsumexp)", flush=True)
    if args.eval_data_path:
        print(f"  - Dataset mode: DUAL-DATASET", flush=True)
        print(f"    Train data: {args.combined_or_train_data_path or 'default'}", flush=True)
        print(f"    Eval data:  {args.eval_data_path}", flush=True)
    else:
        print(f"  - Dataset mode: SINGLE-DATASET", flush=True)
        print(f"    Data path: {args.combined_or_train_data_path or 'default'}", flush=True)

    agent, pipeline, exp_prefix = train_block_discrete_sqil(
        vp2_bins=args.vp2_bins,
        epochs=args.epochs,
        suffix=args.suffix,
        save_dir=args.save_dir,
        combined_or_train_data_path=args.combined_or_train_data_path,
        eval_data_path=args.eval_data_path,
    )

    print("\n" + "=" * 70, flush=True)
    print(" TRAINING COMPLETE", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
