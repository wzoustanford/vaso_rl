"""
IRL Full Recovery: Maximum sum_log_probs Inverse Reinforcement Learning
Recovers reward functions from expert demonstrations using MaxSLP IRL.

The key idea: Expert trajectories should have high probability under the
learned reward function. We maximize the sum_log_probs of the trajectory
distribution to find the reward that best explains expert behavior.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Import project modules
from integrated_data_pipeline_v2 import IntegratedDataPipelineV2
import data_config as config


# ==============================================================================
# STAGE 1: Extract training trajectories with 6-step lookahead
# ==============================================================================

@dataclass
class Trajectory:
    """Container for a single patient trajectory"""
    patient_id: int
    states: np.ndarray      # (T, state_dim)
    actions: np.ndarray     # (T,) or (T, action_dim)
    length: int


class TrajectoryExtractor:
    """
    Extract training trajectories from the data pipeline.
    Each trajectory contains 6 consecutive transitions for lookahead rewards.
    """

    def __init__(self, pipeline: IntegratedDataPipelineV2, window_size: int = 6):
        """
        Args:
            pipeline: Initialized data pipeline with prepared data
            window_size: Number of consecutive transitions per trajectory window
        """
        self.pipeline = pipeline
        self.window_size = window_size
        self.trajectories: List[Trajectory] = []

    def extract_trajectories(self, split: str = 'train') -> List[Trajectory]:
        """
        Extract full patient trajectories from the specified split.

        Args:
            split: 'train', 'val', or 'test'

        Returns:
            List of Trajectory objects
        """
        if split == 'train':
            data = self.pipeline.train_data
            patient_groups = self.pipeline.train_patient_groups
        elif split == 'val':
            data = self.pipeline.val_data
            patient_groups = self.pipeline.val_patient_groups
        else:
            data = self.pipeline.test_data
            patient_groups = self.pipeline.test_patient_groups

        if data is None:
            raise ValueError(f"Data for {split} not prepared. Call prepare_data() first.")

        trajectories = []

        for patient_id, (start_idx, end_idx) in patient_groups.items():
            # Extract patient's states and actions
            states = data['states'][start_idx:end_idx]
            actions = data['actions'][start_idx:end_idx]
            length = end_idx - start_idx

            # Only include trajectories with at least window_size transitions
            if length >= self.window_size:
                traj = Trajectory(
                    patient_id=patient_id,
                    states=states,
                    actions=actions,
                    length=length
                )
                trajectories.append(traj)

        self.trajectories = trajectories
        print(f"Extracted {len(trajectories)} trajectories from {split} split")
        print(f"  Average trajectory length: {np.mean([t.length for t in trajectories]):.1f}")
        print(f"  Min length: {min(t.length for t in trajectories)}")
        print(f"  Max length: {max(t.length for t in trajectories)}")

        return trajectories

    def get_windowed_trajectories(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create sliding window views of trajectories for training.
        Each window contains 6 consecutive (state, action) pairs.
        
        Returns:
            List of (states_window, actions_window) tuples
        """
        windows = []

        for traj in self.trajectories:
            # Create sliding windows over the trajectory
            for start in range(traj.length - self.window_size + 1):
                end = start + self.window_size
                states_window = traj.states[start:end]
                actions_window = traj.actions[start:end]
                windows.append((states_window, actions_window))

        print(f"Created {len(windows)} trajectory windows of size {self.window_size}")
        return windows


# ==============================================================================
# STAGE 2: Build fully connected reward network R(s, a)
# ==============================================================================

class RewardNetwork(nn.Module):
    """
    Fully connected network to predict reward given state and action: R(s, a)

    Architecture:
        Input: concatenated (state, action)
        Hidden layers: FC -> ReLU -> FC -> ReLU -> FC
        Output: scalar reward
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 64, 32]
    ):
        """
        Args:
            state_dim: Dimension of state features
            action_dim: Dimension of action (1 for binary, 2 for dual)
            hidden_dims: List of hidden layer dimensions
        """
        super(RewardNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        input_dim = state_dim + action_dim

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer: single scalar reward
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict reward.

        Args:
            states: (batch, state_dim) or (batch, T, state_dim)
            actions: (batch,) or (batch, action_dim) or (batch, T) or (batch, T, action_dim)

        Returns:
            rewards: (batch,) or (batch, T) scalar rewards
        """
        # Handle action dimension
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        elif actions.dim() == 2 and self.action_dim == 1:
            # (batch, T) -> need to keep as is for trajectory
            if actions.shape[-1] != 1:
                actions = actions.unsqueeze(-1)
        elif actions.dim() == 3 and actions.shape[-1] == 1:
            pass  # Already correct shape

        # Concatenate state and action
        if states.dim() == 2:
            # Single timestep: (batch, state_dim) + (batch, action_dim)
            x = torch.cat([states, actions], dim=-1)
        else:
            # Trajectory: (batch, T, state_dim) + (batch, T, action_dim)
            x = torch.cat([states, actions], dim=-1)

        # Forward through network
        reward = self.network(x)

        return reward.squeeze(-1)


# ==============================================================================
# STAGE 3: Forward pass to compute trajectory rewards R_traj = sum_T R_trans
# ==============================================================================

class TrajectoryRewardComputer:
    """
    Compute trajectory rewards by aggregating transition rewards.
    R_traj = sum_{t=1}^{T} R(s_t, a_t)
    """

    def __init__(self, reward_network: RewardNetwork, device: torch.device):
        self.reward_network = reward_network
        self.device = device
    
    def compute_trajectory_rewards(
        self,
        trajectories: List[Tuple[np.ndarray, np.ndarray]]
    ) -> torch.Tensor:
        """
        Compute total reward for each trajectory.

        Args:
            trajectories: List of (states, actions) tuples
                         states: (T, state_dim), actions: (T,) or (T, action_dim)

        Returns:
            trajectory_rewards: (num_trajectories,) tensor of summed rewards
        """
        trajectory_rewards = []

        for states, actions in trajectories:
            # Convert to tensors
            states_t = torch.FloatTensor(states).to(self.device)
            actions_t = torch.FloatTensor(actions).to(self.device)

            # Ensure actions have correct shape
            if actions_t.dim() == 1:
                actions_t = actions_t.unsqueeze(-1)

            # Compute reward for each transition
            transition_rewards = self.reward_network(states_t, actions_t)

            # Sum across trajectory
            traj_reward = transition_rewards.sum()
            trajectory_rewards.append(traj_reward)

        return torch.stack(trajectory_rewards)

    def compute_batch_trajectory_rewards(
        self,
        states_batch: torch.Tensor,
        actions_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute trajectory rewards for a batch of trajectories.

        Args:
            states_batch: (batch, T, state_dim)
            actions_batch: (batch, T) or (batch, T, action_dim)

        Returns:
            trajectory_rewards: (batch,) tensor
        """
        # Ensure actions have correct shape
        if actions_batch.dim() == 2:
            actions_batch = actions_batch.unsqueeze(-1)

        # Compute rewards for all transitions: (batch, T)
        transition_rewards = self.reward_network(states_batch, actions_batch)

        # Sum across trajectory dimension
        trajectory_rewards = transition_rewards.sum(dim=-1)

        return trajectory_rewards


# ==============================================================================
# STAGE 4: Compute softmax probabilities and sum_log_probs
# ==============================================================================

class MaxSLPIRLObjective:
    """
    Maximum sum_log_probs IRL objective.

    The probability of a trajectory under MaxSLP IRL is:
        p(tau) = exp(R(tau)) / Z

    where Z = sum_tau' exp(R(tau')) is the partition function.

    Using log-sum-exp trick for numerical stability:
        log Z = log(sum exp(R)) = max(R) + log(sum exp(R - max(R)))

    The sum_log_probs objective:
        E = sum_tau log p(tau) = sum_tau [R(tau) - log Z]

    For expert demonstrations, we maximize this (or minimize -E as loss).
    """

    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Softmax temperature (higher = more uniform distribution)
        """
        self.temperature = temperature

    def compute_log_probabilities(
        self,
        trajectory_rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probabilities of trajectories using softmax.

        Args:
            trajectory_rewards: (num_trajectories,) tensor of trajectory rewards

        Returns:
            log_probs: (num_trajectories,) tensor of log probabilities
        """
        # Apply temperature scaling
        scaled_rewards = trajectory_rewards / self.temperature

        # Log-softmax for numerical stability
        log_probs = torch.log_softmax(scaled_rewards, dim=0)

        return log_probs

    def compute_loss(
        self,
        trajectory_rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the IRL sum_log_probs objective.

        E = sum_tau log p(tau)

        For expert trajectories, this should be maximized.

        Args:
            trajectory_rewards: (num_trajectories,) tensor

        Returns:
            sum_log_probs: scalar tensor (sum of log probabilities)
        """
        log_probs = self.compute_log_probabilities(trajectory_rewards)

        # Sum of log probabilities (negative cross-entropy with uniform prior)
        sum_log_probs = log_probs.sum()

        return -sum_log_probs


# ==============================================================================
# STAGE 5: Training loop - minimize loss = -E to train reward model
# ==============================================================================

class MaxSLPIRLTrainer:
    """
    Trainer for Maximum Entropy IRL reward recovery.
    """

    def __init__(
        self,
        reward_network: RewardNetwork,
        learning_rate: float = 1e-3,
        temperature: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            reward_network: The reward network to train
            learning_rate: Learning rate for optimizer
            temperature: Softmax temperature
            device: Torch device (cuda/cpu)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reward_network = reward_network.to(self.device)

        self.optimizer = optim.Adam(reward_network.parameters(), lr=learning_rate)
        self.trajectory_computer = TrajectoryRewardComputer(reward_network, self.device)
        self.objective = MaxSLPIRLObjective(temperature=temperature)

        # Training history
        self.losses = []
        self.entropies = []

    def train_epoch(
        self,
        trajectories: List[Tuple[np.ndarray, np.ndarray]],
        batch_size: int = 64
    ) -> Tuple[float, float]:
        """
        Train for one epoch over all trajectories.

        Args:
            trajectories: List of (states, actions) tuples
            batch_size: Number of trajectories per batch

        Returns:
            Tuple of (average_loss, average_sum_log_probs)
        """
        self.reward_network.train()

        # Shuffle trajectories
        indices = np.random.permutation(len(trajectories))

        epoch_loss = 0.0
        epoch_sum_log_probs = 0.0
        n_batches = 0

        for start in range(0, len(trajectories), batch_size):
            end = min(start + batch_size, len(trajectories))
            batch_indices = indices[start:end]

            # Get batch trajectories
            batch_trajs = [trajectories[i] for i in batch_indices]

            # Compute trajectory rewards
            traj_rewards = self.trajectory_computer.compute_trajectory_rewards(batch_trajs)

            # Compute loss
            loss = self.objective.compute_loss(traj_rewards)
            sum_log_probs = -loss.item()  # sum_log_probs is negative of loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_sum_log_probs += sum_log_probs
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_sum_log_probs = epoch_sum_log_probs / n_batches

        self.losses.append(avg_loss)
        self.entropies.append(avg_sum_log_probs)

        return avg_loss, avg_sum_log_probs

    def train(
        self,
        trajectories: List[Tuple[np.ndarray, np.ndarray]],
        n_epochs: int = 100,
        batch_size: int = 64,
        print_every: int = 10,
        val_trajectories: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    ) -> Dict:
        """
        Full training loop.

        Args:
            trajectories: Training trajectories
            n_epochs: Number of training epochs
            batch_size: Batch size
            print_every: Print progress every N epochs
            val_trajectories: Optional validation trajectories

        Returns:
            Training history dictionary
        """
        print("="*60)
        print("MaxSLP IRL TRAINING")
        print("="*60)
        print(f"  Trajectories: {len(trajectories)}")
        print(f"  Epochs: {n_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Device: {self.device}")
        print("="*60)

        val_losses = []

        for epoch in range(1, n_epochs + 1):
            train_loss, train_sum_log_probs = self.train_epoch(trajectories, batch_size)

            # Validation
            val_loss = None
            if val_trajectories is not None:
                val_loss, val_sum_log_probs = self.evaluate(val_trajectories)
                val_losses.append(val_loss)

            # Print progress
            if epoch % print_every == 0 or epoch == 1:
                msg = f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | sum_log_probs: {train_sum_log_probs:.4f}"
                if val_loss is not None:
                    msg += f" | Val Loss: {val_loss:.4f}"
                print(msg)

        print("="*60)
        print("Training complete!")
        print(f"  Final train loss: {self.losses[-1]:.4f}")
        print(f"  Final sum_log_probs: {self.entropies[-1]:.4f}")

        return {
            'train_losses': self.losses,
            'train_entropies': self.entropies,
            'val_losses': val_losses
        }

    def evaluate(
        self,
        trajectories: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[float, float]:
        """
        Evaluate on a set of trajectories.

        Args:
            trajectories: List of (states, actions) tuples

        Returns:
            Tuple of (loss, sum_log_probs)
        """
        self.reward_network.eval()

        with torch.no_grad():
            traj_rewards = self.trajectory_computer.compute_trajectory_rewards(trajectories)
            loss = self.objective.compute_loss(traj_rewards)
            sum_log_probs = -loss.item()

        return loss.item(), sum_log_probs

    def predict_rewards(
        self,
        states: np.ndarray,
        actions: np.ndarray
    ) -> np.ndarray:
        """
        Predict rewards for given state-action pairs.

        Args:
            states: (N, state_dim) array
            actions: (N,) or (N, action_dim) array

        Returns:
            rewards: (N,) array of predicted rewards
        """
        self.reward_network.eval()

        with torch.no_grad():
            states_t = torch.FloatTensor(states).to(self.device)
            actions_t = torch.FloatTensor(actions).to(self.device)

            if actions_t.dim() == 1:
                actions_t = actions_t.unsqueeze(-1)

            rewards = self.reward_network(states_t, actions_t)

        return rewards.cpu().numpy()

    def save_model(self, experiment_prefix: str, save_dir: str = 'experiment/irl') -> str:
        """
        Save the trained reward network and training history.

        Args:
            experiment_prefix: Prefix for the saved files (e.g., 'exp001', 'binary_v1')
            save_dir: Directory to save the model

        Returns:
            Path to the saved checkpoint file
        """
        os.makedirs(save_dir, exist_ok=True)

        checkpoint_path = os.path.join(save_dir, f'{experiment_prefix}_reward_model.pt')

        checkpoint = {
            'reward_network_state_dict': self.reward_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_dim': self.reward_network.state_dim,
            'action_dim': self.reward_network.action_dim,
            'losses': self.losses,
            'entropies': self.entropies,
            'temperature': self.objective.temperature,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Model saved to: {checkpoint_path}")

        return checkpoint_path

    @classmethod
    def load_model(
        cls,
        experiment_prefix: str,
        save_dir: str = 'experiment/irl',
        hidden_dims: List[int] = [128, 64, 32],
        learning_rate: float = 1e-3,
        device: Optional[torch.device] = None
    ) -> 'MaxSLPIRLTrainer':
        """
        Load a trained reward network from checkpoint.

        Args:
            experiment_prefix: Prefix used when saving the model
            save_dir: Directory where the model was saved
            hidden_dims: Hidden layer dimensions (must match saved model)
            learning_rate: Learning rate for optimizer
            device: Torch device (cuda/cpu)

        Returns:
            Loaded MaxSLPIRLTrainer with trained reward network
        """
        checkpoint_path = os.path.join(save_dir, f'{experiment_prefix}_reward_model.pt')

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Reconstruct reward network
        reward_network = RewardNetwork(
            state_dim=checkpoint['state_dim'],
            action_dim=checkpoint['action_dim'],
            hidden_dims=hidden_dims
        )
        reward_network.load_state_dict(checkpoint['reward_network_state_dict'])

        # Create trainer
        trainer = cls(
            reward_network=reward_network,
            learning_rate=learning_rate,
            temperature=checkpoint['temperature'],
            device=device
        )

        # Load optimizer state
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore training history
        trainer.losses = checkpoint['losses']
        trainer.entropies = checkpoint['entropies']

        print(f"Model loaded from: {checkpoint_path}")
        print(f"  State dim: {checkpoint['state_dim']}")
        print(f"  Action dim: {checkpoint['action_dim']}")
        print(f"  Training epochs completed: {len(trainer.losses)}")

        return trainer


# ==============================================================================
# Main execution
# ==============================================================================

def run_irl_recovery(
    model_type: str = 'binary',
    n_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    hidden_dims: List[int] = [128, 64, 32],
    temperature: float = 1.0,
    window_size: int = 6,
    random_seed: int = 42,
    experiment_prefix: Optional[str] = None,
    save_dir: str = 'experiment/irl'
) -> Tuple[MaxSLPIRLTrainer, Dict]:
    """
    Run the full IRL reward recovery pipeline.

    Args:
        model_type: 'binary' or 'dual'
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        hidden_dims: Hidden layer dimensions for reward network
        temperature: Softmax temperature
        window_size: Trajectory window size
        random_seed: Random seed
        experiment_prefix: Prefix for saving model (e.g., 'exp001'). If None, model won't be saved.
        save_dir: Directory to save the model

    Returns:
        Tuple of (trained_trainer, training_history)
    """
    # Set seeds
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    print("="*70)
    print(" IRL FULL RECOVERY - MAXIMUM sum_log_probs INVERSE RL")
    print("="*70)

    # Stage 1: Load and extract trajectories
    print("\n[STAGE 1] Extracting trajectories...")
    pipeline = IntegratedDataPipelineV2(model_type=model_type, random_seed=random_seed)
    pipeline.prepare_data()

    extractor = TrajectoryExtractor(pipeline, window_size=window_size)
    train_trajectories = extractor.extract_trajectories(split='train')
    val_trajectories = extractor.extract_trajectories(split='val')

    # Get windowed trajectories for training
    train_windows = extractor.get_windowed_trajectories()

    # Also get validation windows
    extractor_val = TrajectoryExtractor(pipeline, window_size=window_size)
    extractor_val.trajectories = val_trajectories
    val_windows = extractor_val.get_windowed_trajectories()

    # Stage 2: Build reward network
    print("\n[STAGE 2] Building reward network...")
    state_dim = pipeline.train_data['states'].shape[1]
    action_dim = 1 if model_type == 'binary' else 2

    reward_network = RewardNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims
    )
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Network:\n{reward_network}")

    # Stage 3-5: Train with MaxSLP IRL
    print("\n[STAGE 3-5] Training with MaxSLP IRL objective...")
    trainer = MaxSLPIRLTrainer(
        reward_network=reward_network,
        learning_rate=learning_rate,
        temperature=temperature
    )

    history = trainer.train(
        trajectories=train_windows,
        n_epochs=n_epochs,
        batch_size=batch_size,
        print_every=10,
        val_trajectories=val_windows
    )

    # Evaluate on test set
    print("\n[EVALUATION] Testing on held-out data...")
    test_trajectories = extractor.extract_trajectories(split='test')
    extractor_test = TrajectoryExtractor(pipeline, window_size=window_size)
    extractor_test.trajectories = test_trajectories
    test_windows = extractor_test.get_windowed_trajectories()

    test_loss, test_sum_log_probs = trainer.evaluate(test_windows)
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test sum_log_probs: {test_sum_log_probs:.4f}")

    # Sample predicted rewards
    print("\n[SAMPLE PREDICTIONS]")
    sample_states = pipeline.train_data['states'][:5]
    sample_actions = pipeline.train_data['actions'][:5]
    sample_rewards = trainer.predict_rewards(sample_states, sample_actions)
    print(f"  Sample predicted rewards: {sample_rewards}")

    # Save model if experiment_prefix is provided
    if experiment_prefix is not None:
        trainer.save_model(experiment_prefix, save_dir)

    return trainer, history


if __name__ == "__main__":
    # Run IRL recovery with default parameters
    trainer, history = run_irl_recovery(
        model_type='binary',
        n_epochs=100,
        batch_size=64,
        learning_rate=1e-3,
        hidden_dims=[128, 64, 32],
        temperature=1.0,
        window_size=6,
        random_seed=42,
        experiment_prefix='irl_binary_v1',
        save_dir='experiment/irl'
    )

    # Example: Load the saved model
    # loaded_trainer = MaxSLPIRLTrainer.load_model(
    #     experiment_prefix='irl_binary_v1',
    #     save_dir='experiment/irl',
    #     hidden_dims=[128, 64, 32]
    # )
