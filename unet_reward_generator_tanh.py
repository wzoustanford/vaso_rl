## imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import json
from datetime import datetime

# Local imports
from medical_sequence_buffer_v2 import MedicalSequenceBufferV2
from run_block_discrete_cql_allalphas import DualBlockDiscreteCQL
from integrated_data_pipeline_v3 import IntegratedDataPipelineV3

## in this python file, we intend to implement a new RL/ML algorithm which combines reward generation using expanded horizons, and learning rewards using imitation learning
## the main ML training flow is the same with most of the q-learning code in this folder 

## Hyperparameters
class Config:
    """Configuration for U-Net Reward Generator training."""
    # Model architecture (state_size inferred from data)
    conv_h_dim: int = 16          # Hidden dimension for conv layers
    # bottleneck_dim = conv_h_dim * 2 = 128

    # Action space: VP1 (binary) x VP2 (discretized)
    vp1_bins: int = 2             # VP1 is binary (0 or 1)
    vp2_bins: int = 5             # VP2 discretized into 5 bins
    action_size: int = vp1_bins * vp2_bins  # = 10

    # Training
    D: int = 10                   # Horizon distance for Q aggregation
    batch_size: int = 128
    num_epochs: int = 100
    learning_rate: float = 1e-3
    gamma: float = 0.99           # Discount factor for reward aggregation
    
    # Evaluation & logging
    num_steps_per_eval_print: int = 50
    eval_q_t_step1: int = 10      # First timestep to evaluate Q
    eval_q_t_step2: int = 20      # Second timestep to evaluate Q

    # Sequence buffer
    sequence_length: int = 40     # Will be set based on data
    overlap: int = 1              # As specified in prompts

    # Paths
    experiment_dir: str = "experiments/unet_reward_gen_fixed_32"

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

## U-Net Architecture for Variable-Length Sequence Reward Generation
class UNetRewardGenerator(nn.Module):
    """
    U-Net that takes variable-length trajectories and outputs per-timestep rewards.

    Input: [batch, seq_len, state_size + action_size]
    Output: [batch, seq_len, 1] (reward per timestep)
    """

    def __init__(self, state_size: int, action_size: int, conv_h_dim: int = 64):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.input_size = state_size + action_size
        self.conv_h_dim = conv_h_dim
        self.bottleneck_dim = conv_h_dim * 2  # 128

        # Encoder: 3 conv layers (no padding, shrinks by 2 per layer)
        # L -> L-2 -> L-4 -> L-6 (bottleneck)
        self.enc1 = nn.Sequential(
            nn.Conv1d(self.input_size, conv_h_dim, kernel_size=3),
            #nn.BatchNorm1d(conv_h_dim),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(conv_h_dim, conv_h_dim, kernel_size=3),
            #nn.BatchNorm1d(conv_h_dim),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(conv_h_dim, conv_h_dim * 2, kernel_size=3),  # bottleneck: [batch, 128, L-6]
            #nn.BatchNorm1d(conv_h_dim * 2),
            nn.ReLU()
        )

        # Decoder: 3 ConvTranspose1d layers (no padding, grows by 2 per layer)
        # L-6 -> L-4 -> L-2 -> L
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(conv_h_dim * 2, conv_h_dim, kernel_size=3),  # 128 -> 64, L-6 -> L-4
            #nn.BatchNorm1d(conv_h_dim),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(conv_h_dim + conv_h_dim, conv_h_dim, kernel_size=3),  # 64+64 -> 64, L-4 -> L-2
            #nn.BatchNorm1d(conv_h_dim),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(conv_h_dim + conv_h_dim, conv_h_dim, kernel_size=3),  # 64+64 -> 64, L-2 -> L
            #nn.BatchNorm1d(conv_h_dim),
            nn.ReLU()
        )

        # Output layer: produces 1 reward per timestep
        self.output_layer = nn.Sequential(
            nn.Conv1d(conv_h_dim, 1, kernel_size=1),
            nn.Tanh()
        )
    
    def encode(self, x: torch.Tensor) -> tuple:
        """
        Encode input sequence through conv layers.

        Args:
            x: [batch, seq_len, state_size + action_size]
        Returns:
            bottleneck: [batch, 128, L-6]
            skip_connections: (e1, e2) for decoder
        """
        # Conv1d expects [batch, channels, seq_len]
        x = x.permute(0, 2, 1)

        # Encoder
        e1 = self.enc1(x)    # [batch, 64, L-2]
        e2 = self.enc2(e1)   # [batch, 64, L-4]
        e3 = self.enc3(e2)   # [batch, 128, L-6] (bottleneck)

        return e3, (e1, e2)

    def decode(self, bottleneck: torch.Tensor, skip_connections: tuple) -> torch.Tensor:
        """
        Decode bottleneck back to per-timestep rewards using transposed convolutions.

        Args:
            bottleneck: [batch, 128, L-6]
            skip_connections: (e1, e2) from encoder
        Returns:
            rewards: [batch, L, 1]
        """
        e1, e2 = skip_connections

        # Decoder with skip connections
        d3 = self.dec3(bottleneck)                  # [batch, 64, L-4]
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # [batch, 64, L-2]
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # [batch, 64, L]

        # Output: reward per timestep
        out = self.output_layer(d1)  # [batch, 1, L]

        # Return as [batch, L, 1]
        return out.permute(0, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, state_size + action_size]
        Returns:
            rewards: [batch, seq_len, 1]
        """
        bottleneck, skip_connections = self.encode(x)
        rewards = self.decode(bottleneck, skip_connections)
        return rewards

## Main Training Loop

def load_trajectories(data_pipeline, config: Config) -> tuple:
    """
    Load patient trajectories into sequence buffer.

    Args:
        data_pipeline: IntegratedDataPipelineV2 or V3
        config: Configuration object
    Returns:
        buffer: MedicalSequenceBufferV2 with loaded trajectories
        state_size: Inferred state dimension
    """
    print("[DEBUG] Creating MedicalSequenceBufferV2...")
    buffer = MedicalSequenceBufferV2(
        capacity=100000,
        sequence_length=config.sequence_length,
        burn_in_length=0,
        overlap=config.overlap,  # overlap=1
        priority_type='uniform'
    )
    print(f"[DEBUG] Buffer created with sequence_length={config.sequence_length}, overlap={config.overlap}")

    print("[DEBUG] Calling data_pipeline.prepare_data()...")
    train_data, _, _ = data_pipeline.prepare_data()
    print(f"[DEBUG] Data prepared. Number of transitions: {len(train_data['states'])}")

    patient_ids = train_data.get('patient_ids', np.arange(len(train_data['states'])))
    unique_patients = len(np.unique(patient_ids))
    print(f"[DEBUG] Unique patients: {unique_patients}")

    print("[DEBUG] Adding transitions to buffer...")
    n_transitions = len(train_data['states'])
    for i in range(n_transitions):
        if i % 10000 == 0:
            print(f"[DEBUG] Added {i}/{n_transitions} transitions...")
        buffer.add_transition(
            state=train_data['states'][i],
            action=train_data['actions'][i],
            reward=train_data['rewards'][i],
            next_state=train_data['next_states'][i],
            done=train_data['dones'][i],
            patient_id=patient_ids[i]
        )
    print(f"[DEBUG] Finished adding all {n_transitions} transitions")

    stats = buffer.get_statistics()
    print(f"Loaded {stats['total_sequences_generated']} sequences from {stats['total_patients_processed']} patients")

    return buffer, train_data['states'].shape[1]  # Return state_size


def compute_training_step(
    model: UNetRewardGenerator,
    states: torch.Tensor,
    actions: torch.Tensor,
    config: Config,
    device: torch.device
) -> tuple:
    """
    Compute loss for one batch of sequences.

    Args:
        model: UNetRewardGenerator
        states: [batch, seq_len, state_size]
        actions: [batch, seq_len, 2] - expert actions (vp1, vp2) continuous
        config: Configuration
        device: torch.device
    Returns:
        loss: scalar loss
        metrics: dict with Q values and accuracy
    """
    batch_size, seq_len, state_size = states.shape
    action_size = config.action_size

    # --- Block 6b-3: Convert expert actions to discrete indices ---
    converter = DualBlockDiscreteCQL(state_dim=1, vp2_bins=config.vp2_bins)
    expert_action_idx = torch.tensor([
        [converter.continuous_to_discrete_action(actions[b, t].cpu().numpy())
         for t in range(seq_len)]
        for b in range(batch_size)
    ], device=device)

    # --- Block 6b-4, 6b-5: Compute Q values and loss ---
    D = config.D
    gamma = config.gamma
    discount = torch.tensor([gamma ** t for t in range(D)], device=device)

    total_loss = 0.0
    q_values_list = []
    correct_predictions = 0
    total_predictions = 0

    # Convert expert actions to one-hot: [batch, seq_len] -> [batch, seq_len, action_size]
    expert_action_one_hot = F.one_hot(expert_action_idx, num_classes=action_size).float()

    # Expand for all action choices at current timestep
    # states_expanded: [batch, seq_len, action_size, state_size]
    states_expanded = states.unsqueeze(2).expand(-1, -1, action_size, -1)
    # expert_action_expanded: [batch, seq_len, action_size, action_size]
    expert_action_expanded = expert_action_one_hot.unsqueeze(2).expand(-1, -1, action_size, -1)

    # Eye matrix for replacing at timestep ct (create once outside loop)
    # action_eye: [batch, action_size, action_size]
    action_eye = torch.eye(action_size, device=device).unsqueeze(0).expand(batch_size, -1, -1)

    for ct in range(seq_len - D):

        # --- Block 6b-1: Replace expert action at ct with all possible actions ---
        cur_action_expanded = expert_action_expanded.clone()
        cur_action_expanded[:, ct, :, :] = action_eye
        state_action = torch.cat([states_expanded, cur_action_expanded], dim=-1)

        # --- Block 6b-2: Reshape and forward through U-Net ---
        state_action_flat = state_action.permute(0, 2, 1, 3).reshape(
            batch_size * action_size, seq_len, state_size + action_size
        )
        rewards_pred = model(state_action_flat)
        rewards_pred = rewards_pred.reshape(batch_size, action_size, seq_len, 1)
        rewards_pred = rewards_pred.squeeze(-1).permute(0, 2, 1)

        future_rewards = rewards_pred[:, ct:ct+D, :]
        Q_values = (future_rewards * discount.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        Q_logits = F.softmax(Q_values, dim=1)
        q_values_list.append(Q_values.mean().item())

        expert_idx_ct = expert_action_idx[:, ct]
        loss_ct = F.cross_entropy(Q_logits, expert_idx_ct, reduction='sum')
        total_loss += loss_ct

        predicted_action = Q_logits.argmax(dim=-1)
        correct_predictions += (predicted_action == expert_idx_ct).sum().item()
        total_predictions += batch_size

    # --- Block 6b-6: Normalize loss ---
    T = seq_len - D
    loss = total_loss / (T * batch_size)

    metrics = {
        'avg_q': np.mean(q_values_list),
        'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0.0
    }

    return loss, metrics


def evaluate_validation(
    model: UNetRewardGenerator,
    val_buffer,
    config: Config,
    device: torch.device
) -> dict:
    """
    Evaluate on validation batch.
    Returns accuracy and Q values at specific timesteps.
    """
    model.eval()
    with torch.no_grad():
        burn_in, training, indices, weights = val_buffer.sample_sequences(config.batch_size)
        states = torch.FloatTensor(training['states']).to(device)
        actions = torch.FloatTensor(training['actions']).to(device)

        batch_size, seq_len, state_size = states.shape
        action_size = config.action_size

        # Convert expert actions to discrete indices
        converter = DualBlockDiscreteCQL(state_dim=1, vp2_bins=config.vp2_bins)
        expert_action_idx = torch.tensor([
            [converter.continuous_to_discrete_action(actions[b, t].cpu().numpy())
             for t in range(seq_len)]
            for b in range(batch_size)
        ], device=device)

        # Convert expert actions to one-hot: [batch, seq_len] -> [batch, seq_len, action_size]
        expert_action_one_hot = F.one_hot(expert_action_idx, num_classes=action_size).float()

        # Expand for all action choices at current timestep
        states_expanded = states.unsqueeze(2).expand(-1, -1, action_size, -1)
        expert_action_expanded = expert_action_one_hot.unsqueeze(2).expand(-1, -1, action_size, -1)

        # Eye matrix for replacing at timestep ct (create once outside loop)
        action_eye = torch.eye(action_size, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        # Compute Q values
        D = config.D
        gamma = config.gamma
        discount = torch.tensor([gamma ** t for t in range(D)], device=device)

        correct = 0
        total = 0
        q_expert_step1 = []
        q_expert_step2 = []
        q_max_step1 = []
        q_max_step2 = []

        for ct in range(seq_len - D):
            # Replace expert action at ct with all possible actions
            cur_action_expanded = expert_action_expanded.clone()
            cur_action_expanded[:, ct, :, :] = action_eye
            state_action = torch.cat([states_expanded, cur_action_expanded], dim=-1)

            # Reshape and forward through U-Net
            state_action_flat = state_action.permute(0, 2, 1, 3).reshape(
                batch_size * action_size, seq_len, state_size + action_size
            )
            rewards_pred = model(state_action_flat)
            rewards_pred = rewards_pred.reshape(batch_size, action_size, seq_len, 1)
            rewards_pred = rewards_pred.squeeze(-1).permute(0, 2, 1)

            # Compute Q values for this timestep
            future_rewards = rewards_pred[:, ct:ct+D, :]
            Q_values = (future_rewards * discount.unsqueeze(0).unsqueeze(-1)).sum(dim=1)

            # Accuracy
            predicted = Q_values.argmax(dim=-1)
            correct += (predicted == expert_action_idx[:, ct]).sum().item()
            total += batch_size

            # Q values at specific timesteps
            if ct == config.eval_q_t_step1:
                expert_q = Q_values.gather(1, expert_action_idx[:, ct].unsqueeze(1)).mean().item()
                q_expert_step1.append(expert_q)
                max_q = Q_values.max(dim=1)[0].mean().item()
                q_max_step1.append(max_q)
            if ct == config.eval_q_t_step2:
                expert_q = Q_values.gather(1, expert_action_idx[:, ct].unsqueeze(1)).mean().item()
                q_expert_step2.append(expert_q)
                max_q = Q_values.max(dim=1)[0].mean().item()
                q_max_step2.append(max_q)

    return {
        'val_accuracy': correct / total if total > 0 else 0.0,
        'val_q_expert_t1': np.mean(q_expert_step1) if q_expert_step1 else 0.0,
        'val_q_expert_t2': np.mean(q_expert_step2) if q_expert_step2 else 0.0,
        'val_q_max_t1': np.mean(q_max_step1) if q_max_step1 else 0.0,
        'val_q_max_t2': np.mean(q_max_step2) if q_max_step2 else 0.0
    }


def train(config: Config, data_pipeline):
    """
    Main training loop for U-Net reward generator.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config.experiment_dir, exist_ok=True)

    # Load train and validation trajectories
    print("[DEBUG] Calling load_trajectories...")
    buffer, state_size = load_trajectories(data_pipeline, config)
    print(f"[DEBUG] Buffer loaded. state_size={state_size}, buffer length={len(buffer)}")

    # Create validation buffer (using same pipeline but could be separate)
    val_buffer = buffer  # TODO: Create separate val buffer if needed

    # Initialize model
    print(f"[DEBUG] Initializing UNetRewardGenerator with state_size={state_size}, action_size={config.action_size}")
    model = UNetRewardGenerator(
        state_size=state_size,
        action_size=config.action_size,
        conv_h_dim=config.conv_h_dim
    ).to(device)
    print("[DEBUG] Model initialized")

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    print("[DEBUG] Optimizer created")

    step = 0
    n_batches = len(buffer) // config.batch_size
    print(f"[DEBUG] Starting training: {config.num_epochs} epochs, {n_batches} batches per epoch")

    for epoch in range(config.num_epochs):
        print(f"[DEBUG] Epoch {epoch+1}/{config.num_epochs} starting...")
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        for batch_idx in range(n_batches):
            if batch_idx % 10 == 0:
                print(f"[DEBUG] Epoch {epoch+1}, Batch {batch_idx}/{n_batches}...")

            print(f"[DEBUG] Sampling sequences...") if batch_idx == 0 else None
            burn_in, training, indices, weights = buffer.sample_sequences(config.batch_size)
            print(f"[DEBUG] Sequences sampled, shapes: states={training['states'].shape}") if batch_idx == 0 else None

            states = torch.FloatTensor(training['states']).to(device)
            actions = torch.FloatTensor(training['actions']).to(device)
            print(f"[DEBUG] Tensors created") if batch_idx == 0 else None

            optimizer.zero_grad()
            print(f"[DEBUG] Calling compute_training_step...") if batch_idx == 0 else None
            loss, metrics = compute_training_step(model, states, actions, config, device)
            print(f"[DEBUG] Loss computed: {loss.item():.4f}") if batch_idx == 0 else None

            loss.backward()
            optimizer.step()
            print(f"[DEBUG] Backward and step done") if batch_idx == 0 else None

            epoch_loss += loss.item()
            epoch_acc += metrics['accuracy']
            num_batches += 1
            step += 1

            # Evaluate every num_steps_per_eval_print
            if step % config.num_steps_per_eval_print == 0:
                val_metrics = evaluate_validation(model, val_buffer, config, device)
                print(f"Step {step} | Train Loss: {epoch_loss/num_batches:.4f} | "
                      f"Train Acc: {epoch_acc/num_batches:.4f} | "
                      f"Val Acc: {val_metrics['val_accuracy']:.4f}")
                print(f"  Q_expert@T{config.eval_q_t_step1}: {val_metrics['val_q_expert_t1']:.4f} | "
                      f"Q_expert@T{config.eval_q_t_step2}: {val_metrics['val_q_expert_t2']:.4f} | "
                      f"Q_max@T{config.eval_q_t_step1}: {val_metrics['val_q_max_t1']:.4f} | "
                      f"Q_max@T{config.eval_q_t_step2}: {val_metrics['val_q_max_t2']:.4f}")
                model.train()  # Back to train mode

        # End of epoch
        print(f"Epoch {epoch+1}/{config.num_epochs} complete")

        # Save model each epoch
        save_path = os.path.join(config.experiment_dir, f"model_epoch_{epoch+1}.pt")
        save_model(model, config, save_path)

    print("Training complete!")
    return model


## Model saving and loading functions

def save_model(model: UNetRewardGenerator, config: Config, filepath: str):
    """
    Save model checkpoint.

    Args:
        model: UNetRewardGenerator model
        config: Configuration object
        filepath: Path to save the model
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'state_size': model.state_size,
        'action_size': model.action_size,
        'conv_h_dim': model.conv_h_dim,
        'config': {
            'D': config.D,
            'gamma': config.gamma,
            'vp1_bins': config.vp1_bins,
            'vp2_bins': config.vp2_bins,
            'sequence_length': config.sequence_length
        }
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str, device: torch.device = None) -> tuple:
    """
    Load model from checkpoint.

    Args:
        filepath: Path to the saved model
        device: Device to load the model to
    Returns:
        model: UNetRewardGenerator model
        config_dict: Dictionary of saved config values
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(filepath, map_location=device)

    model = UNetRewardGenerator(
        state_size=checkpoint['state_size'],
        action_size=checkpoint['action_size'],
        conv_h_dim=checkpoint['conv_h_dim']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded from {filepath}")
    return model, checkpoint['config']


## Main entry point

def main():
    """Main entry point for training U-Net reward generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train U-Net Reward Generator')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--conv_h_dim', type=int, default=64, help='Conv hidden dimension')
    parser.add_argument('--D', type=int, default=10, help='Q-value horizon')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--vp1_bins', type=int, default=2, help='VP1 bins')
    parser.add_argument('--vp2_bins', type=int, default=5, help='VP2 bins')
    parser.add_argument('--experiment_dir', type=str, default='experiments/unet_reward_gen',
                       help='Directory to save models')
    args = parser.parse_args()

    # Initialize config from command line args
    config = Config(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        conv_h_dim=args.conv_h_dim,
        D=args.D,
        gamma=args.gamma,
        vp1_bins=args.vp1_bins,
        vp2_bins=args.vp2_bins,
        experiment_dir=args.experiment_dir+'_'+str(args.conv_h_dim)+'_tanh',
    )

    print(f"Config: epochs={config.num_epochs}, batch_size={config.batch_size}, "
          f"lr={config.learning_rate}, conv_h_dim={config.conv_h_dim}, "
          f"D={config.D}, gamma={config.gamma}")

    # Initialize data pipeline (use manual rewards - we generate our own via U-Net)
    print("Initializing data pipeline...")
    pipeline = IntegratedDataPipelineV3(
        model_type='dual',
        reward_source='manual',
        random_seed=42
    )

    # Train model
    model = train(config, pipeline)

    print("Done!")


if __name__ == "__main__":
    main()
