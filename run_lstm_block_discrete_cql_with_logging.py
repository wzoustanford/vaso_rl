#!/usr/bin/env python3
"""
LSTM Block Discrete CQL Training Script with VP1 x VP2 Action Space
====================================================================
Trains LSTM CQL with discrete action space:
- VP1: Binary vasopressin (0 or 1)
- VP2: Discretized norepinephrine (N bins from 0 to 0.5 mcg/kg/min)
"""

import numpy as np
import torch
import time
import os
import sys
import json
from datetime import datetime
from typing import Dict, Tuple

# Import our components
from integrated_data_pipeline_v2 import IntegratedDataPipelineV2
from medical_sequence_buffer import MedicalSequenceBuffer, SequenceDataLoader
from lstm_block_discrete_cql_network import LSTMBlockDiscreteCQL
from fqe_gaussian_analysis import FQEGaussianAnalysis, save_histogram_plot, save_q_values_to_pickle

# Force unbuffered output
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__


class Logger:
    """Simple logger that writes to both file and stdout."""
    
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.log_file = open(log_path, 'w', buffering=1)  # Line buffered
        
    def log(self, message: str):
        """Write message to both file and stdout."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        print(full_message, flush=True)
        self.log_file.write(full_message + '\n')
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()


def evaluate_lstm_cql(
    agent: LSTMBlockDiscreteCQL,
    sequence_buffer: MedicalSequenceBuffer,
    num_eval_batches: int = 10,
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Evaluate LSTM CQL on validation sequences with detailed metrics.
    """
    agent.q1.eval()
    agent.q2.eval()
    
    metrics = {
        'q_values': [],
        'q1_values': [],
        'q2_values': [],
        'q_variance': []
    }
    
    with torch.no_grad():
        for _ in range(num_eval_batches):
            # Sample sequences
            burn_in_batch, training_batch, _, weights = sequence_buffer.sample_sequences(batch_size)
            
            # Convert to torch
            burn_in_torch, training_torch, weights_torch = SequenceDataLoader.prepare_torch_batch(
                burn_in_batch, training_batch, weights, device=agent.device
            )
            
            # Convert actions to long tensors (indices) and squeeze last dimension
            burn_in_torch['actions'] = burn_in_torch['actions'].long().squeeze(-1)
            training_torch['actions'] = training_torch['actions'].long().squeeze(-1)
            
            # Get Q-values
            hidden1 = agent.q1.init_hidden(batch_size, agent.device)
            hidden2 = agent.q2.init_hidden(batch_size, agent.device)
            
            # Burn-in
            if burn_in_torch['states'].shape[1] > 0:
                _, hidden1 = agent.q1.forward(burn_in_torch['states'], hidden1)
                _, hidden2 = agent.q2.forward(burn_in_torch['states'], hidden2)
            
            # Get Q-values for training sequence
            q1_vals, _ = agent.q1.get_q_values(training_torch['states'], training_torch['actions'], hidden1)
            q2_vals, _ = agent.q2.get_q_values(training_torch['states'], training_torch['actions'], hidden2)
            
            q_vals = torch.min(q1_vals, q2_vals)
            
            metrics['q_values'].append(q_vals.mean().item())
            metrics['q1_values'].append(q1_vals.mean().item())
            metrics['q2_values'].append(q2_vals.mean().item())
            metrics['q_variance'].append(q_vals.var().item())
    
    return {
        'mean_q_value': np.mean(metrics['q_values']),
        'std_q_value': np.std(metrics['q_values']),
        'mean_q1_value': np.mean(metrics['q1_values']),
        'mean_q2_value': np.mean(metrics['q2_values']),
        'mean_q_variance': np.mean(metrics['q_variance'])
    }


def train_lstm_block_discrete_cql(
    alpha: float = 0.0,
    vp2_bins: int = 5,  # Number of discrete bins for VP2 (Norepinephrine)
    sequence_length: int = 20,
    burn_in_length: int = 8,
    overlap: int = 10,
    hidden_dim: int = 64,
    lstm_hidden: int = 64,
    num_lstm_layers: int = 2,
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    tau: float = 0.005,
    gamma: float = 0.95,
    grad_clip: float = 1.0,
    buffer_capacity: int = 50000,
    save_dir: str = 'experiment',
    log_dir: str = 'logs',
    log_every: int = 1  # Log every epoch
):
    """
    Train LSTM-based Block Discrete CQL with VP1 x VP2 discrete action space.
    VP1: Binary vasopressin (0 or 1)
    VP2: Discretized norepinephrine into bins (0 to 0.5 mcg/kg/min)
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use scientific notation for very small alphas in log filename
    if alpha == 0.0:
        alpha_str = "0.0"
    elif alpha < 0.0001:
        alpha_str = f"{alpha:.1e}".replace('.', 'p').replace('-', 'm')
    else:
        alpha_str = f"{alpha:.4f}"
    log_path = os.path.join(log_dir, f'lstm_block_discrete_cql_alpha{alpha_str}_bins{vp2_bins}_{timestamp}.log')
    logger = Logger(log_path)
    
    logger.log("="*70)
    logger.log(f" LSTM BLOCK DISCRETE CQL TRAINING WITH ALPHA={alpha}, VP2_BINS={vp2_bins}")
    logger.log("="*70)
    
    # Save configuration to JSON
    config = {
        'alpha': alpha,
        'vp2_bins': vp2_bins,
        'sequence_length': sequence_length,
        'burn_in_length': burn_in_length,
        'overlap': overlap,
        'hidden_dim': hidden_dim,
        'lstm_hidden': lstm_hidden,
        'num_lstm_layers': num_lstm_layers,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'tau': tau,
        'gamma': gamma,
        'grad_clip': grad_clip,
        'buffer_capacity': buffer_capacity,
        'timestamp': timestamp
    }
    
    config_path = os.path.join(log_dir, f'lstm_block_discrete_cql_config_alpha{alpha_str}_bins{vp2_bins}_{timestamp}.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.log(f"Configuration saved to {config_path}")
    
    # Initialize data pipeline
    logger.log("\nInitializing data pipeline...")
    # Use 'dual' mode to get norepinephrine as separate column
    pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
    train_data, val_data, test_data = pipeline.prepare_data()
    
    # State dimension is without norepinephrine (since it's now an action)
    state_dim = train_data['states'].shape[1]
    
    # Define discrete action bins for VP1 x VP2 (aligned with block discrete CQL)
    # VP1: Binary (0 or 1)
    # VP2: Discretized into bins from 0 to 0.5 mcg/kg/min
    vp2_bin_edges = np.linspace(0, 0.5, vp2_bins + 1)
    vp2_bin_centers = (vp2_bin_edges[:-1] + vp2_bin_edges[1:]) / 2
    num_actions = 2 * vp2_bins  # VP1 (2 options) x VP2 (vp2_bins)
    
    # Helper function to convert continuous [vp1, vp2] to discrete action index
    def continuous_to_discrete_action(continuous_action):
        """
        Convert continuous action [vp1, vp2] to discrete action index.

        Args:
            continuous_action: [vp1, vp2] where vp1 is binary (0 or 1), vp2 is continuous

        Returns:
            action_idx: discrete action index (0 to num_actions-1)
        """
        vp1, vp2 = continuous_action
        vp1_idx = int(vp1)  # 0 or 1

        # Find which bin vp2 falls into
        vp2_bin = np.digitize(vp2, vp2_bin_edges) - 1
        vp2_bin = np.clip(vp2_bin, 0, vp2_bins - 1)

        # Combine into single action index: action_idx = vp1_idx * vp2_bins + vp2_bin
        action_idx = vp1_idx * vp2_bins + vp2_bin
        return action_idx
    
    # Print configuration
    logger.log("\n" + "="*70)
    logger.log("CONFIGURATION:")
    logger.log(f"  State dimension: {state_dim}")
    logger.log(f"  Action space: VP1 (binary) x VP2 ({vp2_bins} bins) = {num_actions} total discrete actions")
    logger.log(f"  VP2 bin edges: {vp2_bin_edges}")
    logger.log(f"  VP2 bin centers: {vp2_bin_centers}")
    logger.log(f"  Sequence length: {sequence_length}")
    logger.log(f"  Burn-in length: {burn_in_length}")
    logger.log(f"  Training length: {sequence_length - burn_in_length}")
    logger.log(f"  Overlap: {overlap}")
    logger.log(f"  Hidden dim: {hidden_dim}")
    logger.log(f"  LSTM hidden: {lstm_hidden}")
    logger.log(f"  LSTM layers: {num_lstm_layers}")
    logger.log(f"  Alpha (CQL penalty): {alpha}")
    logger.log(f"  Tau (soft update): {tau}")
    logger.log(f"  Learning rate: {learning_rate}")
    logger.log(f"  Batch size: {batch_size}")
    logger.log(f"  Epochs: {epochs}")
    logger.log(f"  Gradient clipping: {grad_clip}")
    logger.log("="*70)
    
    # Create sequence buffers
    logger.log("\nCreating sequence buffers...")
    
    train_buffer = MedicalSequenceBuffer(
        capacity=buffer_capacity,
        sequence_length=sequence_length,
        burn_in_length=burn_in_length,
        overlap=overlap,
        priority_type='mortality_weighted'
    )
    
    val_buffer = MedicalSequenceBuffer(
        capacity=buffer_capacity // 5,
        sequence_length=sequence_length,
        burn_in_length=burn_in_length,
        overlap=overlap,
        priority_type='uniform'
    )
    
    # Fill training buffer
    logger.log("Generating training sequences...")
    for patient_id, (start_idx, end_idx) in pipeline.train_patient_groups.items():
        for t in range(start_idx, end_idx):
            # Convert continuous [vp1, vp2] action to discrete action index
            # In dual mode, actions are [vp1, vp2] where vp1 is vasopressin, vp2 is norepinephrine
            continuous_action = train_data['actions'][t]  # [vp1, vp2]
            action_idx = continuous_to_discrete_action(continuous_action)

            train_buffer.add_transition(
                state=train_data['states'][t],
                action=np.array([action_idx]),  # Store as discrete action index
                reward=train_data['rewards'][t],
                next_state=train_data['next_states'][t],
                done=bool(train_data['dones'][t]),
                patient_id=patient_id
            )
    
    logger.log(f"  Generated {len(train_buffer)} training sequences from {len(pipeline.train_patient_groups)} patients")
    
    # Fill validation buffer
    logger.log("Generating validation sequences...")
    for patient_id, (start_idx, end_idx) in pipeline.val_patient_groups.items():
        for t in range(start_idx, end_idx):
            # Convert continuous [vp1, vp2] action to discrete action index
            # In dual mode, actions are [vp1, vp2] where vp1 is vasopressin, vp2 is norepinephrine
            continuous_action = val_data['actions'][t]  # [vp1, vp2]
            action_idx = continuous_to_discrete_action(continuous_action)

            val_buffer.add_transition(
                state=val_data['states'][t],
                action=np.array([action_idx]),  # Store as discrete action index
                reward=val_data['rewards'][t],
                next_state=val_data['next_states'][t],
                done=bool(val_data['dones'][t]),
                patient_id=patient_id
            )
    
    logger.log(f"  Generated {len(val_buffer)} validation sequences from {len(pipeline.val_patient_groups)} patients")
    
    # Initialize LSTM Block Discrete CQL agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.log(f"\nInitializing LSTM Block Discrete CQL agent on {device}...")
    
    agent = LSTMBlockDiscreteCQL(
        state_dim=state_dim,
        num_actions=num_actions,
        action_bins=vp2_bin_centers,  # Use bin centers for conversion
        hidden_dim=hidden_dim,
        lstm_hidden=lstm_hidden,
        num_lstm_layers=num_lstm_layers,
        alpha=alpha,
        gamma=gamma,
        tau=tau,
        lr=learning_rate,
        device=device
    )
    
    # Store grad_clip for later use
    agent.grad_clip = grad_clip
    
    total_params = sum(p.numel() for p in agent.q1.parameters())
    logger.log(f"  Total parameters: {total_params:,}")
    
    # Training loop
    logger.log(f"\nStarting training for {epochs} epochs...")
    logger.log("="*70)
    start_time = time.time()
    
    best_val_q = -float('inf')
    batches_per_epoch = min(len(train_buffer) // batch_size, 500)
    
    logger.log(f"Batches per epoch: {batches_per_epoch}")
    logger.log("")
    
    # CSV header for easy parsing
    logger.log("Epoch,Time,Q1_Loss,Q2_Loss,TD_Loss,CQL_Loss,Train_Q,Train_Q1,Train_Q2,Val_Q,Val_Q1,Val_Q2,Val_Q_Std")
    
    training_history = {
        'train_loss': [],
        'train_q_values': [],
        'val_q_values': [],
        'cql_penalties': [],
        'td_losses': []
    }
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        agent.q1.train()
        agent.q2.train()
        
        epoch_metrics = {
            'q1_loss': 0,
            'q2_loss': 0,
            'q1_td_loss': 0,
            'q2_td_loss': 0,
            'q1_cql_loss': 0,
            'q2_cql_loss': 0,
            'q_values': 0,
            'q1_values': 0,
            'q2_values': 0,
            'grad_norm': 0
        }
        
        for batch_idx in range(batches_per_epoch):
            # Sample sequences
            burn_in_batch, training_batch, indices, weights = train_buffer.sample_sequences(batch_size)
            
            # Convert to torch tensors
            burn_in_torch, training_torch, weights_torch = SequenceDataLoader.prepare_torch_batch(
                burn_in_batch, training_batch, weights, device=device
            )
            
            # Convert actions to long tensors (indices) and squeeze last dimension
            burn_in_torch['actions'] = burn_in_torch['actions'].long().squeeze(-1)
            training_torch['actions'] = training_torch['actions'].long().squeeze(-1)
            
            # Update agent
            metrics = agent.update_sequences(burn_in_torch, training_torch, weights_torch)
            
            # Update priorities based on TD error
            if hasattr(train_buffer, 'update_priorities'):
                td_errors = metrics.get('td_errors', None)
                if td_errors is not None:
                    train_buffer.update_priorities(indices, td_errors.cpu().numpy())
            
            # Accumulate metrics
            for key in metrics:
                if key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= batches_per_epoch
        
        # Store training metrics
        training_history['train_loss'].append(epoch_metrics['q1_loss'])
        training_history['train_q_values'].append(epoch_metrics['q_values'])
        training_history['td_losses'].append(epoch_metrics['q1_td_loss'])
        training_history['cql_penalties'].append(epoch_metrics['q1_cql_loss'])
        
        # Validation phase
        val_metrics = evaluate_lstm_cql(agent, val_buffer, num_eval_batches=10, batch_size=batch_size)
        training_history['val_q_values'].append(val_metrics['mean_q_value'])
        
        # Time for epoch
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        # Log metrics in CSV format
        logger.log(f"{epoch+1},{total_time/60:.2f},"
                  f"{epoch_metrics['q1_loss']:.6f},{epoch_metrics['q2_loss']:.6f},"
                  f"{epoch_metrics['q1_td_loss']:.6f},{epoch_metrics['q1_cql_loss']:.6f},"
                  f"{epoch_metrics['q_values']:.6f},{epoch_metrics['q1_values']:.6f},{epoch_metrics['q2_values']:.6f},"
                  f"{val_metrics['mean_q_value']:.6f},{val_metrics['mean_q1_value']:.6f},"
                  f"{val_metrics['mean_q2_value']:.6f},{val_metrics['std_q_value']:.6f}")
        
        # Save best model
        if val_metrics['mean_q_value'] > best_val_q:
            best_val_q = val_metrics['mean_q_value']
            
            # Use scientific notation for very small alphas to avoid filename collisions
            if alpha == 0.0:
                alpha_str = "0.0"
            elif alpha < 0.0001:
                alpha_str = f"{alpha:.1e}".replace('.', 'p').replace('-', 'm')
            else:
                alpha_str = f"{alpha:.4f}"
            save_path = os.path.join(save_dir, f'lstm_block_discrete_cql_alpha{alpha_str}_bins{vp2_bins}_hdim{hidden_dim}_seql{sequence_length}_best.pt')
            torch.save({
                'q1_state_dict': agent.q1.state_dict(),
                'q2_state_dict': agent.q2.state_dict(),
                'q1_target_state_dict': agent.q1_target.state_dict(),
                'q2_target_state_dict': agent.q2_target.state_dict(),
                'q1_optimizer_state_dict': agent.q1_optimizer.state_dict(),
                'q2_optimizer_state_dict': agent.q2_optimizer.state_dict(),
                'vp2_bin_edges': vp2_bin_edges,
                'vp2_bin_centers': vp2_bin_centers,
                'config': config,
                'epoch': epoch + 1,
                'best_val_q': best_val_q,
                'training_history': training_history
            }, save_path)
            
            logger.log(f"  >>> New best model saved at epoch {epoch+1} with Val Q={best_val_q:.6f}")
        
        # Detailed logging every N epochs
        if (epoch + 1) % 5 == 0:
            logger.log("")
            logger.log(f"Epoch {epoch+1}/{epochs} Summary:")
            logger.log(f"  Total Time: {total_time/60:.1f} min")
            logger.log(f"  Q1 Loss: {epoch_metrics['q1_loss']:.6f}")
            logger.log(f"  Q2 Loss: {epoch_metrics['q2_loss']:.6f}")
            logger.log(f"  TD Loss (Q1): {epoch_metrics['q1_td_loss']:.6f}")
            logger.log(f"  CQL Penalty (Q1): {epoch_metrics['q1_cql_loss']:.6f}")
            logger.log(f"  Train Q-value: {epoch_metrics['q_values']:.6f}")
            logger.log(f"  Val Q-value: {val_metrics['mean_q_value']:.6f} ± {val_metrics['std_q_value']:.6f}")
            logger.log(f"  Best Val Q: {best_val_q:.6f}")
            logger.log("")
    
    # Save final model
    # Use scientific notation for very small alphas to avoid filename collisions
    if alpha == 0.0:
        alpha_str = "0.0"
    elif alpha < 0.0001:
        alpha_str = f"{alpha:.1e}".replace('.', 'p').replace('-', 'm')
    else:
        alpha_str = f"{alpha:.4f}"
    final_save_path = os.path.join(save_dir, f'lstm_bd_cql_vp12_alpha{alpha_str}_bins{vp2_bins}_hdim{hidden_dim}_seql{sequence_length}_final.pt')
    torch.save({
        'q1_state_dict': agent.q1.state_dict(),
        'q2_state_dict': agent.q2.state_dict(),
        'q1_target_state_dict': agent.q1_target.state_dict(),
        'q2_target_state_dict': agent.q2_target.state_dict(),
        'q1_optimizer_state_dict': agent.q1_optimizer.state_dict(),
        'q2_optimizer_state_dict': agent.q2_optimizer.state_dict(),
        'vp2_bin_edges': vp2_bin_edges,
        'vp2_bin_centers': vp2_bin_centers,
        'config': config,
        'epoch': epochs,
        'training_history': training_history
    }, final_save_path)
    
    total_time = time.time() - start_time
    logger.log("\n" + "="*70)
    logger.log(f"✅ LSTM Block Discrete CQL training completed in {total_time/60:.1f} minutes!")
    logger.log("Models saved:")
    logger.log(f"  - {save_path} (best)")
    logger.log(f"  - {final_save_path} (final)")
    logger.log(f"  - Log file: {log_path}")
    
    # Final statistics
    logger.log("\nFinal Statistics:")
    logger.log(f"  Best validation Q-value: {best_val_q:.6f}")
    logger.log(f"  Final train Q-value: {epoch_metrics['q_values']:.6f}")
    logger.log(f"  Final validation Q-value: {val_metrics['mean_q_value']:.6f}")
    logger.log("="*70)
    
    logger.close()
    
    return agent, train_buffer, val_buffer, training_history


def main():
    """Main training function."""
    print("="*70)
    print(" LSTM BLOCK DISCRETE CQL TRAINING WITH NOREPINEPHRINE AS ACTION")
    print("="*70)
    print("\nTraining LSTM-based Block Discrete CQL with Norepinephrine as discrete action")
    print("Log files will be saved to logs/ directory")
    print("Models will be saved to experiment/ directory")
    
    # Train with conservative penalty
    agent, train_buffer, val_buffer, history = train_lstm_block_discrete_cql(
        alpha=0.0,
        vp2_bins=10,  # 5 discrete dosing levels for Norepinephrine (aligned with block discrete CQL)
        sequence_length=5,
        burn_in_length=2,
        overlap=2,
        hidden_dim=64,
        lstm_hidden=64,
        num_lstm_layers=2,
        batch_size=32,
        epochs=500,  # Full 100 epochs
        learning_rate=1e-3,
        tau=0.8,
        gamma=0.95,
        grad_clip=1.0,
        buffer_capacity=50000,
        log_every=1  # Log every epoch
    )
    
    print("\n" + "="*70)
    print(" TRAINING COMPLETE")
    print("="*70)
    print("\nCheck logs/ directory for detailed training logs")
    print("Check experiment/ directory for saved models")


if __name__ == "__main__":
    main()