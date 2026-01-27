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
from integrated_data_pipeline_v3 import IntegratedDataPipelineV3
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
    log_every: int = 1,  # Log every epoch
    reward_model_path: str = None,
    suffix: str = "",
    reward_combine_lambda: float = None,
    combined_or_train_data_path: str = None,
    eval_data_path: str = None,
    irl_vp2_bins: int = None
):
    """
    Train LSTM-based Block Discrete CQL with VP1 x VP2 discrete action space.
    VP1: Binary vasopressin (0 or 1)
    VP2: Discretized norepinephrine into bins (0 to 0.5 mcg/kg/min)

    Args:
        alpha: CQL penalty strength
        vp2_bins: Number of bins for VP2 discretization (for Q-learning action space)
        sequence_length: Total sequence length for LSTM
        burn_in_length: Number of steps for burn-in (hidden state warm-up)
        overlap: Overlap between consecutive sequences
        hidden_dim: Hidden dimension for MLP layers
        lstm_hidden: Hidden dimension for LSTM layers
        num_lstm_layers: Number of LSTM layers
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        tau: Soft update coefficient for target networks
        gamma: Discount factor
        grad_clip: Gradient clipping value
        buffer_capacity: Replay buffer capacity
        save_dir: Directory to save models
        log_dir: Directory to save logs
        log_every: Log every N epochs
        reward_model_path: Path to learned reward model (gcl/iq_learn/maxent/unet)
        suffix: Suffix to add to experiment prefix
        reward_combine_lambda: If None, use pure IRL reward. If in [0, 1], use
            (1 - lambda) * manual_reward + lambda * irl_reward.
        combined_or_train_data_path: Path to training dataset. If eval_data_path is
            also provided, all patients are used for training. Otherwise split into
            train/val/test. If None, uses default config.DATA_PATH.
        eval_data_path: Path to evaluation dataset. If provided, enables dual-dataset
            mode where this dataset is split 50/50 into val/test.
        irl_vp2_bins: Number of VP2 bins used to train the IRL model. If None, uses
            the same value as vp2_bins. This allows loading an IRL model trained with
            different discretization than the Q-learning action space.

    Returns:
        agent: Trained LSTMBlockDiscreteCQL agent
        train_buffer: Training sequence buffer
        val_buffer: Validation sequence buffer
        training_history: Dictionary with training metrics
        experiment_prefix: Prefix used for saving models
    """
    # Determine IRL model vp2_bins (default to vp2_bins if not specified)
    if irl_vp2_bins is None:
        irl_vp2_bins = vp2_bins
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
    if irl_vp2_bins != vp2_bins:
        logger.log(f" IRL MODEL VP2_BINS={irl_vp2_bins} (different from Q-learning)")
    logger.log("="*70)

    # Initialize data pipeline and infer reward type
    logger.log("\nInitializing data pipeline...")
    if reward_model_path is None:
        reward_type = "manual"
        pipeline = IntegratedDataPipelineV3(
            model_type='dual', reward_source='manual', random_seed=42,
            combined_or_train_data_path=combined_or_train_data_path,
            eval_data_path=eval_data_path
        )
    elif 'gcl' in reward_model_path:
        reward_type = "gcl"
        pipeline = IntegratedDataPipelineV3(
            model_type='dual', reward_source='learned', random_seed=42,
            reward_combine_lambda=reward_combine_lambda,
            combined_or_train_data_path=combined_or_train_data_path,
            eval_data_path=eval_data_path
        )
        pipeline.load_gcl_reward_model(reward_model_path)
    elif 'iq_learn' in reward_model_path:
        reward_type = "iq_learn"
        pipeline = IntegratedDataPipelineV3(
            model_type='dual', reward_source='learned', random_seed=42,
            reward_combine_lambda=reward_combine_lambda,
            combined_or_train_data_path=combined_or_train_data_path,
            eval_data_path=eval_data_path
        )
        pipeline.load_iq_learn_reward_model(reward_model_path)
    elif 'maxent' in reward_model_path:
        reward_type = "maxent"
        pipeline = IntegratedDataPipelineV3(
            model_type='dual', reward_source='learned', random_seed=42,
            reward_combine_lambda=reward_combine_lambda,
            combined_or_train_data_path=combined_or_train_data_path,
            eval_data_path=eval_data_path
        )
        pipeline.load_maxent_reward_model(reward_model_path)
    elif 'semi_supervised_unet' in reward_model_path:
        reward_type = "semi_supervised_unet"
        pipeline = IntegratedDataPipelineV3(
            model_type='dual', reward_source='learned', random_seed=42,
            reward_combine_lambda=reward_combine_lambda,
            combined_or_train_data_path=combined_or_train_data_path,
            eval_data_path=eval_data_path
        )
        # Use irl_vp2_bins for loading (IRL model's action space), not vp2_bins (Q-learning action space)
        pipeline.load_semi_supervised_unet_reward_model(reward_model_path, vp1_bins=2, vp2_bins=irl_vp2_bins)
        logger.log(f"  IRL model vp2_bins: {irl_vp2_bins}, Q-learning vp2_bins: {vp2_bins}")
    elif 'unet' in reward_model_path:
        reward_type = "unet"
        pipeline = IntegratedDataPipelineV3(
            model_type='dual', reward_source='learned', random_seed=42,
            reward_combine_lambda=reward_combine_lambda,
            combined_or_train_data_path=combined_or_train_data_path,
            eval_data_path=eval_data_path
        )
        # Use irl_vp2_bins for loading (IRL model's action space), not vp2_bins (Q-learning action space)
        pipeline.load_unet_reward_model(reward_model_path, vp1_bins=2, vp2_bins=irl_vp2_bins)
        logger.log(f"  IRL model vp2_bins: {irl_vp2_bins}, Q-learning vp2_bins: {vp2_bins}")
    else:
        raise ValueError(f"Cannot infer reward model type from path: {reward_model_path}")

    # Use pipeline's get_reward_prefix for correct naming with lambda
    experiment_prefix = pipeline.get_reward_prefix() if hasattr(pipeline, 'get_reward_prefix') else reward_type
    experiment_prefix = f"lstm_{experiment_prefix}{suffix}"

    logger.log(f"Reward type: {reward_type}")
    logger.log(f"Experiment prefix: {experiment_prefix}")

    # Save configuration to JSON (now that reward_type and experiment_prefix are defined)
    config = {
        'alpha': alpha,
        'vp2_bins': vp2_bins,
        'irl_vp2_bins': irl_vp2_bins,
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
        'timestamp': timestamp,
        'reward_model_path': reward_model_path,
        'reward_type': reward_type,
        'experiment_prefix': experiment_prefix,
        'suffix': suffix,
        'reward_combine_lambda': reward_combine_lambda,
        'combined_or_train_data_path': combined_or_train_data_path,
        'eval_data_path': eval_data_path
    }

    config_path = os.path.join(log_dir, f'lstm_block_discrete_cql_config_alpha{alpha_str}_bins{vp2_bins}_{timestamp}.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.log(f"Configuration saved to {config_path}")

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

            # Use consistent naming with experiment_prefix (includes lstm_ prefix and reward type)
            save_path = os.path.join(save_dir, f'{experiment_prefix}_alpha{alpha:.4f}_bins{vp2_bins}_best.pt')
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
    
    # Save final model with consistent naming
    final_save_path = os.path.join(save_dir, f'{experiment_prefix}_alpha{alpha:.4f}_bins{vp2_bins}_final.pt')
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

    return agent, train_buffer, val_buffer, training_history, experiment_prefix


def main():
    """Main training function with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(description='Train LSTM Block Discrete CQL with different configurations')
    parser.add_argument('--vp2_bins', type=int, default=5,
                       help='Number of bins for VP2 discretization (default: 5)')
    parser.add_argument('--alpha', type=float, default=0.0,
                       help='Alpha value for CQL penalty (default: 0.0)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--sequence_length', type=int, default=20,
                       help='Total sequence length for LSTM (default: 20)')
    parser.add_argument('--burn_in_length', type=int, default=8,
                       help='Burn-in length for LSTM hidden state (default: 8)')
    parser.add_argument('--overlap', type=int, default=10,
                       help='Overlap between consecutive sequences (default: 10)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension for MLP layers (default: 64)')
    parser.add_argument('--lstm_hidden', type=int, default=64,
                       help='Hidden dimension for LSTM layers (default: 64)')
    parser.add_argument('--num_lstm_layers', type=int, default=2,
                       help='Number of LSTM layers (default: 2)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--tau', type=float, default=0.8,
                       help='Soft update coefficient (default: 0.8)')
    parser.add_argument('--gamma', type=float, default=0.95,
                       help='Discount factor (default: 0.95)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping value (default: 1.0)')
    parser.add_argument('--reward_model_path', type=str, default=None,
                       help='Path to learned reward model (gcl/iq_learn/maxent/unet). None=manual reward')
    parser.add_argument('--suffix', type=str, default='',
                       help='Suffix to add to experiment prefix (e.g., "_test")')
    parser.add_argument('--save_dir', type=str, default='experiment/ql',
                       help='Directory to save models (default: experiment/ql)')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory to save logs (default: logs)')
    parser.add_argument('--reward_combine_lambda', type=float, default=None,
                       help='Lambda for combining manual and IRL rewards: '
                            '(1-lambda)*manual + lambda*IRL. None=pure IRL reward.')
    parser.add_argument('--combined_or_train_data_path', type=str, default=None,
                       help='Path to training dataset. If eval_data_path is also provided, '
                            'all patients from this dataset are used for training. '
                            'If eval_data_path is None, this dataset is split into train/val/test.')
    parser.add_argument('--eval_data_path', type=str, default=None,
                       help='Path to evaluation dataset (for val/test). If provided, enables '
                            'dual-dataset mode where this dataset is split 50/50 into val/test.')
    parser.add_argument('--irl_vp2_bins', type=int, default=None,
                       help='Number of VP2 bins used to train the IRL model. If None, uses '
                            'the same value as --vp2_bins. This allows loading an IRL model '
                            'trained with different discretization than the Q-learning action space.')
    args = parser.parse_args()

    # Determine IRL model vp2_bins
    irl_vp2_bins = args.irl_vp2_bins if args.irl_vp2_bins is not None else args.vp2_bins

    print("="*70, flush=True)
    print(" LSTM BLOCK DISCRETE CQL TRAINING", flush=True)
    print("="*70, flush=True)
    print("\nConfiguration:", flush=True)
    print("  - VP1: Binary (0 or 1)", flush=True)
    print(f"  - VP2: Discretized into {args.vp2_bins} bins (0 to 0.5 mcg/kg/min)", flush=True)
    if args.irl_vp2_bins is not None and args.irl_vp2_bins != args.vp2_bins:
        print(f"  - IRL model VP2 bins: {irl_vp2_bins} (different from Q-learning)", flush=True)
    print(f"  - Alpha: {args.alpha}", flush=True)
    print(f"  - Epochs: {args.epochs}", flush=True)
    print(f"  - Sequence length: {args.sequence_length}", flush=True)
    print(f"  - Burn-in length: {args.burn_in_length}", flush=True)
    print(f"  - LSTM hidden: {args.lstm_hidden}", flush=True)
    print(f"  - Reward model: {args.reward_model_path or 'manual'}", flush=True)
    print(f"  - Reward combine lambda: {args.reward_combine_lambda}", flush=True)
    print(f"  - Suffix: {args.suffix}", flush=True)
    print(f"  - Total discrete actions: {2 * args.vp2_bins}", flush=True)
    if args.eval_data_path:
        print(f"  - Dataset mode: DUAL-DATASET", flush=True)
        print(f"    Train data: {args.combined_or_train_data_path or 'default'}", flush=True)
        print(f"    Eval data:  {args.eval_data_path}", flush=True)
    else:
        print(f"  - Dataset mode: SINGLE-DATASET", flush=True)
        print(f"    Data path: {args.combined_or_train_data_path or 'default'}", flush=True)

    # Train LSTM CQL
    agent, train_buffer, val_buffer, history, exp_prefix = train_lstm_block_discrete_cql(
        alpha=args.alpha,
        vp2_bins=args.vp2_bins,
        sequence_length=args.sequence_length,
        burn_in_length=args.burn_in_length,
        overlap=args.overlap,
        hidden_dim=args.hidden_dim,
        lstm_hidden=args.lstm_hidden,
        num_lstm_layers=args.num_lstm_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        tau=args.tau,
        gamma=args.gamma,
        grad_clip=args.grad_clip,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        reward_model_path=args.reward_model_path,
        suffix=args.suffix,
        reward_combine_lambda=args.reward_combine_lambda,
        combined_or_train_data_path=args.combined_or_train_data_path,
        eval_data_path=args.eval_data_path,
        irl_vp2_bins=irl_vp2_bins
    )

    print("\n" + "="*70, flush=True)
    print(" TRAINING COMPLETE", flush=True)
    print("="*70, flush=True)
    print(f"\nModels saved in {args.save_dir}/ directory:", flush=True)
    print(f"  {exp_prefix}_alpha{args.alpha:.4f}_bins{args.vp2_bins}_best.pt", flush=True)
    print(f"Check {args.log_dir}/ directory for detailed training logs", flush=True)


if __name__ == "__main__":
    main()