#!/usr/bin/env python3
"""
Unified training script for Binary and Dual CQL with alpha=0.0
Uses consistent hyperparameters across both models:
- tau = 0.8
- lr = 1e-3 
- batch_size = 128
- alpha = 0.0 (no CQL penalty - pure Q-learning)
- epochs = 100
"""

import numpy as np
import torch
import time
import os
import sys

# Import our unified pipeline and models
from integrated_data_pipeline_v2 import IntegratedDataPipelineV2
from train_binary_cql import BinaryCQL

# Force unbuffered output
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

def train_binary_cql():
    """Train Binary CQL with alpha=0.0"""
    print("="*70, flush=True)
    print(" BINARY CQL TRAINING WITH ALPHA=0.0", flush=True)
    print("="*70, flush=True)
    
    # Initialize data pipeline
    print("\nInitializing Binary CQL data pipeline...", flush=True)
    pipeline = IntegratedDataPipelineV2(model_type='binary', random_seed=42)
    train_data, val_data, test_data = pipeline.prepare_data()
    
    # Get state dimension
    state_dim = train_data['states'].shape[1]
    
    # Print settings
    print("\n" + "="*70, flush=True)
    print("SETTINGS:", flush=True)
    print(f"  State dimension: {state_dim}", flush=True)
    print(f"  Action dimension: 1 (binary: 0 or 1)", flush=True)
    print("  ALPHA = 0.0 (no CQL penalty - pure Q-learning)", flush=True)
    print("  TAU = 0.8 (target network update)", flush=True)
    print("  LR = 0.001 (learning rate)", flush=True)
    print("  BATCH_SIZE = 128", flush=True)
    print("  EPOCHS = 100", flush=True)
    print("="*70, flush=True)
    
    # Initialize agent with specified parameters
    agent = BinaryCQL(
        state_dim=state_dim,
        alpha=0.0,
        gamma=0.95,
        tau=0.8,      # As specified
        lr=1e-3,      # As specified  
        grad_clip=1.0
    )
    
    # Training loop
    epochs = 500
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
            agent.save('experiment/binary_cql_unified_alpha00_best.pt')
            print(f"Model saved to experiment/binary_cql_unified_alpha00_best.pt", flush=True)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}: "
                  f"Q1 Loss={train_metrics['q1_loss']:.4f}, "
                  f"CQL1 Loss={train_metrics['cql1_loss']:.4f}, "
                  f"Val Q-value={-val_loss:.4f}, "
                  f"Time={elapsed/60:.1f}min", flush=True)
    
    # Save final model
    agent.save('experiment/binary_cql_unified_alpha00_final.pt')
    
    total_time = time.time() - start_time
    print(f"\n✅ Binary CQL training completed in {total_time/60:.1f} minutes!", flush=True)
    print("Models saved:", flush=True)
    print("  - experiment/binary_cql_unified_alpha00_best.pt", flush=True)
    print("  - experiment/binary_cql_unified_alpha00_final.pt", flush=True)
    
    return agent, pipeline


def main():
    """Train Binary CQL with alpha=0.0"""
    print("="*70, flush=True)
    print(" UNIFIED CQL TRAINING - ALPHA=0.0", flush=True)
    print("="*70, flush=True)
    print("\nThis script trains both Binary and Dual CQL models with:", flush=True)
    print("  - Consistent hyperparameters", flush=True)
    print("  - Same Q(s,a) -> R architecture", flush=True)
    print("  - Comparable Q-values", flush=True)
    print("  - NO CQL penalty (pure Q-learning)", flush=True)
    
    # Train Binary CQL
    binary_agent, binary_pipeline = train_binary_cql()
        
    print("\n" + "="*70, flush=True)
    print(" ALL TRAINING COMPLETE", flush=True)
    print("="*70, flush=True)
    print("\nModels saved in experiment/ directory:", flush=True)
    print("  Binary CQL: binary_cql_unified_alpha00_*.pt", flush=True)


if __name__ == "__main__":
    main()