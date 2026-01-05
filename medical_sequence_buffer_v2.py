"""
Medical Sequence Buffer for R2D2-style Sequential Learning
==========================================================
Implements a sequence replay buffer specifically designed for medical RL,
handling patient episodes and generating overlapping sequences for LSTM training.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import deque
import sys
import os

# Add path for SumTree import
sys.path.append(os.path.join(os.path.dirname(__file__), '../angle/RL/model'))
try:
    from sum_tree import SumTree
except ImportError:
    # Fallback implementation if SumTree not available
    class SumTree:
        """Simple priority queue fallback"""
        def __init__(self, capacity):
            self.capacity = capacity
            self.data = []
            self.priorities = []
            self.n_entries = 0
            
        def add(self, priority, data):
            if len(self.data) >= self.capacity:
                self.data.pop(0)
                self.priorities.pop(0)
            self.data.append(data)
            self.priorities.append(priority)
            self.n_entries = len(self.data)
            
        def sample(self, batch_size):
            if len(self.data) < batch_size:
                raise ValueError(f"Not enough data: {len(self.data)} < {batch_size}")
            
            # Simple weighted sampling
            probs = np.array(self.priorities) / sum(self.priorities)
            indices = np.random.choice(len(self.data), batch_size, p=probs, replace=False)
            
            batch = [self.data[i] for i in indices]
            return batch, indices, [self.priorities[i] for i in indices]
            
        def update(self, idx, priority):
            if idx < len(self.priorities):
                self.priorities[idx] = priority
                
        def total(self):
            return sum(self.priorities)


class MedicalSequenceBufferV2:
    """
    Sequence replay buffer for medical RL with patient-aware sampling.
    
    Key features:
    - Respects patient episode boundaries
    - Generates overlapping sequences within episodes
    - Prioritizes clinically relevant sequences
    - Handles variable-length patient trajectories
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        sequence_length: int = 20,
        burn_in_length: int = 10,
        overlap: int = 10,
        priority_type: str = 'mortality_weighted',
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6
    ):
        """
        Initialize medical sequence buffer.
        
        Args:
            capacity: Maximum number of sequences to store
            sequence_length: Length of each sequence
            burn_in_length: Length of burn-in period for LSTM
            overlap: Overlap between consecutive sequences
            priority_type: How to calculate priorities ('mortality_weighted', 'td_error', 'uniform')
            alpha: Priority exponent for prioritized replay
            beta: Importance sampling exponent
            epsilon: Small constant for numerical stability
        """
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        self.overlap = overlap
        self.priority_type = priority_type
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        
        # Validate parameters
        assert burn_in_length < sequence_length, "Burn-in must be shorter than sequence"
        assert overlap > 0 and overlap <= sequence_length, "Invalid overlap"
        
        # Use SumTree for efficient prioritized sampling
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        
        # Patient episode tracking
        self.current_patient_buffer = []
        self.current_patient_id = None
        self.current_patient_metadata = {}
        
        # Statistics
        self.total_sequences_generated = 0
        self.total_patients_processed = 0
        
    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        patient_id: int,
        metadata: Optional[Dict] = None
    ):
        """
        Add single transition and generate sequences when appropriate.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
            patient_id: Unique patient identifier
            metadata: Optional metadata (e.g., timestep, clinical info)
        """
        # Check for patient boundary
        if patient_id != self.current_patient_id:
            self.current_patient_id = patient_id
            self.current_patient_buffer = []
            self.current_patient_metadata = metadata or {}
            self.total_patients_processed += 1
        
        # Create transition dictionary
        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'patient_id': patient_id
        }
        
        # Add metadata if provided
        if metadata:
            transition['metadata'] = metadata
        
        self.current_patient_buffer.append(transition)
        
        # Generate sequences if buffer is long enough
        if len(self.current_patient_buffer) >= self.sequence_length:
            buffer_len = len(self.current_patient_buffer)
            if (buffer_len - self.sequence_length) % self.overlap == 0:
                self._generate_sequence_at_index(buffer_len - self.sequence_length)
        
        # Clear state when episode ends to handle multiple episodes from same patient
        if done:
            self.current_patient_buffer = []
            self.current_patient_id = None
    
    def _generate_sequence_at_index(self, start_idx: int):
        """Generate a single sequence starting at the given index."""
        end_idx = start_idx + self.sequence_length
        sequence = self.current_patient_buffer[start_idx:end_idx]
        
        # Calculate priority based on clinical relevance
        priority = self._calculate_medical_priority(sequence)
        
        # Extract arrays for efficient storage
        states = np.array([t['state'] for t in sequence])
        actions = np.array([t['action'] for t in sequence])
        rewards = np.array([t['reward'] for t in sequence])
        next_states = np.array([t['next_state'] for t in sequence])
        dones = np.array([t['done'] for t in sequence])
        
        # Create sequence data structure
        sequence_data = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'patient_id': self.current_patient_id,
            'length': self.sequence_length,
            'burn_in_end': self.burn_in_length,
            'metadata': self.current_patient_metadata
        }
        
        # Add to replay buffer
        self.tree.add(priority, sequence_data)
        self.max_priority = max(self.max_priority, priority)
        self.total_sequences_generated += 1
    
    def _calculate_medical_priority(self, sequence: List[Dict]) -> float:
        """
        Calculate priority based on medical relevance.
        
        Higher priority for:
        - Sequences with mortality events
        - High variance in vital signs/rewards
        - Critical interventions
        - Sequences near episode boundaries
        """
        if self.priority_type == 'uniform':
            return 1.0
        
        priority = self.epsilon
        
        # Check for mortality (high negative reward at end)
        final_transition = sequence[-1]
        has_mortality = final_transition['done'] and final_transition['reward'] < -5
        
        if self.priority_type == 'mortality_weighted':
            if has_mortality:
                priority += 10.0  # High priority for mortality sequences
            
            # Reward variance indicates critical moments
            rewards = [t['reward'] for t in sequence]
            reward_variance = np.var(rewards)
            priority += reward_variance
            
            # Check for critical interventions (high actions)
            actions = [t['action'] for t in sequence]
            if len(actions) > 0 and len(actions[0]) > 0:
                max_action = np.max([np.max(np.abs(a)) for a in actions])
                if max_action > 0.5:  # Significant intervention
                    priority += 2.0
        
        elif self.priority_type == 'td_error':
            # Will be updated during training
            priority = self.max_priority
        
        return priority ** self.alpha
    
    def sample_sequences(
        self,
        batch_size: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Sample batch of sequences for training.
        
        Args:
            batch_size: Number of sequences to sample
            
        Returns:
            burn_in_batch: Dictionary of burn-in sequences
            training_batch: Dictionary of training sequences
            indices: Tree indices for priority updates
            weights: Importance sampling weights
        """
        if self.tree.n_entries < batch_size:
            raise ValueError(f"Not enough sequences: {self.tree.n_entries} < {batch_size}")
        
        # Sample sequences from tree
        sequences, tree_indices, priorities = self.tree.sample(batch_size)
        
        # Calculate importance sampling weights
        total = self.tree.total()
        weights = []
        
        for priority in priorities:
            if priority > 0:
                prob = priority / total
                weight = (self.tree.n_entries * prob) ** (-self.beta)
                weights.append(weight)
            else:
                weights.append(1.0)
        
        # Normalize weights
        max_weight = max(weights) if weights else 1.0
        weights = np.array([w / max_weight for w in weights])
        
        # Prepare burn-in and training batches
        burn_in_batch = {
            'states': [],
            'actions': [],
            'patient_ids': []
        }
        
        training_batch = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'patient_ids': []
        }
        
        for seq_data in sequences:
            burn_in_end = seq_data['burn_in_end']
            
            # Burn-in portion
            burn_in_batch['states'].append(seq_data['states'][:burn_in_end])
            burn_in_batch['actions'].append(seq_data['actions'][:burn_in_end])
            burn_in_batch['patient_ids'].append(seq_data['patient_id'])
            
            # Training portion
            training_batch['states'].append(seq_data['states'][burn_in_end:])
            training_batch['actions'].append(seq_data['actions'][burn_in_end:])
            training_batch['rewards'].append(seq_data['rewards'][burn_in_end:])
            training_batch['next_states'].append(seq_data['next_states'][burn_in_end:])
            training_batch['dones'].append(seq_data['dones'][burn_in_end:])
            training_batch['patient_ids'].append(seq_data['patient_id'])
        
        # Convert to numpy arrays
        for key in burn_in_batch:
            if key != 'patient_ids':
                burn_in_batch[key] = np.array(burn_in_batch[key])
        
        for key in training_batch:
            if key != 'patient_ids':
                training_batch[key] = np.array(training_batch[key])
        
        return burn_in_batch, training_batch, tree_indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for sequences based on TD errors.
        
        Args:
            indices: Tree indices of sequences
            priorities: New priority values (e.g., TD errors)
        """
        for idx, priority in zip(indices, priorities):
            # Apply prioritization exponent
            priority = (abs(priority) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def update_beta(self, new_beta: float):
        """Update importance sampling correction factor."""
        self.beta = new_beta
    
    def __len__(self):
        """Return number of sequences in buffer."""
        return self.tree.n_entries
    
    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics."""
        return {
            'num_sequences': self.tree.n_entries,
            'capacity': self.capacity,
            'utilization': self.tree.n_entries / self.capacity,
            'total_sequences_generated': self.total_sequences_generated,
            'total_patients_processed': self.total_patients_processed,
            'max_priority': self.max_priority,
            'current_patient_buffer_size': len(self.current_patient_buffer)
        }
    
    def clear(self):
        """Clear the buffer."""
        self.tree = SumTree(self.capacity)
        self.current_patient_buffer = []
        self.current_patient_id = None
        self.total_sequences_generated = 0
        self.total_patients_processed = 0
        self.max_priority = 1.0


class SequenceDataLoader:
    """
    Helper class to convert sequences into training batches for PyTorch.
    """
    
    @staticmethod
    def prepare_torch_batch(
        burn_in_batch: Dict[str, np.ndarray],
        training_batch: Dict[str, np.ndarray],
        weights: np.ndarray,
        device: str = 'cuda'
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """
        Convert numpy batches to PyTorch tensors.
        
        Args:
            burn_in_batch: Burn-in sequence data
            training_batch: Training sequence data
            weights: Importance sampling weights
            device: Device to place tensors on
            
        Returns:
            Torch tensors for burn-in, training, and weights
        """
        # Convert burn-in batch
        burn_in_torch = {}
        for key, value in burn_in_batch.items():
            if key != 'patient_ids':
                burn_in_torch[key] = torch.FloatTensor(value).to(device)
            else:
                burn_in_torch[key] = value
        
        # Convert training batch
        training_torch = {}
        for key, value in training_batch.items():
            if key != 'patient_ids':
                training_torch[key] = torch.FloatTensor(value).to(device)
            else:
                training_torch[key] = value
        
        # Convert weights
        weights_torch = torch.FloatTensor(weights).to(device)
        
        return burn_in_torch, training_torch, weights_torch
    
    @staticmethod
    def compute_sequence_statistics(
        sequences: List[Dict]
    ) -> Dict[str, float]:
        """
        Compute statistics over a batch of sequences.
        
        Args:
            sequences: List of sequence dictionaries
            
        Returns:
            Dictionary of statistics
        """
        all_rewards = []
        all_lengths = []
        mortality_count = 0
        
        for seq in sequences:
            all_rewards.extend(seq['rewards'].flatten())
            all_lengths.append(seq['length'])
            
            # Check for mortality
            if seq['dones'][-1] and seq['rewards'][-1] < -5:
                mortality_count += 1
        
        return {
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'mean_length': np.mean(all_lengths),
            'mortality_rate': mortality_count / len(sequences)
        }


# Test function
if __name__ == "__main__":
    print("Testing Medical Sequence Buffer V2...")

    # Create buffer
    buffer = MedicalSequenceBufferV2(
        capacity=1000,
        sequence_length=10,
        burn_in_length=5,
        overlap=5
    )
    
    # Simulate patient trajectory
    print("\nAdding patient trajectory...")
    patient_id = 1
    for t in range(25):
        state = np.random.randn(10)
        action = np.array([np.random.random()])
        reward = -0.1 if t < 20 else -10.0  # Terminal negative reward
        next_state = np.random.randn(10)
        done = (t == 24)
        
        buffer.add_transition(
            state, action, reward, next_state, done, patient_id
        )
    
    # Check statistics
    stats = buffer.get_statistics()
    print(f"\nBuffer statistics: {stats}")
    
    # Sample if we have enough sequences
    if len(buffer) >= 2:
        print("\nSampling sequences...")
        burn_in, training, indices, weights = buffer.sample_sequences(2)
        print(f"Burn-in shape: {burn_in['states'].shape}")
        print(f"Training shape: {training['states'].shape}")
        print(f"Weights: {weights}")
    
    print("\nTest completed successfully!")