"""
Integrated data pipeline V2 with reward computation and patient-aware batching
Produces (s, a, r, s', done, patient_id) tuples for CQL training
"""

import numpy as np
import pdb
import torch
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler

import data_config as config
from data_loader import DataLoader
from data_loader import DataSplitter


def compute_outcome_reward(
    state: np.ndarray,
    next_state: np.ndarray,
    action: np.ndarray,
    is_terminal: bool,
    mortality: int,
    state_features: list
) -> float:
    """
    Compute reward based on clinical outcomes
    
    Args:
        state: Current state
        next_state: Next state
        action: Action taken (VP1, VP2 for dual; VP1 only for binary)
        is_terminal: Whether this is the last timestep
        mortality: Whether patient died (1) or survived (0)
        state_features: List of feature names for indexing
    
    Returns:
        Reward value
    """
    reward = 0.0
    
    # Get indices for key features
    mbp_idx = state_features.index('mbp') if 'mbp' in state_features else None
    lactate_idx = state_features.index('lactate') if 'lactate' in state_features else None

    """ 
    if mbp_idx is not None:
        mbp = state[mbp_idx]
        next_mbp = next_state[mbp_idx]
        
        # Blood pressure reward
        if next_mbp < 65:
            reward -= 1.0  # Too low
    
    if lactate_idx is not None:
        lactate = state[lactate_idx]
        next_lactate = next_state[lactate_idx]
        
        # Lactate improvement reward
        if next_lactate > 0.4:
            reward -= 0.5  # Lactate to high
    """

    # Minimize vasopressor use (small penalty for high doses)
    if len(action) == 2:  # Dual continuous
        vp1_dose = action[0]
        vp2_dose = action[1]
        total_vaso = vp1_dose + vp2_dose * 2
    else:  # Binary
        vp1_dose = action if np.isscalar(action) else action[0]
        total_vaso = vp1_dose
    
    if total_vaso > 1.0:
        reward -= 0.1 * (total_vaso - 1.0)
    
    # Terminal rewards based on mortality
    if is_terminal:
        if mortality == 0:
            reward += 10.0  # Survived
        else:
            reward -= 10.0  # Died
    
    return reward


class IntegratedDataPipelineV2:
    """
    Complete data pipeline for CQL training with rewards and patient-aware batching
    """
    
    def __init__(self, model_type: str = 'binary', random_seed: int = config.RANDOM_SEED):
        """
        Initialize integrated pipeline
        
        Args:
            model_type: 'binary' or 'dual' for different CQL models
            random_seed: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_seed = random_seed
        
        # Initialize components
        self.loader = DataLoader(verbose=False, random_seed=random_seed)
        self.splitter = DataSplitter(random_seed=random_seed)
        
        # Scaler for state normalization
        self.scaler = StandardScaler()
        
        # Store processed data as (s, a, r, s', done, patient_id) tuples
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        # Store patient groupings for efficient patient-aware batching
        self.train_patient_groups = {}
        self.val_patient_groups = {}
        self.test_patient_groups = {}
        
        # Random number generators
        self.rng = np.random.RandomState(random_seed)
        
    def prepare_data(self):
        """
        Complete data preparation pipeline producing (s, a, r, s', done) tuples
        """
        print(f"Preparing {self.model_type.upper()} CQL data pipeline V2...")
        print("="*60)
        
        # Load and split data
        print("1. Loading and splitting data...")
        full_data = self.loader.load_data()
        patient_ids = self.loader.get_patient_ids()
        train_patients, val_patients, test_patients = self.splitter.split_patients(patient_ids)
        
        print(f"   Train: {len(train_patients)} patients")
        print(f"   Val:   {len(val_patients)} patients")
        print(f"   Test:  {len(test_patients)} patients")
        
        # Encode categorical features
        print("2. Encoding categorical features...")
        self.loader.encode_categorical_features()
        
        # Get feature lists based on model type
        if self.model_type == 'binary':
            state_features = config.BINARY_STATE_FEATURES
            action_col = config.BINARY_ACTION
        else:
            state_features = config.DUAL_STATE_FEATURES
            action_cols = config.DUAL_ACTIONS
        
        # Process each split separately
        print("3. Processing transitions and computing rewards...")
        
        self.train_data = self._process_split(train_patients, state_features, 'train')
        self.val_data = self._process_split(val_patients, state_features, 'val')
        self.test_data = self._process_split(test_patients, state_features, 'test')
        
        print("\n Data pipeline complete!")
        self._print_summary()
        
        return self.train_data, self.val_data, self.test_data
    
    def _process_split(self, patient_list: np.ndarray, state_features: list, split_name: str) -> Dict:
        """
        Process a data split to create (s, a, r, s', done) tuples
        """
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []
        all_patient_ids = []
        
        # Group transitions by patient for patient-aware batching
        patient_groups = {}
        current_idx = 0
        
        for patient_id in patient_list:
            patient_data = self.loader.get_patient_data(patient_id)
            
            if len(patient_data) < 2:  # Need at least 2 timesteps
                continue
            
            # Extract states
            states = patient_data[state_features].values
            
            # Extract actions based on model type
            if self.model_type == 'binary':
                actions = patient_data[config.BINARY_ACTION].values.astype(float)
                # Add norepinephrine as part of state (already in state_features)
            else:
                vp1 = patient_data['action_vaso'].values.astype(float)
                vp2 = patient_data['norepinephrine'].values.astype(float)
                actions = np.column_stack([vp1, vp2])
            
            # Get patient mortality outcome
            mortality = int(patient_data[config.DEATH_COL].iloc[-1])
            
            # Store starting index for this patient
            patient_start_idx = current_idx
            
            # Create transitions
            for t in range(len(states) - 1):
                all_states.append(states[t])
                all_next_states.append(states[t + 1])
                
                if self.model_type == 'binary':
                    all_actions.append(actions[t])
                else:
                    all_actions.append(actions[t])
                
                # Check if terminal
                is_terminal = (t == len(states) - 2)
                all_dones.append(1.0 if is_terminal else 0.0)
                
                # Compute reward (states not normalized yet)
                action_for_reward = actions[t] if self.model_type == 'dual' else np.array([actions[t]])
                reward = compute_outcome_reward(
                    states[t], states[t + 1], action_for_reward,
                    is_terminal, mortality, state_features
                )
                all_rewards.append(reward)
                
                all_patient_ids.append(patient_id)
                current_idx += 1
            
            # Store patient group info
            patient_groups[patient_id] = (patient_start_idx, current_idx)
        
        # Convert to arrays
        all_states = np.array(all_states, dtype=np.float32)
        all_next_states = np.array(all_next_states, dtype=np.float32)
        all_actions = np.array(all_actions, dtype=np.float32)
        all_rewards = np.array(all_rewards, dtype=np.float32)
        all_dones = np.array(all_dones, dtype=np.float32)
        all_patient_ids = np.array(all_patient_ids)
        
        # Normalize states (fit scaler on train only)
        if split_name == 'train':
            all_states_norm = self.scaler.fit_transform(all_states)
        else:
            all_states_norm = self.scaler.transform(all_states)
        all_next_states_norm = self.scaler.transform(all_next_states)
        
        # Store patient groups
        if split_name == 'train':
            self.train_patient_groups = patient_groups
        elif split_name == 'val':
            self.val_patient_groups = patient_groups
        else:
            self.test_patient_groups = patient_groups
        
        print(f"   {split_name}: {len(all_states)} transitions from {len(patient_groups)} patients")
        print(f"   Reward range: [{all_rewards.min():.2f}, {all_rewards.max():.2f}]")
        
        return {
            'states': all_states_norm,
            'actions': all_actions,
            'rewards': all_rewards,
            'next_states': all_next_states_norm,
            'dones': all_dones,
            'patient_ids': all_patient_ids,
            'n_transitions': len(all_states),
            'n_patients': len(patient_groups),
            'patient_groups': patient_groups,
            'state_features': state_features
        }
    
    def get_batch(self, batch_size: int = 256, split: str = 'train', 
                  same_patient: bool = False, seed: Optional[int] = None) -> Dict:
        """
        Get a batch of transitions
        
        Args:
            batch_size: Size of the batch
            split: 'train', 'val', or 'test'
            same_patient: If True, try to sample transitions from same patients
            seed: Optional seed for reproducibility
            
        Returns:
            Dictionary with batch data
        """
        if split == 'train':
            data = self.train_data
            patient_groups = self.train_patient_groups
        elif split == 'val':
            data = self.val_data
            patient_groups = self.val_patient_groups
        elif split == 'test':
            data = self.test_data
            patient_groups = self.test_patient_groups
        else:
            raise ValueError(f"Invalid split: {split}")
        
        if data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        # Use provided seed or internal RNG
        rng = np.random.RandomState(seed) if seed is not None else self.rng
        
        if same_patient and patient_groups:
            # Sample transitions preferring same patients
            indices = self._sample_patient_aware(patient_groups, batch_size, rng)
        else:
            # Random sampling
            n_transitions = data['n_transitions']
            indices = rng.choice(n_transitions, size=min(batch_size, n_transitions), replace=False)
        
        return {
            'states': data['states'][indices],
            'actions': data['actions'][indices],
            'rewards': data['rewards'][indices],
            'next_states': data['next_states'][indices],
            'dones': data['dones'][indices],
            'patient_ids': data['patient_ids'][indices]
        }
    
    def _sample_patient_aware(self, patient_groups: Dict, batch_size: int, rng) -> np.ndarray:
        """
        Sample transitions preferring to keep same patients together
        
        This is a simplified version that samples some patients and takes all their transitions
        """
        indices = []
        remaining = batch_size
        
        # Get list of patients
        patient_list = list(patient_groups.keys())
        rng.shuffle(patient_list)
        
        for patient_id in patient_list:
            if remaining <= 0:
                break
            
            start_idx, end_idx = patient_groups[patient_id]
            patient_transitions = list(range(start_idx, end_idx))
            
            if len(patient_transitions) <= remaining:
                # Take all transitions from this patient
                indices.extend(patient_transitions)
                remaining -= len(patient_transitions)
            else:
                # Take only what we need
                indices.extend(patient_transitions[:remaining])
                remaining = 0
        
        return np.array(indices)
    
    def get_stepwise_batch(self, batch_size: int = 256, split: str = 'train',
                          vp2_bins: int = 10, seed: Optional[int] = None) -> Dict:
        """
        Get a batch of transitions with stepwise action information
        
        This function returns transitions with VP2 dose change information needed
        for stepwise CQL training. For each transition, it includes:
        - Current VP2 dose (continuous and discretized)
        - Next VP2 dose (continuous and discretized) 
        - VP2 change (discrete action representing the change)
        
        Args:
            batch_size: Size of the batch
            split: 'train', 'val', or 'test'
            vp2_bins: Number of bins for discretizing VP2 doses (default 10)
            seed: Optional seed for reproducibility
            
        Returns:
            Dictionary with batch data including stepwise action information
        """
        if split == 'train':
            data = self.train_data
            patient_groups = self.train_patient_groups
        elif split == 'val':
            data = self.val_data
            patient_groups = self.val_patient_groups
        elif split == 'test':
            data = self.test_data
            patient_groups = self.test_patient_groups
        else:
            raise ValueError(f"Invalid split: {split}")
        
        if data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        if self.model_type != 'dual':
            raise ValueError("get_stepwise_batch requires model_type='dual'")
        
        # Use provided seed or internal RNG
        rng = np.random.RandomState(seed) if seed is not None else self.rng
        
        # Sample indices (exclude last transition of each patient as it has no next action)
        valid_indices = []
        for patient_id, (start_idx, end_idx) in patient_groups.items():
            # Exclude the last transition of each patient
            if end_idx - start_idx > 1:
                valid_indices.extend(range(start_idx, end_idx - 1))
        
        if len(valid_indices) < batch_size:
            # Sample with replacement if not enough valid transitions
            sampled_indices = rng.choice(valid_indices, size=batch_size, replace=True)
        else:
            sampled_indices = rng.choice(valid_indices, size=batch_size, replace=False)
        
        # Get base batch data
        states = data['states'][sampled_indices]
        actions = data['actions'][sampled_indices]  # [vp1, vp2_current]
        rewards = data['rewards'][sampled_indices]
        next_states = data['next_states'][sampled_indices]
        dones = data['dones'][sampled_indices]
        patient_ids = data['patient_ids'][sampled_indices]
        
        # Assert that actions are 2D (dual actions required for stepwise)
        assert actions.ndim > 1, "Stepwise batch requires dual actions (VP1 and VP2)"
        
        # Extract VP2 doses (current and next)
        vp1_actions = actions[:, 0]
        vp2_current = actions[:, 1]
        
        # Get next VP2 doses (from next transition's action)
        next_indices = sampled_indices + 1
        next_actions = data['actions'][next_indices]
        assert next_actions.ndim > 1, "Next actions must also be dual actions"
        vp2_next = next_actions[:, 1]
        
        # Clip VP2 values to [0, 0.5] range for stepwise CQL
        # Most therapeutic doses should be in this range
        vp2_current = np.clip(vp2_current, 0, 0.5)
        vp2_next = np.clip(vp2_next, 0, 0.5)
        
        # Discretize VP2 doses to bins
        vp2_bin_edges = np.linspace(0, 0.5, vp2_bins + 1)
        vp2_current_bins = np.digitize(vp2_current, vp2_bin_edges) - 1
        vp2_current_bins = np.clip(vp2_current_bins, 0, vp2_bins - 1)
        
        vp2_next_bins = np.digitize(vp2_next, vp2_bin_edges) - 1
        vp2_next_bins = np.clip(vp2_next_bins, 0, vp2_bins - 1)
        
        # Compute VP2 change as discrete action
        # Map changes to discrete actions (e.g., -2, -1, 0, +1, +2 bins)
        vp2_changes = vp2_next_bins - vp2_current_bins
        
        return {
            'states': states,
            'actions': actions,  # Original continuous actions [vp1, vp2]
            'vp1_actions': vp1_actions,  # Binary VP1 actions
            'vp2_current': vp2_current,  # Current VP2 dose (continuous)
            'vp2_next': vp2_next,  # Next VP2 dose (continuous)
            'vp2_current_bins': vp2_current_bins,  # Current VP2 bin index
            'vp2_next_bins': vp2_next_bins,  # Next VP2 bin index
            'vp2_changes': vp2_changes,  # VP2 change in bins (-2, -1, 0, +1, +2, etc.)
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'patient_ids': patient_ids
        }
    
    def _print_summary(self):
        """Print summary of prepared data"""
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        
        for name, data in [('Train', self.train_data), 
                           ('Val', self.val_data), 
                           ('Test', self.test_data)]:
            if data is None:
                continue
                
            print(f"\n{name} Set:")
            print(f"  Transitions:    {data['n_transitions']}")
            print(f"  Patients:       {data['n_patients']}")
            print(f"  Avg traj len:   {data['n_transitions']/data['n_patients']:.1f}")
            print(f"  States shape:   {data['states'].shape}")
            print(f"  Actions shape:  {data['actions'].shape}")
            
            # Calculate mortality rate
            unique_patients = np.unique(data['patient_ids'])
            died_count = 0
            for pid in unique_patients:
                # Check if last transition for this patient has done=1 and negative reward
                patient_mask = data['patient_ids'] == pid
                patient_dones = data['dones'][patient_mask]
                patient_rewards = data['rewards'][patient_mask]
                if patient_dones[-1] == 1.0 and patient_rewards[-1] < 0:
                    died_count += 1
            
            mortality_rate = died_count / len(unique_patients)
            print(f"  Mortality:      {mortality_rate*100:.1f}%")
    
    def reset_batch_rng(self):
        """Reset the internal random number generator"""
        self.rng = np.random.RandomState(self.random_seed)


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("="*70)
    print(" INTEGRATED DATA PIPELINE V2 - WITH REWARDS")
    print("="*70)
    
    # Test Binary CQL pipeline
    print("\n--- BINARY CQL PIPELINE ---\n")
    binary_pipeline = IntegratedDataPipelineV2(model_type='binary', random_seed=42)
    train_data, val_data, test_data = binary_pipeline.prepare_data()
    
    # Test standard random batch
    print("\n" + "="*70)
    print(" TESTING BATCH SAMPLING")
    print("="*70)
    
    print("\n1. Random batch sampling:")
    batch = binary_pipeline.get_batch(batch_size=32, split='train', seed=123)
    print(f"   States shape: {batch['states'].shape}")
    print(f"   Actions shape: {batch['actions'].shape}")
    print(f"   Rewards shape: {batch['rewards'].shape}")
    print(f"   Sample reward: {batch['rewards'][0]:.3f}")
    print(f"   Sample done: {batch['dones'][0]}")
    print(f"   Unique patients in batch: {len(np.unique(batch['patient_ids']))}")
    
    # Test patient-aware batch
    print("\n2. Patient-aware batch sampling:")
    batch_patient = binary_pipeline.get_batch(batch_size=32, split='train', 
                                              same_patient=True, seed=123)
    print(f"   Unique patients in batch: {len(np.unique(batch_patient['patient_ids']))}")
    print(f"   Patient IDs: {np.unique(batch_patient['patient_ids'])[:5]}...")
    
    # Verify reproducibility
    batch2 = binary_pipeline.get_batch(batch_size=32, split='train', seed=123)
    match = np.array_equal(batch['states'], batch2['states'])
    print(f"\n3. Reproducibility check: {match}")
    
    # Test Dual CQL pipeline
    print("\n" + "="*70)
    print("\n--- DUAL CONTINUOUS CQL PIPELINE ---\n")
    dual_pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
    train_data, val_data, test_data = dual_pipeline.prepare_data()
    
    # Get a batch
    batch = dual_pipeline.get_batch(batch_size=32, split='train', seed=456)
    print(f"\nDual CQL batch:")
    print(f"   States shape: {batch['states'].shape}")
    print(f"   Actions shape: {batch['actions'].shape}")
    print(f"   Sample action (VP1, VP2): {batch['actions'][0]}")
    print(f"   Sample reward: {batch['rewards'][0]:.3f}")
    
    # Test patient-aware sampling effectiveness
    print("\n" + "="*70)
    print(" COMPARING SAMPLING STRATEGIES")
    print("="*70)
    
    # Random sampling
    batch_random = dual_pipeline.get_batch(batch_size=100, split='train', 
                                           same_patient=False, seed=789)
    unique_random = len(np.unique(batch_random['patient_ids']))
    
    # Patient-aware sampling
    batch_patient = dual_pipeline.get_batch(batch_size=100, split='train', 
                                            same_patient=True, seed=789)
    unique_patient = len(np.unique(batch_patient['patient_ids']))
    
    print(f"\nFor batch size 100:")
    print(f"  Random sampling: {unique_random} unique patients")
    print(f"  Patient-aware:   {unique_patient} unique patients")
    print(f"  Reduction:       {(1 - unique_patient/unique_random)*100:.1f}%")
    
    if unique_patient < unique_random:
        print("\n Patient-aware sampling successfully groups transitions!")