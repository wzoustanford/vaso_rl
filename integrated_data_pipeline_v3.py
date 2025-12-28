"""
Integrated data pipeline V3 with LEARNED reward functions
Loads reward models from MaxEnt IRL, GCL, or IQ-Learn to replace manual reward design.

Produces (s, a, r, s', done, patient_id) tuples for CQL training
where r comes from a learned reward/cost function.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler

import data_config as config
from data_loader import DataLoader
from data_loader import DataSplitter
from maxent_irl_full_recovery import RewardNetwork


class IntegratedDataPipelineV3:
    """
    Data pipeline for CQL training using LEARNED rewards from IRL methods.

    Supports loading rewards from:
    - GCL: reward = -cost_f(s, a)
    - IQ-Learn: reward = Q(s,a) - gamma*V(s')
    - MaxEnt IRL: reward = reward_network(s, a)
    """

    def __init__(
        self,
        model_type: str = 'dual',
        reward_source: str = 'learned',  # 'learned' or 'manual'
        random_seed: int = config.RANDOM_SEED
    ):
        """
        Initialize integrated pipeline with learned rewards.

        Args:
            model_type: 'binary' or 'dual' for different CQL models
            reward_source: 'learned' (from IRL) or 'manual' (original compute_outcome_reward)
            random_seed: Random seed for reproducibility
        """
        self.model_type = model_type
        self.reward_source = reward_source
        self.random_seed = random_seed

        # Initialize components
        self.loader = DataLoader(verbose=False, random_seed=random_seed)
        self.splitter = DataSplitter(random_seed=random_seed)

        # Scaler for state normalization
        self.scaler = StandardScaler()

        # Learned reward model (loaded separately)
        self.reward_model = None
        self.reward_model_type = None  # 'gcl', 'iq_learn', or 'maxent'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def load_gcl_reward_model(self, checkpoint_path: str):
        """
        Load GCL model for reward computation.
        Reward = -cost_f(s, a)

        Args:
            checkpoint_path: Path to GCL checkpoint (.pt file)
        """
        from run_gcl_learn_block_discrete import GCLBlockDiscrete

        self.reward_model = GCLBlockDiscrete.load(checkpoint_path, device=self.device)
        self.reward_model_type = 'gcl'
        self.reward_model.cost_f.eval()
        self.reward_model.q1.eval()
        self.reward_model.q2.eval()

        print(f"Loaded GCL reward model from: {checkpoint_path}")
        print(f"  Reward = -cost_f(s, a)")

    def load_iq_learn_reward_model(self, checkpoint_path: str):
        """
        Load IQ-Learn model for reward computation.
        Reward = Q(s,a) - gamma*V(s')

        Args:
            checkpoint_path: Path to IQ-Learn checkpoint (.pt file)
        """
        from run_iq_learn_block_discrete import IQLearnBlockDiscrete

        self.reward_model = IQLearnBlockDiscrete.load(checkpoint_path, device=self.device)
        self.reward_model_type = 'iq_learn'
        self.reward_model.q1.eval()
        self.reward_model.q2.eval()

        print(f"Loaded IQ-Learn reward model from: {checkpoint_path}")
        print(f"  Reward = Q(s,a) - gamma*V(s')")

    def load_maxent_reward_model(self, checkpoint_path: str):
        """
        Load MaxEnt IRL model for reward computation.
        Reward = reward_network(s, a)

        Args:
            checkpoint_path: Path to MaxEnt IRL checkpoint (.pt file)
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        state_dim = checkpoint['state_dim']
        action_dim = checkpoint['action_dim']

        # Create reward network with same architecture
        self.reward_model = RewardNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[128, 64, 32]
        ).to(self.device)

        self.reward_model.load_state_dict(checkpoint['reward_network_state_dict'])
        self.reward_model_type = 'maxent'
        self.reward_model.eval()

        print(f"Loaded MaxEnt IRL reward model from: {checkpoint_path}")
        print(f"  Reward = R(s, a)")

    def _compute_learned_reward(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> np.ndarray:
        """
        Compute rewards using the loaded reward model.

        Args:
            states: [batch, state_dim] normalized states
            actions: [batch, action_dim] actions
            next_states: [batch, state_dim] normalized next states
            dones: [batch] terminal flags

        Returns:
            rewards: [batch] learned rewards
        """
        if self.reward_model is None:
            raise ValueError("No reward model loaded. Call load_*_reward_model() first.")

        with torch.no_grad():
            states_t = torch.FloatTensor(states).to(self.device)
            actions_t = torch.FloatTensor(actions).to(self.device)
            next_states_t = torch.FloatTensor(next_states).to(self.device)
            dones_t = torch.FloatTensor(dones).to(self.device)

            if self.reward_model_type == 'gcl':
                # GCL: reward = -cost
                action_indices = self.reward_model.continuous_to_discrete_batch(actions_t)
                costs = self.reward_model.cost_f(states_t, action_indices)
                rewards = -costs.squeeze().cpu().numpy()

            elif self.reward_model_type == 'iq_learn':
                # IQ-Learn: reward = Q(s,a) - gamma*V(s')
                rewards = self.reward_model.get_recovered_reward(
                    states_t, actions_t, next_states_t, dones_t
                ).cpu().numpy()

            elif self.reward_model_type == 'maxent':
                # MaxEnt IRL: reward = R(s,a)
                rewards = self.reward_model(states_t, actions_t).cpu().numpy()

            else:
                raise ValueError(f"Unknown reward model type: {self.reward_model_type}")

        return rewards

    def prepare_data(self):
        """
        Complete data preparation pipeline producing (s, a, r, s', done) tuples.
        Uses learned rewards if reward_source='learned' and a model is loaded.
        """
        print(f"Preparing {self.model_type.upper()} CQL data pipeline V3...")
        print(f"Reward source: {self.reward_source}")
        if self.reward_model_type:
            print(f"Reward model: {self.reward_model_type}")
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
        else:
            state_features = config.DUAL_STATE_FEATURES

        # Process each split separately
        print("3. Processing transitions...")

        self.train_data = self._process_split(train_patients, state_features, 'train')
        self.val_data = self._process_split(val_patients, state_features, 'val')
        self.test_data = self._process_split(test_patients, state_features, 'test')

        # If using learned rewards, recompute rewards now that data is normalized
        if self.reward_source == 'learned' and self.reward_model is not None:
            print("4. Computing learned rewards...")
            self._recompute_learned_rewards()

        print("\n Data pipeline complete!")
        self._print_summary()

        return self.train_data, self.val_data, self.test_data

    def _process_split(self, patient_list: np.ndarray, state_features: list, split_name: str) -> Dict:
        """
        Process a data split to create (s, a, r, s', done) tuples.
        Initial rewards are set to 0 if using learned rewards (recomputed later).
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
            else:
                vp1 = patient_data['action_vaso'].values.astype(float)
                vp2 = patient_data['norepinephrine'].values.astype(float)
                actions = np.column_stack([vp1, vp2])

            # Get patient mortality outcome (for manual reward if needed)
            mortality = int(patient_data[config.DEATH_COL].iloc[-1])

            # Store starting index for this patient
            patient_start_idx = current_idx

            # Create transitions
            for t in range(len(states) - 1):
                all_states.append(states[t])
                all_next_states.append(states[t + 1])
                all_actions.append(actions[t])

                # Check if terminal
                is_terminal = (t == len(states) - 2)
                all_dones.append(1.0 if is_terminal else 0.0)

                # Initial reward (will be recomputed if using learned rewards)
                if self.reward_source == 'learned':
                    # Placeholder - will be recomputed after normalization
                    reward = 0.0
                else:
                    # Use manual reward
                    from integrated_data_pipeline_v2 import compute_outcome_reward
                    reward = compute_outcome_reward(
                        states, actions, t,
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

    def _recompute_learned_rewards(self):
        """
        Recompute rewards using the learned reward model on normalized states.
        Called after data processing when reward_source='learned'.
        """
        batch_size = 1024  # Process in batches for memory efficiency

        for split_name, data in [('train', self.train_data),
                                  ('val', self.val_data),
                                  ('test', self.test_data)]:
            if data is None:
                continue

            n_transitions = data['n_transitions']
            new_rewards = np.zeros(n_transitions, dtype=np.float32)

            for start_idx in range(0, n_transitions, batch_size):
                end_idx = min(start_idx + batch_size, n_transitions)

                states_batch = data['states'][start_idx:end_idx]
                actions_batch = data['actions'][start_idx:end_idx]
                next_states_batch = data['next_states'][start_idx:end_idx]
                dones_batch = data['dones'][start_idx:end_idx]

                rewards_batch = self._compute_learned_reward(
                    states_batch, actions_batch, next_states_batch, dones_batch
                )
                new_rewards[start_idx:end_idx] = rewards_batch

            data['rewards'] = new_rewards
            print(f"   {split_name}: Reward range [{new_rewards.min():.3f}, {new_rewards.max():.3f}]")

    def get_batch(self, batch_size: int = 256, split: str = 'train',
                  same_patient: bool = False, seed: Optional[int] = None) -> Dict:
        """
        Get a batch of transitions.

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
        """Sample transitions preferring to keep same patients together."""
        indices = []
        remaining = batch_size

        patient_list = list(patient_groups.keys())
        rng.shuffle(patient_list)

        for patient_id in patient_list:
            if remaining <= 0:
                break

            start_idx, end_idx = patient_groups[patient_id]
            patient_transitions = list(range(start_idx, end_idx))

            if len(patient_transitions) <= remaining:
                indices.extend(patient_transitions)
                remaining -= len(patient_transitions)
            else:
                indices.extend(patient_transitions[:remaining])
                remaining = 0

        return np.array(indices)

    def _print_summary(self):
        """Print summary of prepared data"""
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        print(f"Reward source: {self.reward_source}")
        if self.reward_model_type:
            print(f"Reward model: {self.reward_model_type}")

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
            print(f"  Reward range:   [{data['rewards'].min():.3f}, {data['rewards'].max():.3f}]")
            print(f"  Reward mean:    {data['rewards'].mean():.3f}")

    def reset_batch_rng(self):
        """Reset the internal random number generator"""
        self.rng = np.random.RandomState(self.random_seed)


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("="*70)
    print(" INTEGRATED DATA PIPELINE V3 - WITH LEARNED REWARDS")
    print("="*70)

    # Example 1: Using manual rewards (same as V2)
    print("\n--- MANUAL REWARDS (baseline) ---\n")
    pipeline_manual = IntegratedDataPipelineV3(
        model_type='dual',
        reward_source='manual',
        random_seed=42
    )
    train_data, val_data, test_data = pipeline_manual.prepare_data()

    # Example 2: Using GCL learned rewards
    print("\n--- GCL LEARNED REWARDS ---\n")
    pipeline_gcl = IntegratedDataPipelineV3(
        model_type='dual',
        reward_source='learned',
        random_seed=42
    )

    # Load GCL model (uncomment when model is trained)
    # pipeline_gcl.load_gcl_reward_model('experiment/gcl/gcl_bins5_best.pt')
    # train_data, val_data, test_data = pipeline_gcl.prepare_data()

    print("\nTo use learned rewards:")
    print("  1. Train GCL/IQ-Learn model")
    print("  2. Call pipeline.load_gcl_reward_model(path) or load_iq_learn_reward_model(path)")
    print("  3. Call pipeline.prepare_data()")
