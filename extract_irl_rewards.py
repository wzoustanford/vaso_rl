"""
IRL Reward Extraction Script
Extracts rewards from various IRL models on TEST and VALIDATION sets.
Saves rewards in trajectory form as pickle files.

Models supported:
1. U-Net tanh (generative autoregressive)
2. Semi-supervised U-Net GA
3. Maximum Entropy IRL
4. Guided Cost Learning (GCL)
5. IQ-Learn
6. AIRL

Output: Dictionary with trajectory-indexed rewards saved as pickle.

Usage:
    python extract_irl_rewards.py                # Extract test set only
    python extract_irl_rewards.py --include-val  # Extract test + validation sets
"""

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from datetime import datetime

# Local imports
from integrated_data_pipeline_v3 import IntegratedDataPipelineV3

# Model imports
from unet_reward_generator_tanh import UNetRewardGenerator as UNetTanh
from semi_supervised_unet_reward_generator import (
    UNetRewardGenerator as SemiSupUNet,
    MortalityDiffuser
)
from maxent_irl_full_recovery import RewardNetwork
from run_gcl_learn_block_discrete import GCLBlockDiscrete
from run_iq_learn_block_discrete import IQLearnBlockDiscrete


# ============================================================================
# MODEL PATH CONFIGURATION (PLACEHOLDERS - UPDATE WITH ACTUAL PATHS)
# ============================================================================

MODEL_PATHS = {
    'unet_tanh': 'experiments/unet_reward_gen_h64_tanh/model_epoch_100.pt',
    'semi_sup_unet': 'experiments/semi_supervised_unet_64/model_epoch_100.pt',
    'maxent_irl': 'experiment/irl/maxent_reward_model.pt',
    'gcl': '/scratch/code/vaso_rl/experiment/irl/gcl_tau0.005_lr0.001_clr0.01_cost_model.pt',
    'iq_learn': '/scratch/code/vaso_rl/experiment/irl/iq_learn_temp0.001_tau0.005_lr0.001_divjs_q_model.pt',
    'airl': 'experiment/airl/airl_reward_model.pt',
}

# Output directory
OUTPUT_DIR = 'irl_analysis'

# Configuration
RANDOM_SEED = 42
VP1_BINS = 2
VP2_BINS = 5
MIN_SEQ_LEN = 7  # Minimum for U-Net (3 conv layers shrink by 6)


# ============================================================================
# Helper Functions
# ============================================================================

def continuous_to_discrete_action(actions: np.ndarray, vp2_bins: int = 5) -> np.ndarray:
    """Convert continuous actions [vp1, vp2] to discrete indices."""
    vp2_bin_edges = np.linspace(0, 0.5, vp2_bins + 1)
    vp1 = actions[:, 0].astype(int)
    vp2 = actions[:, 1].clip(0, 0.5)
    vp2_bins_idx = np.digitize(vp2, vp2_bin_edges) - 1
    vp2_bins_idx = np.clip(vp2_bins_idx, 0, vp2_bins - 1)
    return vp1 * vp2_bins + vp2_bins_idx


def extract_trajectories_from_data(data: Dict) -> Dict[int, Dict]:
    """
    Extract trajectory data organized by patient ID.

    Returns:
        Dict mapping patient_id -> {
            'states': np.ndarray,
            'actions': np.ndarray,
            'next_states': np.ndarray,
            'dones': np.ndarray,
            'clinician_rewards': np.ndarray,
            'length': int
        }
    """
    trajectories = {}
    for pid, (start_idx, end_idx) in data['patient_groups'].items():
        trajectories[pid] = {
            'states': data['states'][start_idx:end_idx],
            'actions': data['actions'][start_idx:end_idx],
            'next_states': data['next_states'][start_idx:end_idx],
            'dones': data['dones'][start_idx:end_idx],
            'clinician_rewards': data['rewards'][start_idx:end_idx],
            'length': end_idx - start_idx
        }
    return trajectories


# ============================================================================
# U-Net Tanh Reward Extraction
# ============================================================================

def extract_unet_tanh_rewards(
    model_path: str,
    trajectories: Dict[int, Dict],
    device: torch.device
) -> Dict[int, List[float]]:
    """Extract rewards from U-Net tanh model."""

    print(f"\n{'='*60}")
    print("Loading U-Net Tanh Model")
    print(f"{'='*60}")

    if not os.path.exists(model_path):
        print(f"  WARNING: Model not found at {model_path}")
        return None

    checkpoint = torch.load(model_path, map_location=device)
    state_size = checkpoint['state_size']
    action_size = checkpoint['action_size']
    conv_h_dim = checkpoint['conv_h_dim']

    model = UNetTanh(state_size, action_size, conv_h_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Loaded: state_size={state_size}, action_size={action_size}")

    # Extract rewards per trajectory
    rewards_dict = {}
    n_actions = VP1_BINS * VP2_BINS

    with torch.no_grad():
        for i, (pid, traj) in enumerate(trajectories.items()):
            if (i + 1) % 100 == 0:
                print(f"  Processing trajectory {i+1}/{len(trajectories)}")

            seq_len = traj['length'] + 1  # +1 for terminal state

            if seq_len < MIN_SEQ_LEN:
                rewards_dict[pid] = [0.0] * traj['length']
                continue

            # Reconstruct full trajectory
            full_states = np.vstack([traj['states'], traj['next_states'][-1:]])
            last_action = traj['actions'][-1:]
            full_actions = np.vstack([traj['actions'], last_action])

            # Convert to tensors
            states_t = torch.FloatTensor(full_states).unsqueeze(0).to(device)
            actions_discrete = continuous_to_discrete_action(full_actions, VP2_BINS)
            actions_onehot = F.one_hot(
                torch.LongTensor(actions_discrete), num_classes=n_actions
            ).float().unsqueeze(0).to(device)

            # Forward pass
            state_action = torch.cat([states_t, actions_onehot], dim=-1)
            rewards_pred = model(state_action).squeeze().cpu().numpy()

            # Store rewards (exclude terminal)
            rewards_dict[pid] = rewards_pred[:-1].tolist()

    print(f"  Extracted rewards for {len(rewards_dict)} trajectories")
    return rewards_dict


# ============================================================================
# Semi-Supervised U-Net Reward Extraction
# ============================================================================

def extract_semi_sup_unet_rewards(
    model_path: str,
    trajectories: Dict[int, Dict],
    device: torch.device
) -> Dict[int, List[float]]:
    """Extract rewards from Semi-Supervised U-Net model."""

    print(f"\n{'='*60}")
    print("Loading Semi-Supervised U-Net Model")
    print(f"{'='*60}")

    if not os.path.exists(model_path):
        print(f"  WARNING: Model not found at {model_path}")
        return None

    checkpoint = torch.load(model_path, map_location=device)
    state_size = checkpoint['state_size']
    action_size = checkpoint['action_size']
    conv_h_dim = checkpoint['conv_h_dim']

    model = SemiSupUNet(state_size, action_size, conv_h_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load MortalityDiffuser if present
    mortality_diffuser = None
    if 'mortality_diffuser_state_dict' in checkpoint:
        seq_len = checkpoint.get('mortality_diffuser_seq_len', 40)
        h_dim = checkpoint.get('mortality_diffuser_h_dim', 16)
        mortality_diffuser = MortalityDiffuser(seq_len, h_dim).to(device)
        mortality_diffuser.load_state_dict(checkpoint['mortality_diffuser_state_dict'])
        mortality_diffuser.eval()
        print(f"  Loaded with MortalityDiffuser")

    print(f"  Loaded: state_size={state_size}, action_size={action_size}")

    # Extract rewards (same logic as U-Net tanh, but using SemiSupUNet)
    rewards_dict = {}
    n_actions = VP1_BINS * VP2_BINS

    with torch.no_grad():
        for i, (pid, traj) in enumerate(trajectories.items()):
            if (i + 1) % 100 == 0:
                print(f"  Processing trajectory {i+1}/{len(trajectories)}")

            seq_len = traj['length'] + 1

            if seq_len < MIN_SEQ_LEN:
                rewards_dict[pid] = [0.0] * traj['length']
                continue

            full_states = np.vstack([traj['states'], traj['next_states'][-1:]])
            last_action = traj['actions'][-1:]
            full_actions = np.vstack([traj['actions'], last_action])

            states_t = torch.FloatTensor(full_states).unsqueeze(0).to(device)
            actions_discrete = continuous_to_discrete_action(full_actions, VP2_BINS)
            actions_onehot = F.one_hot(
                torch.LongTensor(actions_discrete), num_classes=n_actions
            ).float().unsqueeze(0).to(device)

            state_action = torch.cat([states_t, actions_onehot], dim=-1)
            rewards_pred = model(state_action).squeeze().cpu().numpy()

            rewards_dict[pid] = rewards_pred[:-1].tolist()

    print(f"  Extracted rewards for {len(rewards_dict)} trajectories")
    return rewards_dict


# ============================================================================
# MaxEnt IRL Reward Extraction
# ============================================================================

def extract_maxent_irl_rewards(
    model_path: str,
    trajectories: Dict[int, Dict],
    device: torch.device
) -> Dict[int, List[float]]:
    """Extract rewards from Maximum Entropy IRL model."""

    print(f"\n{'='*60}")
    print("Loading MaxEnt IRL Model")
    print(f"{'='*60}")

    if not os.path.exists(model_path):
        print(f"  WARNING: Model not found at {model_path}")
        return None

    checkpoint = torch.load(model_path, map_location=device)
    state_dim = checkpoint['state_dim']
    action_dim = checkpoint['action_dim']

    model = RewardNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[128, 64, 32]
    ).to(device)
    model.load_state_dict(checkpoint['reward_network_state_dict'])
    model.eval()

    print(f"  Loaded: state_dim={state_dim}, action_dim={action_dim}")

    # Extract rewards per trajectory
    rewards_dict = {}

    with torch.no_grad():
        for i, (pid, traj) in enumerate(trajectories.items()):
            if (i + 1) % 100 == 0:
                print(f"  Processing trajectory {i+1}/{len(trajectories)}")

            states_t = torch.FloatTensor(traj['states']).to(device)
            actions_t = torch.FloatTensor(traj['actions']).to(device)

            # MaxEnt IRL: R(s, a)
            rewards_pred = model(states_t, actions_t).cpu().numpy()
            rewards_dict[pid] = rewards_pred.tolist()

    print(f"  Extracted rewards for {len(rewards_dict)} trajectories")
    return rewards_dict


# ============================================================================
# GCL Reward Extraction
# ============================================================================

def extract_gcl_rewards(
    model_path: str,
    trajectories: Dict[int, Dict],
    device: torch.device
) -> Dict[int, List[float]]:
    """Extract rewards from Guided Cost Learning model.

    GCL reward = -cost(s, a)
    """

    print(f"\n{'='*60}")
    print("Loading GCL Model")
    print(f"{'='*60}")

    if not os.path.exists(model_path):
        print(f"  WARNING: Model not found at {model_path}")
        return None

    # Load using class method
    agent = GCLBlockDiscrete.load(model_path, device=device)
    agent.cost_f.eval()

    print(f"  Loaded: state_dim={agent.state_dim}, vp2_bins={agent.vp2_bins}")

    # Extract rewards per trajectory
    rewards_dict = {}

    with torch.no_grad():
        for i, (pid, traj) in enumerate(trajectories.items()):
            if (i + 1) % 100 == 0:
                print(f"  Processing trajectory {i+1}/{len(trajectories)}")

            states_t = torch.FloatTensor(traj['states']).to(device)
            actions_t = torch.FloatTensor(traj['actions']).to(device)

            # Convert continuous actions to discrete indices
            action_indices = agent.continuous_to_discrete_batch(actions_t)

            # GCL: reward = -cost(s, a)
            costs = agent.cost_f(states_t, action_indices).squeeze()
            rewards_pred = -costs.cpu().numpy()

            rewards_dict[pid] = rewards_pred.tolist()

    print(f"  Extracted rewards for {len(rewards_dict)} trajectories")
    return rewards_dict


# ============================================================================
# IQ-Learn Reward Extraction
# ============================================================================

def extract_iq_learn_rewards(
    model_path: str,
    trajectories: Dict[int, Dict],
    device: torch.device
) -> Dict[int, List[float]]:
    """Extract rewards from IQ-Learn model.

    IQ-Learn reward = Q(s,a) - gamma * V(s')
    """

    print(f"\n{'='*60}")
    print("Loading IQ-Learn Model")
    print(f"{'='*60}")

    if not os.path.exists(model_path):
        print(f"  WARNING: Model not found at {model_path}")
        return None

    # Load using class method
    agent = IQLearnBlockDiscrete.load(model_path, device=device)
    agent.q1.eval()
    agent.q2.eval()

    print(f"  Loaded: state_dim={agent.state_dim}, vp2_bins={agent.vp2_bins}")
    print(f"  gamma={agent.gamma}, init_temp={agent.init_temp}")

    # Extract rewards per trajectory
    rewards_dict = {}

    with torch.no_grad():
        for i, (pid, traj) in enumerate(trajectories.items()):
            if (i + 1) % 100 == 0:
                print(f"  Processing trajectory {i+1}/{len(trajectories)}")

            states_t = torch.FloatTensor(traj['states']).to(device)
            actions_t = torch.FloatTensor(traj['actions']).to(device)
            next_states_t = torch.FloatTensor(traj['next_states']).to(device)
            dones_t = torch.FloatTensor(traj['dones']).to(device)

            # IQ-Learn: reward = Q(s,a) - gamma * V(s')
            rewards_pred = agent.get_recovered_reward(
                states_t, actions_t, next_states_t, dones_t
            ).cpu().numpy()

            rewards_dict[pid] = rewards_pred.tolist()

    print(f"  Extracted rewards for {len(rewards_dict)} trajectories")
    return rewards_dict


# ============================================================================
# AIRL Reward Extraction
# ============================================================================

def extract_airl_rewards(
    model_path: str,
    trajectories: Dict[int, Dict],
    device: torch.device,
    state_features: List[str],
    scaler
) -> Dict[int, List[float]]:
    """
    Extract rewards from AIRL checkpoint produced by airl.py.
    Reconciles feature order/action schema/normalization with pipeline data.
    """
    print(f"\n{'='*60}")
    print("Loading AIRL Model")
    print(f"{'='*60}")

    if not os.path.exists(model_path):
        print(f"  WARNING: Model not found at {model_path}")
        return None

    try:
        import gymnasium as gym
        from imitation.rewards.reward_nets import BasicShapedRewardNet
        from imitation.util.networks import RunningNorm
    except Exception as exc:
        print(f"  WARNING: AIRL dependencies unavailable ({exc}). Skipping AIRL extraction.")
        return None

    checkpoint = torch.load(model_path, map_location=device)
    required = [
        'reward_net_state_dict', 'state_dim', 'action_dim',
        'state_mean', 'state_std', 'action_mean', 'action_std'
    ]
    missing = [k for k in required if k not in checkpoint]
    if missing:
        print(f"  WARNING: Invalid AIRL checkpoint missing keys {missing}")
        return None

    state_dim = int(checkpoint['state_dim'])
    action_dim = int(checkpoint['action_dim'])

    obs_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(state_dim,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(action_dim,), dtype=np.float32)
    model = BasicShapedRewardNet(
        observation_space=obs_space,
        action_space=action_space,
        normalize_input_layer=RunningNorm,
    ).to(device)
    model.load_state_dict(checkpoint['reward_net_state_dict'])
    model.eval()

    airl_state_cols = checkpoint.get('state_cols', None)
    airl_action_cols = checkpoint.get('action_cols', None)
    state_mean = np.asarray(checkpoint['state_mean'], dtype=np.float32)
    state_std = np.asarray(checkpoint['state_std'], dtype=np.float32)
    action_mean = np.asarray(checkpoint['action_mean'], dtype=np.float32)
    action_std = np.asarray(checkpoint['action_std'], dtype=np.float32)

    print(f"  Loaded: state_dim={state_dim}, action_dim={action_dim}")
    if airl_action_cols is not None:
        print(f"  AIRL action columns: {airl_action_cols}")

    rewards_dict = {}

    with torch.no_grad():
        for i, (pid, traj) in enumerate(trajectories.items()):
            if (i + 1) % 100 == 0:
                print(f"  Processing trajectory {i+1}/{len(trajectories)}")

            # Pipeline trajectories are normalized by pipeline scaler -> convert to raw first.
            states_raw = scaler.inverse_transform(traj['states'])
            next_states_raw = scaler.inverse_transform(traj['next_states'])

            # State alignment
            if airl_state_cols is not None:
                try:
                    idx = [state_features.index(c) for c in airl_state_cols]
                except ValueError:
                    print("  WARNING: AIRL state features mismatch with pipeline. Skipping AIRL extraction.")
                    return None
                states_airl_raw = states_raw[:, idx]
                next_states_airl_raw = next_states_raw[:, idx]
            else:
                states_airl_raw = states_raw
                next_states_airl_raw = next_states_raw

            # Action alignment
            actions = traj['actions']
            if airl_action_cols is None:
                actions_airl_raw = actions
            else:
                action_parts = []
                for action_col in airl_action_cols:
                    if action_col == 'action_vaso':
                        action_parts.append(actions[:, 0:1])
                    elif action_col == 'norepinephrine':
                        if actions.shape[1] < 2:
                            print("  WARNING: AIRL expects norepinephrine action but actions are not dual.")
                            return None
                        action_parts.append(actions[:, 1:2])
                    elif action_col in state_features:
                        sidx = state_features.index(action_col)
                        action_parts.append(states_raw[:, sidx:sidx+1])
                    else:
                        print(f"  WARNING: Unsupported AIRL action column '{action_col}'")
                        return None
                actions_airl_raw = np.concatenate(action_parts, axis=1).astype(np.float32)

            states_airl = ((states_airl_raw - state_mean) / (state_std + 1e-8)).astype(np.float32)
            next_states_airl = ((next_states_airl_raw - state_mean) / (state_std + 1e-8)).astype(np.float32)
            actions_airl = ((actions_airl_raw - action_mean) / (action_std + 1e-8)).astype(np.float32)
            dones_airl = traj['dones'].astype(np.float32)

            if hasattr(model, 'predict_processed'):
                rewards_pred = model.predict_processed(states_airl, actions_airl, next_states_airl, dones_airl)
            elif hasattr(model, 'predict'):
                try:
                    rewards_pred = model.predict(states_airl, actions_airl, next_states_airl, dones_airl)
                except TypeError:
                    rewards_pred = model.predict(states_airl, actions_airl)
            else:
                s_t = torch.as_tensor(states_airl, dtype=torch.float32, device=device)
                a_t = torch.as_tensor(actions_airl, dtype=torch.float32, device=device)
                ns_t = torch.as_tensor(next_states_airl, dtype=torch.float32, device=device)
                d_t = torch.as_tensor(dones_airl, dtype=torch.float32, device=device)
                try:
                    rewards_pred = model(s_t, a_t, ns_t, d_t).detach().cpu().numpy()
                except TypeError:
                    rewards_pred = model(s_t, a_t).detach().cpu().numpy()

            rewards_dict[pid] = np.asarray(rewards_pred, dtype=np.float32).reshape(-1).tolist()

    print(f"  Extracted rewards for {len(rewards_dict)} trajectories")
    return rewards_dict


# ============================================================================
# Clinician Reward Extraction
# ============================================================================

def extract_clinician_rewards(
    trajectories: Dict[int, Dict]
) -> Dict[int, List[float]]:
    """Extract clinician-designed rewards from data."""

    print(f"\n{'='*60}")
    print("Extracting Clinician Rewards")
    print(f"{'='*60}")

    rewards_dict = {}

    for pid, traj in trajectories.items():
        rewards_dict[pid] = traj['clinician_rewards'].tolist()

    print(f"  Extracted rewards for {len(rewards_dict)} trajectories")
    return rewards_dict


# ============================================================================
# Save Function
# ============================================================================

def save_rewards_pickle(
    rewards_data: Dict,
    output_dir: str,
    filename: str
) -> str:
    """Save rewards dictionary to pickle file."""

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(rewards_data, f)

    print(f"  Saved: {filepath}")
    return filepath


# ============================================================================
# Main Function
# ============================================================================

def extract_rewards_for_dataset(
    trajectories: Dict[int, Dict],
    device: torch.device,
    dataset_name: str = "test",
    state_features: Optional[List[str]] = None,
    scaler=None
) -> Dict:
    """Extract rewards from all IRL models for a given dataset."""

    all_rewards = {
        'trajectory_ids': list(trajectories.keys()),
        'trajectory_lengths': {pid: traj['length'] for pid, traj in trajectories.items()},
    }

    # 1. Clinician rewards (always available)
    all_rewards['clinician'] = extract_clinician_rewards(trajectories)

    # 2. U-Net Tanh
    unet_rewards = extract_unet_tanh_rewards(
        MODEL_PATHS['unet_tanh'], trajectories, device
    )
    if unet_rewards is not None:
        all_rewards['unet_tanh'] = unet_rewards

    # 3. Semi-supervised U-Net
    semi_sup_rewards = extract_semi_sup_unet_rewards(
        MODEL_PATHS['semi_sup_unet'], trajectories, device
    )
    if semi_sup_rewards is not None:
        all_rewards['semi_sup_unet'] = semi_sup_rewards

    # 4. MaxEnt IRL
    maxent_rewards = extract_maxent_irl_rewards(
        MODEL_PATHS['maxent_irl'], trajectories, device
    )
    if maxent_rewards is not None:
        all_rewards['maxent_irl'] = maxent_rewards

    # 5. GCL
    gcl_rewards = extract_gcl_rewards(
        MODEL_PATHS['gcl'], trajectories, device
    )
    if gcl_rewards is not None:
        all_rewards['gcl'] = gcl_rewards

    # 6. IQ-Learn
    iq_learn_rewards = extract_iq_learn_rewards(
        MODEL_PATHS['iq_learn'], trajectories, device
    )
    if iq_learn_rewards is not None:
        all_rewards['iq_learn'] = iq_learn_rewards

    # 7. AIRL
    if state_features is None or scaler is None:
        print("  WARNING: Missing state_features/scaler; skipping AIRL reward extraction.")
    else:
        airl_rewards = extract_airl_rewards(
            MODEL_PATHS['airl'], trajectories, device, state_features, scaler
        )
        if airl_rewards is not None:
            all_rewards['airl'] = airl_rewards

    return all_rewards


def main(include_val: bool = False):
    """Main function to extract all IRL rewards on TEST set (and optionally VALIDATION set)."""

    print("="*70)
    print("IRL REWARD EXTRACTION SCRIPT")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Include validation set: {include_val}")
    print("="*70)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # =========================================================================
    # Load data using IntegratedDataPipelineV3
    # =========================================================================
    print(f"\n{'='*60}")
    print("Loading Data")
    print(f"{'='*60}")

    pipeline = IntegratedDataPipelineV3(
        model_type='dual',
        reward_source='manual',  # Get clinician rewards
        random_seed=RANDOM_SEED
    )
    train_data, val_data, test_data = pipeline.prepare_data()

    print(f"\nTEST set statistics:")
    print(f"  Transitions: {test_data['n_transitions']}")
    print(f"  Patients:    {test_data['n_patients']}")

    # Extract trajectory structure for test set
    test_trajectories = extract_trajectories_from_data(test_data)
    print(f"  Trajectories extracted: {len(test_trajectories)}")

    # Extract trajectory structure for validation set if requested
    val_trajectories = None
    if include_val:
        print(f"\nVALIDATION set statistics:")
        print(f"  Transitions: {val_data['n_transitions']}")
        print(f"  Patients:    {val_data['n_patients']}")
        val_trajectories = extract_trajectories_from_data(val_data)
        print(f"  Trajectories extracted: {len(val_trajectories)}")

    # =========================================================================
    # Extract rewards from each model
    # =========================================================================
    print(f"\n{'='*60}")
    print("Extracting TEST Set Rewards")
    print(f"{'='*60}")
    test_rewards = extract_rewards_for_dataset(
        test_trajectories,
        device,
        "test",
        state_features=test_data.get('state_features'),
        scaler=pipeline.scaler,
    )

    # Extract validation rewards if requested
    val_rewards = None
    if include_val and val_trajectories is not None:
        print(f"\n{'='*60}")
        print("Extracting VALIDATION Set Rewards")
        print(f"{'='*60}")
        val_rewards = extract_rewards_for_dataset(
            val_trajectories,
            device,
            "val",
            state_features=val_data.get('state_features'),
            scaler=pipeline.scaler,
        )

    # =========================================================================
    # Build combined rewards structure
    # =========================================================================

    # Test-only rewards
    test_all_rewards = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'random_seed': RANDOM_SEED,
            'n_trajectories': len(test_trajectories),
            'n_transitions': test_data['n_transitions'],
            'eval_set': 'test',
            'model_paths': MODEL_PATHS,
        },
        **test_rewards
    }

    # Combined val+test rewards (if validation is included)
    combined_rewards = None
    if include_val and val_rewards is not None:
        # Merge trajectory IDs and lengths
        combined_traj_ids = test_rewards['trajectory_ids'] + val_rewards['trajectory_ids']
        combined_traj_lengths = {**test_rewards['trajectory_lengths'], **val_rewards['trajectory_lengths']}

        combined_rewards = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'random_seed': RANDOM_SEED,
                'n_trajectories': len(combined_traj_ids),
                'n_transitions': test_data['n_transitions'] + val_data['n_transitions'],
                'eval_set': 'val+test',
                'model_paths': MODEL_PATHS,
            },
            'trajectory_ids': combined_traj_ids,
            'trajectory_lengths': combined_traj_lengths,
        }

        # Merge rewards for each model
        for model_name in ['clinician', 'unet_tanh', 'semi_sup_unet', 'maxent_irl', 'gcl', 'iq_learn', 'airl']:
            if model_name in test_rewards and model_name in val_rewards:
                combined_rewards[model_name] = {**test_rewards[model_name], **val_rewards[model_name]}
            elif model_name in test_rewards:
                combined_rewards[model_name] = test_rewards[model_name]
            elif model_name in val_rewards:
                combined_rewards[model_name] = val_rewards[model_name]

    # =========================================================================
    # Save all rewards
    # =========================================================================
    print(f"\n{'='*60}")
    print("Saving Rewards")
    print(f"{'='*60}")

    # Save test-only combined file
    save_rewards_pickle(
        test_all_rewards,
        OUTPUT_DIR,
        'all_irl_rewards_test.pkl'
    )

    # Save combined val+test file if validation included
    if combined_rewards is not None:
        save_rewards_pickle(
            combined_rewards,
            OUTPUT_DIR,
            'all_irl_rewards_val_test.pkl'
        )

    # Save individual files for each model (test only)
    for model_name in ['clinician', 'unet_tanh', 'semi_sup_unet', 'maxent_irl', 'gcl', 'iq_learn', 'airl']:
        if model_name in test_rewards:
            individual_data = {
                'metadata': test_all_rewards['metadata'],
                'trajectory_ids': test_rewards['trajectory_ids'],
                'trajectory_lengths': test_rewards['trajectory_lengths'],
                'rewards': test_rewards[model_name]
            }
            save_rewards_pickle(
                individual_data,
                OUTPUT_DIR,
                f'{model_name}_rewards_test.pkl'
            )

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"\nTEST set models extracted:")
    for model_name in ['clinician', 'unet_tanh', 'semi_sup_unet', 'maxent_irl', 'gcl', 'iq_learn', 'airl']:
        if model_name in test_rewards:
            n_traj = len(test_rewards[model_name])
            total_rewards = sum(len(r) for r in test_rewards[model_name].values())
            print(f"  {model_name:20s}: {n_traj} trajectories, {total_rewards} rewards")
        else:
            print(f"  {model_name:20s}: NOT FOUND")

    if combined_rewards is not None:
        print(f"\nCOMBINED (val+test) models extracted:")
        for model_name in ['clinician', 'unet_tanh', 'semi_sup_unet', 'maxent_irl', 'gcl', 'iq_learn', 'airl']:
            if model_name in combined_rewards:
                n_traj = len(combined_rewards[model_name])
                total_rewards = sum(len(r) for r in combined_rewards[model_name].values())
                print(f"  {model_name:20s}: {n_traj} trajectories, {total_rewards} rewards")

    print(f"\nFiles saved to: {OUTPUT_DIR}/")
    print(f"  - all_irl_rewards_test.pkl (test only)")
    if combined_rewards is not None:
        print(f"  - all_irl_rewards_val_test.pkl (combined val+test)")
    print(f"  - <model_name>_rewards_test.pkl (individual)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Extract IRL rewards')
    parser.add_argument('--include-val', action='store_true',
                       help='Include validation set in extraction')
    args = parser.parse_args()
    main(include_val=args.include_val)
