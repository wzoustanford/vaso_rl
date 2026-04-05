### import relevant libraries
import torch, pdb
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from integrated_data_pipeline_v2 import IntegratedDataPipelineV2
from integrated_data_pipeline_v3 import IntegratedDataPipelineV3

# Parse command line arguments
parser = argparse.ArgumentParser(description='Sequential Policy Estimator IS evaluation for Block Discrete CQL')
parser.add_argument('--model_path', type=str, required=True,
                   help='Path to trained CQL model checkpoint')
parser.add_argument('--vp2_bins', type=int, default=5,
                   help='Number of bins for VP2 discretization (default: 5)')
parser.add_argument('--eval_set', type=str, default='test', choices=['val', 'test'],
                   help='Which data split to evaluate on (default: test)')
parser.add_argument('--reward_type', type=str, default='manual', choices=['manual', 'irl'],
                   help='Reward type: manual (clinician-defined) or irl (learned from IRL model)')
parser.add_argument('--irl_model_path', type=str, default=None,
                   help='Path to IRL model for reward computation (required if reward_type=irl)')
parser.add_argument('--combined_or_train_data_path', type=str, default=None,
                   help='Path to training dataset. If eval_data_path is also provided, '
                        'all patients from this dataset are used for training. '
                        'If eval_data_path is None, this dataset is split into train/val/test.')
parser.add_argument('--eval_data_path', type=str, default=None,
                   help='Path to evaluation dataset (for val/test). If provided, enables '
                        'dual-dataset mode where this dataset is split 50/50 into val/test.')
parser.add_argument('--use_lstm', action='store_true',
                   help='Use LSTM Q-network instead of standard feedforward network')
parser.add_argument('--disable_is_clipping', action='store_true',
                   help='Disable percentile-based clipping for IS weights and trajectory multiplied weights')
parser.add_argument('--is_clip_lower_pct', type=float, default=0.5,
                   help='Lower percentile for IS weight clipping (default: 0.5)')
parser.add_argument('--is_clip_upper_pct', type=float, default=99.5,
                   help='Upper percentile for IS weight clipping (default: 99.5)')
args = parser.parse_args()

# Validate arguments
if args.reward_type == 'irl' and args.irl_model_path is None:
    parser.error("--irl_model_path is required when --reward_type=irl")
if not (0.0 <= args.is_clip_lower_pct < args.is_clip_upper_pct <= 100.0):
    parser.error("--is_clip_lower_pct and --is_clip_upper_pct must satisfy 0 <= lower < upper <= 100")

n_bins = args.vp2_bins
model_path = args.model_path
eval_set = args.eval_set
reward_type = args.reward_type
irl_model_path = args.irl_model_path
use_lstm = args.use_lstm
use_is_clipping = not args.disable_is_clipping

# Define constants
STATE_DIM = 17  # 17 state features for dual model

### define the block discrete model
# - load the trained block discrete model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model parameters
n_actions = 2 * n_bins  # VP1 (2 options: 0,1) x VP2 (5 bins) = 10 total actions
state_dim = STATE_DIM  # 17 features

# Initialize and load Q-networks based on model type
if use_lstm:
    # Import LSTM network class
    from lstm_block_discrete_cql_network import LSTMDiscreteQNetwork

    # LSTM uses num_actions directly (not 2*n_bins structure)
    q1_network = LSTMDiscreteQNetwork(state_dim=state_dim, num_actions=n_actions).to(device)
    q2_network = LSTMDiscreteQNetwork(state_dim=state_dim, num_actions=n_actions).to(device)

    # Load the trained model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    q1_network.load_state_dict(checkpoint['q1_state_dict'])
    q2_network.load_state_dict(checkpoint['q2_state_dict'])
    q1_network.eval()
    q2_network.eval()

    print(f"Loaded LSTM model from: {model_path}")
    print(f"State dimension: {state_dim}")
    print(f"Number of actions: {n_actions} (VP1: 2 binary × VP2: {n_bins} bins)")
    print("Using min(Q1, Q2) for action selection (double Q-learning)")
    print("Model type: LSTM Q-Network")
else:
    # Import the network class from the training script
    from run_block_discrete_cql_allalphas import DualBlockDiscreteQNetwork

    # Initialize Q-networks (we use both Q1 and Q2, then take min)
    q1_network = DualBlockDiscreteQNetwork(state_dim=state_dim, vp2_bins=n_bins).to(device)
    q2_network = DualBlockDiscreteQNetwork(state_dim=state_dim, vp2_bins=n_bins).to(device)

    # Load the trained model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    q1_network.load_state_dict(checkpoint['q1_state_dict'])
    q2_network.load_state_dict(checkpoint['q2_state_dict'])
    q1_network.eval()
    q2_network.eval()

    print(f"Loaded model from: {model_path}")
    print(f"State dimension: {state_dim}")
    print(f"Number of actions: {n_actions} (VP1: 2 binary × VP2: {n_bins} bins)")
    print("Using min(Q1, Q2) for action selection (double Q-learning)")
    print("Model type: Feedforward Q-Network")

###
# prepare and load the data with random seed 42

print("\n" + "="*70)
print("LOADING DATA")
print("="*70)
print(f"Evaluation set: {eval_set}")
print(f"Reward type: {reward_type}")
if irl_model_path:
    print(f"IRL model: {irl_model_path}")

# Initialize data pipeline based on reward type
if reward_type == 'irl':
    # Use V3 pipeline with learned rewards
    pipeline = IntegratedDataPipelineV3(
        model_type='dual', reward_source='learned', random_seed=42,
        combined_or_train_data_path=args.combined_or_train_data_path,
        eval_data_path=args.eval_data_path
    )

    # Detect IRL model type from path and load accordingly
    if 'iq_learn' in irl_model_path.lower():
        pipeline.load_iq_learn_reward_model(irl_model_path)
    elif 'gcl' in irl_model_path.lower():
        pipeline.load_gcl_reward_model(irl_model_path)
    elif 'maxent' in irl_model_path.lower():
        pipeline.load_maxent_reward_model(irl_model_path)
    elif 'semi_supervised_unet' in irl_model_path.lower():
        pipeline.load_semi_supervised_unet_reward_model(irl_model_path, vp1_bins=2, vp2_bins=n_bins)
    elif 'unet' in irl_model_path.lower():
        pipeline.load_unet_reward_model(irl_model_path, vp1_bins=2, vp2_bins=n_bins)
    else:
        # Try to infer from file content or default to iq_learn
        print(f"Warning: Could not detect IRL model type from path, trying IQ-Learn...")
        pipeline.load_iq_learn_reward_model(irl_model_path)

    train_data, val_data, test_data = pipeline.prepare_data()
else:
    # Use V3 pipeline with manual rewards (supports dual-dataset mode)
    pipeline = IntegratedDataPipelineV3(
        model_type='dual', reward_source='manual', random_seed=42,
        combined_or_train_data_path=args.combined_or_train_data_path,
        eval_data_path=args.eval_data_path
    )
    train_data, val_data, test_data = pipeline.prepare_data()

# Select evaluation data based on eval_set argument
if eval_set == 'val':
    eval_data = val_data
    eval_set_name = "Validation"
else:
    eval_data = test_data
    eval_set_name = "Test"

# Print data statistics
print(f"\nData splits:")
print(f"  Training:   {len(train_data['states'])} transitions")
print(f"  Validation: {len(val_data['states'])} transitions")
print(f"  Test:       {len(test_data['states'])} transitions")
print(f"  Evaluating: {eval_set_name} ({len(eval_data['states'])} transitions)")

# Check unique patients in each split
train_patients = len(np.unique(train_data['patient_ids']))
val_patients = len(np.unique(val_data['patient_ids']))
test_patients = len(np.unique(test_data['patient_ids']))
eval_patients = len(np.unique(eval_data['patient_ids']))

print(f"\nPatient counts:")
print(f"  Training:   {train_patients} patients")
print(f"  Validation: {val_patients} patients")
print(f"  Test:       {test_patients} patients")
print(f"  Evaluating: {eval_set_name} ({eval_patients} patients)")

# Verify state and action dimensions
print(f"\nData shapes:")
print(f"  States:  {train_data['states'].shape}")
print(f"  Actions: {train_data['actions'].shape}")
print(f"  Rewards: {train_data['rewards'].shape}")
print(f"  Reward type: {reward_type}")

## use the trained model to predict the optimal actions predicted by the model
## for example to get model actions: model_actions = model.select_action(states)
## for example to get clinician actions: clinician_actions = data.actions
## note the resulting action should have both vp1 action and vp2 action

print("\n" + "="*70)
print("GENERATING MODEL ACTIONS")
print("="*70)

# Define VP2 bin edges (exactly as in training script)
vp2_bin_edges = np.linspace(0, 0.5, n_bins + 1)
print(f"VP2 bin edges: {vp2_bin_edges}")

# Helper function to select actions and return DISCRETE indices (0-9) - Feedforward version
def select_action_batch_discrete(states, q1_net, q2_net, vp2_bins, device):
    """
    Select best actions for a batch of states using min(Q1, Q2)
    Returns: numpy array of discrete action indices [0-9]
    """
    with torch.no_grad():
        if states.ndim == 1:
            states = states.reshape(1, -1)

        batch_size = states.shape[0]
        state_tensor = torch.FloatTensor(states).to(device)

        # Create all possible discrete actions for each state in batch
        total_actions = 2 * vp2_bins  # 10 actions
        all_actions = torch.arange(total_actions).to(device)
        all_actions = all_actions.unsqueeze(0).expand(batch_size, -1)

        # Expand states to match actions
        state_expanded = state_tensor.unsqueeze(1).expand(-1, total_actions, -1)
        state_expanded = state_expanded.reshape(-1, state_dim)
        actions_flat = all_actions.reshape(-1)

        # Compute Q-values for all actions
        q1_values = q1_net(state_expanded, actions_flat).reshape(batch_size, total_actions)
        q2_values = q2_net(state_expanded, actions_flat).reshape(batch_size, total_actions)
        q_values = torch.min(q1_values, q2_values)

        # Get best action indices (0-9)
        best_action_indices = q_values.argmax(dim=1).cpu().numpy()

        return best_action_indices


# Helper function to select actions using LSTM networks
def select_action_batch_discrete_lstm(states, q1_net, q2_net, device):
    """
    Select best actions for a batch of states using LSTM Q-networks with min(Q1, Q2).
    LSTM networks output Q-values for all actions at once, no need to loop.

    Note: For WIS evaluation, we use fresh hidden states for each state
    (stateless inference). This is simpler and appropriate since we're
    evaluating per-transition, not per-trajectory.

    Returns: numpy array of discrete action indices [0-9]
    """
    with torch.no_grad():
        if states.ndim == 1:
            states = states.reshape(1, -1)

        batch_size = states.shape[0]
        state_tensor = torch.FloatTensor(states).to(device)

        # Initialize fresh hidden states
        hidden1 = q1_net.init_hidden(batch_size, device)
        hidden2 = q2_net.init_hidden(batch_size, device)

        # Forward pass - LSTM returns Q-values for all actions at once
        # forward_single_step: state [batch_size, state_dim] -> q_values [batch_size, num_actions]
        q1_values, _ = q1_net.forward_single_step(state_tensor, hidden1)
        q2_values, _ = q2_net.forward_single_step(state_tensor, hidden2)

        # Take minimum of Q1 and Q2
        q_values = torch.min(q1_values, q2_values)

        # Get best action indices
        best_action_indices = q_values.argmax(dim=1).cpu().numpy()

        return best_action_indices

# Helper function to convert continuous actions to discrete indices
def continuous_to_discrete_action(actions, vp2_edges, vp2_bins):
    """
    Convert continuous actions [vp1, vp2] to discrete action indices (0-9)
    action_idx = vp1 * n_bins + vp2_bin
    """
    vp1 = actions[:, 0].astype(int)  # Binary 0 or 1
    vp2 = actions[:, 1].clip(0, 0.5)

    # Convert VP2 continuous to bin index
    vp2_bins_idx = np.digitize(vp2, vp2_edges) - 1
    vp2_bins_idx = np.clip(vp2_bins_idx, 0, len(vp2_edges) - 2)

    # Combine: action_idx = vp1 * n_bins + vp2_bin
    action_indices = vp1 * vp2_bins + vp2_bins_idx
    return action_indices

# Get model actions as discrete indices (0-9)
print("Computing model actions (discrete) for training data...")
if use_lstm:
    train_model_actions_discrete = select_action_batch_discrete_lstm(train_data['states'], q1_network, q2_network, device)
else:
    train_model_actions_discrete = select_action_batch_discrete(train_data['states'], q1_network, q2_network, n_bins, device)

print(f"Computing model actions (discrete) for {eval_set_name.lower()} data...")
if use_lstm:
    eval_model_actions_discrete = select_action_batch_discrete_lstm(eval_data['states'], q1_network, q2_network, device)
else:
    eval_model_actions_discrete = select_action_batch_discrete(eval_data['states'], q1_network, q2_network, n_bins, device)

# Convert clinician continuous actions to discrete indices (0-9)
train_clinician_actions_discrete = continuous_to_discrete_action(train_data['actions'], vp2_bin_edges, n_bins)
eval_clinician_actions_discrete = continuous_to_discrete_action(eval_data['actions'], vp2_bin_edges, n_bins)

print(f"\nDiscrete action indices generated:")
print(f"  Train model actions:     {train_model_actions_discrete.shape}")
print(f"  Train clinician actions: {train_clinician_actions_discrete.shape}")
print(f"  {eval_set_name} model actions:      {eval_model_actions_discrete.shape}")
print(f"  {eval_set_name} clinician actions:  {eval_clinician_actions_discrete.shape}")

# Display sample discrete actions
print(f"\nSample discrete action indices (first 10):")
print(f"  Model (train):     {train_model_actions_discrete[:10]}")
print(f"  Clinician (train): {train_clinician_actions_discrete[:10]}")

# Show action distribution (0-9)
print(f"\nAction distribution (0-{n_actions-1}):")
for action_idx in range(n_actions):
    print(f"  Action {action_idx}: Model (train)={np.sum(train_model_actions_discrete==action_idx)}, "
          f"Clinician (train)={np.sum(train_clinician_actions_discrete==action_idx)}, "
          f"Model ({eval_set_name.lower()})={np.sum(eval_model_actions_discrete==action_idx)}, "
          f"Clinician ({eval_set_name.lower()})={np.sum(eval_clinician_actions_discrete==action_idx)}")

## Train sequential policy estimators for trajectory-level IS weights
from train_seq_policy_estimator import SeqPolicyEstimatorTrainer

print("\n" + "="*70)
print("TRAINING SEQUENTIAL POLICY ESTIMATORS")
print("="*70)

# Train model policy estimator: P(expert-like | states, model_actions)
print("\n1. Training seq estimator for MODEL actions...")
model_estimator = SeqPolicyEstimatorTrainer(
    state_dim=state_dim, n_actions=n_actions, sequence_length=40)
model_estimator.train(train_data['states'], train_model_actions_discrete,
                      train_data['patient_ids'])

# Train clinician policy estimator: P(expert-like | states, clinician_actions)
print("\n2. Training seq estimator for CLINICIAN actions...")
clinician_estimator = SeqPolicyEstimatorTrainer(
    state_dim=state_dim, n_actions=n_actions, sequence_length=40)
clinician_estimator.train(train_data['states'], train_clinician_actions_discrete,
                          train_data['patient_ids'])

# Predict on eval set — both evaluate the actual clinician trajectory
print("\n" + "="*70)
print("COMPUTING TRAJECTORY-LEVEL IS WEIGHTS")
print("="*70)

pis_model_policy = model_estimator.predict(
    eval_data['states'], eval_clinician_actions_discrete, eval_data['patient_ids'])
pis_clinician_policy = clinician_estimator.predict(
    eval_data['states'], eval_clinician_actions_discrete, eval_data['patient_ids'])

# Compute trajectory-level IS weights and rewards
eps = 1e-10
eval_patient_ids = eval_data['patient_ids']
eval_rewards = eval_data['rewards']
unique_patients = np.unique(eval_patient_ids)

traj_weights = []
traj_rewards = []

for pid in unique_patients:
    w = pis_model_policy[pid] / (pis_clinician_policy[pid] + eps)
    traj_weights.append(w)
    mask = eval_patient_ids == pid
    traj_rewards.append(eval_rewards[mask].sum())

traj_weights = np.array(traj_weights)
traj_rewards = np.array(traj_rewards)

# WIS estimate: sum(w * R) / sum(w)
wis_estimate = (traj_weights * traj_rewards).sum() / (traj_weights.sum() + eps)
clinician_mean = traj_rewards.mean()

print(f"\nTrajectory-level WIS evaluation:")
print(f"  Number of trajectories: {len(unique_patients)}")
print(f"  Clinician policy (raw):  {clinician_mean:.4f}")
print(f"  Model policy (WIS):      {wis_estimate:.4f}")
print(f"  Difference:              {wis_estimate - clinician_mean:.4f}")

print(f"\nIS weight statistics:")
print(f"  Mean: {traj_weights.mean():.4f}, Std: {traj_weights.std():.4f}")
print(f"  Min: {traj_weights.min():.4f}, Max: {traj_weights.max():.4f}")
