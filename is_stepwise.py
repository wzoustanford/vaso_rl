### import relevant libraries
import torch, pdb
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import os
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from integrated_data_pipeline_v2 import IntegratedDataPipelineV2

# Import stepwise model classes
from run_unified_stepwise_cql_allalphas import (
    StepwiseActionSpace,
    UnifiedStepwiseCQL,
    StepwiseQNetwork,
    prepare_stepwise_batch
)

# Define stepwise action space parameters
MAX_STEP = 0.2  # Maximum VP2 change per step (mcg/kg/min)
MIN_STEP = 0.05  # Minimum VP2 step size (fixed)

# VP2 discretization parameters
N_VP2_BINS = 10  # VP2 bins from 0.05 to 0.50 in 0.05 increments

# State dimensions:
# - Base features: 17 (from dual pipeline)
# - VP2 one-hot: 10 bins
# - Total: 27 dimensions
BASE_STATE_DIM = 17
TOTAL_STATE_DIM = BASE_STATE_DIM + N_VP2_BINS  # 27

### define the stepwise directional model
# - load the trained stepwise CQL model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize stepwise action space
action_space = StepwiseActionSpace(max_step=MAX_STEP)

# Model parameters
n_vp2_changes = len(action_space.VP2_CHANGES)  # 9 changes: [-0.2, -0.15, ..., +0.15, +0.2]
n_actions = 2 * n_vp2_changes  # VP1 (2 binary) × VP2 changes (9) = 18 total actions

# Initialize stepwise CQL agent
stepwise_agent = UnifiedStepwiseCQL(
    state_dim=TOTAL_STATE_DIM,  # 27 (includes VP2 one-hot)
    max_step=MAX_STEP,
    alpha=0.0,
    gamma=0.95,
    tau=0.8,
    lr=1e-3,
    device=device
)

# Load the trained model checkpoint
model_path = 'experiment/epoch500_new_trained_oviss_reward_models/stepwise_cql_alpha0.000000_maxstep0.2_final.pt'

checkpoint = torch.load(model_path, map_location=device)
stepwise_agent.q1.load_state_dict(checkpoint['q1_state_dict'])
stepwise_agent.q2.load_state_dict(checkpoint['q2_state_dict'])
stepwise_agent.q1.eval()
stepwise_agent.q2.eval()

print(f"Loaded model from: {model_path}")
print(f"State dimension: {TOTAL_STATE_DIM} (Base: {BASE_STATE_DIM} + VP2 bins: {N_VP2_BINS})")
print(f"VP2 changes: {action_space.VP2_CHANGES}")
print(f"Number of actions: {n_actions} (VP1: 2 binary × VP2 changes: {n_vp2_changes})")
print("Using min(Q1, Q2) for action selection (double Q-learning)") 

###
# prepare and load the data with random seed 42 with all the data in a sequence, for example:
# pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
# train_data, val_data, test_data = pipeline.prepare_data()

print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

# Initialize data pipeline with random seed 42
# Stepwise model uses 'dual' pipeline (states don't include VP2, we add it later)
pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
train_data, val_data, test_data = pipeline.prepare_data()

# Print data statistics
print(f"\nData splits:")
print(f"  Training:   {len(train_data['states'])} transitions")
print(f"  Validation: {len(val_data['states'])} transitions")
print(f"  Test:       {len(test_data['states'])} transitions")

# Check unique patients in each split
train_patients = len(np.unique(train_data['patient_ids']))
val_patients = len(np.unique(val_data['patient_ids']))
test_patients = len(np.unique(test_data['patient_ids']))

print(f"\nPatient counts:")
print(f"  Training:   {train_patients} patients")
print(f"  Validation: {val_patients} patients")
print(f"  Test:       {test_patients} patients")

# Verify state and action dimensions
print(f"\nData shapes:")
print(f"  Base states:  {train_data['states'].shape} (will add VP2 one-hot)")
print(f"  Actions:      {train_data['actions'].shape} (VP1, VP2)")
print(f"  Rewards:      {train_data['rewards'].shape}")

print("\n" + "="*70)
print("GENERATING MODEL ACTIONS")
print("="*70)

# For stepwise model, VP2 is discretized into bins and one-hot encoded
# VP2 bins: [0.05, 0.10, 0.15, ..., 0.50] (10 bins)
print(f"VP2 discretization: {N_VP2_BINS} bins from {MIN_STEP} to 0.50")
print(f"VP2 bin centers: {action_space.vp2_bin_centers}")
print(f"VP2 stepwise changes: {action_space.VP2_CHANGES}")

# Helper function to select stepwise actions for a batch of states
def select_action_batch_stepwise(base_states, vp2_doses, agent, action_space, device):
    """
    Select best stepwise actions for a batch of states using min(Q1, Q2)

    Args:
        base_states: Base state features [batch_size, 17]
        vp2_doses: Current VP2 doses [batch_size]
        agent: UnifiedStepwiseCQL agent
        action_space: StepwiseActionSpace instance
        device: torch device

    Returns:
        numpy array of discrete action indices [0 to n_actions-1]
    """
    with torch.no_grad():
        batch_size = base_states.shape[0]

        # Convert VP2 doses to one-hot encodings
        vp2_one_hot = np.zeros((batch_size, action_space.n_vp2_bins))
        for i in range(batch_size):
            bin_idx = action_space.vp2_to_bin(vp2_doses[i])
            vp2_one_hot[i, bin_idx] = 1.0

        # Augment states with VP2 one-hot encoding
        extended_states = np.concatenate([base_states, vp2_one_hot], axis=1)
        state_tensor = torch.FloatTensor(extended_states).to(device)

        # Expand states for all actions
        state_expanded = state_tensor.unsqueeze(1).expand(-1, agent.n_actions, -1)
        state_flat = state_expanded.reshape(-1, TOTAL_STATE_DIM)

        # Create action indices for all actions
        action_indices = torch.arange(agent.n_actions, device=device)
        action_indices = action_indices.unsqueeze(0).expand(batch_size, -1)
        action_flat = action_indices.reshape(-1)

        # Compute Q-values for all actions
        q1_values = agent.q1(state_flat, action_flat).reshape(batch_size, agent.n_actions)
        q2_values = agent.q2(state_flat, action_flat).reshape(batch_size, agent.n_actions)
        q_values = torch.min(q1_values, q2_values)

        # Get valid action masks for each state (batched)
        vp2_doses_tensor = torch.FloatTensor(vp2_doses).to(device)
        valid_masks = action_space.get_valid_actions_batch(vp2_doses_tensor)
        
        # Mask invalid actions
        #q_values[~valid_masks] = -float('inf')

        # Select best valid action for each state
        best_action_indices = q_values.argmax(dim=1).cpu().numpy()
        
        return best_action_indices, q_values

# Helper function to convert clinician actions to stepwise discrete actions
def clinician_to_stepwise_action(current_vp2, next_vp2, vp1_action, action_space):
    """
    Convert clinician's continuous actions to stepwise discrete action indices.

    The stepwise action represents the CHANGE in VP2 from current to next state.

    Args:
        current_vp2: Current VP2 doses [batch_size]
        next_vp2: Next VP2 doses (from actions[:, 1]) [batch_size]
        vp1_action: VP1 binary actions (from actions[:, 0]) [batch_size]
        action_space: StepwiseActionSpace instance

    Returns:
        Discrete action indices [batch_size]
    """
    batch_size = len(current_vp2)

    # Discretize VP2 doses to bins
    current_vp2_bins = np.array([action_space.vp2_to_bin(vp2) for vp2 in current_vp2])
    next_vp2_bins = np.array([action_space.vp2_to_bin(vp2) for vp2 in next_vp2])

    # Compute VP2 change in bins (just like in training)
    vp2_changes = next_vp2_bins - current_vp2_bins

    # Map bin changes to VP2_CHANGES indices
    # VP2_CHANGES is e.g., [-0.2, -0.15, -0.1, -0.05, 0, +0.05, +0.1, +0.15, +0.2]
    # vp2_changes are in bins (e.g., -4, -3, -2, -1, 0, +1, +2, +3, +4)
    middle_idx = len(action_space.VP2_CHANGES) // 2  # Index for 0 change
    vp2_change_idx = np.clip(vp2_changes + middle_idx, 0, len(action_space.VP2_CHANGES) - 1)

    # Convert VP1 to binary (0 or 1)
    vp1_binary = (vp1_action > 0.5).astype(int)

    # Combine into discrete action: vp1 * n_vp2_changes + vp2_change_idx
    discrete_actions = vp1_binary * action_space.n_vp2_actions + vp2_change_idx

    return discrete_actions

# Get model stepwise actions
print("Computing model stepwise actions for training data...")
train_model_actions_stepwise, train_q_values = select_action_batch_stepwise(
    train_data['states'],
    train_data['actions'][:, 1].clip(0.05, 0.5),  # Current VP2 doses
    stepwise_agent,
    action_space,
    device
)

print("Computing model stepwise actions for test data...")
test_model_actions_stepwise, test_q_values = select_action_batch_stepwise(
    test_data['states'],
    test_data['actions'][:, 1].clip(0.05, 0.5),  # Current VP2 doses
    stepwise_agent,
    action_space,
    device
)

# Convert clinician actions to stepwise discrete actions
print("Converting clinician actions to stepwise format...")
# For clinician: need to get next VP2 to compute the change
# We'll use the next transition's VP2 action
train_next_vp2 = np.roll(train_data['actions'][:, 1].clip(0.05, 0.5), -1)
test_next_vp2 = np.roll(test_data['actions'][:, 1].clip(0.05, 0.5), -1)

train_clinician_actions_stepwise = clinician_to_stepwise_action(
    train_data['actions'][:, 1].clip(0.05, 0.5),  # Current VP2
    train_next_vp2,                # Next VP2
    train_data['actions'][:, 0],  # VP1
    action_space
)

test_clinician_actions_stepwise = clinician_to_stepwise_action(
    test_data['actions'][:, 1].clip(0.05, 0.5),   # Current VP2
    test_next_vp2,                 # Next VP2
    test_data['actions'][:, 0],   # VP1
    action_space
)

print(f"\nStepwise discrete action indices generated:")
print(f"  Train model actions:     {train_model_actions_stepwise.shape}")
print(f"  Train clinician actions: {train_clinician_actions_stepwise.shape}")
print(f"  Test model actions:      {test_model_actions_stepwise.shape}")
print(f"  Test clinician actions:  {test_clinician_actions_stepwise.shape}")

# Display sample stepwise actions
print(f"\nSample stepwise action indices (first 10):")
print(f"  Model (train):     {train_model_actions_stepwise[:10]}")
print(f"  Clinician (train): {train_clinician_actions_stepwise[:10]}")

# Decode and display action meanings
print(f"\nAction space breakdown (VP1 × VP2 changes):")
print(f"  Total actions: {n_actions}")
print(f"  VP1: 2 options (0, 1)")
print(f"  VP2 changes: {n_vp2_changes} options {action_space.VP2_CHANGES}")

# Show action distribution with decoded meanings
print(f"\nAction distribution (0-{n_actions-1}):")
print(f"  {'Action':>6} | {'VP1':>3} | {'VP2 Change':>11} | {'Model(train)':>12} | {'Clin(train)':>11} | {'Model(test)':>11} | {'Clin(test)':>10}")
print(f"  {'-'*6}-+-{'-'*3}-+-{'-'*11}-+-{'-'*12}-+-{'-'*11}-+-{'-'*11}-+-{'-'*10}")

for action_idx in range(n_actions):
    vp1, vp2_change_idx = action_space.decode_action(action_idx)
    vp2_change = action_space.VP2_CHANGES[vp2_change_idx]

    model_train_count = np.sum(train_model_actions_stepwise == action_idx)
    clin_train_count = np.sum(train_clinician_actions_stepwise == action_idx)
    model_test_count = np.sum(test_model_actions_stepwise == action_idx)
    clin_test_count = np.sum(test_clinician_actions_stepwise == action_idx)

    print(f"  {action_idx:6d} | {vp1:3d} | {vp2_change:+6.2f} mcg | {model_train_count:12d} | {clin_train_count:11d} | {model_test_count:11d} | {clin_test_count:10d}") 


print("\n" + "="*70)
print("TRAINING BEHAVIOR POLICY CLASSIFIERS (STEPWISE JOINT ACTION SPACE)")
print("="*70)

# Augment states with VP2 one-hot encoding (same as stepwise model uses)
print(f"\nAugmenting states with VP2 one-hot encoding ({N_VP2_BINS} bins)...")
# Training states
train_vp2_one_hot = np.zeros((len(train_data['states']), N_VP2_BINS))
for i in range(len(train_data['states'])):
    bin_idx = action_space.vp2_to_bin(train_data['actions'][i, 1])
    train_vp2_one_hot[i, bin_idx] = 1.0
train_states_extended = np.concatenate([train_data['states'], train_vp2_one_hot], axis=1)

# Test states
test_vp2_one_hot = np.zeros((len(test_data['states']), N_VP2_BINS))
for i in range(len(test_data['states'])):
    bin_idx = action_space.vp2_to_bin(test_data['actions'][i, 1])
    test_vp2_one_hot[i, bin_idx] = 1.0
test_states_extended = np.concatenate([test_data['states'], test_vp2_one_hot], axis=1)

print(f"   Original state dim: {train_data['states'].shape[1]}")
print(f"   Extended state dim: {train_states_extended.shape[1]} (base + VP2 one-hot)")

# Train softmax classifier for model stepwise actions
# Action space: VP1 (binary) × VP2 (stepwise directional changes)
print(f"\n1. Training softmax classifier for model stepwise actions ({n_actions} classes)...")
clf_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
clf_model.fit(train_states_extended, train_model_actions_stepwise)
train_acc_model = accuracy_score(train_model_actions_stepwise, clf_model.predict(train_states_extended))
test_acc_model = accuracy_score(test_model_actions_stepwise, clf_model.predict(test_states_extended))
print(f"   Train accuracy: {train_acc_model:.4f}")
print(f"   Test accuracy:  {test_acc_model:.4f}")

# Train softmax classifier for clinician stepwise actions
# Clinician actions converted to stepwise format (VP2 changes)
print(f"\n2. Training softmax classifier for clinician stepwise actions ({n_actions} classes)...")
clf_clinician = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
clf_clinician.fit(train_states_extended, train_clinician_actions_stepwise)
train_acc_clinician = accuracy_score(train_clinician_actions_stepwise, clf_clinician.predict(train_states_extended))
test_acc_clinician = accuracy_score(test_clinician_actions_stepwise, clf_clinician.predict(test_states_extended))
print(f"   Train accuracy: {train_acc_clinician:.4f}")
print(f"   Test accuracy:  {test_acc_clinician:.4f}")

# Get probabilities on test data
print("\n" + "="*70)
print("COMPUTING BEHAVIOR POLICY PROBABILITIES")
print("="*70)

# Model policy: π_model(a_clinician | s)
test_probs_model = clf_model.predict_proba(test_states_extended)
#test_probs_model = torch.nn.Softmax(dim=1)(test_q_values) + 1e-8
test_prob_model = test_probs_model[np.arange(len(test_clinician_actions_stepwise)), test_clinician_actions_stepwise.astype(int)]
# Clinician policy: π_clinician(a_clinician | s)
test_probs_clinician = clf_clinician.predict_proba(test_states_extended)
test_prob_clinician = test_probs_clinician[np.arange(len(test_clinician_actions_stepwise)), test_clinician_actions_stepwise.astype(int)]

print(f"\nTest probability statistics:")
print(f"  Model policy π(a_clinician|s)     - Mean: {test_prob_model.mean():.4f}, Std: {test_prob_model.std():.4f}, Min: {test_prob_model.min():.4f}, Max: {test_prob_model.max():.4f}")
print(f"  Clinician policy π(a_clinician|s) - Mean: {test_prob_clinician.mean():.4f}, Std: {test_prob_clinician.std():.4f}, Min: {test_prob_clinician.min():.4f}, Max: {test_prob_clinician.max():.4f}")

print(f"\nSample probabilities (first 10 test transitions):")
print(f"  Index | Model π | Clinician π | Ratio")
print(f"  " + "-"*50)
for i in range(min(10, len(test_prob_model))):
    ratio = test_prob_model[i] / (test_prob_clinician[i] + 1e-10)
    print(f"  {i:5d} | {test_prob_model[i]:.4f} | {test_prob_clinician[i]:.4f} | {ratio:.4f}") 


# Finally for each of the transitions (indices) in the test data, compute pi_lg_vp1_ma(s) / (eps + pi_lg_vp1_ca(s)) * pi_sm_vp2_ma(s) / (esp + pi_sm_vp2_ca(s)) * R where R is the reward on that transition index or state, this is the expected reward using the model actions on that state
# then compute the per transition average of the expected reward (using the model recommended actions), as well as the per patient average of the expected reward (using the model recommended actions)
# compare the two values with the average reward (the raw reward) per transition, and the average reward per patient
# print out the results and save it in latex table in a separate latex file

print("\n" + "="*70)
print("COMPUTING IMPORTANCE SAMPLING WEIGHTS AND REWARDS")
print("="*70)

# Set epsilon for numerical stability
eps = 1e-10

# Compute importance sampling weights (single ratio, no chain rule!)
# IS_weight = π_model(a_clinician|s) / π_clinician(a_clinician|s)
is_weight = test_prob_model / (test_prob_clinician + eps)

print(f"\nIS weight statistics:")
print(f"  Mean: {is_weight.mean():.4f}, Std: {is_weight.std():.4f}, Min: {is_weight.min():.4f}, Max: {is_weight.max():.4f}")

# Get test rewards
test_rewards = test_data['rewards']

# Compute IS-weighted rewards
is_weighted_rewards = is_weight * test_rewards

print(f"\nReward statistics:")
print(f"  Raw rewards (clinician)     - Mean: {test_rewards.mean():.4f}, Std: {test_rewards.std():.4f}, Sum: {test_rewards.sum():.2f}")
print(f"  IS-weighted rewards (model) - Mean: {is_weighted_rewards.mean():.4f}, Std: {is_weighted_rewards.std():.4f}, Sum: {is_weighted_rewards.sum():.2f}")

# Per-transition average
avg_raw_reward_per_transition = test_rewards.mean()
avg_is_reward_per_transition = is_weighted_rewards.mean()

print(f"\nPer-transition averages:")
print(f"  Clinician policy (raw):     {avg_raw_reward_per_transition:.4f}")
print(f"  Model policy (IS-weighted): {avg_is_reward_per_transition:.4f}")
print(f"  Difference:                 {avg_is_reward_per_transition - avg_raw_reward_per_transition:.4f}")

# Per-patient average (efficient computation)
test_patient_ids = test_data['patient_ids']
unique_patients = np.unique(test_patient_ids)
n_patients = len(unique_patients)

avg_raw_reward_per_patient = test_rewards.sum() / n_patients
avg_is_reward_per_patient = is_weighted_rewards.sum() / n_patients

print(f"\nPer-patient averages (efficient computation):")
print(f"  Clinician policy (raw):     {avg_raw_reward_per_patient:.4f}")
print(f"  Model policy (IS-weighted): {avg_is_reward_per_patient:.4f}")
print(f"  Difference:                 {avg_is_reward_per_patient - avg_raw_reward_per_patient:.4f}")

# Compute per-patient statistics (std, min, max) - requires loop
print(f"\nComputing per-patient statistics...")
patient_raw_rewards = []
patient_raw_rewards_ave = []

patient_is_rewards = []

for patient_id in unique_patients:
    patient_mask = test_patient_ids == patient_id
    patient_raw_reward = test_rewards[patient_mask].sum()
    patient_is_reward = is_weighted_rewards[patient_mask].sum()
    patient_raw_rewards.append(patient_raw_reward)
    patient_raw_rewards_ave.append(patient_raw_reward/len(test_rewards[patient_mask]))
    patient_is_rewards.append(patient_is_reward)

patient_raw_rewards = np.array(patient_raw_rewards)
patient_raw_rewards_ave = np.array(patient_raw_rewards_ave)
patient_is_rewards = np.array(patient_is_rewards)

print(f"\nPer-patient reward statistics:")
print(f"  Raw (clinician):    Mean: {patient_raw_rewards.mean():.4f}, Std: {patient_raw_rewards.std():.4f}, Min: {patient_raw_rewards.min():.2f}, Max: {patient_raw_rewards.max():.2f}")
print(f"  IS (model):         Mean: {patient_is_rewards.mean():.4f}, Std: {patient_is_rewards.std():.4f}, Min: {patient_is_rewards.min():.2f}, Max: {patient_is_rewards.max():.2f}")

# Create latex directory
os.makedirs('latex', exist_ok=True)

# Save results to LaTeX table
latex_output = r"""\begin{table}[h]
\centering
\caption{Importance Sampling Off-Policy Evaluation Results (Block Discrete CQL, $\alpha=0.0$)}
\label{tab:is_ope_results}
\begin{tabular}{lcc}
\hline
\textbf{Metric} & \textbf{Clinician Policy} & \textbf{Model Policy (IS)} \\
\hline
Per-Transition Avg Reward & %.4f & %.4f \\
Per-Patient Avg Reward & %.4f & %.4f \\
\hline
Per-Patient Reward Std & %.4f & %.4f \\
Per-Patient Reward Min & %.2f & %.2f \\
Per-Patient Reward Max & %.2f & %.2f \\
\hline
Number of Patients & %d & %d \\
Number of Transitions & %d & %d \\
\hline
\multicolumn{3}{l}{\textbf{Improvement (Model - Clinician)}} \\
\hline
Per-Transition & \multicolumn{2}{c}{%.4f} \\
Per-Patient & \multicolumn{2}{c}{%.4f} \\
\hline
\end{tabular}
\end{table}
""" % (
    avg_raw_reward_per_transition, avg_is_reward_per_transition,
    avg_raw_reward_per_patient, avg_is_reward_per_patient,
    patient_raw_rewards.std(), patient_is_rewards.std(),
    patient_raw_rewards.min(), patient_is_rewards.min(),
    patient_raw_rewards.max(), patient_is_rewards.max(),
    n_patients, n_patients,
    len(test_rewards), len(test_rewards),
    avg_is_reward_per_transition - avg_raw_reward_per_transition,
    avg_is_reward_per_patient - avg_raw_reward_per_patient
)

latex_file = 'latex/is_ope_results.tex'
with open(latex_file, 'w') as f:
    f.write(latex_output)

print(f"\n" + "="*70)
print(f"Results saved to: {latex_file}")
print("="*70)

# Weighted Importance Sampling (WIS) - Per-Transition Level
print("\n" + "="*70)
print("WEIGHTED IMPORTANCE SAMPLING (WIS) - PER-TRANSITION")
print("="*70)

# V_WIS = Σᵢ [wᵢ · rᵢ] / Σᵢ wᵢ
# where wᵢ is the IS weight for transition i, rᵢ is the reward for transition i

# Sum of weighted rewards
sum_weighted_rewards = (is_weight * test_rewards).sum()

# Sum of weights
sum_weights = is_weight.sum()

# WIS estimate
wis_per_transition = sum_weighted_rewards / sum_weights if sum_weights > 0 else 0.0

# Standard IS (for comparison)
is_per_transition = (is_weight * test_rewards).mean()

# Clinician average (raw)
clinician_per_transition = test_rewards.mean()

print(f"\nPer-transition evaluation:")
print(f"  Clinician policy (raw):        {clinician_per_transition:.4f}")
print(f"  Model policy (standard IS):    {is_per_transition:.4f}")
print(f"  Model policy (weighted IS):    {wis_per_transition:.4f}")

print(f"\nImprovement over clinician:")
print(f"  Standard IS:  {is_per_transition - clinician_per_transition:.4f}")
print(f"  Weighted IS:  {wis_per_transition - clinician_per_transition:.4f}")

print(f"\nWeight normalization:")
print(f"  Sum of weights: {sum_weights:.4f}")
print(f"  Number of transitions: {len(test_rewards)}")
print(f"  Average weight: {sum_weights / len(test_rewards):.4f}")

# (4) Weighted Importance Sampling (WIS) - Per-Trajectory Level
print("\n" + "="*70)
print("WEIGHTED IMPORTANCE SAMPLING (WIS) - PER-TRAJECTORY")
print("="*70)

# For each patient trajectory: WIS_τ = Σᵢ(wᵢ·rᵢ) / Σᵢ(wᵢ) for transitions in that trajectory
# Then average across all patients

weights_per_trajectory_list = [] 
total_rewards_per_trajectory_list = []
weighted_rewards_per_trajectory_list = []

for patient_id in unique_patients:
    patient_mask = test_patient_ids == patient_id
    """
    mean_model_prob = test_prob_model[patient_mask].mean()
    mean_clinician_prob = test_prob_clinician[patient_mask].mean()
    patient_rewards = test_rewards[patient_mask]
    traj_is_weight = mean_model_prob /(1e-8 + mean_clinician_prob)

    if not traj_is_weight == 0:
        #weights_per_trajectory_list.append(patient_weights.prod())
        weights_per_trajectory_list.append(traj_is_weight)
        total_rewards_per_trajectory_list.append(patient_rewards.sum())
    else: 
        print("zero encountered in trajectory level multiplied weights")
    """
    
    # Get weights and rewards for this trajectory
    patient_weights = is_weight[patient_mask]
    patient_rewards = test_rewards[patient_mask]
    est_total_reward_per_traj = (patient_weights *  patient_rewards).sum() / patient_weights.sum() * len(patient_weights)
    
    weights_per_trajectory_list.append(patient_weights.mean())
    total_rewards_per_trajectory_list.append(patient_rewards.sum())
    weighted_rewards_per_trajectory_list.append(est_total_reward_per_traj)
    

#wisw_ci_diff_lower = np.percentile(weights_per_trajectory_list, 5)
#wisw_ci_diff_upper = np.percentile(weights_per_trajectory_list, 95)

#weights_per_trajectory_list = np.clip(weights_per_trajectory_list, a_min = wisw_ci_diff_lower, a_max = wisw_ci_diff_upper)

#weights_per_trajectory_list = np.clip(weights_per_trajectory_list, a_min = 0.0, a_max = 10)

# Compute mean WIS across all trajectories
weights_per_trajectory_list = np.array(weights_per_trajectory_list)
total_rewards_per_trajectory_list = np.array(total_rewards_per_trajectory_list)
weighted_rewards_per_trajectory_list = np.array(weighted_rewards_per_trajectory_list)

wis_trajectory_level = (weights_per_trajectory_list * weighted_rewards_per_trajectory_list).sum() / weights_per_trajectory_list.sum() 

# Also compute raw clinician per-trajectory for comparison
clinician_per_trajectory_mean = total_rewards_per_trajectory_list.mean()

# Compute 95% confidence interval using bootstrapping for the DIFFERENCE
print(f"\nComputing 95% CI for difference using bootstrapping (1000 iterations)...")

# First, compute the difference for each trajectory
# For clinician: just the raw reward sum per patient
# For model: WIS per trajectory (already computed)
# Difference: s = WIS_model - raw_clinician for each patient

# Bootstrap from the bag of differences
n_bootstrap = 1000
bootstrap_differences = []

np.random.seed(42)  # For reproducibility
for _ in range(n_bootstrap):
    # Resample differences with replacement
    bootstrap_indices = np.random.choice(len(total_rewards_per_trajectory_list), size=len(total_rewards_per_trajectory_list), replace=True)

    bootstrap_weights = weights_per_trajectory_list[bootstrap_indices]
    bootstrap_rewards = total_rewards_per_trajectory_list[bootstrap_indices]
    weighted_bootstrap_rewards = weighted_rewards_per_trajectory_list[bootstrap_indices]

    if np.sum(bootstrap_weights) > 0:
        bootstrap_wis = np.sum(bootstrap_weights * weighted_bootstrap_rewards) / np.sum(bootstrap_weights)
    else:
        bootstrap_wis = 0
    
    bootstrap_behavior = np.mean(bootstrap_rewards)
    bootstrap_diff = bootstrap_wis - bootstrap_behavior
    bootstrap_differences.append(bootstrap_diff)

bootstrap_differences = np.array(bootstrap_differences)

# Compute 95% CI using percentile method
ci_diff_lower = np.percentile(bootstrap_differences, 2.5)
ci_diff_upper = np.percentile(bootstrap_differences, 97.5)

# Mean difference
improvement_mean = wis_trajectory_level - clinician_per_trajectory_mean

print(f"\nPer-trajectory WIS evaluation with 95% CI:")
print(f"  Clinician policy (raw):        {clinician_per_trajectory_mean:.4f}")
print(f"  Model policy (WIS):            {wis_trajectory_level:.4f}")
print(f"  Difference (Model - Clinician): {improvement_mean:.4f} (95% CI: [{ci_diff_lower:.4f}, {ci_diff_upper:.4f}])")
