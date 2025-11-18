### import relevant libraries
import torch
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

# Define constants
STATE_DIM = 17  # 17 state features for dual model
ACTION_BINS_5 = [0, 0.05, 0.1, 0.2, 0.5]  # VP2 discretization bins 


### define the block discrete model
# - load the trained block discrete model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Import the network class from the training script
from run_block_discrete_cql_allalphas import DualBlockDiscreteQNetwork

# Model parameters
n_bins = 5
action_bins = ACTION_BINS_5  # [0, 0.05, 0.1, 0.2, 0.5]
n_actions = 2 * n_bins  # VP1 (2 options: 0,1) x VP2 (5 bins) = 10 total actions
state_dim = STATE_DIM  # 17 features

# Initialize Q-networks (we use both Q1 and Q2, then take min)
q1_network = DualBlockDiscreteQNetwork(state_dim=state_dim, vp2_bins=n_bins).to(device)
q2_network = DualBlockDiscreteQNetwork(state_dim=state_dim, vp2_bins=n_bins).to(device)

# Load the trained model checkpoint
model_path = '/scratch/zouwil/code/ucsf_rl/experiment/block_discrete_cql_alpha0.0000_bins5_best.pt'
checkpoint = torch.load(model_path, map_location=device)
q1_network.load_state_dict(checkpoint['q1_state_dict'])
q2_network.load_state_dict(checkpoint['q2_state_dict'])
q1_network.eval()
q2_network.eval()

print(f"Loaded model from: {model_path}")
print(f"State dimension: {state_dim}")
print(f"Number of actions: {n_actions} (VP1: 2 binary × VP2: {n_bins} bins)")
print(f"Action bins for VP2: {action_bins}")
print("Using min(Q1, Q2) for action selection (double Q-learning)") 



###
# prepare and load the data with random seed 42 with all the data in a sequence, for example:
# pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
# train_data, val_data, test_data = pipeline.prepare_data()

print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

# Initialize data pipeline with random seed 42
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
print(f"  States:  {train_data['states'].shape}")
print(f"  Actions: {train_data['actions'].shape}")
print(f"  Rewards: {train_data['rewards'].shape}")


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

# Helper function to select actions using the loaded Q-networks
def select_action_batch(states, q1_net, q2_net, vp2_bins, vp2_edges, device):
    """
    Select best actions for a batch of states using min(Q1, Q2)
    Exactly mirrors the select_action method in DualBlockDiscreteCQL
    Returns: numpy array of shape [batch_size, 2] with [vp1, vp2] actions
    """
    with torch.no_grad():
        if states.ndim == 1:
            states = states.reshape(1, -1)

        batch_size = states.shape[0]
        state_tensor = torch.FloatTensor(states).to(device)

        # Create all possible discrete actions for each state in batch
        # Shape: [batch_size, total_actions]
        total_actions = 2 * vp2_bins
        all_actions = torch.arange(total_actions).to(device)
        all_actions = all_actions.unsqueeze(0).expand(batch_size, -1)

        # Expand states to match actions
        # Shape: [batch_size * total_actions, state_dim]
        state_expanded = state_tensor.unsqueeze(1).expand(-1, total_actions, -1)
        state_expanded = state_expanded.reshape(-1, state_dim)

        # Flatten actions for network input
        # Shape: [batch_size * total_actions]
        actions_flat = all_actions.reshape(-1)

        # Compute Q-values for all actions
        q1_values = q1_net(state_expanded, actions_flat).reshape(batch_size, total_actions)
        q2_values = q2_net(state_expanded, actions_flat).reshape(batch_size, total_actions)
        q_values = torch.min(q1_values, q2_values)

        # Get best action for each batch element
        best_action_indices = q_values.argmax(dim=1).cpu().numpy()

        # Vectorized conversion from discrete indices to continuous actions
        # Extract VP1 (binary) and VP2 bin indices
        vp1_actions = (best_action_indices // vp2_bins).astype(float)
        vp2_bin_indices = best_action_indices % vp2_bins

        # Convert VP2 bin indices to continuous values using bin centers
        vp2_bin_centers = (vp2_edges[:-1] + vp2_edges[1:]) / 2
        vp2_actions = vp2_bin_centers[vp2_bin_indices]

        # Stack VP1 and VP2 into action array
        actions = np.stack([vp1_actions, vp2_actions], axis=1)

        return actions if batch_size > 1 else actions[0]

# Get model actions for train and test data
print("Computing model actions for training data...")
train_model_actions = select_action_batch(train_data['states'], q1_network, q2_network, n_bins, vp2_bin_edges, device)

print("Computing model actions for test data...")
test_model_actions = select_action_batch(test_data['states'], q1_network, q2_network, n_bins, vp2_bin_edges, device)

# Get clinician actions (already in the data)
train_clinician_actions = train_data['actions']
test_clinician_actions = test_data['actions']

print(f"\nModel actions generated:")
print(f"  Train: {train_model_actions.shape}")
print(f"  Test:  {test_model_actions.shape}")

print(f"\nClinician actions:")
print(f"  Train: {train_clinician_actions.shape}")
print(f"  Test:  {test_clinician_actions.shape}")

# Display sample actions
print(f"\nSample model actions (first 5):")
print(train_model_actions[:5])
print(f"\nSample clinician actions (first 5):")
print(train_clinician_actions[:5])

# Separate VP1 and VP2 for training classifiers
train_model_vp1 = train_model_actions[:, 0]
train_model_vp2 = train_model_actions[:, 1]
train_clinician_vp1 = train_clinician_actions[:, 0]
train_clinician_vp2 = train_clinician_actions[:, 1]

test_model_vp1 = test_model_actions[:, 0]
test_model_vp2 = test_model_actions[:, 1]
test_clinician_vp1 = test_clinician_actions[:, 0]
test_clinician_vp2 = test_clinician_actions[:, 1]

print(f"\nVP1 (binary) distribution:")
print(f"  Model (train):     VP1=0: {(train_model_vp1==0).sum()}, VP1=1: {(train_model_vp1==1).sum()}")
print(f"  Clinician (train): VP1=0: {(train_clinician_vp1==0).sum()}, VP1=1: {(train_clinician_vp1==1).sum()}")
print(f"  Model (test):      VP1=0: {(test_model_vp1==0).sum()}, VP1=1: {(test_model_vp1==1).sum()}")
print(f"  Clinician (test):  VP1=0: {(test_clinician_vp1==0).sum()}, VP1=1: {(test_clinician_vp1==1).sum()}")

# Convert VP2 continuous values to discrete bin indices for classifier training
def vp2_continuous_to_discrete(vp2_values, vp2_edges):
    """Convert continuous VP2 values to discrete bin indices (0 to n_bins-1)"""
    bin_indices = np.digitize(vp2_values, vp2_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(vp2_edges) - 2)
    return bin_indices

# Convert VP2 to discrete for classifiers
train_model_vp2_discrete = vp2_continuous_to_discrete(train_model_vp2, vp2_bin_edges)
train_clinician_vp2_discrete = vp2_continuous_to_discrete(train_clinician_vp2, vp2_bin_edges)
test_model_vp2_discrete = vp2_continuous_to_discrete(test_model_vp2, vp2_bin_edges)
test_clinician_vp2_discrete = vp2_continuous_to_discrete(test_clinician_vp2, vp2_bin_edges)

print(f"\nVP2 (discrete bins) distribution:")
for bin_idx in range(n_bins):
    print(f"  Bin {bin_idx}: Model (train)={np.sum(train_model_vp2_discrete==bin_idx)}, "
          f"Clinician (train)={np.sum(train_clinician_vp2_discrete==bin_idx)}, "
          f"Model (test)={np.sum(test_model_vp2_discrete==bin_idx)}, "
          f"Clinician (test)={np.sum(test_clinician_vp2_discrete==bin_idx)}") 

## train a logistic classifier (lg_vp1_ma) to predict the vp1 model actions (binary) using the states on the training data, then use it on the test data to get the probability of clinician vp1 actions
## train a logistic classifier (lg_vp1_ca) to predict the vp1 clinician actions (binary) using the states on the training data, then use it on the test data to get the probability of clinician vp1 actions
## train a softmax classifier (sm_vp2_ma) to predict the vp2 model actions (block discrete) using the states on the training data, then use it on the test data to get the probability of clinician vp2 actions
## train a softmax classifier (sm_vp2_ca) to predict the vp2 clinician (block discrete) using the states on the training data, then use it on the test data to get the probability of clinician vp2 actions
## save the test data probabilities, and display some of them (each of the four)

print("\n" + "="*70)
print("TRAINING BEHAVIOR POLICY CLASSIFIERS")
print("="*70)

MAX_ITER = 100
# Train logistic classifier for VP1 model actions
print("\n1. Training logistic classifier for VP1 model actions (lg_vp1_ma)...")
lg_vp1_ma = LogisticRegression(max_iter=MAX_ITER, random_state=42)
lg_vp1_ma.fit(train_data['states'], train_model_vp1)
train_acc_vp1_ma = accuracy_score(train_model_vp1, lg_vp1_ma.predict(train_data['states']))
test_acc_vp1_ma = accuracy_score(test_model_vp1, lg_vp1_ma.predict(test_data['states']))
print(f"   Train accuracy: {train_acc_vp1_ma:.4f}")
print(f"   Test accuracy:  {test_acc_vp1_ma:.4f}")

# Train logistic classifier for VP1 clinician actions
print("\n2. Training logistic classifier for VP1 clinician actions (lg_vp1_ca)...")
lg_vp1_ca = LogisticRegression(max_iter=MAX_ITER, random_state=42)
lg_vp1_ca.fit(train_data['states'], train_clinician_vp1)
train_acc_vp1_ca = accuracy_score(train_clinician_vp1, lg_vp1_ca.predict(train_data['states']))
test_acc_vp1_ca = accuracy_score(test_clinician_vp1, lg_vp1_ca.predict(test_data['states']))
print(f"   Train accuracy: {train_acc_vp1_ca:.4f}")
print(f"   Test accuracy:  {test_acc_vp1_ca:.4f}")

# Train softmax classifier for VP2 model actions
print("\n3. Training softmax classifier for VP2 model actions (sm_vp2_ma)...")
sm_vp2_ma = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=MAX_ITER, random_state=42)
sm_vp2_ma.fit(train_data['states'], train_model_vp2_discrete)
train_acc_vp2_ma = accuracy_score(train_model_vp2_discrete, sm_vp2_ma.predict(train_data['states']))
test_acc_vp2_ma = accuracy_score(test_model_vp2_discrete, sm_vp2_ma.predict(test_data['states']))
print(f"   Train accuracy: {train_acc_vp2_ma:.4f}")
print(f"   Test accuracy:  {test_acc_vp2_ma:.4f}")

# Train softmax classifier for VP2 clinician actions
print("\n4. Training softmax classifier for VP2 clinician actions (sm_vp2_ca)...")
sm_vp2_ca = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=MAX_ITER, random_state=42)
sm_vp2_ca.fit(train_data['states'], train_clinician_vp2_discrete)
train_acc_vp2_ca = accuracy_score(train_clinician_vp2_discrete, sm_vp2_ca.predict(train_data['states']))
test_acc_vp2_ca = accuracy_score(test_clinician_vp2_discrete, sm_vp2_ca.predict(test_data['states']))
print(f"   Train accuracy: {train_acc_vp2_ca:.4f}")
print(f"   Test accuracy:  {test_acc_vp2_ca:.4f}")

# Get probabilities on test data
print("\n" + "="*70)
print("COMPUTING BEHAVIOR POLICY PROBABILITIES")
print("="*70)

# VP1 model action probabilities - get probability of CLINICIAN actions under MODEL policy
test_probs_vp1_ma = lg_vp1_ma.predict_proba(test_data['states'])
test_prob_vp1_ma = test_probs_vp1_ma[np.arange(len(test_clinician_vp1)), test_clinician_vp1.astype(int)]

# VP1 clinician action probabilities - get probability of CLINICIAN actions under CLINICIAN policy
test_probs_vp1_ca = lg_vp1_ca.predict_proba(test_data['states'])
test_prob_vp1_ca = test_probs_vp1_ca[np.arange(len(test_clinician_vp1)), test_clinician_vp1.astype(int)]

# VP2 model action probabilities - get probability of CLINICIAN actions under MODEL policy
test_probs_vp2_ma = sm_vp2_ma.predict_proba(test_data['states'])
test_prob_vp2_ma = test_probs_vp2_ma[np.arange(len(test_clinician_vp2_discrete)), test_clinician_vp2_discrete.astype(int)]

# VP2 clinician action probabilities - get probability of CLINICIAN actions under CLINICIAN policy
test_probs_vp2_ca = sm_vp2_ca.predict_proba(test_data['states'])
test_prob_vp2_ca = test_probs_vp2_ca[np.arange(len(test_clinician_vp2_discrete)), test_clinician_vp2_discrete.astype(int)]

print(f"\nTest probability statistics:")
print(f"  VP1 model action probs     - Mean: {test_prob_vp1_ma.mean():.4f}, Std: {test_prob_vp1_ma.std():.4f}, Min: {test_prob_vp1_ma.min():.4f}, Max: {test_prob_vp1_ma.max():.4f}")
print(f"  VP1 clinician action probs - Mean: {test_prob_vp1_ca.mean():.4f}, Std: {test_prob_vp1_ca.std():.4f}, Min: {test_prob_vp1_ca.min():.4f}, Max: {test_prob_vp1_ca.max():.4f}")
print(f"  VP2 model action probs     - Mean: {test_prob_vp2_ma.mean():.4f}, Std: {test_prob_vp2_ma.std():.4f}, Min: {test_prob_vp2_ma.min():.4f}, Max: {test_prob_vp2_ma.max():.4f}")
print(f"  VP2 clinician action probs - Mean: {test_prob_vp2_ca.mean():.4f}, Std: {test_prob_vp2_ca.std():.4f}, Min: {test_prob_vp2_ca.min():.4f}, Max: {test_prob_vp2_ca.max():.4f}")

print(f"\nSample probabilities (first 10 test transitions):")
print(f"  Index | VP1_MA  | VP1_CA  | VP2_MA  | VP2_CA")
print(f"  " + "-"*50)
for i in range(min(10, len(test_prob_vp1_ma))):
    print(f"  {i:5d} | {test_prob_vp1_ma[i]:.4f} | {test_prob_vp1_ca[i]:.4f} | {test_prob_vp2_ma[i]:.4f} | {test_prob_vp2_ca[i]:.4f}") 


# Finally for each of the transitions (indices) in the test data, compute pi_lg_vp1_ma(s) / (eps + pi_lg_vp1_ca(s)) * pi_sm_vp2_ma(s) / (esp + pi_sm_vp2_ca(s)) * R where R is the reward on that transition index or state, this is the expected reward using the model actions on that state
# then compute the per transition average of the expected reward (using the model recommended actions), as well as the per patient average of the expected reward (using the model recommended actions)
# compare the two values with the average reward (the raw reward) per transition, and the average reward per patient
# print out the results and save it in latex table in a separate latex file

print("\n" + "="*70)
print("COMPUTING IMPORTANCE SAMPLING WEIGHTS AND REWARDS")
print("="*70)

# Set epsilon for numerical stability
eps = 1e-10

# Compute importance sampling weights
# IS_weight = [π_model(a_clinician|s) / π_clinician(a_clinician|s)] for VP1 and VP2
is_weight_vp1 = test_prob_vp1_ma / (test_prob_vp1_ca + eps)
is_weight_vp2 = test_prob_vp2_ma / (test_prob_vp2_ca + eps)
is_weight_total = is_weight_vp1 * is_weight_vp2

print(f"\nIS weight statistics:")
print(f"  VP1 weights     - Mean: {is_weight_vp1.mean():.4f}, Std: {is_weight_vp1.std():.4f}, Min: {is_weight_vp1.min():.4f}, Max: {is_weight_vp1.max():.4f}")
print(f"  VP2 weights     - Mean: {is_weight_vp2.mean():.4f}, Std: {is_weight_vp2.std():.4f}, Min: {is_weight_vp2.min():.4f}, Max: {is_weight_vp2.max():.4f}")
print(f"  Total weights   - Mean: {is_weight_total.mean():.4f}, Std: {is_weight_total.std():.4f}, Min: {is_weight_total.min():.4f}, Max: {is_weight_total.max():.4f}")

# Get test rewards
test_rewards = test_data['rewards']

# Compute IS-weighted rewards
is_weighted_rewards = is_weight_total * test_rewards

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
patient_is_rewards = []

for patient_id in unique_patients:
    patient_mask = test_patient_ids == patient_id
    patient_raw_reward = test_rewards[patient_mask].sum()
    patient_is_reward = is_weighted_rewards[patient_mask].sum()
    patient_raw_rewards.append(patient_raw_reward)
    patient_is_rewards.append(patient_is_reward)

patient_raw_rewards = np.array(patient_raw_rewards)
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
sum_weighted_rewards = (is_weight_total * test_rewards).sum()

# Sum of weights
sum_weights = is_weight_total.sum()

# WIS estimate
wis_per_transition = sum_weighted_rewards / sum_weights if sum_weights > 0 else 0.0

# Standard IS (for comparison)
is_per_transition = (is_weight_total * test_rewards).mean()

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
