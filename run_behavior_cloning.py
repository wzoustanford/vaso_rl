#!/usr/bin/env python3
"""
Behavior cloning trainer for block-discrete dual vasopressor actions.

The policy learns pi(a | s) by supervised classification on expert clinician
transitions. Continuous expert actions [VP1, VP2] are discretized into the same
joint action space used by the block-discrete baselines:

    action_idx = VP1 * vp2_bins + VP2_bin
"""

import argparse
import os
import random
import sys
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from integrated_data_pipeline_v3 import IntegratedDataPipelineV3


sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BehaviorCloningPolicy(nn.Module):
    """
    MLP policy for block-discrete actions.

    Input:
        state: [batch_size, state_dim]
    Output:
        logits: [batch_size, total_actions]
    """

    def __init__(
        self,
        state_dim: int,
        vp2_bins: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.vp2_bins = vp2_bins
        self.total_actions = 2 * vp2_bins
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.total_actions),
        )

        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return unnormalized action logits."""
        return self.net(states)

    def action_probabilities(self, states: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(states), dim=-1)

    def select_action_discrete(self, states: np.ndarray, device: torch.device = None) -> np.ndarray:
        """Return greedy block-discrete action indices."""
        was_single = states.ndim == 1
        if was_single:
            states = states.reshape(1, -1)

        if device is None:
            device = next(self.parameters()).device

        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(states).to(device)
            action_idx = self.forward(state_tensor).argmax(dim=1).cpu().numpy()

        return action_idx[0] if was_single else action_idx

    def select_action(self, states: np.ndarray, device: torch.device = None) -> np.ndarray:
        """Return greedy actions converted to continuous [VP1, VP2] bin centers."""
        action_idx = self.select_action_discrete(states, device)
        was_single = np.isscalar(action_idx)
        action_idx = np.asarray([action_idx]) if was_single else np.asarray(action_idx)
        actions = discrete_to_continuous_actions(action_idx, self.vp2_bins)
        return actions[0] if was_single else actions


class BehaviorCloningAgent:
    """Supervised behavior cloning wrapper around the block-discrete policy."""

    def __init__(
        self,
        state_dim: int,
        vp2_bins: int = 5,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        grad_clip: float = 1.0,
        device: torch.device = None,
    ):
        self.state_dim = state_dim
        self.vp2_bins = vp2_bins
        self.total_actions = 2 * vp2_bins
        self.hidden_dim = hidden_dim
        self.grad_clip = grad_clip
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = BehaviorCloningPolicy(
            state_dim=state_dim,
            vp2_bins=vp2_bins,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def update(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, float]:
        target_actions = torch.LongTensor(
            continuous_to_discrete_actions(actions.detach().cpu().numpy(), self.vp2_bins)
        ).to(self.device)
        logits = self.policy(states)
        loss = self.loss_fn(logits, target_actions)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            pred_actions = logits.argmax(dim=1)
            accuracy = (pred_actions == target_actions).float().mean().item()
            expert_prob = probs.gather(1, target_actions.unsqueeze(1)).mean().item()
            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean().item()

        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "expert_prob": expert_prob,
            "entropy": entropy,
        }

    def evaluate(self, states: np.ndarray, actions: np.ndarray, batch_size: int = 4096) -> Dict[str, float]:
        self.policy.eval()
        loss_sum = 0.0
        correct_sum = 0.0
        expert_prob_sum = 0.0
        entropy_sum = 0.0
        n_samples = len(states)

        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                state_batch = torch.FloatTensor(states[start:end]).to(self.device)
                target_actions = torch.LongTensor(
                    continuous_to_discrete_actions(actions[start:end], self.vp2_bins)
                ).to(self.device)
                logits = self.policy(state_batch)
                probs = torch.softmax(logits, dim=-1)
                pred_actions = logits.argmax(dim=1)

                batch_size_actual = end - start
                loss = self.loss_fn(logits, target_actions)
                entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1)

                loss_sum += loss.item() * batch_size_actual
                correct_sum += (pred_actions == target_actions).float().sum().item()
                expert_prob_sum += probs.gather(1, target_actions.unsqueeze(1)).sum().item()
                entropy_sum += entropy.sum().item()

        return {
            "loss": loss_sum / n_samples,
            "accuracy": correct_sum / n_samples,
            "expert_prob": expert_prob_sum / n_samples,
            "entropy": entropy_sum / n_samples,
        }

    def select_action(self, states: np.ndarray) -> np.ndarray:
        return self.policy.select_action(states, self.device)

    def select_action_discrete(self, states: np.ndarray) -> np.ndarray:
        return self.policy.select_action_discrete(states, self.device)

    def save(self, filepath: str, extra: Dict = None) -> None:
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "state_dim": self.state_dim,
            "vp2_bins": self.vp2_bins,
            "total_actions": self.total_actions,
            "hidden_dim": self.hidden_dim,
            "dropout": self.policy.dropout,
            "model_type": "behavior_cloning_block_discrete",
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, filepath)

    @classmethod
    def load(cls, filepath: str, device: torch.device = None) -> "BehaviorCloningAgent":
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(filepath, map_location=device)
        agent = cls(
            state_dim=checkpoint["state_dim"],
            vp2_bins=checkpoint.get("vp2_bins", 5),
            hidden_dim=checkpoint.get("hidden_dim", 128),
            dropout=checkpoint.get("dropout", 0.0),
            device=device,
        )
        agent.policy.load_state_dict(checkpoint["policy_state_dict"])
        agent.policy.eval()
        return agent


def continuous_to_discrete_actions(
    actions: np.ndarray,
    vp2_bins: int,
    threshold_vp1: bool = False,
) -> np.ndarray:
    """Convert continuous [VP1, VP2] actions to block-discrete joint indices."""
    vp2_edges = np.linspace(0.0, 0.5, vp2_bins + 1)
    if threshold_vp1:
        vp1 = (actions[:, 0] >= 0.5).astype(int)
    else:
        vp1 = actions[:, 0].astype(int)
    vp2 = np.clip(actions[:, 1], 0.0, 0.5)
    vp2_idx = np.digitize(vp2, vp2_edges) - 1
    vp2_idx = np.clip(vp2_idx, 0, vp2_bins - 1)
    return vp1 * vp2_bins + vp2_idx


def discrete_to_continuous_actions(action_idx: np.ndarray, vp2_bins: int) -> np.ndarray:
    """Convert block-discrete action indices to [VP1, VP2] bin centers."""
    action_idx = np.asarray(action_idx).astype(int)
    vp2_edges = np.linspace(0.0, 0.5, vp2_bins + 1)
    vp1 = (action_idx // vp2_bins).astype(np.float32)
    vp2_bin = action_idx % vp2_bins
    vp2 = ((vp2_edges[vp2_bin] + vp2_edges[vp2_bin + 1]) / 2.0).astype(np.float32)
    return np.stack([vp1, vp2], axis=1)


def train_discrete_probability_model(states: np.ndarray, actions: np.ndarray):
    """Train a multinomial classifier, with a constant fallback for one-class data."""
    actions = actions.astype(int)
    classes = np.unique(actions)
    if len(classes) == 1:
        return {"type": "constant", "class": int(classes[0])}

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    clf.fit(states, actions)
    return {"type": "multinomial", "model": clf}


def predict_discrete_action_probability(
    prob_model,
    states: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """Return P(model_action == requested discrete action | state)."""
    actions = actions.astype(int)
    if prob_model["type"] == "constant":
        probs = np.full(len(actions), 1e-10)
        probs[actions == prob_model["class"]] = 1.0
        return probs

    clf = prob_model["model"]
    pred_proba = clf.predict_proba(states)
    probs = np.full(len(actions), 1e-10)
    for col_idx, class_value in enumerate(clf.classes_.astype(int)):
        mask = actions == class_value
        probs[mask] = pred_proba[mask, col_idx]
    return probs


def discrete_probability_accuracy(prob_model, states: np.ndarray, target_actions: np.ndarray) -> float:
    target_actions = target_actions.astype(int)
    if prob_model["type"] == "constant":
        pred = np.full(len(target_actions), prob_model["class"], dtype=int)
    else:
        pred = prob_model["model"].predict(states)
    return accuracy_score(target_actions, pred)


def compute_wis_metrics(
    is_weight: np.ndarray,
    rewards: np.ndarray,
    patient_ids: np.ndarray,
    clip_lower: float,
    clip_upper: float,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute transition-level WIS and Method 2 trajectory-level WIS metrics."""
    is_weight = np.asarray(is_weight, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    patient_ids = np.asarray(patient_ids)

    clipped_weight = np.clip(is_weight, clip_lower, clip_upper)
    clinician_transition = rewards.mean()
    standard_is_transition = (clipped_weight * rewards).mean()
    sum_weights = clipped_weight.sum()
    wis_transition = (clipped_weight * rewards).sum() / sum_weights if sum_weights > 0 else 0.0

    # Method 2 from is_block_discrete.py:
    # R_j = sum_t r_t, w_j = prod_t ratio_t, R_WIS = sum_j w_j R_j / sum_j w_j.
    unique_patients = np.unique(patient_ids)
    weights_per_trajectory = []
    total_rewards_per_trajectory = []

    for patient_id in unique_patients:
        patient_mask = patient_ids == patient_id
        patient_step_ratios = clipped_weight[patient_mask]
        patient_rewards = rewards[patient_mask]

        trajectory_weight = np.prod(patient_step_ratios)
        trajectory_weight = np.clip(trajectory_weight, clip_lower, clip_upper)

        total_rewards_per_trajectory.append(patient_rewards.sum())
        weights_per_trajectory.append(trajectory_weight)

    weights_per_trajectory = np.asarray(weights_per_trajectory, dtype=np.float64)
    total_rewards_per_trajectory = np.asarray(total_rewards_per_trajectory, dtype=np.float64)
    weighted_rewards_per_trajectory = total_rewards_per_trajectory.copy()

    if weights_per_trajectory.sum() > 0:
        wis_trajectory = (
            weights_per_trajectory * total_rewards_per_trajectory
        ).sum() / weights_per_trajectory.sum()
    else:
        wis_trajectory = 0.0

    clinician_trajectory = total_rewards_per_trajectory.mean()

    rng = np.random.RandomState(seed)
    bootstrap_differences = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(total_rewards_per_trajectory), size=len(total_rewards_per_trajectory), replace=True)
        bootstrap_weights = weights_per_trajectory[idx]
        bootstrap_clinician_rewards = total_rewards_per_trajectory[idx]
        bootstrap_weighted_rewards = weighted_rewards_per_trajectory[idx]

        if bootstrap_weights.sum() > 0:
            bootstrap_wis = (bootstrap_weights * bootstrap_weighted_rewards).sum() / bootstrap_weights.sum()
        else:
            bootstrap_wis = 0.0
        bootstrap_differences.append(bootstrap_wis - bootstrap_clinician_rewards.mean())

    bootstrap_differences = np.asarray(bootstrap_differences)
    transition_square_sum = np.square(clipped_weight).sum()
    trajectory_square_sum = np.square(weights_per_trajectory).sum()
    ess_transition = clipped_weight.sum() ** 2 / transition_square_sum if transition_square_sum > 0 else 0.0
    ess_trajectory = (
        weights_per_trajectory.sum() ** 2 / trajectory_square_sum
        if trajectory_square_sum > 0 else 0.0
    )

    return {
        "trajectory_wis_method": "method2_product_ratios",
        "clinician_transition": clinician_transition,
        "standard_is_transition": standard_is_transition,
        "wis_transition": wis_transition,
        "wis_transition_diff": wis_transition - clinician_transition,
        "clinician_trajectory": clinician_trajectory,
        "wis_trajectory": wis_trajectory,
        "wis_trajectory_diff": wis_trajectory - clinician_trajectory,
        "wis_trajectory_diff_ci_lower": np.percentile(bootstrap_differences, 2.5),
        "wis_trajectory_diff_ci_upper": np.percentile(bootstrap_differences, 97.5),
        "sum_weights": sum_weights,
        "mean_weight": clipped_weight.mean(),
        "min_weight": clipped_weight.min(),
        "max_weight": clipped_weight.max(),
        "ess_transition": ess_transition,
        "ess_transition_frac": ess_transition / len(clipped_weight),
        "ess_trajectory": ess_trajectory,
        "ess_trajectory_frac": ess_trajectory / len(weights_per_trajectory),
        "n_transitions": len(rewards),
        "n_patients": len(unique_patients),
    }


def evaluate_behavior_cloning_wis(
    agent: BehaviorCloningAgent,
    train_data: Dict,
    eval_data: Dict,
    eval_set_name: str,
    clip_lower_pct: float = 0.5,
    clip_upper_pct: float = 99.5,
    n_bootstrap: int = 1000,
    save_dir: str = "experiment/behavior_cloning",
    experiment_prefix: str = "bc",
    seed: int = 42,
    vp2_bins: int = 5,
) -> Dict[str, float]:
    """
    Estimate WIS for the behavior cloning policy using the same block-discrete
    joint action approximation as is_block_discrete_gail.py.
    """
    print("\n" + "=" * 70, flush=True)
    print(f"WEIGHTED IMPORTANCE SAMPLING EVALUATION ({eval_set_name.upper()})", flush=True)
    print("=" * 70, flush=True)

    n_actions = 2 * vp2_bins
    train_model_actions_discrete = agent.select_action_discrete(train_data["states"])
    eval_model_actions_discrete = agent.select_action_discrete(eval_data["states"])
    train_clinician_actions_discrete = continuous_to_discrete_actions(
        train_data["actions"], vp2_bins=vp2_bins, threshold_vp1=False
    )
    eval_clinician_actions_discrete = continuous_to_discrete_actions(
        eval_data["actions"], vp2_bins=vp2_bins, threshold_vp1=False
    )

    print("\nDiscrete action indices generated:", flush=True)
    print(f"  VP2 bins: {vp2_bins}", flush=True)
    print(f"  Total joint actions: {n_actions}", flush=True)
    print(f"  Train model actions:     {train_model_actions_discrete.shape}", flush=True)
    print(f"  Train clinician actions: {train_clinician_actions_discrete.shape}", flush=True)
    print(f"  {eval_set_name} model actions:      {eval_model_actions_discrete.shape}", flush=True)
    print(f"  {eval_set_name} clinician actions:  {eval_clinician_actions_discrete.shape}", flush=True)

    print(f"\nAction distribution (0-{n_actions - 1}):", flush=True)
    for action_idx in range(n_actions):
        print(
            f"  Action {action_idx}: "
            f"Model (train)={np.sum(train_model_actions_discrete == action_idx)}, "
            f"Clinician (train)={np.sum(train_clinician_actions_discrete == action_idx)}, "
            f"Model ({eval_set_name})={np.sum(eval_model_actions_discrete == action_idx)}, "
            f"Clinician ({eval_set_name})={np.sum(eval_clinician_actions_discrete == action_idx)}",
            flush=True,
        )

    print("\nTraining behavior policy classifiers (joint action space)...", flush=True)
    model_prob = train_discrete_probability_model(train_data["states"], train_model_actions_discrete)
    clinician_prob = train_discrete_probability_model(train_data["states"], train_clinician_actions_discrete)

    print(
        f"  Model action classifier accuracy train/{eval_set_name}: "
        f"{discrete_probability_accuracy(model_prob, train_data['states'], train_model_actions_discrete):.4f} / "
        f"{discrete_probability_accuracy(model_prob, eval_data['states'], eval_model_actions_discrete):.4f}",
        flush=True,
    )
    print(
        f"  Clinician action classifier accuracy train/{eval_set_name}: "
        f"{discrete_probability_accuracy(clinician_prob, train_data['states'], train_clinician_actions_discrete):.4f} / "
        f"{discrete_probability_accuracy(clinician_prob, eval_data['states'], eval_clinician_actions_discrete):.4f}",
        flush=True,
    )

    eval_prob_model = predict_discrete_action_probability(
        model_prob, eval_data["states"], eval_clinician_actions_discrete
    )
    eval_prob_clinician = predict_discrete_action_probability(
        clinician_prob, eval_data["states"], eval_clinician_actions_discrete
    )

    eps = 1e-10
    is_weight = eval_prob_model / (eval_prob_clinician + eps)

    clip_lower = np.percentile(is_weight, clip_lower_pct)
    clip_upper = np.percentile(is_weight, clip_upper_pct)

    print("\nProbability statistics:", flush=True)
    print(
        f"  Model π(a_clinician|s):     mean={eval_prob_model.mean():.4f}, "
        f"std={eval_prob_model.std():.4f}, min={eval_prob_model.min():.4f}, "
        f"max={eval_prob_model.max():.4f}",
        flush=True,
    )
    print(
        f"  Clinician π(a_clinician|s): mean={eval_prob_clinician.mean():.4f}, "
        f"std={eval_prob_clinician.std():.4f}, min={eval_prob_clinician.min():.4f}, "
        f"max={eval_prob_clinician.max():.4f}",
        flush=True,
    )
    print("\nIS weight clipping:", flush=True)
    print(f"  Percentiles: [{clip_lower_pct}, {clip_upper_pct}]", flush=True)
    print(f"  Bounds: [{clip_lower:.6f}, {clip_upper:.6f}]", flush=True)

    metrics = compute_wis_metrics(
        is_weight=is_weight,
        rewards=eval_data["rewards"],
        patient_ids=eval_data["patient_ids"],
        clip_lower=clip_lower,
        clip_upper=clip_upper,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    metrics.update({
        "clip_lower": clip_lower,
        "clip_upper": clip_upper,
        "clip_lower_pct": clip_lower_pct,
        "clip_upper_pct": clip_upper_pct,
        "vp2_bins": vp2_bins,
        "n_actions": n_actions,
    })

    print("\nPer-transition WIS:", flush=True)
    print(f"  Clinician policy raw:       {metrics['clinician_transition']:.4f}", flush=True)
    print(f"  Model policy standard IS:   {metrics['standard_is_transition']:.4f}", flush=True)
    print(f"  Model policy weighted IS:   {metrics['wis_transition']:.4f}", flush=True)
    print(f"  WIS difference:             {metrics['wis_transition_diff']:.4f}", flush=True)

    print("\nPer-trajectory WIS:", flush=True)
    print("  Method: is_block_discrete.py Method 2 (product of per-step ratios)", flush=True)
    print(f"  Clinician policy raw:       {metrics['clinician_trajectory']:.4f}", flush=True)
    print(f"  Model policy weighted IS:   {metrics['wis_trajectory']:.4f}", flush=True)
    print(
        f"  WIS difference:             {metrics['wis_trajectory_diff']:.4f} "
        f"(95% CI: [{metrics['wis_trajectory_diff_ci_lower']:.4f}, "
        f"{metrics['wis_trajectory_diff_ci_upper']:.4f}])",
        flush=True,
    )

    print("\nDiagnostics:", flush=True)
    print(f"  Mean clipped weight:        {metrics['mean_weight']:.4f}", flush=True)
    print(f"  Clipped weight range:       [{metrics['min_weight']:.4f}, {metrics['max_weight']:.4f}]", flush=True)
    print(f"  Transition ESS/N:           {metrics['ess_transition_frac']:.4f}", flush=True)
    print(f"  Trajectory ESS/N:           {metrics['ess_trajectory_frac']:.4f}", flush=True)

    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, f"{experiment_prefix}_{eval_set_name}_wis_results.npz")
    np.savez(results_path, **metrics)
    print(f"\nWIS metrics saved to: {results_path}", flush=True)

    os.makedirs("latex", exist_ok=True)
    latex_path = os.path.join("latex", f"is_ope_results_{experiment_prefix}_{eval_set_name}_bc.tex")
    latex_output = r"""\begin{table}[h]
\centering
\caption{Weighted Importance Sampling Results (Behavior Cloning)}
\label{tab:is_ope_results_bc}
\begin{tabular}{lcc}
\hline
\textbf{Metric} & \textbf{Clinician Policy} & \textbf{BC Policy (WIS)} \\
\hline
Per-Transition Avg Reward & %.4f & %.4f \\
Per-Trajectory Avg Reward & %.4f & %.4f \\
\hline
Transition ESS / N & \multicolumn{2}{c}{%.4f} \\
Trajectory ESS / N & \multicolumn{2}{c}{%.4f} \\
Number of Patients & \multicolumn{2}{c}{%d} \\
Number of Transitions & \multicolumn{2}{c}{%d} \\
\hline
Per-Transition Difference & \multicolumn{2}{c}{%.4f} \\
Per-Trajectory Difference & \multicolumn{2}{c}{%.4f [%.4f, %.4f]} \\
\hline
\end{tabular}
\end{table}
""" % (
        metrics["clinician_transition"], metrics["wis_transition"],
        metrics["clinician_trajectory"], metrics["wis_trajectory"],
        metrics["ess_transition_frac"], metrics["ess_trajectory_frac"],
        metrics["n_patients"], metrics["n_transitions"],
        metrics["wis_transition_diff"],
        metrics["wis_trajectory_diff"],
        metrics["wis_trajectory_diff_ci_lower"],
        metrics["wis_trajectory_diff_ci_upper"],
    )
    with open(latex_path, "w") as f:
        f.write(latex_output)
    print(f"LaTeX WIS table saved to: {latex_path}", flush=True)

    return metrics


def train_behavior_cloning(
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    dropout: float = 0.0,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    seed: int = 42,
    eval_set: str = "test",
    skip_wis: bool = False,
    wis_clip_lower_pct: float = 0.5,
    wis_clip_upper_pct: float = 99.5,
    wis_bootstrap: int = 1000,
    vp2_bins: int = 5,
    suffix: str = "",
    save_dir: str = "experiment/behavior_cloning",
    combined_or_train_data_path: str = None,
    eval_data_path: str = None,
) -> Tuple[BehaviorCloningAgent, IntegratedDataPipelineV3, str]:
    """Train a block-discrete behavior cloning policy with cross-entropy loss."""
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    print("\nInitializing Behavior Cloning data pipeline...", flush=True)
    pipeline = IntegratedDataPipelineV3(
        model_type="dual",
        reward_source="manual",
        random_seed=seed,
        combined_or_train_data_path=combined_or_train_data_path,
        eval_data_path=eval_data_path,
    )

    experiment_prefix = f"bc_{suffix}" if suffix else "bc"

    print("=" * 70, flush=True)
    print(" BEHAVIOR CLONING TRAINING", flush=True)
    print("=" * 70, flush=True)

    train_data, val_data, test_data = pipeline.prepare_data()
    state_dim = train_data["states"].shape[1]
    action_dim = train_data["actions"].shape[1] if train_data["actions"].ndim > 1 else 1

    if action_dim != 2:
        raise ValueError(f"Behavior cloning expects dual continuous actions [VP1, VP2], got action_dim={action_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = BehaviorCloningAgent(
        state_dim=state_dim,
        vp2_bins=vp2_bins,
        hidden_dim=hidden_dim,
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        grad_clip=grad_clip,
        device=device,
    )

    print("\nSETTINGS:", flush=True)
    print(f"  Device: {device}", flush=True)
    print(f"  State dimension: {state_dim}", flush=True)
    print(f"  Action space: {2 * vp2_bins} block-discrete actions (VP1: 2 x VP2: {vp2_bins})", flush=True)
    print("  Loss: Cross-entropy between softmax policy and expert discrete action", flush=True)
    print(f"  Hidden dim: {hidden_dim}", flush=True)
    print(f"  Dropout: {dropout}", flush=True)
    print(f"  LR: {lr}", flush=True)
    print(f"  Weight decay: {weight_decay}", flush=True)
    print(f"  Batch size: {batch_size}", flush=True)
    print(f"  Epochs: {epochs}", flush=True)
    print(f"  Train transitions: {len(train_data['states'])}", flush=True)
    print(f"  Val transitions: {len(val_data['states'])}", flush=True)
    print(f"  Test transitions: {len(test_data['states'])}", flush=True)
    print("=" * 70, flush=True)

    n_batches = max(1, len(train_data["states"]) // batch_size)
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(epochs):
        agent.policy.train()
        epoch_metrics = {"loss": 0.0, "accuracy": 0.0, "expert_prob": 0.0, "entropy": 0.0}

        for _ in range(n_batches):
            batch = pipeline.get_batch(batch_size=batch_size, split="train")
            states = torch.FloatTensor(batch["states"]).to(agent.device)
            actions = torch.FloatTensor(batch["actions"]).to(agent.device)

            metrics = agent.update(states, actions)
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]

        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches

        val_metrics = agent.evaluate(val_data["states"], val_data["actions"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path = f"{save_dir}/{experiment_prefix}_best.pt"
            agent.save(
                best_path,
                extra={
                    "epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                    "seed": seed,
                    "train_metrics": epoch_metrics,
                    "val_metrics": val_metrics,
                },
            )
            print(f"  Saving best model (val loss={best_val_loss:.6f}) at {best_path}", flush=True)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch + 1}: "
                f"train_loss={epoch_metrics['loss']:.6f}, "
                f"train_acc={epoch_metrics['accuracy']:.4f}, "
                f"train_expert_prob={epoch_metrics['expert_prob']:.4f}, "
                f"val_loss={val_metrics['loss']:.6f}, "
                f"val_acc={val_metrics['accuracy']:.4f}, "
                f"val_expert_prob={val_metrics['expert_prob']:.4f}, "
                f"Time={elapsed / 60:.1f}min",
                flush=True,
            )

    final_path = f"{save_dir}/{experiment_prefix}_final.pt"
    final_val_metrics = agent.evaluate(val_data["states"], val_data["actions"])
    test_metrics = agent.evaluate(test_data["states"], test_data["actions"])
    agent.save(
        final_path,
        extra={
            "seed": seed,
            "final_val_metrics": final_val_metrics,
            "test_metrics": test_metrics,
        },
    )

    total_time = time.time() - start_time
    print(f"\nBehavior cloning training completed in {total_time / 60:.1f} minutes!", flush=True)
    print(f"Best val loss: {best_val_loss:.6f}", flush=True)
    print(
        "Final test metrics: "
        f"loss={test_metrics['loss']:.6f}, "
        f"accuracy={test_metrics['accuracy']:.4f}, "
        f"expert_prob={test_metrics['expert_prob']:.4f}, "
        f"entropy={test_metrics['entropy']:.4f}",
        flush=True,
    )
    print("Models saved:", flush=True)
    print(f"  - {save_dir}/{experiment_prefix}_best.pt", flush=True)
    print(f"  - {final_path}", flush=True)

    if not skip_wis:
        wis_eval_data = val_data if eval_set == "val" else test_data
        evaluate_behavior_cloning_wis(
            agent=agent,
            train_data=train_data,
            eval_data=wis_eval_data,
            eval_set_name=eval_set,
            clip_lower_pct=wis_clip_lower_pct,
            clip_upper_pct=wis_clip_upper_pct,
            n_bootstrap=wis_bootstrap,
            save_dir=save_dir,
            experiment_prefix=experiment_prefix,
            seed=seed,
            vp2_bins=vp2_bins,
        )

    return agent, pipeline, experiment_prefix


def main() -> None:
    parser = argparse.ArgumentParser(description="Train block-discrete softmax behavior cloning policy")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--hidden_dim", type=int, default=128, help="MLP hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Adam weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max gradient norm; <=0 disables clipping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval_set", type=str, default="test", choices=["val", "test"], help="Split for post-training WIS")
    parser.add_argument("--skip_wis", action="store_true", help="Skip post-training WIS evaluation")
    parser.add_argument("--wis_clip_lower_pct", type=float, default=0.5, help="Lower percentile for WIS weight clipping")
    parser.add_argument("--wis_clip_upper_pct", type=float, default=99.5, help="Upper percentile for WIS weight clipping")
    parser.add_argument("--wis_bootstrap", type=int, default=1000, help="Bootstrap iterations for trajectory WIS CI")
    parser.add_argument("--vp2_bins", type=int, default=5, help="Number of VP2 bins for block-discrete BC and WIS")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for experiment naming")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="experiment/behavior_cloning",
        help="Directory to save model checkpoints",
    )
    parser.add_argument("--combined_or_train_data_path", type=str, default=None, help="Path to training dataset")
    parser.add_argument("--eval_data_path", type=str, default=None, help="Path to evaluation dataset")
    args = parser.parse_args()

    if not (0.0 <= args.wis_clip_lower_pct < args.wis_clip_upper_pct <= 100.0):
        parser.error("--wis_clip_lower_pct and --wis_clip_upper_pct must satisfy 0 <= lower < upper <= 100")
    if args.vp2_bins <= 0:
        parser.error("--vp2_bins must be positive")

    print("=" * 70, flush=True)
    print(" BLOCK-DISCRETE SOFTMAX BEHAVIOR CLONING", flush=True)
    print("=" * 70, flush=True)
    print("Policy: neural network mapping observed state to action logits", flush=True)
    print("Objective: minimize cross-entropy with expert block-discrete action", flush=True)
    if args.eval_data_path:
        print("Dataset mode: DUAL-DATASET", flush=True)
        print(f"  Train data: {args.combined_or_train_data_path or 'default'}", flush=True)
        print(f"  Eval data:  {args.eval_data_path}", flush=True)
    else:
        print("Dataset mode: SINGLE-DATASET", flush=True)
        print(f"  Data path: {args.combined_or_train_data_path or 'default'}", flush=True)

    train_behavior_cloning(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        seed=args.seed,
        eval_set=args.eval_set,
        skip_wis=args.skip_wis,
        wis_clip_lower_pct=args.wis_clip_lower_pct,
        wis_clip_upper_pct=args.wis_clip_upper_pct,
        wis_bootstrap=args.wis_bootstrap,
        vp2_bins=args.vp2_bins,
        suffix=args.suffix,
        save_dir=args.save_dir,
        combined_or_train_data_path=args.combined_or_train_data_path,
        eval_data_path=args.eval_data_path,
    )

    print("\n" + "=" * 70, flush=True)
    print(" TRAINING COMPLETE", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
