"""
Transformer policy model for clinician-action modeling.

This module reuses the transformer encoder shape from
transformer_reward_generator_tanh.py, but replaces the scalar tanh reward head
with an action-logit policy head. It trains pi(a_t | trajectory context) by
behavior cloning on the dual-action data used elsewhere in this repo.
"""

import argparse
import os
import re
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from integrated_data_pipeline_v3 import IntegratedDataPipelineV3
from medical_sequence_buffer_v2 import MedicalSequenceBufferV2


class Config:
    """Configuration for transformer policy training."""

    d_model: int = 128
    nhead: int = 2
    num_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1

    vp1_bins: int = 2
    vp2_bins: int = 5
    action_size: int = vp1_bins * vp2_bins

    batch_size: int = 128
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0

    sequence_length: int = 40
    overlap: int = 1

    num_steps_per_eval_print: int = 50
    experiment_dir: str = "experiments/transformer_policy"

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.action_size = self.vp1_bins * self.vp2_bins


class TransformerPolicyModel(nn.Module):
    """
    Transformer policy over block-discrete vasopressor actions.

    Input: [batch, seq_len, state_size]
    Output: [batch, seq_len, action_size] action logits
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        d_ff: int = 64,
        dropout: float = 0.1,
        max_seq_length: int = 512,
    ):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout_p = dropout
        self.max_seq_length = max_seq_length

        self.input_projection = nn.Linear(state_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.policy_head = nn.Linear(d_model, action_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = states.shape
        if seq_len > self.max_seq_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_seq_length}"
            )

        positions = torch.arange(seq_len, device=states.device)
        x = self.input_projection(states)
        x = x + self.position_embedding(positions).unsqueeze(0)
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.ln_f(x)
        return self.policy_head(x)

    @torch.no_grad()
    def action_probabilities(self, states: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(states), dim=-1)

    @torch.no_grad()
    def select_action(self, states: torch.Tensor) -> torch.Tensor:
        """Return greedy discrete action indices for each timestep."""
        return self.forward(states).argmax(dim=-1)


def continuous_to_discrete_actions(actions: np.ndarray, vp2_bins: int) -> np.ndarray:
    """
    Convert continuous [vp1, vp2] actions to block-discrete indices.

    Matches DualBlockDiscreteCQL.continuous_to_discrete_action without
    constructing Q-networks just for discretization.
    """
    vp2_bin_edges = np.linspace(0, 0.5, vp2_bins + 1)
    vp1 = actions[..., 0].astype(np.int64)
    vp2 = np.clip(actions[..., 1], 0.0, 0.5)
    vp2_idx = np.digitize(vp2, vp2_bin_edges) - 1
    vp2_idx = np.clip(vp2_idx, 0, vp2_bins - 1).astype(np.int64)
    return vp1 * vp2_bins + vp2_idx


def discrete_to_continuous_actions(action_idx: np.ndarray, vp2_bins: int) -> np.ndarray:
    """Convert block-discrete action indices back to [vp1, vp2] bin centers."""
    action_idx = np.asarray(action_idx)
    vp2_bin_edges = np.linspace(0, 0.5, vp2_bins + 1)
    vp1 = (action_idx // vp2_bins).astype(np.float32)
    vp2_bin = action_idx % vp2_bins
    vp2 = ((vp2_bin_edges[vp2_bin] + vp2_bin_edges[vp2_bin + 1]) / 2).astype(np.float32)
    return np.stack([vp1, vp2], axis=-1)


def load_trajectories(data_pipeline, config: Config, split: str = "train") -> Tuple[MedicalSequenceBufferV2, int]:
    """Load a prepared split into MedicalSequenceBufferV2."""
    if data_pipeline.train_data is None:
        train_data, val_data, test_data = data_pipeline.prepare_data()
    else:
        train_data, val_data, test_data = data_pipeline.train_data, data_pipeline.val_data, data_pipeline.test_data

    data_by_split = {"train": train_data, "val": val_data, "test": test_data}
    data = data_by_split[split]

    buffer = MedicalSequenceBufferV2(
        capacity=100000,
        sequence_length=config.sequence_length,
        burn_in_length=0,
        overlap=config.overlap,
        priority_type="uniform",
    )

    patient_ids = data.get("patient_ids", np.arange(len(data["states"])))
    n_transitions = len(data["states"])
    for i in range(n_transitions):
        buffer.add_transition(
            state=data["states"][i],
            action=data["actions"][i],
            reward=data["rewards"][i],
            next_state=data["next_states"][i],
            done=data["dones"][i],
            patient_id=patient_ids[i],
        )

    stats = buffer.get_statistics()
    print(
        f"{split}: loaded {stats['total_sequences_generated']} sequences "
        f"from {stats['total_patients_processed']} patients"
    )
    return buffer, data["states"].shape[1]


def compute_batch_loss(
    model: TransformerPolicyModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    config: Config,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute behavior-cloning loss and metrics for one sequence batch."""
    action_idx_np = continuous_to_discrete_actions(actions.detach().cpu().numpy(), config.vp2_bins)
    action_idx = torch.LongTensor(action_idx_np).to(device)

    logits = model(states)
    loss = F.cross_entropy(
        logits.reshape(-1, config.action_size),
        action_idx.reshape(-1),
    )

    pred = logits.argmax(dim=-1)
    accuracy = (pred == action_idx).float().mean().item()
    probs = F.softmax(logits, dim=-1)
    expert_prob = probs.gather(-1, action_idx.unsqueeze(-1)).mean().item()
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean().item()

    metrics = {
        "accuracy": accuracy,
        "expert_prob": expert_prob,
        "entropy": entropy,
    }
    return loss, metrics


def evaluate(model: TransformerPolicyModel, buffer, config: Config, device: torch.device) -> Dict[str, float]:
    model.eval()
    eval_batch_size = min(config.batch_size, len(buffer))
    n_batches = max(1, min(10, len(buffer) // eval_batch_size))
    losses = []
    accuracies = []
    expert_probs = []
    entropies = []

    with torch.no_grad():
        for _ in range(n_batches):
            _, training, _, _ = buffer.sample_sequences(eval_batch_size)
            states = torch.FloatTensor(training["states"]).to(device)
            actions = torch.FloatTensor(training["actions"]).to(device)
            loss, metrics = compute_batch_loss(model, states, actions, config, device)
            losses.append(loss.item())
            accuracies.append(metrics["accuracy"])
            expert_probs.append(metrics["expert_prob"])
            entropies.append(metrics["entropy"])

    model.train()
    return {
        "val_loss": float(np.mean(losses)),
        "val_accuracy": float(np.mean(accuracies)),
        "val_expert_prob": float(np.mean(expert_probs)),
        "val_entropy": float(np.mean(entropies)),
    }


def train(
    config: Config,
    data_pipeline,
    resume_model_path: Optional[str] = None,
    time_one_batch: bool = False,
) -> TransformerPolicyModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(config.experiment_dir, exist_ok=True)

    train_buffer, state_size = load_trajectories(data_pipeline, config, split="train")
    val_buffer, _ = load_trajectories(data_pipeline, config, split="val")

    if len(train_buffer) == 0:
        raise ValueError(
            f"No train sequences generated. Use a smaller --sequence_length than {config.sequence_length}."
        )
    if len(val_buffer) == 0:
        print("Validation buffer is empty; validation will use train sequences.")
        val_buffer = train_buffer

    start_epoch = 0
    if resume_model_path:
        model, checkpoint_config = load_model(resume_model_path, device=device)
        if model.action_size != config.action_size:
            raise ValueError(
                f"Checkpoint action_size={model.action_size} does not match requested "
                f"action_size={config.action_size}"
            )
        if model.d_model != config.d_model or model.num_layers != config.num_layers:
            raise ValueError(
                "Checkpoint architecture does not match requested transformer settings. "
                f"checkpoint=(d_model={model.d_model}, num_layers={model.num_layers}) "
                f"requested=(d_model={config.d_model}, num_layers={config.num_layers})"
            )
        config.sequence_length = checkpoint_config.get("sequence_length", config.sequence_length)
        match = re.search(r"model_epoch_(\d+)\.pt$", os.path.basename(resume_model_path))
        if match:
            start_epoch = int(match.group(1))
    else:
        model = TransformerPolicyModel(
            state_size=state_size,
            action_size=config.action_size,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_seq_length=config.sequence_length,
        ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    n_batches = len(train_buffer) // config.batch_size
    if n_batches == 0:
        raise ValueError(
            f"Not enough train sequences for batch_size={config.batch_size}: "
            f"buffer has {len(train_buffer)} sequences."
        )

    step = 0
    epochs_to_run = 1 if time_one_batch else config.num_epochs
    for epoch in range(epochs_to_run):
        absolute_epoch = start_epoch + epoch + 1
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_expert_prob = 0.0
        epoch_entropy = 0.0
        num_batches = 0

        model.train()
        for batch_idx in range(n_batches):
            _, training, _, _ = train_buffer.sample_sequences(config.batch_size)
            states = torch.FloatTensor(training["states"]).to(device)
            actions = torch.FloatTensor(training["actions"]).to(device)

            optimizer.zero_grad()

            if time_one_batch:
                batch_transitions = states.shape[0] * states.shape[1]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                loss, metrics = compute_batch_loss(model, states, actions, config, device)
                loss.backward()
                if config.grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                save_path = os.path.join(config.experiment_dir, "timing_batch_policy_model.pt")
                save_model(model, config, save_path)
                print(f"TRANSFORMER_POLICY_BATCH_SHAPE={tuple(states.shape)}")
                print(f"TRANSFORMER_POLICY_BATCH_TRANSITIONS={batch_transitions}")
                print(f"TRANSFORMER_POLICY_BATCH_TIME_SECONDS={end - start}")
                print(f"TRANSFORMER_POLICY_MODEL_PATH={save_path}")
                return model

            loss, metrics = compute_batch_loss(model, states, actions, config, device)
            loss.backward()
            if config.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += metrics["accuracy"]
            epoch_expert_prob += metrics["expert_prob"]
            epoch_entropy += metrics["entropy"]
            num_batches += 1
            step += 1

            if step % config.num_steps_per_eval_print == 0:
                val_metrics = evaluate(model, val_buffer, config, device)
                print(
                    f"Step {step} | Train Loss: {epoch_loss/num_batches:.4f} | "
                    f"Train Acc: {epoch_acc/num_batches:.4f} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val Acc: {val_metrics['val_accuracy']:.4f}"
                )

        print(
            f"Epoch {absolute_epoch} complete | "
            f"loss={epoch_loss/num_batches:.4f} | "
            f"acc={epoch_acc/num_batches:.4f} | "
            f"expert_prob={epoch_expert_prob/num_batches:.4f} | "
            f"entropy={epoch_entropy/num_batches:.4f}"
        )

        save_path = os.path.join(config.experiment_dir, f"model_epoch_{absolute_epoch}.pt")
        save_model(model, config, save_path)

    print("Training complete!")
    return model


def save_model(model: TransformerPolicyModel, config: Config, filepath: str):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "transformer_policy",
            "state_size": model.state_size,
            "action_size": model.action_size,
            "d_model": model.d_model,
            "nhead": model.nhead,
            "num_layers": model.num_layers,
            "d_ff": model.d_ff,
            "dropout": model.dropout_p,
            "max_seq_length": model.max_seq_length,
            "config": {
                "vp1_bins": config.vp1_bins,
                "vp2_bins": config.vp2_bins,
                "sequence_length": config.sequence_length,
                "d_model": config.d_model,
                "nhead": config.nhead,
                "num_layers": config.num_layers,
                "d_ff": config.d_ff,
                "dropout": config.dropout,
            },
        },
        filepath,
    )
    print(f"Model saved to {filepath}")


def load_model(filepath: str, device: torch.device = None) -> Tuple[TransformerPolicyModel, Dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(filepath, map_location=device)
    model = TransformerPolicyModel(
        state_size=checkpoint["state_size"],
        action_size=checkpoint["action_size"],
        d_model=checkpoint["d_model"],
        nhead=checkpoint["nhead"],
        num_layers=checkpoint["num_layers"],
        d_ff=checkpoint["d_ff"],
        dropout=checkpoint["dropout"],
        max_seq_length=checkpoint["max_seq_length"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded from {filepath}")
    return model, checkpoint.get("config", {})


def main():
    parser = argparse.ArgumentParser(description="Train transformer policy model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm; <=0 disables")
    parser.add_argument("--d_model", type=int, default=64, help="Transformer hidden dimension")
    parser.add_argument("--nhead", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--d_ff", type=int, default=128, help="Transformer feedforward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout")
    parser.add_argument("--sequence_length", type=int, default=40, help="Trajectory window length")
    parser.add_argument("--vp1_bins", type=int, default=2, help="VP1 bins")
    parser.add_argument("--vp2_bins", type=int, default=5, help="VP2 bins")
    parser.add_argument("--experiment_dir", type=str, default="experiments/transformer_policy", help="Save directory")
    parser.add_argument("--resume_model_path", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--combined_or_train_data_path", type=str, default=None, help="Training dataset path")
    parser.add_argument("--eval_data_path", type=str, default=None, help="Evaluation dataset path")
    parser.add_argument("--time_one_batch", action="store_true", help="Run one batch, print timing, and exit")
    args = parser.parse_args()

    grad_clip = args.grad_clip if args.grad_clip and args.grad_clip > 0 else None
    config = Config(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=grad_clip,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        sequence_length=args.sequence_length,
        vp1_bins=args.vp1_bins,
        vp2_bins=args.vp2_bins,
        experiment_dir=args.experiment_dir + f"_{args.d_model}d_{args.num_layers}l",
    )

    print(
        f"Config: epochs={config.num_epochs}, batch_size={config.batch_size}, "
        f"lr={config.learning_rate}, d_model={config.d_model}, nhead={config.nhead}, "
        f"num_layers={config.num_layers}, d_ff={config.d_ff}, dropout={config.dropout}, "
        f"sequence_length={config.sequence_length}, action_size={config.action_size}"
    )

    pipeline = IntegratedDataPipelineV3(
        model_type="dual",
        reward_source="manual",
        random_seed=42,
        combined_or_train_data_path=args.combined_or_train_data_path,
        eval_data_path=args.eval_data_path,
    )

    train(
        config,
        pipeline,
        resume_model_path=args.resume_model_path,
        time_one_batch=args.time_one_batch,
    )


if __name__ == "__main__":
    main()
