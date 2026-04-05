#!/usr/bin/env python3
"""
Sequential Policy Estimator using LSTM.
Predicts action probabilities from state/action sequences.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from medical_sequence_buffer_v2 import MedicalSequenceBufferV2


class SeqPolicyEstimator(nn.Module):
    """
    LSTM-based sequential policy estimator.
    Input: [state, one_hot_action] sequence [batch, seq_len, state_dim + n_actions]
    Output: probability from last hidden state [batch, 1] via sigmoid
    """

    def __init__(self, state_dim: int, n_actions: int,
                 hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lstm = nn.LSTM(
            input_size=state_dim + n_actions,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return torch.sigmoid(self.output(last_hidden))


class SeqPolicyEstimatorTrainer:
    """Trains SeqPolicyEstimator using sequential replay buffer with 50/50 expert/random split."""

    def __init__(self, state_dim: int, n_actions: int, sequence_length: int = 40,
                 hidden_dim: int = 64, num_layers: int = 2, lr: float = 1e-3):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SeqPolicyEstimator(
            state_dim=state_dim, n_actions=n_actions,
            hidden_dim=hidden_dim, num_layers=num_layers,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def _load_buffer(self, states, actions, patient_ids):
        """Load transitions into sequential replay buffer.
        Args:
            states: [n_transitions, state_dim] numpy array
            actions: [n_transitions] numpy int array of discrete action indices
            patient_ids: [n_transitions] numpy array
        """
        buffer = MedicalSequenceBufferV2(
            capacity=100000,
            sequence_length=self.sequence_length,
            burn_in_length=0,
            overlap=1,
            priority_type='uniform',
        )
        n = len(states)
        for i in range(n):
            is_last = (i == n - 1) or (patient_ids[i] != patient_ids[i + 1])
            buffer.add_transition(
                state=states[i],
                action=actions[i],
                reward=0.0,
                next_state=states[min(i + 1, n - 1)],
                done=is_last,
                patient_id=patient_ids[i],
            )
        stats = buffer.get_statistics()
        print(f"Buffer: {stats['total_sequences_generated']} sequences "
              f"from {stats['total_patients_processed']} patients")
        return buffer

    def _make_input(self, states, action_indices):
        """
        Concatenate states with one-hot actions.
        Args:
            states: [batch, seq_len, state_dim] tensor
            action_indices: [batch, seq_len] long tensor of discrete action indices
        Returns:
            [batch, seq_len, state_dim + n_actions] tensor
        """
        one_hot = F.one_hot(action_indices, num_classes=self.n_actions).float()
        return torch.cat([states, one_hot], dim=-1)

    def train(self, states, actions, patient_ids, epochs: int = 100, batch_size: int = 128):
        """
        Train the sequential policy estimator.
        50% expert data (label=1), 50% same states with random actions (label=0).
        Args:
            states: [n_transitions, state_dim] numpy array
            actions: [n_transitions] numpy int array of discrete action indices
            patient_ids: [n_transitions] numpy array for trajectory grouping
        """
        buffer = self._load_buffer(states, actions, patient_ids)
        half_batch = batch_size // 2
        n_batches = buffer.tree.n_entries // half_batch

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for _ in range(n_batches):
                _, batch, _, _ = buffer.sample_sequences(half_batch)
                s = torch.FloatTensor(batch['states']).to(self.device)
                a = torch.LongTensor(batch['actions']).to(self.device)

                # Expert half: real actions, label = 1
                expert_input = self._make_input(s, a)
                expert_labels = torch.ones(half_batch, 1).to(self.device)

                # Random half: same states, random actions, label = 0
                random_a = torch.randint(0, self.n_actions, a.shape).to(self.device)
                random_input = self._make_input(s, random_a)
                random_labels = torch.zeros(half_batch, 1).to(self.device)

                inputs = torch.cat([expert_input, random_input], dim=0)
                labels = torch.cat([expert_labels, random_labels], dim=0)

                preds = self.model(inputs)
                loss = F.binary_cross_entropy(preds, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: loss={epoch_loss/n_batches:.4f}")

    def predict(self, states, actions, patient_ids):
        """
        Predict sigmoid logits for variable-length trajectories.
        One prediction per patient trajectory (from last hidden state).
        Args:
            states: [n_transitions, state_dim] numpy array
            actions: [n_transitions] numpy int array of discrete action indices
            patient_ids: [n_transitions] numpy array
        Returns:
            patient_preds: dict mapping patient_id -> sigmoid prediction
        """
        self.model.eval()
        unique_patients = np.unique(patient_ids)
        patient_preds = {}

        with torch.no_grad():
            for pid in unique_patients:
                mask = patient_ids == pid
                s = torch.FloatTensor(states[mask]).unsqueeze(0).to(self.device)
                a = torch.LongTensor(actions[mask]).unsqueeze(0).to(self.device)
                x = self._make_input(s, a)
                pred = self.model(x)
                patient_preds[pid] = pred.item()

        return patient_preds
