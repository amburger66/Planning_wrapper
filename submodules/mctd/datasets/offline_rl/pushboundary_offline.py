"""
Offline RL dataset wrapper for the custom PushBoundary trajectories.

This dataset exposes the tuple format expected by:
  `submodules/mctd/algorithms/diffusion_forcing/df_planning.py`

Return values (for each item):
  observation:  [n_frames, observation_dim]  float32
  action:       [n_frames, action_dim]       float32
  reward:       [n_frames]                  float32
  nonterminal:  [n_frames]                   bool
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from omegaconf import DictConfig


def _resolve_repo_path(p: str | Path) -> Path:
    p = Path(p).expanduser()
    if p.is_absolute():
        return p

    # File: <repo_root>/submodules/mctd/datasets/offline_rl/<this_file>.py
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / p


class PushBoundaryOfflineRLDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        _ = split  # PushBoundary dataset is already offline; we ignore split.
        self.cfg = cfg

        self.episode_len = int(cfg.episode_len)
        self.n_frames = self.episode_len + 1

        # Shapes / dims.
        # Expected yaml fields:
        # - observation_shape: [obs_dim]
        # - action_dim: action_dim
        # Optional (for 2D slicing): state_indices, action_indices
        self.observation_dim = int(cfg.observation_shape[0])
        self.action_dim = int(cfg.action_dim)

        self._state_indices = (
            list(cfg.state_indices) if "state_indices" in cfg else None
        )
        self._action_indices = (
            list(cfg.action_indices) if "action_indices" in cfg else None
        )

        if self._state_indices is not None:
            if len(self._state_indices) != self.observation_dim:
                raise ValueError(
                    f"state_indices length {len(self._state_indices)} != observation_shape[0] {self.observation_dim}"
                )
        if self._action_indices is not None:
            if len(self._action_indices) != self.action_dim:
                raise ValueError(
                    f"action_indices length {len(self._action_indices)} != action_dim {self.action_dim}"
                )

        states_path = _resolve_repo_path(cfg.states_path)
        actions_path = _resolve_repo_path(cfg.actions_path)

        # Use mmap so we don't eagerly load the full arrays into each worker.
        states = np.load(str(states_path), mmap_mode="r")
        actions = np.load(str(actions_path), mmap_mode="r")

        if states.ndim != 3:
            raise ValueError(f"states must be 3D [N, T, SD], got shape={states.shape}")
        if actions.ndim != 3:
            raise ValueError(f"actions must be 3D [N, T, AD], got shape={actions.shape}")

        n_traj, t_total, sd = states.shape
        n_traj_a, t_total_a, ad = actions.shape
        if n_traj != n_traj_a or t_total != t_total_a:
            raise ValueError(
                f"states/actions shape mismatch: states={states.shape} actions={actions.shape}"
            )

        if self._state_indices is not None:
            if sd <= max(self._state_indices):
                raise ValueError(
                    f"states last dim {sd} too small for state_indices {self._state_indices}"
                )
        else:
            if sd != self.observation_dim:
                raise ValueError(
                    f"states last dim mismatch: got {sd}, expected {self.observation_dim}"
                )

        if self._action_indices is not None:
            if ad <= max(self._action_indices):
                raise ValueError(
                    f"actions last dim {ad} too small for action_indices {self._action_indices}"
                )
        else:
            if ad != self.action_dim:
                raise ValueError(
                    f"actions last dim mismatch: got {ad}, expected {self.action_dim}"
                )

        if t_total < self.n_frames:
            raise ValueError(
                f"Trajectory length too short: states/actions have T_total={t_total} "
                f"but require n_frames={self.n_frames}."
            )

        self.states = states
        self.actions = actions
        self.n_traj = n_traj

        # Reward is not used by `df_planning.yaml` when `use_reward=False`, but the
        # dataset contract requires it.
        self._reward_template = torch.zeros(self.n_frames, dtype=torch.float32)

        nonterminal = torch.ones(self.n_frames, dtype=torch.bool)
        nonterminal[-1] = False
        self._nonterminal_template = nonterminal

        # Basic sanity prints (only on rank zero in lightning; dataset isn't aware of rank).
        # Keep them lightweight.
        slice_info = ""
        if self._state_indices is not None or self._action_indices is not None:
            slice_info = f" (sliced to obs={self.observation_dim}D, act={self.action_dim}D)"
        print(
            f"[PushBoundaryOfflineRLDataset] loaded states={states.shape} actions={actions.shape} "
            f"n_frames={self.n_frames}{slice_info}"
        )

    def __len__(self) -> int:
        return self.n_traj

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_np = np.asarray(self.states[idx, : self.n_frames], dtype=np.float32)  # [n_frames, SD]
        act_np = np.asarray(self.actions[idx, : self.n_frames], dtype=np.float32)  # [n_frames, AD]

        if self._state_indices is not None:
            obs_np = obs_np[:, self._state_indices]
        if self._action_indices is not None:
            act_np = act_np[:, self._action_indices]

        # Convert to torch tensors. reward/nonterminal are templates.
        observation = torch.from_numpy(obs_np.copy())  # float32
        action = torch.from_numpy(act_np.copy())  # float32
        reward = self._reward_template
        nonterminal = self._nonterminal_template

        return observation, action, reward, nonterminal

