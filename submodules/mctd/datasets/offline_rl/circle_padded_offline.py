"""
Offline RL dataset for padded NPZ demos (e.g. circle PushBoundary) with valid_lengths.

Same tuple contract as PushBoundaryOfflineRLDataset:
  observation, action, reward, nonterminal

NPZ keys: states [N, T, SD], actions [N, T, AD], valid_lengths [N] (int).
Shorter trajectories are zero-padded; nonterminal masks the padded suffix for df_planning.
"""

# from __future__ import annotations

# from pathlib import Path
# from typing import Tuple

# import numpy as np
# import torch
# from omegaconf import DictConfig


# def _resolve_repo_path(p: str | Path) -> Path:
#     p = Path(p).expanduser()
#     if p.is_absolute():
#         return p

#     repo_root = Path(__file__).resolve().parents[4]
#     return repo_root / p


# def _nonterminal_from_valid_length(n_frames: int, t_valid: int) -> torch.Tensor:
#     """
#     t_valid: number of valid timesteps (same convention as stored valid_lengths).
#     Matches PushBoundary template when t_valid == n_frames: all True except last False.
#     When t_valid < n_frames: False from index (t_valid - 1) through end (padded tail masked).
#     """
#     nt = torch.ones(n_frames, dtype=torch.bool)
#     if t_valid <= 0:
#         return torch.zeros(n_frames, dtype=torch.bool)
#     if t_valid < n_frames:
#         nt[t_valid - 1 :] = False
#     else:
#         nt[-1] = False
#     return nt


# class CirclePaddedOfflineRLDataset(torch.utils.data.Dataset):
#     def __init__(self, cfg: DictConfig, split: str = "training"):
#         super().__init__()
#         _ = split
#         self.cfg = cfg

#         self.episode_len = int(cfg.episode_len)
#         self.n_frames = self.episode_len + 1

#         self.observation_dim = int(cfg.observation_shape[0])
#         self.action_dim = int(cfg.action_dim)

#         self._state_indices = (
#             list(cfg.state_indices) if "state_indices" in cfg else None
#         )
#         self._action_indices = (
#             list(cfg.action_indices) if "action_indices" in cfg else None
#         )

#         if self._state_indices is not None:
#             if len(self._state_indices) != self.observation_dim:
#                 raise ValueError(
#                     f"state_indices length {len(self._state_indices)} != observation_shape[0] {self.observation_dim}"
#                 )
#         if self._action_indices is not None:
#             if len(self._action_indices) != self.action_dim:
#                 raise ValueError(
#                     f"action_indices length {len(self._action_indices)} != action_dim {self.action_dim}"
#                 )

#         npz_path = _resolve_repo_path(cfg.npz_path)
#         data = np.load(str(npz_path), mmap_mode="r")
#         if "states" not in data or "actions" not in data or "valid_lengths" not in data:
#             raise ValueError(
#                 f"NPZ must contain 'states', 'actions', 'valid_lengths'; keys={list(data.files)}"
#             )

#         states = data["states"]
#         actions = data["actions"]
#         valid_lengths = np.asarray(data["valid_lengths"], dtype=np.int64)

#         if states.ndim != 3:
#             raise ValueError(f"states must be 3D [N, T, SD], got shape={states.shape}")
#         if actions.ndim != 3:
#             raise ValueError(f"actions must be 3D [N, T, AD], got shape={actions.shape}")

#         n_traj, t_total, sd = states.shape
#         n_traj_a, t_total_a, ad = actions.shape
#         if n_traj != n_traj_a or t_total != t_total_a:
#             raise ValueError(
#                 f"states/actions shape mismatch: states={states.shape} actions={actions.shape}"
#             )
#         if len(valid_lengths) != n_traj:
#             raise ValueError(
#                 f"valid_lengths length {len(valid_lengths)} != n_traj {n_traj}"
#             )

#         if self._state_indices is not None:
#             if sd <= max(self._state_indices):
#                 raise ValueError(
#                     f"states last dim {sd} too small for state_indices {self._state_indices}"
#                 )
#         else:
#             if sd != self.observation_dim:
#                 raise ValueError(
#                     f"states last dim mismatch: got {sd}, expected {self.observation_dim}"
#                 )

#         if self._action_indices is not None:
#             if ad <= max(self._action_indices):
#                 raise ValueError(
#                     f"actions last dim {ad} too small for action_indices {self._action_indices}"
#                 )
#         else:
#             if ad != self.action_dim:
#                 raise ValueError(
#                     f"actions last dim mismatch: got {ad}, expected {self.action_dim}"
#                 )

#         if t_total < self.n_frames:
#             raise ValueError(
#                 f"NPZ time dimension too short: T_total={t_total} < n_frames={self.n_frames}."
#             )

#         self.states = states
#         self.actions = actions
#         self._valid_lengths = valid_lengths

#         # Keep only trajectories with at least one valid step.
#         self._indices = [
#             i for i in range(n_traj) if int(self._valid_lengths[i]) >= 1
#         ]
#         n_drop = n_traj - len(self._indices)
#         if n_drop > 0:
#             print(
#                 f"[CirclePaddedOfflineRLDataset] warning: dropped {n_drop} trajectories "
#                 f"with valid_lengths < 1"
#             )

#         self._reward_template = torch.zeros(self.n_frames, dtype=torch.float32)

#         slice_info = ""
#         if self._state_indices is not None or self._action_indices is not None:
#             slice_info = f" (sliced to obs={self.observation_dim}D, act={self.action_dim}D)"
#         print(
#             f"[CirclePaddedOfflineRLDataset] loaded npz={npz_path} "
#             f"states={states.shape} actions={actions.shape} "
#             f"n_frames={self.n_frames} usable_traj={len(self._indices)}{slice_info}"
#         )

#     def __len__(self) -> int:
#         return len(self._indices)

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         traj_idx = self._indices[idx]
#         obs_np = np.asarray(self.states[traj_idx, : self.n_frames], dtype=np.float32)
#         act_np = np.asarray(self.actions[traj_idx, : self.n_frames], dtype=np.float32)

#         if self._state_indices is not None:
#             obs_np = obs_np[:, self._state_indices]
#         if self._action_indices is not None:
#             act_np = act_np[:, self._action_indices]

#         t_valid = int(min(int(self._valid_lengths[traj_idx]), self.n_frames))

#         # Fill any zero-padded tail with the last valid frame so that normalization
#         # never sees out-of-distribution zeros.  Zero padding is especially harmful
#         # when some dimensions have near-zero std (e.g. 1e-8): (0 - mean) / 1e-8
#         # produces ±1e8 which overflows float16 in mixed-precision training → NaN.
#         if t_valid < self.n_frames and t_valid > 0:
#             obs_np[t_valid:] = obs_np[t_valid - 1]
#             act_np[t_valid:] = act_np[t_valid - 1]

#         nonterminal = _nonterminal_from_valid_length(self.n_frames, t_valid)

#         observation = torch.from_numpy(obs_np.copy())
#         action = torch.from_numpy(act_np.copy())
#         reward = self._reward_template

#         return observation, action, reward, nonterminal


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

    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / p


class CirclePaddedOfflineRLDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        _ = split
        self.cfg = cfg

        self.episode_len = int(cfg.episode_len)
        self.n_frames = self.episode_len + 1

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

        npz_path = _resolve_repo_path(cfg.npz_path)
        data = np.load(str(npz_path), mmap_mode="r")
        
        if "states" not in data or "actions" not in data:
            raise ValueError(
                f"NPZ must contain 'states' and 'actions'; keys={list(data.files)}"
            )

        states = data["states"]
        actions = data["actions"]

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
                f"NPZ time dimension too short: T_total={t_total} < n_frames={self.n_frames}."
            )

        self.states = states
        self.actions = actions
        self.n_traj = n_traj

        # ── Precompute fixed templates for fixed-length trajectories ──
        self._reward_template = torch.zeros(self.n_frames, dtype=torch.float32)
        
        self._nonterminal_template = torch.ones(self.n_frames, dtype=torch.bool)
        self._nonterminal_template[-1] = False

        slice_info = ""
        if self._state_indices is not None or self._action_indices is not None:
            slice_info = f" (sliced to obs={self.observation_dim}D, act={self.action_dim}D)"
        print(
            f"[OfflineRLDataset] loaded npz={npz_path} "
            f"states={states.shape} actions={actions.shape} "
            f"n_frames={self.n_frames} usable_traj={self.n_traj}{slice_info}"
        )

    def __len__(self) -> int:
        return self.n_traj

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_np = np.asarray(self.states[idx, : self.n_frames], dtype=np.float32)
        act_np = np.asarray(self.actions[idx, : self.n_frames], dtype=np.float32)

        if self._state_indices is not None:
            obs_np = obs_np[:, self._state_indices]
        if self._action_indices is not None:
            act_np = act_np[:, self._action_indices]

        observation = torch.from_numpy(obs_np.copy())
        action = torch.from_numpy(act_np.copy())

        return observation, action, self._reward_template, self._nonterminal_template