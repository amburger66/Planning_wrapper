#!/usr/bin/env python3
"""
Compute per-dimension mean and std for PushBoundary 2D slices.

State (4D): indices [0, 1, 9, 10] -> [tcp_x, tcp_y, block_x, block_y]
Action (2D): indices [0, 1] -> [next_tcp_x, next_tcp_y]

Uses first 991 frames per trajectory (same trim as pushboundary_offline).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

STATE_INDICES = [0, 1, 9, 10]
ACTION_INDICES = [0, 1]
N_FRAMES = 991


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--states",
        default="demos/PushBoundary/states.npy",
        help="Path to states.npy",
    )
    p.add_argument(
        "--actions",
        default="demos/PushBoundary/actions.npy",
        help="Path to actions.npy",
    )
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    states_path = repo_root / args.states
    actions_path = repo_root / args.actions

    states = np.load(str(states_path), mmap_mode="r")
    actions = np.load(str(actions_path), mmap_mode="r")

    # Trim to first N_FRAMES per trajectory
    states_trim = np.asarray(states[:, :N_FRAMES, :], dtype=np.float64)
    actions_trim = np.asarray(actions[:, :N_FRAMES, :], dtype=np.float64)

    obs = states_trim[:, :, STATE_INDICES]  # (N, T, 4)
    act = actions_trim[:, :, ACTION_INDICES]  # (N, T, 2)

    obs_flat = obs.reshape(-1, obs.shape[-1])
    act_flat = act.reshape(-1, act.shape[-1])

    obs_mean = np.mean(obs_flat, axis=0)
    obs_std = np.std(obs_flat, axis=0)
    act_mean = np.mean(act_flat, axis=0)
    act_std = np.std(act_flat, axis=0)

    # Avoid zero std
    obs_std = np.where(obs_std < 1e-8, 1.0, obs_std)
    act_std = np.where(act_std < 1e-8, 1.0, act_std)

    print("observation_mean:", obs_mean.tolist())
    print("observation_std:", obs_std.tolist())
    print("action_mean:", act_mean.tolist())
    print("action_std:", act_std.tolist())


if __name__ == "__main__":
    main()
