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

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--states",
        default="demos/PushBoundary/states.npy",
        help="Path to states.npy",
    )
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    states_path = repo_root / args.states
    # actions_path = repo_root / args.actions

    
    states = np.load(str(states_path), mmap_mode="r")

    obs_flat = states.reshape(-1, states.shape[-1])
        
    obs_mean = np.mean(obs_flat, axis=0)
    obs_std = np.std(obs_flat, axis=0)

    # Avoid zero std
    obs_std = np.where(obs_std < 1e-8, 1.0, obs_std)
    # act_std = np.where(act_std < 1e-8, 1.0, act_std)

    print("observation_mean:", obs_mean.tolist())
    print("observation_std:", obs_std.tolist())
    # print("action_mean:", act_mean.tolist())
    # print("action_std:", act_std.tolist())


if __name__ == "__main__":
    main()