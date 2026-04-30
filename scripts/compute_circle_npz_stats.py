#!/usr/bin/env python3
"""
Compute observation/action mean and std from a padded circle NPZ, using only valid timesteps.

Example (from repo root):
  python scripts/compute_circle_npz_stats.py --npz demos/circle_demos_full.npz

Paste the printed YAML fragments into submodules/mctd/configurations/dataset/circle_2d_offline.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--npz",
        type=Path,
        required=True,
        help="Path to .npz with states, actions, valid_lengths",
    )
    p.add_argument(
        "--state-indices",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17",
        help="Comma-separated indices into full state (default: 2D PushBoundary slice)",
    )
    p.add_argument(
        "--action-indices",
        type=str,
        default="0,1,2,3,4,5,6,7,8",
        help="Comma-separated indices into full action",
    )
    args = p.parse_args()

    state_indices = [int(x.strip()) for x in args.state_indices.split(",") if x.strip()]
    action_indices = [
        int(x.strip()) for x in args.action_indices.split(",") if x.strip()
    ]

    data = np.load(args.npz)
    states = data["states"].astype(np.float64)
    actions = data["actions"].astype(np.float64)
    valid_lengths = data["valid_lengths"].astype(np.int64)

    obs_chunks = []
    act_chunks = []
    for n in range(states.shape[0]):
        t_valid = int(valid_lengths[n])
        if t_valid <= 0:
            continue
        obs_chunks.append(states[n, :t_valid][:, state_indices])
        act_chunks.append(actions[n, :t_valid][:, action_indices])

    if not obs_chunks:
        raise SystemExit("No trajectories with valid_lengths >= 1")

    obs_all = np.concatenate(obs_chunks, axis=0)
    act_all = np.concatenate(act_chunks, axis=0)

    obs_mean = obs_all.mean(axis=0)
    obs_std = obs_all.std(axis=0)
    act_mean = act_all.mean(axis=0)
    act_std = act_all.std(axis=0)

    # Avoid zero std
    obs_std = np.maximum(obs_std, 1e-8)
    act_std = np.maximum(act_std, 1e-8)

    # Frame-to-frame displacement statistics (per valid trajectory, then pooled).
    # Used to calibrate warp_threshold for calculate_values() in df_planning.py.
    disp_norms = []
    for chunk in obs_chunks:
        if len(chunk) < 2:
            continue
        diffs = chunk[1:] - chunk[:-1]  # (T-1, obs_dim)
        norms = np.linalg.norm(diffs, axis=-1)  # (T-1,)
        disp_norms.append(norms)
    disp_all = np.concatenate(disp_norms)  # (total_steps,)
    disp_p50 = float(np.percentile(disp_all, 50))
    disp_p95 = float(np.percentile(disp_all, 95))
    disp_p99 = float(np.percentile(disp_all, 99))
    disp_max = float(disp_all.max())

    def fmt(a: np.ndarray) -> str:
        return "[" + ", ".join(f"{float(x):.8g}" for x in a.tolist()) + "]"

    print(
        f"# From {args.npz.resolve()}  (N={states.shape[0]}, valid steps pooled={obs_all.shape[0]})"
    )
    print(f"observation_mean: {fmt(obs_mean)}")
    print(f"observation_std:  {fmt(obs_std)}")
    print(f"action_mean:      {fmt(act_mean)}")
    print(f"action_std:       {fmt(act_std)}")
    print()
    print("# Frame-to-frame displacement norms (unnormalized obs space, 4-D Euclidean)")
    print(f"#   p50  = {disp_p50:.6f}   (typical step size)")
    print(f"#   p95  = {disp_p95:.6f}   (fast-but-plausible motion)")
    print(f"#   p99  = {disp_p99:.6f}   (near-maximum real motion)")
    print(f"#   max  = {disp_max:.6f}   (absolute maximum in dataset)")
    print(f"#")
    print(
        f"# Suggested warp_threshold for circle_2d: ~3–5× p99, e.g. {disp_p99 * 4:.4f}"
    )
    print(f"# (catches diffusion teleportation while allowing all real motion)")
    print(f"# Add to your inference command: --warp_threshold {disp_p99 * 4:.4f}")


if __name__ == "__main__":
    main()
