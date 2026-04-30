#!/usr/bin/env python3
"""
Inference script for pushblock_offline diffusion planning.

Current pushblock_offline observation layout (6D):
  obs = [tcp_x, tcp_y, block_x, block_y, yaw_a, yaw_b]
where (yaw_a, yaw_b) is a continuous 2D encoding of block yaw (often cos,sin).

Usage (run from submodules/mctd/):
  python scripts/inference.py --checkpoint path/to/model.ckpt --mode unguided
  python scripts/inference.py --checkpoint path/to/model.ckpt --mode guided --num_samples 4

  # 4-value shorthand: tcp_x,tcp_y,block_x,block_y (yaw dims filled with obs mean)
  python scripts/inference.py --checkpoint path/to/model.ckpt --mode guided --start='-0.2,0.0,-0.15,0.0' --goal='0.1,0.1,0.05,0.05'

  # Full obs spec: provide all obs_dim values directly (e.g. 6 values for pushblock_offline)
  python scripts/inference.py --checkpoint path/to/model.ckpt --mode guided --start='...' --goal='...'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import glob
from pathlib import Path

import numpy as np
try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore

_SCRIPT_DIR = Path(__file__).resolve().parent
_MCTD_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_MCTD_ROOT))

import numpy as np
try:
    from shapely.geometry import Point, Polygon  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    Point = None  # type: ignore
    Polygon = None  # type: ignore

# Default indices for the 6D pushblock_offline observation vector:
#   [tcp_x, tcp_y, block_x, block_y, yaw_a, yaw_b, obs1_x, obs1_y...]
_TCP_XY_IDX = (0, 1)
_BLOCK_XY_IDX = (2, 3)
_BLOCK_YAW2_IDX = (4, 5)  # visualization expects (cos, sin) ordering

# 4-value shorthand: (tcp_x, tcp_y, block_x, block_y) → obs dims 0,1,2,3.
_SHORTHAND_4D_MAP = [0, 1, 2, 3]

# Set from args in main() before load_config() is called.
_FRAME_STACK: int = 10

BLOCK_HALF = 0.025
CIRCLE_RADIUS = 0.025
STICK_RADIUS = 0.01
OBSTACLE_RADIUS = 0.018


def valid_path(obs: np.ndarray, block_shape: str) -> bool:
    if Point is None or Polygon is None:
        raise ModuleNotFoundError(
            "shapely is required for valid_path(); install shapely or run without path validation."
        )
    # Baseline areas for percentage calculations
    gripper_total_area = np.pi * (STICK_RADIUS**2)
    obstacle_total_area = np.pi * (OBSTACLE_RADIUS**2)

    for t in range(obs.shape[0]):
        # 1. Extract Positions
        tx, ty = obs[t, list(_TCP_XY_IDX)]
        bx, by = obs[t, list(_BLOCK_XY_IDX)]
        if not np.isfinite([tx, ty, bx, by]).all():
            # Treat non-finite rollouts as invalid instead of crashing shapely.
            return False

        # Extract up to 4 obstacles, assuming they start at index 6
        obstacles_xy = []
        if obs.shape[1] >= 14:
            for i in range(6, 14, 2):
                ox, oy = obs[t, i], obs[t, i + 1]
                if not np.isfinite([ox, oy]).all():
                    return False
                obstacles_xy.append((ox, oy))

        # 2. Build Shapely Geometries
        gripper_geom = Point(tx, ty).buffer(STICK_RADIUS)
        obs_geoms = [Point(ox, oy).buffer(OBSTACLE_RADIUS) for ox, oy in obstacles_xy]

        if block_shape == "square":
            c, s = obs[t, list(_BLOCK_YAW2_IDX)]
            if not np.isfinite([c, s]).all():
                return False

            # Define square corners in local frame [-BLOCK_HALF, BLOCK_HALF]^2
            corners_local = [
                (BLOCK_HALF, BLOCK_HALF),
                (-BLOCK_HALF, BLOCK_HALF),
                (-BLOCK_HALF, -BLOCK_HALF),
                (BLOCK_HALF, -BLOCK_HALF),
            ]

            # Rotate using rotation matrix [[c, -s], [s, c]] and translate to world frame
            corners_world = []
            for lx, ly in corners_local:
                wx = bx + (c * lx - s * ly)
                wy = by + (s * lx + c * ly)
                corners_world.append((wx, wy))

            try:
                block_geom = Polygon(corners_world)
            except Exception:
                return False

        elif block_shape == "circle":
            block_geom = Point(bx, by).buffer(CIRCLE_RADIUS)
        else:
            raise ValueError(f"Unknown block_shape: {block_shape}")

        # 3. Perform Area Checks

        # Condition A: > 75% of the gripper is inside the block
        if gripper_geom.intersection(block_geom).area > (0.75 * gripper_total_area):
            print(f"TCP penetrated > 75% into the block at time {t}")
            return False

        # Condition B: > 75% of the gripper is inside ANY obstacle
        for idx, obs_geom in enumerate(obs_geoms):
            if gripper_geom.intersection(obs_geom).area > (0.75 * gripper_total_area):
                print(f"TCP penetrated > 75% into obstacle {idx+1} at time {t}")
                return False

        # Condition C: > 30% of ANY obstacle is covered by the block
        for idx, obs_geom in enumerate(obs_geoms):
            if block_geom.intersection(obs_geom).area > (0.30 * obstacle_total_area):
                print(f"Block covered > 30% of obstacle {idx+1} at time {t}")
                return False

    return True


def goal_reached(obs: np.ndarray, goal: np.ndarray, goal_threshold: float) -> bool:
    for t in range(obs.shape[0]):
        if not np.isfinite(obs[t, list(_BLOCK_XY_IDX)]).all():
            continue
        if (
            np.linalg.norm(obs[t, list(_BLOCK_XY_IDX)] - goal[list(_BLOCK_XY_IDX)])
            < goal_threshold
        ):
            print(f"Goal reached at time {t}")
            return True
    return False


def sample_goal_state_batched(states, max_retries=100):
    """
    Samples goal states for a batch of environments simultaneously.

    Args:
        states: numpy array of shape (B, 14)
        max_retries: int, maximum number of resampling attempts

    Returns:
        goals: numpy array of shape (B, 2) containing the (x, y) coordinates.
        valid: boolean array of shape (B,) indicating which environments
               successfully found a collision-free goal.
    """
    B = states.shape[0]

    # Extract geometries for the entire batch
    block_pos = states[:, 2:4]  # Shape: (B, 2)
    # Reshape the 8 obstacle coordinates into 4 (x,y) pairs per batch
    obstacles = states[:, 6:14].reshape(B, 4, 2)  # Shape: (B, 4, 2)

    # Calculate safe distance
    safe_dist = (np.sqrt(2) * BLOCK_HALF) + OBSTACLE_RADIUS + 0.002

    # Initialize output arrays and our completion mask
    goals = np.zeros((B, 2))
    valid = np.zeros(B, dtype=bool)

    for _ in range(max_retries):
        if np.all(valid):
            break  # Exit early if all batch elements have a valid goal

        # Extract indices of environments that still need a valid goal
        active_idx = np.where(~valid)[0]
        N = len(active_idx)

        # Subset the data to only compute for active environments
        act_block_pos = block_pos[active_idx]  # (N, 2)
        act_obstacles = obstacles[active_idx]  # (N, 4, 2)

        # 1. Randomly pick two distinct obstacles per active batch element.
        # Generating random numbers and sorting them is a highly efficient way
        # to sample without replacement across a batch.
        rand_indices = np.random.rand(N, 4).argsort(axis=1)[:, :2]
        idx1 = rand_indices[:, 0]
        idx2 = rand_indices[:, 1]

        o1 = act_obstacles[np.arange(N), idx1]  # (N, 2)
        o2 = act_obstacles[np.arange(N), idx2]  # (N, 2)

        # 2. Randomized midpoint
        t = np.random.uniform(0.3, 0.7, size=(N, 1))
        midpoint = o1 + t * (o2 - o1)  # (N, 2)

        # 3. Normal vectors
        direction = o2 - o1  # (N, 2)
        dir_norm = np.linalg.norm(direction, axis=1, keepdims=True)

        # Prevent division by zero if any obstacles overlap perfectly
        dir_norm[dir_norm < 1e-5] = 1e-5

        normal = np.empty_like(direction)
        normal[:, 0] = -direction[:, 1]
        normal[:, 1] = direction[:, 0]
        normal = normal / dir_norm  # (N, 2)

        # 4. Flip normals pointing towards the block
        vec_to_block = act_block_pos - midpoint  # (N, 2)
        dot_products = np.sum(normal * vec_to_block, axis=1, keepdims=True)  # (N, 1)
        # np.where conditional array replacement based on dot product
        normal = np.where(dot_products > 0, -normal, normal)

        # 5. Random offset
        max_offset = safe_dist + np.random.uniform(0.02, 0.08, size=(N, 1))
        offset_dist = np.random.uniform(safe_dist, max_offset, size=(N, 1))

        proposed_goals = midpoint + (normal * offset_dist)  # (N, 2)

        # 6. Collision Check against ALL 4 obstacles for these N elements
        # proposed_goals[:, np.newaxis, :] changes shape to (N, 1, 2)
        # allowing it to broadcast correctly against act_obstacles (N, 4, 2)
        dists = np.linalg.norm(
            act_obstacles - proposed_goals[:, np.newaxis, :], axis=2
        )  # (N, 4)
        min_dists = np.min(dists, axis=1)  # (N,)
        is_free = min_dists >= safe_dist  # (N,)

        # Map the active indices back to global batch indices and update
        successful_global_idx = active_idx[is_free]
        goals[successful_global_idx] = proposed_goals[is_free]
        valid[successful_global_idx] = True

    return goals, valid


def _mkdir(path: Path) -> None:
    """mkdir -p with 0o777 permissions, bypassing the process umask.

    When the script runs inside Docker as root, directories created with the
    default umask (022) end up as root:root 755 on the bind-mounted host
    filesystem — the host user can read but not write them.  Forcing 0o777
    makes every new directory component world-writable so the host user can
    delete, rename, or overwrite outputs without sudo.
    """
    prev = os.umask(0)
    try:
        path.mkdir(mode=0o777, parents=True, exist_ok=True)
    finally:
        os.umask(prev)


def _now_s() -> float:
    return time.perf_counter()


def _cuda_sync_if_needed(device: torch.device) -> None:
    # For accurate GPU timings, synchronize around timed regions.
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _summarize_times(records: list[dict]) -> dict:
    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"count": 0}
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "count": int(arr.size),
            "mean_sec": float(arr.mean()),
            "min_sec": float(arr.min()),
            "max_sec": float(arr.max()),
        }

    by_mode: dict[str, list[float]] = {}
    by_mode_and_scale: dict[str, list[float]] = {}
    for r in records:
        mode = str(r.get("mode", "unknown"))
        t = r.get("time_sec")
        if t is None:
            continue
        by_mode.setdefault(mode, []).append(float(t))
        if mode == "guided" and "guidance_scale" in r:
            key = f"{mode}/guidance_scale={r['guidance_scale']}"
            by_mode_and_scale.setdefault(key, []).append(float(t))

    out = {"by_mode": {k: _stats(v) for k, v in sorted(by_mode.items())}}
    if by_mode_and_scale:
        out["by_mode_and_guidance_scale"] = {
            k: _stats(v) for k, v in sorted(by_mode_and_scale.items())
        }
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="PushBoundary2D inference")
    parser.add_argument(
        "--checkpoint",
        default="/workspace/checkpoints/epoch=1063-step=50000.ckpt",
        help="Path to .ckpt or wandb run id (8 chars)",
    )
    parser.add_argument(
        "--mode",
        choices=["unguided", "guided", "mctd", "mctd_fast"],
        default="unguided",
        help="Inference mode",
    )
    parser.add_argument(
        "--num_samples", type=int, default=4, help="Number of trajectories"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("inference_results/cylinder"),
        help="Output directory for visualizations (GIF/MP4) and trajectories",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help=(
            "Start observation for guided mode. Either 4 values (tcp_x,tcp_y,block_x,block_y) "
            "as a shorthand — remaining dims filled with obs mean — or the full obs_dim "
            "observation vector (e.g. 6 values for pushblock_offline). "
            "Use --start='-0.2,0,-0.15,0' for negative values."
        ),
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help=(
            "Goal observation for guided mode. Same format as --start: 4-value shorthand "
            "(tcp_x,tcp_y,block_x,block_y) or full obs_dim vector."
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help=(
            "Guidance scale for guided mode when not using --guidance_scales (default: from config). "
            "Ignored when --guidance_scales is set."
        ),
    )
    parser.add_argument(
        "--guidance_scales",
        type=str,
        default=None,
        help=(
            "Guided mode only: comma-separated guidance scales (e.g. 0.1,0.5,1,2). "
            "Each run writes to output_dir/guided/scale_<tag>/. "
            "Omit for a single run under output_dir/guided/ using --guidance_scale or checkpoint default."
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Planning horizon in env steps (default: episode_len)",
    )
    parser.add_argument(
        "--frame_stack",
        type=int,
        default=10,
        help="frame_stack used during training (default: 4)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--format",
        choices=["gif", "mp4"],
        default="gif",
        help="Output format for trajectory visualizations (default: gif)",
    )
    parser.add_argument(
        "--no_reset_seed_between_guidance_scales",
        action="store_true",
        help=(
            "Guided guidance sweep only: do not reset torch/numpy RNG before each guidance scale "
            "(default resets so start/goal and noise match across scales)."
        ),
    )
    parser.add_argument(
        "--block_shape",
        choices=["square", "circle"],
        default="square",
        help="BEV block geometry in GIF/MP4 (circle matches envs/push_boundary.py CIRCLE_RADIUS)",
    )
    # Offline visualization from stored plan_*.npz files (no checkpoint needed).
    parser.add_argument(
        "--viz_from_npz_glob",
        type=str,
        default=None,
        help=(
            "If set, renders static overlay PNG(s) directly from stored plan_*.npz files "
            "(e.g. 'inference_obstacles/ckpt_400k_autoregressive/mctd/plan_*.npz') and exits."
        ),
    )
    parser.add_argument(
        "--viz_out_dir",
        type=Path,
        default=None,
        help="Optional output directory for overlay PNGs (default: alongside each .npz).",
    )
    parser.add_argument(
        "--viz_opacity_min",
        type=float,
        default=0.05,
        help="Minimum alpha for earliest timesteps in overlay (default: 0.05).",
    )
    parser.add_argument(
        "--viz_opacity_max",
        type=float,
        default=0.95,
        help="Maximum alpha for latest timesteps in overlay (default: 0.95).",
    )
    parser.add_argument(
        "--viz_gamma",
        type=float,
        default=1.0,
        help="Opacity ramp exponent gamma (default: 1.0 for linear).",
    )
    parser.add_argument(
        "--viz_max_steps",
        type=int,
        default=None,
        help="Optional max number of overlaid steps (subsamples evenly if set).",
    )
    parser.add_argument(
        "--viz_step_stride",
        type=int,
        default=5,
        help="Overlay stride in timesteps (default: 5). Always includes the final step.",
    )
    # MCTD-specific overrides (used only with --mode mctd)
    parser.add_argument(
        "--max_search_num",
        type=int,
        default=None,
        help="MCTD max search iterations (default: from config)",
    )
    parser.add_argument(
        "--num_denoising_steps",
        type=int,
        default=None,
        help="MCTD denoising steps per tree level (default: from config)",
    )
    parser.add_argument(
        "--mctd_guidance_scales",
        type=str,
        default="0,0.1,0.5,1",
        help="MCTD-only: comma-separated guidance scales for tree actions (e.g. 0,1,2,5,10)",
    )
    parser.add_argument(
        "--warp_threshold",
        type=float,
        default=0.015,
        help=(
            "MCTD-only: warp detection threshold in unnormalized obs-space units (TCP xy and block xy coordinates only)"
        ),
    )
    parser.add_argument(
        "--goal_threshold",
        type=float,
        default=0.0175,
        help=(
            "Goal-achievement threshold in unnormalized obs-space units (block xy coordinates only)"
        ),
    )
    parser.add_argument(
        "--scheduling_matrix",
        type=str,
        default="pyramid",
        help="Scheduling matrix (default: pyramid)",
    )
    parser.add_argument(
        "--divergence_threshold",
        type=float,
        default=None,
        help="MCTD-only: pairwise divergence threshold in unnormalized obs-space units (default: null)",
    )
    return parser.parse_args()


def _render_static_overlays_from_npz_glob(args: argparse.Namespace) -> int:
    from pushboundary_viz import render_pushboundary_plan_static_overlay

    pattern = str(args.viz_from_npz_glob).strip()
    if not pattern:
        sys.stderr.write("--viz_from_npz_glob was empty.\n")
        return 2

    npz_paths = sorted(glob.glob(pattern))
    if len(npz_paths) == 0:
        sys.stderr.write(f"No files matched --viz_from_npz_glob={pattern!r}\n")
        return 2

    n_ok = 0
    for p_str in npz_paths:
        p = Path(p_str)
        try:
            data = np.load(p, allow_pickle=False)
            states = np.asarray(data["states"])
            goal_marker = np.asarray(data["target_xy"]) if "target_xy" in data else None
            start_marker = (
                np.asarray(states[0, list(_BLOCK_XY_IDX)], dtype=np.float64)
                if states.ndim == 2 and states.shape[0] > 0 and states.shape[1] >= 4
                else None
            )
            out_dir = args.viz_out_dir if args.viz_out_dir is not None else p.parent
            out_path = Path(out_dir) / f"{p.stem}_overlay.png"
            render_pushboundary_plan_static_overlay(
                states=states,
                out_path=out_path,
                tcp_xy_indices=_TCP_XY_IDX,
                block_xy_indices=_BLOCK_XY_IDX,
                block_yaw_indices=_BLOCK_YAW2_IDX if states.shape[1] >= 6 else None,
                start_marker=start_marker,
                goal_marker=goal_marker,
                block_shape=str(args.block_shape),
                opacity_min=float(args.viz_opacity_min),
                opacity_max=float(args.viz_opacity_max),
                gamma=float(args.viz_gamma),
                step_stride=int(args.viz_step_stride),
                max_steps=args.viz_max_steps,
            )
            print(f"Wrote overlay: {out_path}")
            n_ok += 1
        except Exception as e:
            sys.stderr.write(f"[viz] Failed on {p}: {e}\n")

    if n_ok == 0:
        return 1
    return 0


def _parse_guidance_scales(s: str | None) -> list[float] | None:
    if s is None or not str(s).strip():
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return None
    out: list[float] = []
    seen: set[float] = set()
    for p in parts:
        v = float(p)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _guidance_scale_dirname(g: float) -> str:
    """Filesystem-safe tag for a guidance scale (used in scale_<tag> directories)."""
    s = f"{g:.6g}"
    return s.replace(".", "p").replace("-", "m")


def _build_hydra_overrides(args) -> list[str]:
    overrides = [
        "experiment=exp_planning",
        "dataset=pushblock_offline",
        "algorithm=df_planning",
        "+name=pushblock_inference",
        "wandb.mode=disabled",
        "algorithm.no_sim_env=true",
        f"algorithm.viz_block_shape={args.block_shape}",
        f"algorithm.frame_stack={_FRAME_STACK}",
        f"algorithm.scheduling_matrix={args.scheduling_matrix}",
    ]

    if args.mode in ("mctd", "mctd_fast"):
        overrides.append("algorithm.mctd=true")
        if args.mode == "mctd_fast":
            overrides.append("+algorithm._name=df_planning_fast")
        if args.max_search_num is not None:
            overrides.append(f"algorithm.mctd_max_search_num={args.max_search_num}")
        if args.num_denoising_steps is not None:
            overrides.append(
                f"algorithm.mctd_num_denoising_steps={args.num_denoising_steps}"
            )
        if (
            args.mctd_guidance_scales is not None
            and str(args.mctd_guidance_scales).strip()
        ):
            overrides.append(
                f"algorithm.mctd_guidance_scales=[{args.mctd_guidance_scales}]"
            )
        overrides.append(f"algorithm.warp_threshold={args.warp_threshold}")
        overrides.append(f"algorithm.goal_threshold={args.goal_threshold}")
        if args.divergence_threshold is not None:
            overrides.append(
                f"algorithm.mctd_pairwise_divergence_threshold={args.divergence_threshold}"
            )

    return overrides


def load_config(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    config_path = str(_MCTD_ROOT / "configurations")
    with initialize_config_dir(version_base=None, config_dir=config_path):
        cfg = compose(config_name="config", overrides=_build_hydra_overrides(args))
    with open_dict(cfg):
        cfg.experiment._name = "exp_planning"
        cfg.dataset._name = "pushblock_offline"
        # cfg.algorithm._name = "df_planning"
    return cfg


def resolve_checkpoint(checkpoint_arg: str) -> Path:
    from utils.ckpt_utils import download_latest_checkpoint, is_run_id

    path = Path(checkpoint_arg)
    print(path)
    if path.exists():
        return path
    if is_run_id(checkpoint_arg):
        # Minimal config for wandb path resolution
        class _Args:
            mode = "guided"
            block_shape = "square"
            max_search_num = None
            num_denoising_steps = None
            mctd_guidance_scales = None
            warp_threshold = 0.015
            goal_threshold = 0.03

        cfg = load_config(_Args())
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{checkpoint_arg}"
        return download_latest_checkpoint(run_path, Path("outputs/downloaded"))
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_arg}")


def build_algo_and_dataset(args):
    from experiments import build_experiment

    cfg = load_config(args)
    # print(cfg)
    experiment = build_experiment(cfg, logger=None, ckpt_path=None)
    algo = experiment._build_algo()
    dataset = experiment._build_dataset("validation")
    return algo, dataset, cfg


def load_checkpoint(algo, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "state_dict" in ckpt:
        algo.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        algo.load_state_dict(ckpt, strict=True)
    algo.eval()
    algo.to(device)
    return algo


def run_unguided(
    algo,
    dataset,
    num_samples: int,
    output_dir: Path,
    device: torch.device,
    output_format: str = "gif",
    mode_dir: Path | None = None,
    block_shape: str = "square",
    horizon: int | None = None,
):
    """
    Unguided rollout via ``algo.plan(..., guidance_scale=0)``.

    For each batch, the first observation of each trajectory (from the shuffled
    validation loader) is used as the start; ``goal`` is a dummy (cloned start)
    and is ignored because guidance is disabled. Length is ``H + 1`` frames
    (start plus ``H`` planned steps), where ``H`` is ``--horizon`` if set, else
    ``algo.episode_len``.

    If ``mode_dir`` is None, writes under ``output_dir / "unguided"``.
    """
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset, batch_size=min(num_samples, 4), shuffle=True, num_workers=0
    )
    if mode_dir is None:
        mode_dir = output_dir / "unguided"
    _mkdir(mode_dir)

    algo.eval()
    H = horizon if horizon is not None else algo.episode_len
    obs_mean = np.array(algo.observation_mean, dtype=np.float32)
    obs_std = np.array(algo.observation_std, dtype=np.float32)
    obs_mean_t = torch.from_numpy(obs_mean).to(device).float()
    obs_std_t = torch.from_numpy(obs_std).to(device).float()
    model_device = next(algo.parameters()).device
    times_path = mode_dir / "inference_times.jsonl"

    trajectories = []
    sample_idx = 0
    tcp_xy_indices, block_xy_indices, block_yaw_indices = _infer_viz_indices(algo)

    for batch_idx, batch in enumerate(dataloader):
        if sample_idx >= num_samples:
            break
        batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
        observations = batch[0][..., : algo.observation_dim].float()
        batch_size = observations.shape[0]
        start_norm_batch = (observations[:, 0] - obs_mean_t) / obs_std_t

        for i in range(batch_size):
            if sample_idx >= num_samples:
                break
            start_i = start_norm_batch[i : i + 1].to(model_device).contiguous()
            goal_i = start_i.clone()
            _cuda_sync_if_needed(model_device)
            t0 = _now_s()
            with torch.no_grad():
                plan_hist = algo.plan(start_i, goal_i, H, guidance_scale=0)
            _cuda_sync_if_needed(model_device)
            dt = _now_s() - t0
            _append_jsonl(
                times_path,
                {
                    "mode": "unguided",
                    "sample_idx": int(sample_idx),
                    "time_sec": float(dt),
                    "guidance_scale": 0.0,
                    "horizon": int(H),
                    "frame_stack": int(getattr(algo, "frame_stack", -1)),
                    "obs_dim": int(getattr(algo, "observation_dim", -1)),
                    "device": str(model_device),
                },
            )

            pf = plan_hist[-1][:, 0:1, :]
            plan_unnorm = algo._unnormalize_x(pf)
            obs_unnorm, _, _ = algo.split_bundle(plan_unnorm)
            start_unnorm = (start_i[0] * obs_std_t + obs_mean_t).unsqueeze(0)
            start_obs = start_unnorm[:, : algo.observation_dim].unsqueeze(1)
            full_traj = torch.cat(
                [start_obs, obs_unnorm[:, : algo.observation_dim]],
                0,
            )
            states = full_traj.detach().cpu().numpy().astype(np.float32)
            trajectories.append(states)
            states_2d = states.squeeze(1)
            np.savez(
                mode_dir / f"trajectory_{sample_idx}.npz",
                states=states_2d.astype(np.float32),
                source="diffuser_inference",
                version=np.int32(1),
                mode="unguided",
                horizon=np.int32(H),
            )
            viz_dir = (
                mode_dir / "gifs" if output_format == "gif" else mode_dir / "videos"
            )
            algo._log_or_save_pushboundary_2d_gif(
                namespace="unguided",
                states=states_2d,
                sample_idx=sample_idx,
                tcp_xy_indices=tcp_xy_indices,
                block_xy_indices=block_xy_indices,
                block_yaw_indices=block_yaw_indices,
                gif_out_dir=viz_dir,
                output_format=output_format,
                block_shape=block_shape,
            )
            sample_idx += 1

    return trajectories


def _expand_obs_shorthand(values: list[float], obs_mean: np.ndarray) -> np.ndarray:
    """
    Expand a user-provided observation specification to the full obs_dim.

    - If len(values) == obs_dim: used directly.
    - If len(values) == 4: treated as (tcp_x, tcp_y, block_x, block_y), mapped to
      dims 0,1,2,3; all other dims (including yaw encoding) filled from obs_mean.
    - Otherwise: raises ValueError.
    """
    obs_dim = len(obs_mean)
    if len(values) == obs_dim:
        return np.array(values, dtype=np.float32)
    if len(values) == 4:
        obs = obs_mean.copy()
        for dst, src in zip(_SHORTHAND_4D_MAP, values):
            obs[dst] = src
        return obs.astype(np.float32)
    raise ValueError(
        f"--start/--goal must have 4 values (tcp_x,tcp_y,block_x,block_y shorthand) "
        f"or {obs_dim} values (full observation); got {len(values)}."
    )


def _normalize_obs(obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((obs - mean) / std).float()


def _unnormalize_obs_tensor(
    obs_norm: torch.Tensor, obs_mean: np.ndarray, obs_std: np.ndarray
) -> torch.Tensor:
    """Unnormalize an observation-only tensor using obs stats."""
    mean_t = torch.from_numpy(np.asarray(obs_mean, dtype=np.float32)).to(
        obs_norm.device
    )
    std_t = torch.from_numpy(np.asarray(obs_std, dtype=np.float32)).to(obs_norm.device)
    return obs_norm * std_t + mean_t


def _infer_viz_indices(
    algo,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int] | None]:
    """
    Determine indices for TCP XY, block XY, and (optionally) a 2D yaw encoding in the
    current observation representation.

    For the circle 2D offline dataset, observations are 4D:
        [tcp_x, tcp_y, block_x, block_y]
    because the dataset slices the original 18D pushblock state via
    state_indices [0, 1, 9, 10].
    """
    obs_dim = int(getattr(algo, "observation_dim", 0))
    if obs_dim == 4:
        # No yaw channels exist in the 4D representation.
        return (0, 1), (2, 3), None
    if obs_dim == 6:
        return

    # Fallback: treat unknown layouts as no-yaw for visualization.
    return _TCP_XY_IDX, _BLOCK_XY_IDX, _BLOCK_YAW2_IDX


def _block_xy_from_obs_unnorm(obs_unnorm_1d: torch.Tensor, algo) -> np.ndarray:
    """Extract unnormalised block XY from a 1D observation tensor."""
    obs_dim = int(getattr(algo, "observation_dim", 0))
    if obs_dim == 4:
        xy = obs_unnorm_1d[[2, 3]]
    elif obs_dim == 6:
        xy = obs_unnorm_1d[[2, 3]]
    else:
        xy = obs_unnorm_1d[[2, 3]]
    return xy.detach().cpu().numpy().astype(np.float32)


def _get_start_goal_from_dataset(dataset, num_samples, algo, device):
    from torch.utils.data import DataLoader

    obs_mean = np.array(algo.observation_mean, dtype=np.float32)
    obs_std = np.array(algo.observation_std, dtype=np.float32)
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    obs, *_ = batch
    obs = obs[:, :, : algo.observation_dim]
    start = obs[:, 0].float().to(device)
    goal = obs[:, -1].float().to(device)
    mean_t = torch.from_numpy(obs_mean).to(device)
    std_t = torch.from_numpy(obs_std).to(device)
    start_norm = ((start - mean_t) / std_t).float()
    goal_norm = ((goal - mean_t) / std_t).float()
    return start_norm, goal_norm


def _get_start_goal_from_dataset_with_unnorm(
    dataset, num_samples: int, algo, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Sample start/goal and return both normalized tensors and unnormalized numpy arrays."""
    from torch.utils.data import DataLoader

    obs_mean = np.array(algo.observation_mean, dtype=np.float32)
    obs_std = np.array(algo.observation_std, dtype=np.float32)

    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    obs, *_ = batch
    obs = obs[:, :, : algo.observation_dim]

    start_unnorm = obs[:, 0].float().cpu().numpy().astype(np.float32)
    goal_unnorm = obs[:, -1].float().cpu().numpy().astype(np.float32)

    start_norm = _normalize_obs(start_unnorm, obs_mean, obs_std).to(device)
    goal_norm = _normalize_obs(goal_unnorm, obs_mean, obs_std).to(device)

    return start_norm, goal_norm, start_unnorm, goal_unnorm


def run_mctd(
    algo,
    dataset,
    args,
    output_dir: Path,
    device: torch.device,
):
    """Run MCTD planning (algo.p_mctd_plan) for each start/goal pair."""
    mode_tag = str(getattr(args, "mode", "mctd"))
    obs_mean = np.array(algo.observation_mean, dtype=np.float32)
    obs_std = np.array(algo.observation_std, dtype=np.float32)

    if args.start is not None and args.goal is not None:
        start_vals = [float(x) for x in args.start.split(",")]
        goal_obs = [float(x) for x in args.goal.split(",")]
        start_obs = _expand_obs_shorthand(start_vals, obs_mean)[None]
        start_unnorm = np.repeat(start_obs.astype(np.float32), args.num_samples, axis=0)
        goal_unnorm = np.repeat(goal_obs.astype(np.float32), args.num_samples, axis=0)
        start_norm = _normalize_obs(start_unnorm, obs_mean, obs_std).to(device)
        goal_norm = _normalize_obs(goal_unnorm, obs_mean, obs_std).to(device)
    else:
        assert (
            dataset is not None
        ), "MCTD mode requires a dataset for sampling start/goal."

        start_norm, goal_norm, start_unnorm, goal_unnorm = (
            _get_start_goal_from_dataset_with_unnorm(
                dataset, args.num_samples, algo, device
            )
        )

        goal_states = sample_goal_state_batched(start_unnorm, max_retries=100)[0]
        goal_states_norm = _normalize_obs(goal_states, obs_mean[2:4], obs_std[2:4])

        goal_nan = torch.full_like(start_norm, float("nan"))
        goal_nan[:, 2:4] = (
            goal_states_norm  # block xy for validity checking and visualization, but not guidance
        )
        goal_norm = goal_nan

        goal_unnorm_nan = np.full_like(start_unnorm, float("nan"), dtype=np.float32)
        goal_unnorm_nan[:, 2:4] = (
            goal_states  # block xy for validity checking and visualization, but not guidance
        )
        goal_unnorm = goal_unnorm_nan

    horizon = args.horizon if args.horizon is not None else algo.episode_len

    _mkdir(output_dir)
    viz_dir = output_dir / ("gifs" if args.format == "gif" else "videos")
    times_path = output_dir / "inference_times.jsonl"

    tcp_xy_indices, block_xy_indices, block_yaw_indices = _infer_viz_indices(algo)

    model_device = next(algo.parameters()).device

    for i in range(args.num_samples):
        print(f"[{i+1}/{args.num_samples}] Running {mode_tag} search...")
        start_i = start_norm[i : i + 1].to(model_device).float().contiguous()
        goal_i = goal_norm[i : i + 1].to(model_device).float().contiguous()
        start_unnorm_i = start_unnorm[i : i + 1]
        goal_unnorm_i = goal_unnorm[i : i + 1]

        _cuda_sync_if_needed(model_device)
        t0 = _now_s()
        with torch.no_grad():
            solved, plan_hist = algo.p_mctd_plan(
                start_i,
                goal_i,
                horizon,
                None,  # conditions
                start_unnorm_i,
                goal_unnorm_i,
            )
        _cuda_sync_if_needed(model_device)
        dt = _now_s() - t0
        _append_jsonl(
            times_path,
            {
                "mode": mode_tag,
                "sample_idx": int(i),
                "time_sec": float(dt),
                "horizon": int(horizon),
                "frame_stack": int(getattr(algo, "frame_stack", -1)),
                "obs_dim": int(getattr(algo, "observation_dim", -1)),
                "device": str(model_device),
                "max_search_num": args.max_search_num,
                "num_denoising_steps": args.num_denoising_steps,
                "mctd_guidance_scales": args.mctd_guidance_scales,
                "warp_threshold": float(args.warp_threshold),
                "goal_threshold": float(args.goal_threshold),
            },
        )

        snap = getattr(algo, "_mctd_last_pairwise_divergence_snapshot", None)
        if (
            getattr(algo, "mctd_pairwise_divergence_threshold", None) is not None
            and snap is not None
        ):
            pd_path = output_dir / "pairwise_first_divergence.jsonl"
            _append_jsonl(
                pd_path,
                {
                    "mode": mode_tag,
                    "sample_idx": int(i),
                    "solved": bool(solved),
                    "time_sec": float(dt),
                    **snap,
                },
            )
            algo._mctd_pairwise_divergence_last_save_path = str(pd_path.resolve())

        # p_mctd_plan may return obs-only (normalized) or full bundle. Handle both.
        # Shapes observed in this codebase:
        # - (1, t, 1, obs_dim)  (obs-only, normalized)
        # - (1, t, 1, bundle_dim) (full bundle, normalized)
        if plan_hist.ndim != 4:
            raise ValueError(
                f"Unexpected p_mctd_plan output shape: {tuple(plan_hist.shape)}"
            )
        plan_last_dim = int(plan_hist.shape[-1])
        plan_final = plan_hist[-1]  # (t, 1, C)
        if plan_last_dim == int(algo.observation_dim):
            obs_traj_unnorm = _unnormalize_obs_tensor(plan_final, obs_mean, obs_std)
            states = obs_traj_unnorm[:, 0, :].detach().cpu().numpy().astype(np.float32)
        else:
            plan_unnorm = algo._unnormalize_x(plan_final)
            obs_traj, _, _ = algo.split_bundle(plan_unnorm)
            states = obs_traj[:, 0, :].detach().cpu().numpy().astype(np.float32)

        start_marker = start_unnorm_i[0, list(block_xy_indices)].astype(np.float32)
        goal_marker = goal_unnorm_i[0, list(block_xy_indices)].astype(np.float32)

        valid = valid_path(states, args.block_shape)

        if valid and solved:
            out_path = output_dir / f"plan_{i}_success.npz"
            suffix = "success"
        elif not valid and solved:
            out_path = output_dir / f"plan_{i}_failed_solved.npz"
            suffix = "failed_solved"
        else:
            out_path = output_dir / f"plan_{i}_failed.npz"
            suffix = "failed"

        np.savez(
            out_path,
            states=states.astype(np.float32),
            target_xy=goal_marker.astype(np.float32),
            solved=np.bool_(solved and valid),
            source="mctd_inference",
            version=np.int32(1),
            mode=mode_tag,
            horizon=np.int32(horizon),
            seed=np.int32(args.seed),
        )
        print(f"  Saved plan to {out_path}")
        algo._log_or_save_pushboundary_2d_gif(
            namespace=mode_tag,
            states=states,
            sample_idx=i,
            suffix=suffix,
            tcp_xy_indices=tcp_xy_indices,
            block_xy_indices=block_xy_indices,
            block_yaw_indices=block_yaw_indices,
            gif_out_dir=viz_dir,
            start_marker=start_marker,
            goal_marker=goal_marker,
            output_format=args.format,
            block_shape=args.block_shape,
        )

    print(f"\nAll outputs saved under {output_dir}")


def run_guided(
    algo,
    dataset,
    args,
    output_dir: Path,
    device: torch.device,
    output_format: str = "gif",
    *,
    guidance_scale: float | None = None,
    mode_dir: Path | None = None,
):
    """
    Goal-conditioned planning. If ``mode_dir`` is None, writes under ``output_dir / "guided"``.
    If ``guidance_scale`` is None, uses ``args.guidance_scale`` or the checkpoint default.
    """
    obs_mean = np.array(algo.observation_mean, dtype=np.float32)
    obs_std = np.array(algo.observation_std, dtype=np.float32)

    if args.start is not None and args.goal is not None:
        start_vals = [float(x) for x in args.start.split(",")]
        goal_obs = [float(x) for x in args.goal.split(",")]
        start_obs = _expand_obs_shorthand(start_vals, obs_mean)
        start_norm = _normalize_obs(start_obs[None], obs_mean, obs_std).to(device)
        goal_norm = _normalize_obs(goal_obs[None], obs_mean, obs_std).to(device)
        start_norm = start_norm.repeat(args.num_samples, 1)
        goal_norm = goal_norm.repeat(args.num_samples, 1)
    else:
        assert (
            dataset is not None
        ), "MCTD mode requires a dataset for sampling start/goal."

        start_norm, goal_norm, start_unnorm, goal_unnorm = (
            _get_start_goal_from_dataset_with_unnorm(
                dataset, args.num_samples, algo, device
            )
        )

        goal_states = sample_goal_state_batched(start_unnorm, max_retries=100)[0]
        goal_states_norm = _normalize_obs(goal_states, obs_mean[2:4], obs_std[2:4])

        goal_nan = torch.full_like(start_norm, float("nan"))
        goal_nan[:, 2:4] = (
            goal_states_norm  # block xy for validity checking and visualization, but not guidance
        )
        goal_norm = goal_nan

        goal_unnorm_nan = np.full_like(start_unnorm, float("nan"), dtype=np.float32)
        goal_unnorm_nan[:, 2:4] = (
            goal_states  # block xy for validity checking and visualization, but not guidance
        )
        goal_unnorm = goal_unnorm_nan

    horizon = args.horizon if args.horizon is not None else algo.episode_len
    if guidance_scale is None:
        guidance_scale = (
            args.guidance_scale
            if args.guidance_scale is not None
            else algo.guidance_scale
        )

    if mode_dir is None:
        mode_dir = output_dir / "guided"
    _mkdir(mode_dir)
    viz_dir = mode_dir / "gifs" if output_format == "gif" else mode_dir / "videos"

    model_device = next(algo.parameters()).device
    times_path = mode_dir / "inference_times.jsonl"
    trajectories = []
    batch_size = start_norm.shape[0]
    tcp_xy_indices, block_xy_indices, block_yaw_indices = _infer_viz_indices(algo)
    for i in range(batch_size):
        start_i = start_norm[i : i + 1].to(model_device).float().contiguous()
        goal_i = goal_norm[i : i + 1].to(model_device).float().contiguous()
        _cuda_sync_if_needed(model_device)
        t0 = _now_s()
        with torch.no_grad():
            plan_hist = algo.plan(
                start_i, goal_i, horizon, guidance_scale=guidance_scale
            )
        _cuda_sync_if_needed(model_device)
        dt = _now_s() - t0
        _append_jsonl(
            times_path,
            {
                "mode": "guided",
                "sample_idx": int(i),
                "time_sec": float(dt),
                "guidance_scale": float(guidance_scale),
                "horizon": int(horizon),
                "frame_stack": int(getattr(algo, "frame_stack", -1)),
                "obs_dim": int(getattr(algo, "observation_dim", -1)),
                "device": str(model_device),
            },
        )
        plan_final = plan_hist[-1]
        plan_unnorm = algo._unnormalize_x(plan_final)
        obs_unnorm, _, _ = algo.split_bundle(plan_unnorm)
        start_unnorm = (
            start_norm[i] * torch.from_numpy(obs_std).to(device)
            + torch.from_numpy(obs_mean).to(device)
        ).unsqueeze(0)
        start_obs = start_unnorm[:, : algo.observation_dim].unsqueeze(1)
        full_traj = torch.cat(
            [start_obs, obs_unnorm[:, : algo.observation_dim]],
            0,
        )
        states = full_traj.detach().cpu().numpy().astype(np.float32)
        trajectories.append(states)
        states_2d = states.squeeze(1)
        goal_unnorm = goal_norm[i] * torch.from_numpy(obs_std).to(
            device
        ) + torch.from_numpy(obs_mean).to(device)
        start_marker = _block_xy_from_obs_unnorm(start_unnorm[0], algo)
        goal_marker = _block_xy_from_obs_unnorm(goal_unnorm, algo)

        # TODO: add solved flag and check for invalid path
        valid = valid_path(states_2d, args.block_shape)
        solved = goal_reached(
            states_2d, goal_unnorm.detach().cpu().numpy(), args.goal_threshold
        )

        if valid and solved:
            out_path = mode_dir / f"plan_{i}_success.npz"
            suffix = "success"
        elif not valid and solved:
            out_path = mode_dir / f"plan_{i}_failed_solved.npz"
            suffix = "failed_solved"
        else:
            out_path = mode_dir / f"plan_{i}_failed.npz"
            suffix = "failed"

        np.savez(
            out_path,
            states=states_2d.astype(np.float32),
            target_xy=goal_marker.astype(np.float32),
            solved=np.bool_(solved and valid),
            source="diffuser_inference",
            version=np.int32(1),
            mode="guided",
            guidance_scale=np.float32(guidance_scale),
            horizon=np.int32(horizon),
        )

        algo._log_or_save_pushboundary_2d_gif(
            namespace="guided",
            states=states_2d,
            sample_idx=i,
            suffix=suffix,
            tcp_xy_indices=tcp_xy_indices,
            block_xy_indices=block_xy_indices,
            block_yaw_indices=block_yaw_indices,
            gif_out_dir=viz_dir,
            start_marker=start_marker,
            goal_marker=goal_marker,
            output_format=output_format,
            block_shape=args.block_shape,
        )
    return trajectories


def main():
    global _FRAME_STACK
    args = parse_args()
    _FRAME_STACK = args.frame_stack
    if args.viz_from_npz_glob is not None:
        raise SystemExit(_render_static_overlays_from_npz_glob(args))
    if torch is None:
        raise ModuleNotFoundError(
            "torch is required for model inference; install torch or use --viz_from_npz_glob."
        )
    if (
        args.guidance_scales is not None
        and str(args.guidance_scales).strip()
        and args.mode != "guided"
    ):
        sys.stderr.write("--guidance_scales is only supported with --mode guided.\n")
        sys.exit(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = resolve_checkpoint(args.checkpoint)
    print(f"Loading checkpoint from {ckpt_path}")

    algo, dataset, _ = build_algo_and_dataset(args)
    algo = load_checkpoint(algo, ckpt_path, device)

    output_dir = Path(args.output_dir)
    _mkdir(output_dir)

    # sample random start state from the dataset
    dataset_states = dataset.states
    start_state = dataset_states[np.random.choice(len(dataset_states))]
    print(f"Sampled random start state from dataset: {start_state}")

    if args.mode == "unguided":
        run_unguided(
            algo,
            dataset,
            args.num_samples,
            output_dir,
            device,
            args.format,
            block_shape=args.block_shape,
            horizon=args.horizon,
        )
    elif args.mode == "guided":
        scale_list = _parse_guidance_scales(args.guidance_scales)
        if scale_list is None:
            run_guided(algo, dataset, args, output_dir, device, args.format)
        else:
            guided_root = output_dir / "guided"
            _mkdir(guided_root)
            sweep_runs: list[dict] = []
            for g in scale_list:
                if not args.no_reset_seed_between_guidance_scales:
                    torch.manual_seed(args.seed)
                    np.random.seed(args.seed)
                tag = _guidance_scale_dirname(g)
                subdir = guided_root / f"scale_{tag}"
                trajs = run_guided(
                    algo,
                    dataset,
                    args,
                    output_dir,
                    device,
                    args.format,
                    guidance_scale=g,
                    mode_dir=subdir,
                )
                sweep_runs.append(
                    {
                        "guidance_scale": g,
                        "relative_dir": str(subdir.relative_to(output_dir)),
                        "num_trajectories": len(trajs),
                        "trajectory_npy_glob": "trajectory_*.npy",
                    }
                )
            manifest = {
                "mode": "guided_guidance_sweep",
                "seed": args.seed,
                "reset_seed_between_guidance_scales": not args.no_reset_seed_between_guidance_scales,
                "num_samples_requested": args.num_samples,
                "runs": sweep_runs,
            }
            manifest_path = guided_root / "guidance_sweep_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(
                f"Guidance sweep: wrote {len(scale_list)} runs under {guided_root}/scale_<tag>/ "
                f"and {manifest_path.name}"
            )
    elif args.mode == "mctd":
        run_mctd(algo, dataset, args, output_dir / "mctd", device)
    elif args.mode == "mctd_fast":
        run_mctd(algo, dataset, args, output_dir / "mctd_fast", device)

    # Write summary aggregating all timing files written during this run.
    records: list[dict] = []
    for p in output_dir.rglob("inference_times.jsonl"):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        except Exception:
            pass
    if records:
        summary = _summarize_times(records)
        _write_json(output_dir / "inference_times_summary.json", summary)

    print(f"Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
