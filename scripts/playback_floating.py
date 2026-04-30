#!/usr/bin/env python3
"""
scripts/playback_floating.py

Playback and evaluation for block-pushing with the floating gripper.

Two main entry points
---------------------

1. evaluate_predictions(...)
   Takes predicted gripper actions and predicted block states, runs them in
   simulation, and returns per-step accuracy metrics.

   Inputs:
     initial_block_pos   (3,)    world-frame XYZ of the block at t=0
     initial_block_quat  (4,)    [w,x,y,z] block orientation at t=0
     initial_gripper_xy  (2,)    world-frame XY of the gripper at t=0
     predicted_actions   (T, 2)  gripper delta-XY actions to replay
     predicted_block_pos (T, 3)  model's predicted block XYZ at each step
     predicted_block_quat(T, 4)  model's predicted block orientation (optional)

   Returns a dict with:
     actual_block_pos    (T, 3)  simulated block positions
     actual_block_quat   (T, 4)  simulated block orientations
     position_errors     (T,)    L2 distance between predicted and actual (metres)
     yaw_errors          (T,)    absolute yaw error in degrees
     mean_position_error float
     final_position_error float
     out_of_bounds_steps int     number of steps the block was outside boundary
     trajectory_success  bool    True if block never left boundary

2. evaluate_target(...)
   Takes gripper actions and a target location, runs them in simulation, and
   reports whether the block reaches the target within the overlap threshold.

   Inputs:
     initial_block_pos   (3,)
     initial_block_quat  (4,)
     initial_gripper_xy  (2,)
     actions             (T, 2)
     target_xy           (2,)    target block XY position
     success_radius      float   block centroid must be within this distance (m)

   Returns a dict with:
     success             bool
     min_distance        float   closest the block came to the target
     final_distance      float   distance at end of rollout
     actual_block_pos    (T, 3)
     actual_block_quat   (T, 4)
     out_of_bounds_steps int

CLI usage
---------
    # Replay a saved NPZ of predictions
    python scripts/playback_floating.py --npz path/to/predictions.npz --render

    # Replay from an HDF5 demo trajectory (ground-truth actions)
    python scripts/playback_floating.py --h5 demos/all_demos.h5 --traj traj_0 --render

NPZ format (for --npz)
-----------------------
    initial_block_pos   (3,)
    initial_block_quat  (4,)
    initial_gripper_xy  (2,)
    predicted_actions   (T, 2)
    predicted_block_pos (T, 3)         optional
    predicted_block_quat (T, 4)        optional
    target_xy           (2,)           optional
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import sapien
import gymnasium as gym
import torch

import envs  # noqa: F401 — registers PushBoundary + FloatingGripper

from utils.output_conversions import convert, CANONICAL_BLOCK_POS

from envs.push_boundary import (
    BOUNDARY_CENTER_X,
    BOUNDARY_CENTER_Y,
    BOUNDARY_HALF_X,
    BOUNDARY_HALF_Y,
    GRIPPER_Z_FIXED,
    OUT_MARGIN,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _unwrap(env):
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e


def _yaw_from_quat_wxyz(q: np.ndarray) -> float:
    """Extract yaw (radians) from a [w, x, y, z] quaternion."""
    q = np.asarray(q, dtype=np.float64).ravel()
    r = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses [x,y,z,w]
    return float(r.as_euler("xyz")[2])


def _quat_wxyz_to_sapien(q: np.ndarray) -> list:
    """[w,x,y,z] → sapien.Pose q= list."""
    q = np.asarray(q, dtype=np.float64).ravel()
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def _block_out_of_bounds(block_xy: np.ndarray) -> bool:
    x, y = float(block_xy[0]), float(block_xy[1])
    return (
        x < BOUNDARY_CENTER_X - BOUNDARY_HALF_X - OUT_MARGIN
        or x > BOUNDARY_CENTER_X + BOUNDARY_HALF_X + OUT_MARGIN
        or y < BOUNDARY_CENTER_Y - BOUNDARY_HALF_Y - OUT_MARGIN
        or y > BOUNDARY_CENTER_Y + BOUNDARY_HALF_Y + OUT_MARGIN
    )


def _build_env(
    render: bool,
    target_xy: Optional[np.ndarray] = None,
    video_dir: Optional[str] = None,
    video_name: str = "playback",
):
    """
    Create a PushBoundaryEnv.

    render=True  → render_mode="all" (offscreen, headless-safe).
    video_dir    → wrap with RecordEpisode so frames are written to disk.
                   If render=False and video_dir is set, still records video.
    """
    needs_render = render or (video_dir is not None)

    kwargs: dict = dict(
        obs_mode="state_dict",
        control_mode="floating_vel",
        render_mode="all" if needs_render else None,
        sim_backend="cpu",
        num_envs=1,
        robot_uids="floating_gripper",
    )
    if target_xy is not None:
        kwargs["target_xy"] = (float(target_xy[0]), float(target_xy[1]))

    env = gym.make("PushBoundary", **kwargs)

    if video_dir is not None:
        from mani_skill.utils.wrappers.record import RecordEpisode

        Path(video_dir).mkdir(parents=True, exist_ok=True)
        env = RecordEpisode(
            env,
            output_dir=video_dir,
            save_trajectory=False,
            save_video=True,
            video_fps=20,
            trajectory_name=video_name,
        )

    return env


def _set_gripper_xy(base_env, gx: float, gy: float) -> None:
    """Teleport the floating gripper to (gx, gy) via joint-space (initial setup only)."""
    robot = base_env.agent.robot
    try:
        n_dof = robot.dof if hasattr(robot, "dof") else len(robot.get_qpos())
        if hasattr(n_dof, "item"):
            n_dof = int(n_dof.item())
        qpos = torch.zeros(n_dof, dtype=torch.float32)
        qpos[0] = gx - BOUNDARY_CENTER_X
        qpos[1] = gy - BOUNDARY_CENTER_Y
        robot.set_qpos(qpos.unsqueeze(0).expand(base_env.num_envs, -1))
        robot.set_qvel(
            torch.zeros_like(qpos).unsqueeze(0).expand(base_env.num_envs, -1)
        )
    except Exception:
        robot.set_pose(sapien.Pose(p=[gx, gy, GRIPPER_Z_FIXED], q=[1, 0, 0, 0]))


def _setup_episode(
    base_env,
    initial_block_pos: np.ndarray,
    initial_block_quat: np.ndarray,
    initial_gripper_xy: np.ndarray,
) -> None:
    """
    Teleport block and gripper to their initial positions for frame 0.
    Subsequent actions are sent through the controller via env.step().
    """
    block_pos = np.asarray(initial_block_pos, dtype=np.float32).ravel()
    block_quat = np.asarray(initial_block_quat, dtype=np.float32).ravel()
    gxy = np.asarray(initial_gripper_xy, dtype=np.float32).ravel()

    base_env.block.set_pose(
        sapien.Pose(
            p=[float(block_pos[0]), float(block_pos[1]), float(block_pos[2])],
            q=_quat_wxyz_to_sapien(block_quat),
        )
    )
    try:
        base_env.block.set_velocity([0.0, 0.0, 0.0])
        base_env.block.set_angular_velocity([0.0, 0.0, 0.0])
    except Exception:
        pass

    _set_gripper_xy(base_env, float(gxy[0]), float(gxy[1]))


def _run_actions(
    env,
    base_env,
    actions: np.ndarray,
    initial_gripper_xy: np.ndarray,
    playback_mode: str = "action",
) -> tuple:
    """
    Step the environment through `actions` (T, 2) and collect per-step
    block positions/quaternions.

    Parameters
    ----------
    actions            : (T, 2)  gripper delta-XY per step
    initial_gripper_xy : (2,)    gripper XY at frame 0 (used by set_pose mode)
    playback_mode      : "action" or "set_pose"
        "action"   — sends each delta through the controller via env.step().
                     Faithful replay: controller + physics drive everything.
        "set_pose" — teleports the gripper to its cumulative predicted position
                     each step, then calls env.step(zeros) to advance physics.
                     Useful for debugging: isolates block-prediction error from
                     controller tracking error.

    Returns
    -------
    block_pos  (T, 3)
    block_quat (T, 4)  [w,x,y,z]
    oob_steps  int
    """
    T = len(actions)
    block_pos = np.zeros((T, 3), dtype=np.float32)
    block_quat = np.zeros((T, 4), dtype=np.float32)
    oob_steps = 0

    gripper_xy = np.asarray(initial_gripper_xy, dtype=np.float32).ravel()[:2].copy()

    for t in range(T):
        action = np.asarray(actions[t], dtype=np.float32).ravel()[:2]

        if playback_mode == "set_pose":
            gripper_xy = gripper_xy + action
            _set_gripper_xy(base_env, float(gripper_xy[0]), float(gripper_xy[1]))
            env.step(np.zeros(2, dtype=np.float32))
        else:
            env.step(action)

        bp = _to_np(base_env.block.pose.p).ravel()
        bq = _to_np(base_env.block.pose.q).ravel()
        block_pos[t] = bp[:3]
        block_quat[t] = bq[:4]

        if _block_out_of_bounds(bp[:2]):
            oob_steps += 1

    return block_pos, block_quat, oob_steps


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_predictions(
    initial_block_pos: np.ndarray,
    initial_block_quat: np.ndarray,
    initial_gripper_xy: np.ndarray,
    predicted_actions: np.ndarray,
    predicted_block_pos: np.ndarray,
    predicted_block_quat: Optional[np.ndarray] = None,
    render: bool = False,
    video_dir: Optional[str] = None,
    video_name: str = "playback_predict",
    playback_mode: str = "action",
    pause_on_done: float = 0.0,
) -> dict:
    """
    Replay predicted gripper actions in simulation, then compare the
    resulting block trajectory against the model's block-state predictions.

    Parameters
    ----------
    initial_block_pos    : (3,)
    initial_block_quat   : (4,)   [w,x,y,z]
    initial_gripper_xy   : (2,)
    predicted_actions    : (T, 2) Gripper delta-XY actions to execute.
    predicted_block_pos  : (T, 3) Model's predicted block positions per step.
    predicted_block_quat : (T, 4) Model's predicted block quaternions (optional).
    render               : bool
    video_dir            : str or None  Directory to save video.
    video_name           : str
    playback_mode        : "action" (default) or "set_pose"
        "action"   — actions go through the controller; faithful replay.
        "set_pose" — gripper is teleported to its cumulative predicted position
                     each step; useful for debugging model predictions directly.
    pause_on_done        : float

    Returns
    -------
    dict with keys:
        actual_block_pos     (T, 3)
        actual_block_quat    (T, 4)
        position_errors      (T,)    L2 error per step (metres)
        yaw_errors           (T,)    |Δyaw| per step (degrees)
        mean_position_error  float
        final_position_error float
        max_position_error   float
        out_of_bounds_steps  int
        trajectory_success   bool
    """
    predicted_actions = np.asarray(predicted_actions, dtype=np.float32)
    predicted_block_pos = np.asarray(predicted_block_pos, dtype=np.float32)
    T = len(predicted_actions)

    env = _build_env(render, video_dir=video_dir, video_name=video_name)
    base_env = _unwrap(env)
    env.reset()
    _setup_episode(base_env, initial_block_pos, initial_block_quat, initial_gripper_xy)

    actual_pos, actual_quat, oob_steps = _run_actions(
        env, base_env, predicted_actions, initial_gripper_xy, playback_mode
    )

    if render and pause_on_done > 0:
        time.sleep(pause_on_done)
    env.close()

    # ── Position errors ───────────────────────────────────────────────────────
    pos_errors = np.linalg.norm(actual_pos - predicted_block_pos, axis=1)  # (T,)

    # ── Yaw errors ────────────────────────────────────────────────────────────
    yaw_errors = np.zeros(T, dtype=np.float32)
    if predicted_block_quat is not None:
        pq = np.asarray(predicted_block_quat, dtype=np.float32)
        for t in range(T):
            pred_yaw = _yaw_from_quat_wxyz(pq[t])
            actual_yaw = _yaw_from_quat_wxyz(actual_quat[t])
            diff_deg = abs(np.degrees(actual_yaw - pred_yaw))
            yaw_errors[t] = min(diff_deg, 360.0 - diff_deg)  # wrap to [0, 180]

    return dict(
        actual_block_pos=actual_pos,
        actual_block_quat=actual_quat,
        position_errors=pos_errors,
        yaw_errors=yaw_errors,
        mean_position_error=float(pos_errors.mean()),
        final_position_error=float(pos_errors[-1]) if T > 0 else 0.0,
        max_position_error=float(pos_errors.max()) if T > 0 else 0.0,
        out_of_bounds_steps=oob_steps,
        trajectory_success=(oob_steps == 0),
    )


def evaluate_target(
    initial_block_pos: np.ndarray,
    initial_block_quat: np.ndarray,
    initial_gripper_xy: np.ndarray,
    actions: np.ndarray,
    target_xy: np.ndarray,
    success_radius: float = 0.05,
    render: bool = False,
    video_dir: Optional[str] = None,
    video_name: str = "playback_target",
    playback_mode: str = "action",
    pause_on_done: float = 0.0,
) -> dict:
    """
    Execute gripper actions in simulation and evaluate whether the block
    reaches a target position within `success_radius` metres.

    Parameters
    ----------
    initial_block_pos   : (3,)
    initial_block_quat  : (4,) [w,x,y,z]
    initial_gripper_xy  : (2,)
    actions             : (T, 2) gripper delta-XY actions.
    target_xy           : (2,)  target block XY in world frame.
    success_radius      : float Block centroid must come within this distance.
    render              : bool
    pause_on_done       : float

    Returns
    -------
    dict with keys:
        success              bool   Block came within success_radius of target.
        min_distance         float  Minimum distance achieved (metres).
        final_distance       float  Distance at end of rollout.
        first_success_step   int    Step index of first success, or -1 if none.
        actual_block_pos     (T, 3)
        actual_block_quat    (T, 4)
        distances_to_target  (T,)
        out_of_bounds_steps  int
    """
    actions = np.asarray(actions, dtype=np.float32)
    target_xy = np.asarray(target_xy, dtype=np.float32).ravel()

    env = _build_env(
        render, target_xy=target_xy, video_dir=video_dir, video_name=video_name
    )
    base_env = _unwrap(env)
    env.reset()
    _setup_episode(base_env, initial_block_pos, initial_block_quat, initial_gripper_xy)

    actual_pos, actual_quat, oob_steps = _run_actions(
        env, base_env, actions, initial_gripper_xy, playback_mode
    )

    if render and pause_on_done > 0:
        time.sleep(pause_on_done)
    env.close()

    distances = np.linalg.norm(actual_pos[:, :2] - target_xy, axis=1)  # (T,)
    success_mask = distances <= success_radius
    first_success_step = int(np.argmax(success_mask)) if success_mask.any() else -1

    return dict(
        success=bool(success_mask.any()),
        min_distance=float(distances.min()),
        final_distance=float(distances[-1]) if len(distances) > 0 else float("inf"),
        first_success_step=first_success_step,
        actual_block_pos=actual_pos,
        actual_block_quat=actual_quat,
        distances_to_target=distances,
        out_of_bounds_steps=oob_steps,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 reader helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_h5_trajectory(h5_path: str, traj_key: str = "traj_0") -> dict:
    """
    Load a single trajectory from a RecordEpisode HDF5 file.

    Returns a dict with:
        actions         (T, 2)  recorded gripper delta-XY actions
        block_pos       (T, 3)  block positions from observations
        block_quat      (T, 4)  block quaternions
        gripper_pos     (T, 3)  TCP positions
        initial_block_pos  (3,)
        initial_block_quat (4,)
        initial_gripper_xy (2,)
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        traj = f[traj_key]
        actions = np.array(traj["actions"], dtype=np.float32)  # (T, 2)

        # Observations are stored with shape (T+1, ...) — one per state
        extra = traj["obs"]["extra"]

        block_raw = np.array(extra["block_pose"], dtype=np.float32)  # (T+1, 7)
        gripper_raw = np.array(extra["tcp_pose"], dtype=np.float32)  # (T+1, 7)

    T = len(actions)

    # States at steps 0..T (i.e., before each action is applied)
    block_pos = block_raw[:T, :3]
    block_quat = block_raw[:T, 3:7]  # [w,x,y,z] in ManiSkill convention
    gripper_pos = gripper_raw[:T, :3]

    return dict(
        actions=actions,
        block_pos=block_pos,
        block_quat=block_quat,
        gripper_pos=gripper_pos,
        initial_block_pos=block_pos[0],
        initial_block_quat=block_quat[0],
        initial_gripper_xy=gripper_pos[0, :2],
    )


def load_npz_predictions(npz_path: str) -> dict:
    """
    Load a standardised predictions NPZ (one already produced by convert_from_3D
    or saved manually in the same format).

    Required keys:
        predicted_actions    (T, 2)

    Optional keys (sensible defaults applied if missing):
        initial_block_quat   (4,) [w,x,y,z]  — defaults to identity
        initial_gripper_xy   (2,)             — defaults to CANONICAL_BLOCK_POS[:2]
        predicted_block_pos  (T, 3)
        predicted_block_quat (T, 4)
        target_xy            (2,)

    Note: initial_block_pos is NOT read from the file.  The block always starts
    at CANONICAL_BLOCK_POS (defined in output_conversions.py) so that all NPZ
    inputs share the same world-frame anchor.
    """
    data = dict(np.load(npz_path, allow_pickle=False))
    return data


def load_raw_npz(npz_path: str, conversion_mode: str) -> dict:
    """
    Load a raw model-output NPZ and convert it to the standardised playback dict.

    The NPZ must contain whatever keys the chosen converter expects.
    For '3d' mode these are:
        pred_positions  (T, N, 3)
        obj_ids         (T, N) or (N,)
        hand_id         scalar int
        block_id        scalar int
        all_targets     (T, N, 3)

    Optional:
        hand_template   (n_hand,  3)
        block_template  (n_block, 3)

    Parameters
    ----------
    npz_path        : str   Path to the raw predictions NPZ.
    conversion_mode : str   '3d' or '2d'.

    Returns
    -------
    Standardised playback dict ready for evaluate_predictions / evaluate_target.
    """
    raw = dict(np.load(npz_path, allow_pickle=True))

    # np.load returns 0-d arrays for scalars saved with np.savez — unwrap them.
    for key in ("hand_id", "block_id"):
        if key in raw and isinstance(raw[key], np.ndarray) and raw[key].ndim == 0:
            raw[key] = raw[key].item()

    return convert(raw, conversion_mode)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────


def print_prediction_results(results: dict) -> None:
    print("\n── Prediction Accuracy ─────────────────────────────────────────")
    print(f"  Steps evaluated       : {len(results['position_errors'])}")
    print(f"  Mean position error   : {results['mean_position_error']*100:.2f} cm")
    print(f"  Final position error  : {results['final_position_error']*100:.2f} cm")
    print(f"  Max position error    : {results['max_position_error']*100:.2f} cm")
    if results["yaw_errors"].any():
        print(f"  Mean yaw error        : {results['yaw_errors'].mean():.1f}°")
        print(f"  Max yaw error         : {results['yaw_errors'].max():.1f}°")
    print(f"  Out-of-bounds steps   : {results['out_of_bounds_steps']}")
    print(f"  Boundary success      : {'✓' if results['trajectory_success'] else '✗'}")
    print("─" * 66)


def print_target_results(results: dict, success_radius: float) -> None:
    print("\n── Target Evaluation ───────────────────────────────────────────")
    print(
        f"  Success (r={success_radius*100:.1f} cm)  : {'✓' if results['success'] else '✗'}"
    )
    print(f"  Min distance to target: {results['min_distance']*100:.2f} cm")
    print(f"  Final distance        : {results['final_distance']*100:.2f} cm")
    if results["first_success_step"] >= 0:
        print(f"  First success at step : {results['first_success_step']}")
    print(f"  Out-of-bounds steps   : {results['out_of_bounds_steps']}")
    print("─" * 66)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Playback and evaluate floating-gripper push predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input source (mutually exclusive) ─────────────────────────────────────
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--npz",
        type=str,
        default=None,
        help="Standardised predictions NPZ (already converted).",
    )
    src.add_argument(
        "--raw_npz",
        type=str,
        default=None,
        help="Raw model-output NPZ to convert before playback.",
    )
    src.add_argument(
        "--raw_npz_dir",
        type=str,
        default=None,
        help=(
            "Directory of raw model-output NPZ files to batch-evaluate. "
            "If it contains scale_* subdirectories, each is treated as a group."
        ),
    )
    src.add_argument(
        "--h5",
        type=str,
        default=None,
        help="HDF5 demo file (ground-truth actions replayed as predictions).",
    )

    # ── Conversion (only used with --raw_npz) ────────────────────────────────
    p.add_argument(
        "--conversion_mode",
        type=str,
        default="2d",
        choices=["3d", "2d"],
        help="Which converter to apply to --raw_npz input.",
    )

    # ── Batch-eval options (only used with --raw_npz_dir) ────────────────────
    p.add_argument(
        "--group_by_scale_subdirs",
        action="store_true",
        help="Group by scale_* subdirectories when present (default: true).",
    )
    p.add_argument(
        "--no_group_by_scale_subdirs",
        dest="group_by_scale_subdirs",
        action="store_false",
        help="Do not group by scale_* subdirectories.",
    )
    p.set_defaults(group_by_scale_subdirs=True)
    p.add_argument(
        "--max_trajs_per_group",
        type=int,
        default=None,
        help="Max number of trajectories to evaluate per group (scale).",
    )
    p.add_argument(
        "--summary_json",
        type=str,
        default=None,
        help="Optional path to write per-group summary metrics JSON.",
    )
    p.add_argument(
        "--metrics_out_dir",
        type=str,
        default=None,
        help="Optional directory to write per-trajectory metrics and a JSONL manifest.",
    )
    p.add_argument(
        "--save_videos",
        action="store_true",
        help="Save one video per trajectory during batch evaluation.",
    )
    p.add_argument(
        "--videos_out_dir",
        type=str,
        default=None,
        help="Root directory for per-trajectory videos (required if --save_videos).",
    )
    p.add_argument(
        "--video_name_template",
        type=str,
        default="{stem}",
        help=(
            "Template for per-trajectory video name (no extension). "
            "Fields: {group}, {stem}, {index}."
        ),
    )

    # ── Evaluation mode ───────────────────────────────────────────────────────
    p.add_argument(
        "--traj",
        type=str,
        default="traj_0",
        help="Trajectory key inside the HDF5 file.",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="predict",
        choices=["predict", "target"],
        help="'predict': compare block trajectory to predictions. "
        "'target': check if block reaches a target XY.",
    )
    p.add_argument(
        "--target_xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Target XY for --mode target (overrides NPZ value).",
    )
    p.add_argument(
        "--success_radius",
        type=float,
        default=0.05,
        help="Success radius in metres for --mode target.",
    )
    p.add_argument(
        "--playback_mode",
        type=str,
        default="action",
        choices=["action", "set_pose"],
        help="'action' (default): sends delta-XY through the controller. "
        "'set_pose': teleports gripper to cumulative predicted position "
        "each step; useful for debugging model predictions directly.",
    )
    p.add_argument(
        "--render",
        action="store_true",
        help="Enable offscreen rendering (headless-safe).",
    )
    p.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Directory to save playback video. "
        "Implies rendering even without --render.",
    )
    p.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="Seconds to sleep after rollout completes.",
    )
    p.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Truncate trajectories to this many steps.",
    )
    return p.parse_args()


def _find_groups_in_dir(
    root: Path, group_by_scale_subdirs: bool
) -> list[tuple[str, list[Path]]]:
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"--raw_npz_dir not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"--raw_npz_dir must be a directory: {root}")

    scale_dirs: list[Path] = []
    if group_by_scale_subdirs:
        scale_dirs = sorted(
            [p for p in root.iterdir() if p.is_dir() and p.name.startswith("scale_")]
        )

    groups: list[tuple[str, list[Path]]] = []
    if scale_dirs:
        for d in scale_dirs:
            files = sorted(
                [p for p in d.iterdir() if p.is_file() and p.suffix == ".npz"]
            )
            groups.append((d.name, files))
    else:
        files = sorted(
            [p for p in root.iterdir() if p.is_file() and p.suffix == ".npz"]
        )
        groups.append((root.name, files))
    return groups


def _write_jsonl_line(path: Path, record: dict) -> None:
    def _jsonify(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            return x.item()
        if isinstance(x, dict):
            return {str(k): _jsonify(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_jsonify(v) for v in x]
        return x

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonify(record), sort_keys=True) + "\n")


def _save_metrics_npz(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict = {}
    for k, v in record.items():
        if v is None:
            continue
        if isinstance(v, np.ndarray):
            arrays[k] = v
        elif isinstance(v, (float, int, bool, str)):
            arrays[k] = np.array(v)
        else:
            arrays[k] = np.array(json.dumps(v))
    np.savez(path, **arrays)


def _summarize_group(per_traj: list[dict]) -> dict:
    n_total = len(per_traj)
    n_success = sum(1 for r in per_traj if bool(r.get("success", False)))
    success_rate = float(n_success / n_total) if n_total > 0 else 0.0

    steps_proxy_all = [int(r["steps_proxy"]) for r in per_traj if "steps_proxy" in r]
    mean_steps_proxy = (
        float(np.mean(steps_proxy_all)) if steps_proxy_all else float("nan")
    )

    steps_to_success = [
        int(r["steps_to_success"]) for r in per_traj if r.get("success", False)
    ]
    mean_steps_to_success = (
        float(np.mean(steps_to_success)) if steps_to_success else float("nan")
    )
    median_steps_to_success = (
        float(np.median(steps_to_success)) if steps_to_success else float("nan")
    )

    return dict(
        n_total=n_total,
        n_success=n_success,
        success_rate=success_rate,
        mean_steps_proxy_all=mean_steps_proxy,
        mean_steps_to_success_successes=mean_steps_to_success,
        median_steps_to_success_successes=median_steps_to_success,
    )


def _batch_evaluate_raw_npz_dir(args) -> None:
    root = Path(args.raw_npz_dir).expanduser()
    groups = _find_groups_in_dir(root, args.group_by_scale_subdirs)
    if args.save_videos and not args.videos_out_dir:
        raise ValueError("--videos_out_dir is required when --save_videos is set.")

    metrics_root = (
        Path(args.metrics_out_dir).expanduser() if args.metrics_out_dir else None
    )
    manifest_path = metrics_root / "manifest.jsonl" if metrics_root else None
    videos_root = (
        Path(args.videos_out_dir).expanduser() if args.videos_out_dir else None
    )

    all_group_summaries: dict[str, dict] = {}

    for group_name, files in groups:
        if not files:
            print(f"[{group_name}] No .npz files found, skipping.")
            all_group_summaries[group_name] = dict(
                n_total=0, n_success=0, success_rate=0.0
            )
            continue

        if args.max_trajs_per_group is not None:
            files = files[: int(args.max_trajs_per_group)]

        per_traj_records: list[dict] = []
        print(f"\n[{group_name}] Evaluating {len(files)} trajectories")

        for idx, npz_path in enumerate(files):
            data = load_raw_npz(str(npz_path), args.conversion_mode)
            actions = data["predicted_actions"]
            initial_block_pos = data["initial_block_pos"]
            initial_block_quat = data["initial_block_quat"]
            initial_gripper_xy = data["initial_gripper_xy"]
            target_xy = (
                np.array(args.target_xy, dtype=np.float32)
                if args.target_xy is not None
                else data.get("target_xy", None)
            )
            if target_xy is None:
                raise ValueError(
                    f"[{group_name}] {npz_path.name}: missing target_xy in file and no --target_xy provided."
                )

            video_dir = None
            video_name = None
            video_path = None
            if args.save_videos:
                assert videos_root is not None
                stem = npz_path.stem
                traj_tag = args.video_name_template.format(
                    group=group_name, stem=stem, index=idx
                )
                # ManiSkill RecordEpisode writes videos as e.g. 0.mp4 in output_dir.
                # Use a unique directory per trajectory to avoid overwriting.
                video_dir = str(videos_root / group_name / traj_tag)
                video_name = "playback_target"
                video_path = str(Path(video_dir) / "0.mp4")

            results = evaluate_target(
                initial_block_pos=initial_block_pos,
                initial_block_quat=initial_block_quat,
                initial_gripper_xy=initial_gripper_xy,
                actions=actions,
                target_xy=target_xy,
                success_radius=args.success_radius,
                render=args.render or args.save_videos,
                video_dir=video_dir,
                video_name=video_name or "playback_target",
                playback_mode=args.playback_mode,
                pause_on_done=args.pause,
            )

            T = int(len(actions))
            fss = int(results.get("first_success_step", -1))
            steps_to_success = int(fss + 1) if fss >= 0 else None
            steps_proxy = int(steps_to_success) if steps_to_success is not None else T

            traj_record = dict(
                group=group_name,
                index=int(idx),
                input_npz=str(npz_path),
                conversion_mode=str(args.conversion_mode),
                mode="target",
                success=bool(results.get("success", False)),
                first_success_step=fss,
                steps_to_success=steps_to_success,
                steps_proxy=steps_proxy,
                T=T,
                success_radius=float(args.success_radius),
                target_xy=np.asarray(target_xy, dtype=np.float32),
                min_distance=float(results.get("min_distance", float("nan"))),
                final_distance=float(results.get("final_distance", float("nan"))),
                out_of_bounds_steps=int(results.get("out_of_bounds_steps", 0)),
                video_path=video_path,
            )
            per_traj_records.append(traj_record)

            if metrics_root is not None:
                metrics_path = (
                    metrics_root / group_name / f"{npz_path.stem}_metrics.npz"
                )
                metrics_payload = dict(
                    **traj_record,
                    distances_to_target=np.asarray(results.get("distances_to_target")),
                    actual_block_pos=np.asarray(results.get("actual_block_pos")),
                    actual_block_quat=np.asarray(results.get("actual_block_quat")),
                )
                _save_metrics_npz(metrics_path, metrics_payload)
                traj_record_for_manifest = dict(
                    **traj_record,
                    metrics_npz=str(metrics_path),
                )
                assert manifest_path is not None
                _write_jsonl_line(manifest_path, traj_record_for_manifest)

        summary = _summarize_group(per_traj_records)
        all_group_summaries[group_name] = summary
        print(
            f"[{group_name}] success_rate={summary['success_rate']:.3f} "
            f"({summary['n_success']}/{summary['n_total']}), "
            f"mean_steps_proxy_all={summary['mean_steps_proxy_all']:.2f}, "
            f"median_steps_to_success_successes={summary['median_steps_to_success_successes']:.2f}"
        )

    if args.summary_json:
        out_path = Path(args.summary_json).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_group_summaries, indent=2), encoding="utf-8")
        print(f"\nWrote summary JSON to {out_path}")


def main():
    args = parse_args()

    # ── Load and (optionally) convert data ────────────────────────────────────
    if args.raw_npz_dir is not None:
        _batch_evaluate_raw_npz_dir(args)
        return

    if args.h5 is not None:
        print(f"Loading H5: {args.h5}  traj={args.traj}")
        data = load_h5_trajectory(args.h5, args.traj)
        actions = data["actions"]
        initial_block_pos = data["initial_block_pos"]
        initial_block_quat = data["initial_block_quat"]
        initial_gripper_xy = data["initial_gripper_xy"]
        predicted_block_pos = data["block_pos"]
        predicted_block_quat = data["block_quat"]
        target_xy = np.array(args.target_xy) if args.target_xy else None

    elif args.raw_npz is not None:
        print(
            f"Loading raw NPZ: {args.raw_npz}  conversion_mode={args.conversion_mode}"
        )
        data = load_raw_npz(args.raw_npz, args.conversion_mode)
        actions = data["predicted_actions"]
        initial_block_pos = data["initial_block_pos"]
        initial_block_quat = data["initial_block_quat"]
        initial_gripper_xy = data["initial_gripper_xy"]
        predicted_block_pos = data.get("predicted_block_pos", None)
        predicted_block_quat = data.get("predicted_block_quat", None)
        # CLI --target_xy overrides whatever the converter extracted
        target_xy = (
            np.array(args.target_xy, dtype=np.float32)
            if args.target_xy is not None
            else data.get("target_xy", None)
        )

        # Print a brief conversion summary
        T = len(actions)
        print(f"  Converted {args.conversion_mode.upper()} predictions → {T} steps")
        if target_xy is not None:
            print(f"  Extracted target XY: {target_xy}")
        else:
            print("  No target found in predictions (use --target_xy to set one)")

    else:  # --npz  (standardised NPZ, already converted)
        print(f"Loading NPZ: {args.npz}")
        data = load_npz_predictions(args.npz)
        actions = data["predicted_actions"]

        # Block always starts at the canonical world position — never read from file.
        initial_block_pos = CANONICAL_BLOCK_POS.copy()

        # Quat may be present if the NPZ was saved from convert_from_3D output
        # (which derives it from block particles).  If not, identity is correct:
        # a cube's yaw doesn't matter for the relative block–gripper geometry.
        initial_block_quat = np.asarray(
            data.get("initial_block_quat", np.array([1.0, 0.0, 0.0, 0.0])),
            dtype=np.float32,
        )

        # Gripper XY encodes the relative offset; if missing, default to block centre.
        initial_gripper_xy = np.asarray(
            data.get("initial_gripper_xy", CANONICAL_BLOCK_POS[:2].copy()),
            dtype=np.float32,
        )

        predicted_block_pos = data.get("predicted_block_pos", None)
        predicted_block_quat = data.get("predicted_block_quat", None)
        target_xy = (
            np.array(args.target_xy, dtype=np.float32)
            if args.target_xy is not None
            else data.get("target_xy", None)
        )

    # ── Truncate if requested ─────────────────────────────────────────────────
    if args.max_steps is not None:
        actions = actions[: args.max_steps]
        if predicted_block_pos is not None:
            predicted_block_pos = predicted_block_pos[: args.max_steps]
        if predicted_block_quat is not None:
            predicted_block_quat = predicted_block_quat[: args.max_steps]

    print(f"  T={len(actions)} steps")
    print(f"  Block start:   {initial_block_pos}")
    print(f"  Gripper start: {initial_gripper_xy}")

    # ── Run evaluation ────────────────────────────────────────────────────────
    if args.mode == "target":
        if target_xy is None:
            raise ValueError(
                "--mode target requires --target_xy X Y, or a target embedded "
                "in the input file (raw_npz all_targets, or npz target_xy key)."
            )
        print(f"  Target XY:     {target_xy}  radius={args.success_radius} m")

        results = evaluate_target(
            initial_block_pos=initial_block_pos,
            initial_block_quat=initial_block_quat,
            initial_gripper_xy=initial_gripper_xy,
            actions=actions,
            target_xy=target_xy,
            success_radius=args.success_radius,
            render=args.render,
            video_dir=args.video_dir,
            playback_mode=args.playback_mode,
            pause_on_done=args.pause,
        )
        print_target_results(results, args.success_radius)

    else:  # predict
        if predicted_block_pos is None:
            print(
                "Warning: no predicted block positions available — "
                "running actions only, errors will be reported as zero."
            )
            predicted_block_pos = np.zeros((len(actions), 3), dtype=np.float32)

        results = evaluate_predictions(
            initial_block_pos=initial_block_pos,
            initial_block_quat=initial_block_quat,
            initial_gripper_xy=initial_gripper_xy,
            predicted_actions=actions,
            predicted_block_pos=predicted_block_pos,
            predicted_block_quat=predicted_block_quat,
            render=args.render,
            video_dir=args.video_dir,
            playback_mode=args.playback_mode,
            pause_on_done=args.pause,
        )
        print_prediction_results(results)


if __name__ == "__main__":
    main()
