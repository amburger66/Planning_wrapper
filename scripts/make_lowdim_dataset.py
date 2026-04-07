"""
scripts/make_lowdim_dataset.py

Convert a RecordEpisode HDF5 file into a low-dimensional NPZ dataset.
Pads sequences to the maximum length and saves their actual valid lengths.

Rotation representation
-----------------------
All orientations use the 6D rotation representation from:
  "On the Continuity of Rotation Representations in Neural Networks"
  Zhou et al., CVPR 2019.

Given rotation matrix R = [r1 | r2 | r3], the 6D rep is the first two
columns: [r1, r2] (shape 6).  This is continuous everywhere, unlike
quaternions (double cover, discontinuous) or Euler angles (gimbal lock).

Recovery:
  a1 = normalize(r1)
  a2 = normalize(r2 - (r2 · a1) * a1)
  a3 = a1 × a2

State  (18-dim): tcp  pos(3) + tcp  6D-rot(6)  ||  block pos(3) + block 6D-rot(6)
Action  (9-dim): next-step tcp pos(3) + tcp 6D-rot(6)  — absolute pose, not a delta

Source fields (both scalar-first quaternion: qw, qx, qy, qz):
  obs/extra/tcp_pose   (T+1, 7)
  obs/extra/block_pose (T+1, 7)

Alignment:
  state[t]  = tcp_pose[t]  ||  block_pose[t]
  action[t] = tcp_pose[t+1]          <- absolute next-step TCP pose

The last state (t = T-1) has no t+1 pose, so we repeat tcp_pose[T-1] as the
final action (i.e. "stay where you are"), keeping array lengths equal.

Usage
-----
    python scripts/make_lowdim_dataset.py \
        --h5  demos/PushBoundary/scripted/scripted.h5 \
        --out datasets/push_lowdim.npz

    python scripts/make_lowdim_dataset.py --h5 ... --out ... --min_len 4000
    python scripts/make_lowdim_dataset.py --h5 ... --out ... --no_outlier_filter
    python scripts/make_lowdim_dataset.py --h5 ... --out ... --skip_first 10
    python scripts/make_lowdim_dataset.py --h5 ... --out ... --knocked_over_thresh 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Rotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    (N, 4) quaternions (qw, qx, qy, qz) → (N, 3, 3) rotation matrices.
    Handles batch dimension.
    """
    q  = q / np.linalg.norm(q, axis=-1, keepdims=True)   # normalise
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = np.stack([
        1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y),
          2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x),
          2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y),
    ], axis=-1).reshape(*q.shape[:-1], 3, 3)
    return R


def rotmat_to_6d(R: np.ndarray) -> np.ndarray:
    """
    (N, 3, 3) rotation matrices → (N, 6) 6D representation.
    Takes the first two columns: [R[:, :, 0], R[:, :, 1]] flattened.
    """
    return np.concatenate([R[..., 0], R[..., 1]], axis=-1)   # (N, 6)


def rot6d_to_rotmat(r6: np.ndarray) -> np.ndarray:
    """
    (N, 6) 6D representation → (N, 3, 3) rotation matrices (Gram-Schmidt).
    """
    r1 = r6[..., :3]
    r2 = r6[..., 3:]

    a1 = r1 / np.linalg.norm(r1, axis=-1, keepdims=True)
    a2 = r2 - (r2 * a1).sum(axis=-1, keepdims=True) * a1
    a2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    a3 = np.cross(a1, a2)

    return np.stack([a1, a2, a3], axis=-1)   # (N, 3, 3)  columns = axes


def pose7_to_9d(pose: np.ndarray) -> np.ndarray:
    """
    (N, 7) [pos(3), qw, qx, qy, qz] → (N, 9) [pos(3), 6D-rot(6)].
    """
    pos  = pose[:, :3]                                  # (N, 3)
    quat = pose[:, 3:]                                  # (N, 4) wxyz
    R    = _quat_wxyz_to_rotmat(quat)                   # (N, 3, 3)
    rot6 = rotmat_to_6d(R)                              # (N, 6)
    return np.concatenate([pos, rot6], axis=-1)         # (N, 9)


# ─────────────────────────────────────────────────────────────────────────────
# Knockover detection
# ─────────────────────────────────────────────────────────────────────────────

def _y_tipping_angles_from_ref(R_seq: np.ndarray, R_ref: np.ndarray) -> np.ndarray:
    """
    Y-axis tipping angle (radians) of each rotation in R_seq relative to R_ref.
    """
    up_seq = R_seq[:, :, 2]                                 # (N, 3)
    up_ref = R_ref[:, 2]                                    # (3,)

    _xz = np.array([1.0, 0.0, 1.0], dtype=np.float64)
    up_seq_xz = up_seq * _xz                                # (N, 3)
    up_ref_xz = up_ref * _xz                                # (3,)

    up_seq_xz /= np.linalg.norm(up_seq_xz, axis=-1, keepdims=True).clip(min=1e-8)
    up_ref_xz /= max(float(np.linalg.norm(up_ref_xz)), 1e-8)

    cos_angle = np.clip((up_seq_xz * up_ref_xz).sum(axis=-1), -1.0, 1.0)
    return np.arccos(cos_angle)                             # (N,) radians


def _block_knocked_over(
    block_pose_raw: np.ndarray,
    skip_first: int,
    threshold_deg: float,
) -> tuple[bool, float]:
    """
    Return (knocked_over, max_deviation_deg).
    """
    R_all = _quat_wxyz_to_rotmat(block_pose_raw[:, 3:])   # (T+1, 3, 3)
    R_ref = R_all[skip_first]                             # reference = first kept frame
    angles_rad = _y_tipping_angles_from_ref(R_all[skip_first:], R_ref)
    max_deg = float(np.rad2deg(angles_rad.max()))
    return max_deg > threshold_deg, max_deg


# ─────────────────────────────────────────────────────────────────────────────
# H5 helpers
# ─────────────────────────────────────────────────────────────────────────────

def _traj_keys(f: h5py.File) -> list[str]:
    keys = [k for k in f.keys() if k.startswith("traj_")]
    keys.sort(key=lambda k: int(k.split("_")[1]))
    return keys


def _traj_len(traj: h5py.Group, block_actor_key: str) -> int:
    # -1 because frame 0 is always skipped (obs lags set_pose by one step)
    return traj[f"env_states/actors/{block_actor_key}"].shape[0] - 1


def _iqr_min_len(lengths: list[int]) -> int:
    arr  = np.array(lengths, dtype=float)
    q1, q3 = np.percentile(arr, [25, 75])
    lower   = q1 - 1.5 * (q3 - q1)
    non_out = arr[arr >= lower]
    return int((non_out if len(non_out) else arr).min())


def _extract_trajectory(
    traj: h5py.Group,
    skip_first: int,
    knocked_over_thresh: float,
    verbose: bool,
    key: str,
    max_states: int | None = None,
    art_key: str = "floating_gripper",
    block_actor_key: str = "push_block",
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Extracts up to max_states steps of a trajectory minus skipped frames.
    Returns:
      states  (min(T - skip_first, max_states), 18)
      actions (min(T - skip_first, max_states),  9)
    or None if the block is knocked over or trajectory is too short after skip.

    Data source: env_states/actors/{block_actor_key}  (T_raw, 13)
                 env_states/articulations/{art_key}   (T_raw, 17)
    Frame 0 is always dropped (obs at reset lags set_pose by one step).
    TCP is computed as: root_pos + [joint_x, joint_y, 0], identity quat.
    art_states layout: pos(3) quat(4) qpos(2) qvel(2) lin_vel(3) ang_vel(3) → 17
    FLOAT_QPOS = slice(13, 15)  (joint positions after vel and other fields)
    """
    block_states_raw = np.array(traj[f"env_states/actors/{block_actor_key}"], dtype=np.float32)
    art_states_raw   = np.array(traj[f"env_states/articulations/{art_key}"],  dtype=np.float32)

    # Drop frame 0 (reset observation lag)
    block_states = block_states_raw[1:]   # (T, 13)
    art_states   = art_states_raw[1:]     # (T, 17)
    T = block_states.shape[0]

    if T <= skip_first:
        return None

    # ── Build TCP pose (pos + identity quat) ──────────────────────────────
    root_pos = art_states[:, 0:3].copy()          # (T, 3)
    joint_xy = art_states[:, 13:15]               # (T, 2)  FLOAT_QPOS
    root_pos[:, 0] += joint_xy[:, 0]
    root_pos[:, 1] += joint_xy[:, 1]
    tcp_quat     = np.tile(np.array([1., 0., 0., 0.], dtype=np.float32), (T, 1))
    tcp_pose_raw = np.concatenate([root_pos, tcp_quat], axis=1)   # (T, 7)

    # ── Block pose (pos + quat wxyz) ──────────────────────────────────────
    block_pose_raw = np.concatenate([
        block_states[:, 0:3],   # ACTOR_POS
        block_states[:, 3:7],   # ACTOR_QUAT (wxyz)
    ], axis=-1)                                    # (T, 7)

    # ── Knockover check ────────────────────────────────────────────────────
    if knocked_over_thresh < 180.0:
        knocked, max_dev = _block_knocked_over(block_pose_raw, skip_first, knocked_over_thresh)
        if knocked:
            if verbose:
                print(f"  [knock] {key}  max_block_rot={max_dev:.1f}° > {knocked_over_thresh}°")
            return None
        elif verbose:
            print(f"  [ok]    {key}  max_block_rot={max_dev:.1f}°")

    # ── Convert every pose to 9D ───────────────────────────────────────────
    tcp_9d   = pose7_to_9d(tcp_pose_raw)    # (T, 9)
    block_9d = pose7_to_9d(block_pose_raw)  # (T, 9)

    # ── States: steps skip_first..T-1 (capped at max_states) ─────────────
    start = skip_first
    end   = start + max_states if max_states is not None else T
    end   = min(end, T)
    states = np.concatenate(
        [tcp_9d[start:end], block_9d[start:end]], axis=-1
    )   # (T - skip_first, 18)

    # ── Actions: absolute TCP pose at the NEXT step (t+1) ─────────────────
    next_tcp = np.concatenate([
        tcp_9d[start + 1 : end],            # steps start+1..end-1
        tcp_9d[end : end + 1],              # step `end` (or repeat last if at boundary)
    ], axis=0)                              

    # Guard 
    kept = end - start
    if next_tcp.shape[0] < kept:
        repeat = np.tile(tcp_9d[-1:], (kept - next_tcp.shape[0], 1))
        next_tcp = np.concatenate([next_tcp, repeat], axis=0)

    actions = next_tcp.astype(np.float32)   # (kept, 9)
    return states.astype(np.float32), actions


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    h5_path: str,
    out_path: str,
    min_len_arg: int | None,
    no_outlier_filter: bool,
    skip_first: int,
    knocked_over_thresh: float,
    verbose: bool,
    max_states: int | None = 50,
) -> None:
    with h5py.File(h5_path, "r") as f:
        keys = _traj_keys(f)
        if not keys:
            raise ValueError(f"No traj_* keys found in {h5_path}")

        print(f"Found {len(keys)} trajectories in {h5_path}")

        # ── Auto-discover articulation / actor key names ───────────────────
        sample_traj  = f[keys[0]]
        art_keys     = list(sample_traj["env_states/articulations"].keys())
        actor_keys   = list(sample_traj["env_states/actors"].keys())
        _robot_cands = ["floating_gripper", "robot", "panda_stick", "gripper"]
        art_key      = next(
            (k for c in _robot_cands for k in art_keys if c in k.lower()),
            art_keys[0],
        )
        _block_cands    = ["push_block", "block", "cube"]
        block_actor_key = next(
            (k for c in _block_cands for k in actor_keys if c in k.lower()),
            actor_keys[0],
        )
        print(f"  articulation key: '{art_key}'  actor key: '{block_actor_key}'")

        lengths  = [_traj_len(f[k], block_actor_key) for k in keys]
        arr_len  = np.array(lengths)
        print(f"\nTrajectory length stats:")
        print(f"  min={arr_len.min()}  max={arr_len.max()}  "
              f"mean={arr_len.mean():.1f}  median={np.median(arr_len):.1f}")

        if skip_first > 0:
            print(f"\nSkipping first {skip_first} frames of each trajectory")

        # Determine the cutoff for dropping trajectories
        if min_len_arg is not None:
            min_len = min_len_arg
            print(f"\nUsing user-specified min_len={min_len} to drop short trajectories")
        elif no_outlier_filter:
            min_len = int(arr_len.min())
            print(f"\nNo outlier filter → drop threshold is global minimum: {min_len}")
        else:
            min_len = _iqr_min_len(lengths)
            q1, q3 = np.percentile(arr_len, [25, 75])
            lower   = q1 - 1.5 * (q3 - q1)
            n_out   = int((arr_len < lower).sum())
            print(f"\nIQR outlier filter:  Q1={q1:.0f}  Q3={q3:.0f}  lower_fence={lower:.0f}")
            if n_out:
                print(f"  → {n_out} short outlier(s) will be dropped")
            print(f"  → minimum required length: {min_len}")

        if skip_first >= min_len:
            raise ValueError(
                f"skip_first={skip_first} >= min_len={min_len}: no frames would remain."
            )

        if knocked_over_thresh < 180.0:
            print(f"\nKnockover filter: dropping trajectories where block rotates "
                  f"> {knocked_over_thresh}° from its pose at frame {skip_first}")

        all_states:  list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        valid_lengths: list[int] = []
        
        skipped_short    = 0
        skipped_knocked  = 0

        # Extract all valid sequences at their native lengths
        for k in keys:
            traj_len = _traj_len(f[k], block_actor_key)

            # 1. Drop if it doesn't meet the minimum length
            if traj_len < min_len:
                skipped_short += 1
                continue

            # 2. Extract unpadded data and check for knockovers
            result = _extract_trajectory(
                f[k], skip_first, knocked_over_thresh, verbose, k, max_states,
                art_key=art_key, block_actor_key=block_actor_key,
            )
            
            if result is None:
                skipped_knocked += 1
                continue
                
            s, a = result
            all_states.append(s)
            all_actions.append(a)
            valid_lengths.append(s.shape[0])

        if not all_states:
            raise RuntimeError(
                f"No trajectories survived filtering "
                f"(min_len={min_len}, skip_first={skip_first}, "
                f"knocked_over_thresh={knocked_over_thresh})."
            )

        print(f"\nKept {len(all_states)} / {len(keys)} trajectories")
        if skipped_short:
            print(f"  dropped {skipped_short} (too short < {min_len})")
        if skipped_knocked:
            print(f"  dropped {skipped_knocked} (block knocked over > {knocked_over_thresh}°)")

    # ── Zero-padding to uniform shape ──────────────────────────────────────
    max_len = max(valid_lengths)
    N = len(all_states)
    
    print(f"\nPadding sequences to max valid length: {max_len}")
    
    # Initialize padded arrays with zeros
    states_np  = np.zeros((N, max_len, 18), dtype=np.float32)
    actions_np = np.zeros((N, max_len,  9), dtype=np.float32)
    valid_lengths_np = np.array(valid_lengths, dtype=np.int32)
    
    # Fill in the actual data
    for i, (s, a) in enumerate(zip(all_states, all_actions)):
        L = s.shape[0]
        states_np[i, :L, :] = s
        actions_np[i, :L, :] = a

    print(f"\nDataset shape:")
    print(f"  states:        {states_np.shape} dtype={states_np.dtype}")
    print(f"  actions:       {actions_np.shape} dtype={actions_np.dtype}")
    print(f"  valid_lengths: {valid_lengths_np.shape}     dtype={valid_lengths_np.dtype}")
    print(f"\nState layout  (18-dim):")
    print(f"  [0:3]   tcp position")
    print(f"  [3:9]   tcp 6D rotation  (first two cols of rotation matrix)")
    print(f"  [9:12]  block position")
    print(f"  [12:18] block 6D rotation")
    print(f"\nAction layout  (9-dim):")
    print(f"  [0:3]   next-step tcp position   (absolute, world frame)")
    print(f"  [3:9]   next-step tcp 6D rotation (absolute, world frame)")
    print(f"\nSample (traj 0, step 0):")
    print(f"  tcp  pos    = {states_np[0, 0, 0:3]}")
    print(f"  tcp  rot6d  = {states_np[0, 0, 3:9]}")
    print(f"  block pos   = {states_np[0, 0, 9:12]}")
    print(f"  block rot6d = {states_np[0, 0, 12:18]}")
    print(f"  action      = {actions_np[0, 0]}")
    print(f"  valid_len   = {valid_lengths_np[0]}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save states, actions, and valid lengths together
    np.savez(out_path, states=states_np, actions=actions_np, valid_lengths=valid_lengths_np)
    print(f"\nSaved → {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5",               required=True)
    p.add_argument("--out",              required=True)
    p.add_argument(
        "--min_len",         
        type=int,   
        default=None,
        help="Drop trajectories shorter than this length. If not provided, determined by IQR."
    )
    p.add_argument("--no_outlier_filter",action="store_true")
    p.add_argument(
        "--skip_first",
        type=int,
        default=0,
        help="Drop this many leading frames from every trajectory (default: 0). "
             "Useful to skip the robot settling / initialisation period.",
    )
    p.add_argument(
        "--knocked_over_thresh",
        type=float,
        default=30.0,
        help="Geodesic angle threshold in degrees (default: 30). Trajectories where "
             "the block's orientation deviates more than this from its pose at frame "
             "skip_first are dropped entirely. Set to 180 to disable.",
    )
    p.add_argument(
        "--max_states",
        type=int,
        default=50,
        help="Keep only the first N states from each trajectory (default: 50). "
             "Set to 0 to keep all states.",
    )
    p.add_argument("--verbose",          action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        args.h5,
        args.out,
        args.min_len,
        args.no_outlier_filter,
        args.skip_first,
        args.knocked_over_thresh,
        args.verbose,
        max_states=args.max_states if args.max_states > 0 else None,
    )