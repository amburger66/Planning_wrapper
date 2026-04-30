"""
utils/output_conversions.py

Convert raw model output (particle trajectories) into the standardised dict
that playback_floating.py's evaluate_predictions / evaluate_target expect.

Approach
--------
No external templates are used.  Frame 0 particles serve as the Kabsch
reference for all subsequent frames.  Hand (gripper) position is simply the
particle centroid — only XY is needed since the gripper is 2D.  Block
orientation is recovered via Kabsch relative to frame 0.

The entire scene is shifted so the block starts at CANONICAL_BLOCK_POS in the
simulator; the gripper's relative offset from the block at t=0 is preserved.

Standardised output dict
------------------------
    initial_block_pos    (3,)    always CANONICAL_BLOCK_POS
    initial_block_quat   (4,)    [w,x,y,z] yaw estimated from cube-surface residual minimisation
    initial_gripper_xy   (2,)    canonical_block_xy + relative offset at t=0
    predicted_actions    (T-1, 2)  gripper delta-XY, one per step
    predicted_block_pos  (T-1, 3)  block XYZ after each action, world frame
    predicted_block_quat (T-1, 4)  block [w,x,y,z] after each action
    target_xy            (2,) or None   world-frame target XY from all_targets
    tcp_positions        (T, 3)    full gripper centroid trajectory, world frame
    block_positions      (T, 3)    full block position trajectory, world frame

Modes
-----
    convert_from_3D   — particle cloud predictions (implemented)
    convert_from_2D   — 2D top-down predictions (stub, not yet implemented)
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Kabsch algorithm  (used for block rotation from frame-0 reference)
# ─────────────────────────────────────────────────────────────────────────────

def kabsch(source: np.ndarray, target: np.ndarray):
    """
    Find rotation R and translation t minimising ||target - (source @ R.T + t)||².
    Kabsch–Umeyama SVD method.  source/target: (N, 3).
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    c_s = source.mean(0)
    c_t = target.mean(0)
    H   = (source - c_s).T @ (target - c_t)
    U, _, Vt = np.linalg.svd(H)
    d   = np.linalg.det(Vt.T @ U.T)
    R   = (Vt.T @ np.diag([1.0, 1.0, d]) @ U.T).astype(np.float32)
    t   = (c_t - R @ c_s).astype(np.float32)
    return R, t


def kabsch_batch(source: np.ndarray, targets: np.ndarray):
    """Vectorised Kabsch.  source (N,3), targets (T,N,3) → Rs (T,3,3), ts (T,3)."""
    T = targets.shape[0]
    Rs = np.empty((T, 3, 3), dtype=np.float32)
    ts = np.empty((T, 3),    dtype=np.float32)
    for i in range(T):
        Rs[i], ts[i] = kabsch(source, targets[i])
    return Rs, ts


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → unit quaternion [w, x, y, z] (Shepperd method)."""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s;  x = (R[2,1]-R[1,2])*s;  y = (R[0,2]-R[2,0])*s;  z = (R[1,0]-R[0,1])*s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1]-R[1,2])/s;  x = 0.25*s;  y = (R[0,1]+R[1,0])/s;  z = (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2]-R[2,0])/s;  x = (R[0,1]+R[1,0])/s;  y = 0.25*s;  z = (R[1,2]+R[2,1])/s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0]-R[0,1])/s;  x = (R[0,2]+R[2,0])/s;  y = (R[1,2]+R[2,1])/s;  z = 0.25*s
    q = np.array([w, x, y, z], dtype=np.float32)
    return q / np.linalg.norm(q)


def _estimate_yaw_from_particles(centered_particles: np.ndarray,
                                  half_size: float = 0.025,
                                  n_angles: int = 360) -> float:
    """
    Estimate the yaw of a cube from its centered surface particles.

    Assumes zero pitch and roll (the cube is flat on the table, enforced by
    gravity in simulation).  Only yaw needs to be found.

    Method: grid-search over yaw ∈ [0, π/2) and find the angle that minimises
    the cube-surface residual.  For a correctly-oriented cube, every surface
    particle satisfies max(|x_local|, |y_local|, |z_local|) = half_size.
    No point correspondence is required — this works purely from geometry.

    Due to 4-fold symmetry the search only needs [0, π/2); the returned angle
    is the best fit within that range.
    """
    angles    = np.linspace(0.0, np.pi / 2.0, n_angles, endpoint=False)
    residuals = np.empty(n_angles)
    pts       = centered_particles.astype(np.float64)

    for i, theta in enumerate(angles):
        c, s   = np.cos(theta), np.sin(theta)
        R_inv  = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        p_loc  = pts @ R_inv.T                        # rotate to candidate local frame
        max_abs = np.abs(p_loc).max(axis=1)           # should equal half_size everywhere
        residuals[i] = np.mean((max_abs - half_size) ** 2)

    return float(angles[int(residuals.argmin())])
# ─────────────────────────────────────────────────────────────────────────────
# The block is always placed at this position in simulation regardless of what
# absolute coordinates appear in the prediction frame.  Only the *relative*
# displacement between the block and the gripper at t=0 is preserved.

_CANONICAL_BLOCK_XY = (-0.135, 0.00)   # BOUNDARY_CENTER_X, BOUNDARY_CENTER_Y
_CANONICAL_BLOCK_Z  = 0.026            # CUBE_HALF + 1e-3

CANONICAL_BLOCK_POS = np.array(
    [_CANONICAL_BLOCK_XY[0], _CANONICAL_BLOCK_XY[1], _CANONICAL_BLOCK_Z],
    dtype=np.float32,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3D conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_from_3D(prediction_dict: dict) -> dict:
    """
    Convert raw 3D particle-cloud predictions into the standardised playback dict.

    No external templates are required.  Frame 0 particles are used as the
    reference for Kabsch fitting on subsequent frames.

    Input
    -----
    prediction_dict must contain:
        pred_positions  : (T, N, 3)      predicted particle positions
        obj_ids         : (T, N) or (N,) integer particle IDs (stable over time)
        hand_id         : int            ID for gripper particles
        block_id        : int            ID for block particles
        all_targets     : (T, N, 3)      target positions; NaN where no target

    Output (standardised playback dict)
    ------------------------------------
        initial_block_pos    (3,)       always CANONICAL_BLOCK_POS
        initial_block_quat   (4,)       [w,x,y,z] — identity at t=0 by construction
        initial_gripper_xy   (2,)       canonical_block_xy + relative offset at t=0
        predicted_actions    (T-1, 2)   delta-XY derived from hand centroid trajectory
        predicted_block_pos  (T-1, 3)   block positions after each action, world frame
        predicted_block_quat (T-1, 4)   block quaternions after each action
        target_xy            (2,) or None
        tcp_positions        (T, 3)     full TCP trajectory, world frame (diagnostic)
        block_positions      (T, 3)     full block trajectory, world frame (diagnostic)

    Coordinate mapping
    ------------------
        world_offset = CANONICAL_BLOCK_POS - block_centroid[0]

    Applied to every position so the block starts at CANONICAL_BLOCK_POS while
    all relative distances are preserved.
    """
    pred_positions = np.asarray(prediction_dict["predicted_positions"], dtype=np.float32)
    all_targets    = np.asarray(prediction_dict["all_targets"],    dtype=np.float32)
    block_id        = int(prediction_dict["hand_ids"])
    hand_id       = int(prediction_dict["block_ids"])

    raw_ids = np.asarray(prediction_dict["obj_ids"])
    ids_1d  = raw_ids[0] if raw_ids.ndim == 2 else raw_ids

    T, N = pred_positions.shape[:2]

    # ── Separate particles ────────────────────────────────────────────────────
    hand_mask  = ids_1d == hand_id
    block_mask = ids_1d == block_id

    if not hand_mask.any():
        raise ValueError(f"No particles with hand_id={hand_id}. "
                         f"Unique IDs: {np.unique(ids_1d).tolist()}")
    if not block_mask.any():
        raise ValueError(f"No particles with block_id={block_id}. "
                         f"Unique IDs: {np.unique(ids_1d).tolist()}")

    hand_particles  = pred_positions[:, hand_mask,  :]   # (T, n_hand,  3)
    block_particles = pred_positions[:, block_mask, :]   # (T, n_block, 3)

    # ── Hand: centroid XY only (floating gripper is 2D) ──────────────────────
    hand_centroids = hand_particles.mean(axis=1)         # (T, 3)
    tcp_xy_pred    = hand_centroids[:, :2]               # (T, 2)

    # ── Block: rotation estimation ────────────────────────────────────────────
    # Center each frame's particles on their own centroid.  Kabsch across
    # frames is valid here because particle indices are consistent over time
    # (the model tracks the same particles, not re-samples each frame).
    block_centroids = block_particles.mean(axis=1, keepdims=True)   # (T, 1, 3)
    block_centred   = block_particles - block_centroids             # (T, n_block, 3)

    # Absolute yaw at t=0: grid-search minimising the cube-surface residual.
    # This avoids the correspondence problem entirely — no template needed.
    yaw_0 = _estimate_yaw_from_particles(block_centred[0], half_size=0.025)
    c0, s0 = np.cos(yaw_0), np.sin(yaw_0)
    R0 = np.array([[c0, -s0, 0.0],
                   [s0,  c0, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float32)   # Z-rotation only

    # Relative rotation of each frame w.r.t. frame 0 via Kabsch.
    # Correspondence is valid because particle i is the same object point at
    # every timestep, so aligning frame-0 particles to frame-t particles gives
    # the true rigid-body rotation between those two states.
    Rs_rel, _ = kabsch_batch(block_centred[0], block_centred)       # (T, 3, 3)

    # Absolute rotation at each frame: R_abs[t] = R_rel[t] @ R0
    Rs_abs = np.stack([Rs_rel[t] @ R0 for t in range(T)])           # (T, 3, 3)

    block_pos_pred = block_centroids[:, 0, :]                       # (T, 3)
    block_quats    = np.stack([rotation_matrix_to_quaternion(Rs_abs[t])
                               for t in range(T)])                  # (T, 4)

    # ── World-frame offset ────────────────────────────────────────────────────
    world_offset      = CANONICAL_BLOCK_POS - block_pos_pred[0]  # (3,)
    tcp_positions_world   = hand_centroids  + world_offset        # (T, 3)
    block_positions_world = block_pos_pred  + world_offset        # (T, 3)
    tcp_xy_world          = tcp_positions_world[:, :2]            # (T, 2)

    # ── Initial state ─────────────────────────────────────────────────────────
    initial_block_pos  = CANONICAL_BLOCK_POS.copy()               # (3,)
    initial_block_quat = block_quats[0].copy()    # absolute rotation from canonical template fit
    initial_gripper_xy = tcp_xy_world[0].copy()                   # (2,)

    # ── Actions and aligned predictions ──────────────────────────────────────
    predicted_actions    = np.diff(tcp_xy_world, axis=0).astype(np.float32)      # (T-1, 2)
    predicted_block_pos  = block_positions_world[1:].astype(np.float32)          # (T-1, 3)
    predicted_block_quat = block_quats[1:].astype(np.float32)                    # (T-1, 4)

    # ── Target extraction ─────────────────────────────────────────────────────
    block_targets = all_targets[:, block_mask, :]        # (T, n_block, 3)
    target_xy: Optional[np.ndarray] = None

    for t in range(T - 1, -1, -1):
        frame       = block_targets[t]
        finite_mask = np.isfinite(frame).all(axis=-1)
        if finite_mask.any():
            centroid_world = frame[finite_mask].mean(0) + world_offset
            target_xy      = centroid_world[:2].astype(np.float32)
            break

    return dict(
        initial_block_pos    = initial_block_pos,
        initial_block_quat   = initial_block_quat,
        initial_gripper_xy   = initial_gripper_xy,
        predicted_actions    = predicted_actions,
        predicted_block_pos  = predicted_block_pos,
        predicted_block_quat = predicted_block_quat,
        target_xy            = target_xy,
        tcp_positions        = tcp_positions_world,
        block_positions      = block_positions_world,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2D conversion  (stub — to be implemented)
# ─────────────────────────────────────────────────────────────────────────────

def convert_from_2D(prediction_dict: dict) -> dict:
    """
    Convert 2D top-down predictions into the standardised playback dict.

    Expected input (raw 2D NPZ / dict)
    ---------------------------------
        states    (T, 4) or (T, 6) in prediction frame.

        - 4D legacy layout:
            [tcp_x, tcp_y, block_x, block_y]
          (no yaw; block is treated as axis-aligned)

        - 6D pushblock_offline layout:
            [tcp_x, tcp_y, block_x, block_y, yaw_a, yaw_b]
          where (yaw_a, yaw_b) is a continuous 2D encoding of yaw. By default this
          is interpreted as (cos(yaw), sin(yaw)) so yaw = atan2(yaw_b, yaw_a).

        Optional:
            yaw_encoding: str  "cos_sin" (default) or "sin_cos" to swap (yaw_a,yaw_b).
        target_xy (2,) optional, in the same prediction frame.

    Output (standardised playback dict)
    ----------------------------------
        initial_block_pos   (3,)   always CANONICAL_BLOCK_POS
        initial_block_quat  (4,)   [w,x,y,z] (yaw recovered if provided by 6D states)
        initial_gripper_xy  (2,)
        predicted_actions   (T-1, 2)
        predicted_block_pos (T-1, 3)
        predicted_block_quat (T-1, 4) optional (present for 6D input)
        target_xy           (2,) or None   world-frame target XY

    Coordinate mapping
    ------------------
    The entire scene is shifted so the predicted block starts at CANONICAL_BLOCK_POS
    in the simulator while preserving the relative gripper–block offset at t=0.
    """
    states = np.asarray(prediction_dict["states"], dtype=np.float32)
    if states.ndim != 2 or states.shape[1] not in (4, 6):
        raise ValueError(f"'states' must have shape (T, 4) or (T, 6); got {states.shape}")
    if states.shape[0] < 2:
        raise ValueError(f"'states' must have at least 2 frames; got T={states.shape[0]}")

    tcp_xy = states[:, 0:2]
    block_xy = states[:, 2:4]
    has_yaw = states.shape[1] == 6

    world_offset_xy = CANONICAL_BLOCK_POS[:2] - block_xy[0]
    tcp_xy_world = (tcp_xy + world_offset_xy).astype(np.float32)
    block_xy_world = (block_xy + world_offset_xy).astype(np.float32)

    z_const = float(CANONICAL_BLOCK_POS[2])
    block_positions_world = np.concatenate(
        [block_xy_world, np.full((block_xy_world.shape[0], 1), z_const, dtype=np.float32)],
        axis=1,
    )

    initial_block_pos = CANONICAL_BLOCK_POS.copy()
    initial_gripper_xy = tcp_xy_world[0].copy()

    predicted_actions = np.diff(tcp_xy_world, axis=0).astype(np.float32)
    predicted_block_pos = block_positions_world[1:].astype(np.float32)

    predicted_block_quat = None
    if has_yaw:
        yaw_a = states[:, 4].astype(np.float32)
        yaw_b = states[:, 5].astype(np.float32)
        enc = str(prediction_dict.get("yaw_encoding", "cos_sin")).strip().lower()
        if enc == "sin_cos":
            yaw_a, yaw_b = yaw_b, yaw_a
        # Interpret (yaw_a, yaw_b) as (cos(yaw), sin(yaw))
        yaws = np.arctan2(yaw_b, yaw_a).astype(np.float32)  # (T,)
        half = 0.5 * yaws
        w = np.cos(half).astype(np.float32)
        z = np.sin(half).astype(np.float32)
        # [w,x,y,z] quaternion for yaw-only rotation
        block_quat_world = np.stack([w, np.zeros_like(w), np.zeros_like(w), z], axis=1)
        initial_block_quat = block_quat_world[0].copy()
        predicted_block_quat = block_quat_world[1:].astype(np.float32)
    else:
        initial_block_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    target_xy = prediction_dict.get("target_xy", None)
    if target_xy is not None:
        target_xy = np.asarray(target_xy, dtype=np.float32).ravel()
        if target_xy.shape != (2,):
            raise ValueError(f"'target_xy' must have shape (2,); got {target_xy.shape}")
        target_xy = (target_xy + world_offset_xy).astype(np.float32)

    out = dict(
        initial_block_pos=initial_block_pos,
        initial_block_quat=initial_block_quat,
        initial_gripper_xy=initial_gripper_xy,
        predicted_actions=predicted_actions,
        predicted_block_pos=predicted_block_pos,
        target_xy=target_xy,
        tcp_positions=np.concatenate(
            [tcp_xy_world, np.full((tcp_xy_world.shape[0], 1), z_const, dtype=np.float32)],
            axis=1,
        ).astype(np.float32),
        block_positions=block_positions_world.astype(np.float32),
    )
    if predicted_block_quat is not None:
        out["predicted_block_quat"] = predicted_block_quat
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

_CONVERTERS = {
    "3d": convert_from_3D,
    "2d": convert_from_2D,
}


def convert(prediction_dict: dict, mode: str) -> dict:
    """
    Route to the appropriate converter.

    Parameters
    ----------
    prediction_dict : dict   Raw model output (format depends on mode).
    mode            : str    '3d' or '2d'.

    Returns
    -------
    Standardised playback dict (see module docstring).
    """
    mode = mode.lower().strip()
    if mode not in _CONVERTERS:
        raise ValueError(f"Unknown conversion mode '{mode}'. "
                         f"Available: {list(_CONVERTERS)}")
    return _CONVERTERS[mode](prediction_dict)