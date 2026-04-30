#!/usr/bin/env python3
"""
scripts/collect_demos_single.py

Scripted face-approach push demo collection for PushBoundary (floating gripper).

Each episode:
  1. Spawns the block at centre with a random yaw.
  2. Spawns the gripper at a collision-free position.
  3. Runs a two-phase policy: APPROACH standoff → PUSH.
  4. Resamples a new face every ~20-40 push steps.

Usage:
    python scripts/collect_demos_single.py --num_episodes 50 --record_dir demos/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import truncnorm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import sapien
import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode

import envs  # noqa: F401 — registers PushBoundary + FloatingGripper

from envs.push_boundary import (
    BOUNDARY_CENTER_X as BCX,
    BOUNDARY_CENTER_Y as BCY,
    BOUNDARY_HALF_X   as BHX,
    BOUNDARY_HALF_Y   as BHY,
    GRIPPER_Z_FIXED,
    CUBE_Z_SPAWN,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

GRIPPER_RADIUS   = 0.01    # cylinder radius (metres)
CORNER_CLEARANCE = 0.015

STANDOFF_DIST_MIN = 0.03
STANDOFF_DIST_MAX = 0.07
STANDOFF_DIST_MU  = 0.05
STANDOFF_DIST_STD = 0.01
CORNER_JITTER     = 0.01

WALL_MARGIN       = 0.10
SPAWN_CLEARANCE   = GRIPPER_RADIUS + 0.005
SPAWN_MAX_DIST    = 0.10
GRIPPER_SPAWN_STD = 0.045

SPAWN_MIN_X = BCX - BHX + WALL_MARGIN
SPAWN_MAX_X = BCX + BHX - WALL_MARGIN
SPAWN_MIN_Y = BCY - BHY + WALL_MARGIN
SPAWN_MAX_Y = BCY + BHY - WALL_MARGIN


# ─────────────────────────────────────────────────────────────────────────────
# 2D collision helpers
# ─────────────────────────────────────────────────────────────────────────────

def _circle_hits_obb(circle_xy, r, block_xy, hx, hy, angle):
    c, s = np.cos(angle), np.sin(angle)
    dx = circle_xy[0] - block_xy[0]
    dy = circle_xy[1] - block_xy[1]
    lx =  c * dx + s * dy
    ly = -s * dx + c * dy
    cx = np.clip(lx, -hx, hx)
    cy = np.clip(ly, -hy, hy)
    return (lx - cx) ** 2 + (ly - cy) ** 2 < r * r


def _segment_hits_obb(p1, p2, center, hx, hy, angle):
    c, s  = np.cos(angle), np.sin(angle)
    R_inv = np.array([[c, s], [-s, c]])
    p1_l  = R_inv @ (p1 - center)
    p2_l  = R_inv @ (p2 - center)
    d     = p2_l - p1_l
    tmin, tmax = 0.0, 1.0
    for i, half in enumerate([hx, hy]):
        lo, hi = -half, half
        if abs(d[i]) < 1e-9:
            if p1_l[i] <= lo or p1_l[i] >= hi:
                return False
        else:
            t1 = (lo - p1_l[i]) / d[i]
            t2 = (hi - p1_l[i]) / d[i]
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin >= tmax:
                return False
    return tmin < 1.0 and tmax > 0.0


def _capsule_hits_obb(p1, p2, r, block_xy, hx, hy, angle):
    hits_exp = _segment_hits_obb(p1, p2, block_xy, hx + r, hy + r, angle)
    ep1 = _circle_hits_obb(p1, r, block_xy, hx, hy, angle)
    ep2 = _circle_hits_obb(p2, r, block_xy, hx, hy, angle)
    return hits_exp or ep1 or ep2


# ─────────────────────────────────────────────────────────────────────────────
# Path planning
# ─────────────────────────────────────────────────────────────────────────────

def _truncated_gaussian(rng, mu, sigma, lo, hi):
    a, b = (lo - mu) / sigma, (hi - mu) / sigma
    if a >= b:
        return rng.uniform(lo, hi)
    return float(truncnorm.rvs(a, b, loc=mu, scale=sigma,
                               random_state=rng.integers(2**31)))


def _jittered_corners(block_xy, hx, hy, angle, base_pad, rng):
    c, s = np.cos(angle), np.sin(angle)
    corners = []
    for sx in (+1, -1):
        for sy in (+1, -1):
            pad = base_pad + rng.uniform(0.0, CORNER_JITTER)
            lx  = (hx + pad) * sx
            ly  = (hy + pad) * sy
            wx  = block_xy[0] + c * lx - s * ly
            wy  = block_xy[1] + s * lx + c * ly
            corners.append(np.array([wx, wy]))
    return corners


def _plan_approach(ee_xy, standoff, block_xy, hx, hy, angle, rng, max_attempts=8):
    r        = GRIPPER_RADIUS
    base_pad = GRIPPER_RADIUS + CORNER_CLEARANCE
    NUDGE    = r * 2.5

    def hits_from_ee(a, b):
        d = b - a
        dist = np.linalg.norm(d)
        if dist < 1e-6:
            return False
        a_check = a + d / dist * min(NUDGE, dist * 0.4)
        return _capsule_hits_obb(a_check, b, r, block_xy, hx, hy, angle)

    def hits(a, b):
        return _capsule_hits_obb(a, b, r, block_xy, hx, hy, angle)

    if not hits_from_ee(ee_xy, standoff):
        return []

    for _ in range(max_attempts):
        corners   = _jittered_corners(block_xy, hx, hy, angle, base_pad, rng)
        best      = None
        best_cost = float("inf")

        for c in corners:
            if not hits_from_ee(ee_xy, c) and not hits(c, standoff):
                cost = np.linalg.norm(c - ee_xy) + np.linalg.norm(standoff - c)
                if cost < best_cost:
                    best_cost, best = cost, [c]

        if best is not None:
            return best

        for i, c1 in enumerate(corners):
            for j, c2 in enumerate(corners):
                if i == j:
                    continue
                if not hits_from_ee(ee_xy, c1) and not hits(c1, c2) and not hits(c2, standoff):
                    cost = (np.linalg.norm(c1 - ee_xy)
                          + np.linalg.norm(c2 - c1)
                          + np.linalg.norm(standoff - c2))
                    if cost < best_cost:
                        best_cost, best = cost, [c1, c2]

        if best is not None:
            return best

    corners = _jittered_corners(block_xy, hx, hy, angle, base_pad, rng)
    return [min(corners, key=lambda c: np.linalg.norm(c - ee_xy))]


# ─────────────────────────────────────────────────────────────────────────────
# Misc helpers
# ─────────────────────────────────────────────────────────────────────────────

def _obb_support(hx, hy, angle, direction):
    ax = np.array([ np.cos(angle),  np.sin(angle)])
    ay = np.array([-np.sin(angle),  np.cos(angle)])
    return hx * abs(float(np.dot(direction, ax))) + hy * abs(float(np.dot(direction, ay)))


def _wall_dist(direction):
    dx, dy = float(direction[0]), float(direction[1])
    cands = []
    if dx >  1e-9: cands.append((SPAWN_MAX_X - BCX) / dx)
    if dx < -1e-9: cands.append((SPAWN_MIN_X - BCX) / dx)
    if dy >  1e-9: cands.append((SPAWN_MAX_Y - BCY) / dy)
    if dy < -1e-9: cands.append((SPAWN_MIN_Y - BCY) / dy)
    return min(c for c in cands if c > 0)


def _yaw_to_sapien_quat(yaw):
    q = Rotation.from_euler("z", yaw).as_quat()
    return [q[3], q[0], q[1], q[2]]


def _get_xyz(pose_p):
    try:
        import torch
        if isinstance(pose_p, torch.Tensor):
            pose_p = pose_p.cpu().numpy()
    except ImportError:
        pass
    return np.asarray(pose_p, dtype=np.float64).reshape(-1)[:3].copy()


def _get_xyzw_quat(pose_q):
    try:
        import torch
        if isinstance(pose_q, torch.Tensor):
            pose_q = pose_q.cpu().numpy()
    except ImportError:
        pass
    q = np.asarray(pose_q, dtype=np.float64).reshape(-1)
    return np.array([q[1], q[2], q[3], q[0]])


def _unwrap(env):
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e


# ─────────────────────────────────────────────────────────────────────────────
# Gripper teleport
# ─────────────────────────────────────────────────────────────────────────────

def _set_gripper_xy(base_env, gx, gy):
    robot = base_env.agent.robot
    try:
        import torch
        n_dof = robot.dof if hasattr(robot, "dof") else len(robot.get_qpos())
        if hasattr(n_dof, "item"):
            n_dof = int(n_dof.item())
        if n_dof > 0:
            qpos    = torch.zeros(n_dof, dtype=torch.float32)
            qpos[0] = gx - BCX
            qpos[1] = gy - BCY
            robot.set_qpos(qpos.unsqueeze(0).expand(base_env.num_envs, -1))
            robot.set_qvel(
                torch.zeros_like(qpos).unsqueeze(0).expand(base_env.num_envs, -1)
            )
            return
    except Exception:
        pass
    robot.set_pose(sapien.Pose(p=[gx, gy, GRIPPER_Z_FIXED], q=[1, 0, 0, 0]))


# ─────────────────────────────────────────────────────────────────────────────
# Policy
# ─────────────────────────────────────────────────────────────────────────────

class FaceApproachPushPolicy:
    """
    Two-phase per-episode policy:
      APPROACH  Collision-free path to a standoff point behind the block face.
      PUSH      Drive in the direction standoff → contact; resample every ~30 steps.
    """

    MAX_STEP              = 0.015
    ARRIVE_THRESH         = 0.015
    RESAMPLE_INTERVAL_MIN = 20
    RESAMPLE_INTERVAL_MAX = 40
    CONTACT_DIST          = 0.04

    def __init__(self):
        self._phase          = "approach"
        self._push_dir       = np.array([1.0, 0.0])
        self._face_index     = 0
        self._contact_offset = 0.0
        self._standoff_offset = 0.0
        self._standoff_dist  = STANDOFF_DIST_MU
        self._standoff_pt    = np.zeros(2)
        self._contact_pt     = np.zeros(2)
        self._waypoints: list = []
        self._hx             = 0.025
        self._hy             = 0.025
        self._block_yaw      = 0.0
        self._rng            = None
        self._step           = 0
        self._next_resample  = 0
        self._style          = "face"

    def _draw_interval(self):
        return int(self._rng.integers(self.RESAMPLE_INTERVAL_MIN,
                                      self.RESAMPLE_INTERVAL_MAX + 1))

    def reset(self, ee_xy, block_xy, block_yaw, hx, hy, rng, initial_face=None):
        self._hx, self._hy = hx, hy
        self._block_yaw    = block_yaw
        self._rng          = rng
        self._step         = 0
        self._new_face(ee_xy, block_xy, block_yaw,
                       prev_face=None, forced_face=initial_face)
        self._next_resample = self._draw_interval()

    @staticmethod
    def _face_normal(face_index, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        return [np.array([ c,  s]),
                np.array([-c, -s]),
                np.array([-s,  c]),
                np.array([ s, -c])][face_index]

    @staticmethod
    def _face_tangent(face_index, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        return [np.array([-s,  c]),
                np.array([-s,  c]),
                np.array([ c,  s]),
                np.array([ c,  s])][face_index]

    def _face_normal_extent(self, fi):
        return self._hx if fi < 2 else self._hy

    def _face_tangent_extent(self, fi):
        return self._hy if fi < 2 else self._hx

    def _sample_contact_offset(self, fi):
        half_ext = self._face_tangent_extent(fi)
        max_off  = half_ext * 0.65
        return _truncated_gaussian(self._rng, 0.0, max_off * 0.45, -max_off, max_off)

    def _sample_standoff_offset(self, fi, contact_offset):
        half_ext  = self._face_tangent_extent(fi)
        max_total = half_ext * 1.5
        lo        = -max_total - contact_offset
        hi        =  max_total - contact_offset
        delta     = _truncated_gaussian(self._rng, 0.0, half_ext * 0.4, lo, hi)
        return contact_offset + delta

    def _compute_geometry(self, block_xy, fi, yaw):
        normal  = self._face_normal(fi, yaw)
        tangent = self._face_tangent(fi, yaw)
        n_ext   = self._face_normal_extent(fi)
        contact_pt  = block_xy + normal * n_ext + tangent * self._contact_offset
        standoff_pt = block_xy - normal * self._standoff_dist + tangent * self._standoff_offset
        d = contact_pt - standoff_pt
        dist = np.linalg.norm(d)
        push_dir = d / dist if dist > 1e-6 else normal.copy()
        return standoff_pt, contact_pt, push_dir

    def _new_face(self, ee_xy, block_xy, block_yaw, prev_face, forced_face=None):
        self._style      = self._rng.choice(["face", "directed"])
        new_fi           = forced_face if forced_face is not None \
                           else int(self._rng.integers(4))
        same_face        = (new_fi == prev_face)
        in_contact       = np.linalg.norm(ee_xy - block_xy) < self.CONTACT_DIST
        self._face_index = new_fi

        self._standoff_dist   = _truncated_gaussian(
            self._rng, STANDOFF_DIST_MU, STANDOFF_DIST_STD,
            STANDOFF_DIST_MIN, STANDOFF_DIST_MAX)
        self._contact_offset  = self._sample_contact_offset(self._face_index)
        self._standoff_offset = self._sample_standoff_offset(
            self._face_index, self._contact_offset)

        standoff_pt, contact_pt, push_dir = self._compute_geometry(
            block_xy, self._face_index, block_yaw)
        self._standoff_pt = standoff_pt
        self._contact_pt  = contact_pt
        self._push_dir    = push_dir.copy()

        if same_face and in_contact:
            self._phase     = "push"
            self._waypoints = []
        else:
            waypoints = _plan_approach(ee_xy, standoff_pt,
                                       block_xy, self._hx, self._hy, block_yaw,
                                       self._rng)
            self._waypoints = waypoints + [standoff_pt]
            self._phase = "approach"

    def act(self, ee_xy, block_xy, block_yaw=None):
        if block_yaw is not None:
            self._block_yaw = block_yaw

        if self._phase == "push":
            self._step += 1
            box_length   = 2.0 * max(self._hx, self._hy)
            lost_contact = np.linalg.norm(ee_xy - block_xy) > 1.5 * box_length
            if lost_contact or self._step >= self._next_resample:
                self._new_face(ee_xy, block_xy, self._block_yaw,
                               prev_face=self._face_index)
                self._next_resample = self._step + self._draw_interval()

        if self._style == "directed":
            _, _, push_dir = self._compute_geometry(
                block_xy, self._face_index, self._block_yaw)
        else:
            push_dir = self._push_dir

        if self._phase == "approach":
            while (self._waypoints and
                   np.linalg.norm(ee_xy - self._waypoints[0]) < self.ARRIVE_THRESH):
                self._waypoints.pop(0)
            target = ee_xy if not self._waypoints else self._waypoints[0]
            if not self._waypoints:
                self._phase = "push"
        else:
            target = ee_xy + push_dir * self.MAX_STEP

        delta = target - ee_xy
        dist  = np.linalg.norm(delta)
        if dist < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return (delta / dist * min(dist, self.MAX_STEP)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Custom initialisation
# ─────────────────────────────────────────────────────────────────────────────

def custom_init(base_env, rng, block_z):
    """Spawn block at centre with random yaw; spawn gripper at a safe offset."""
    hx = base_env.block_dims.half_x
    hy = base_env.block_dims.half_y

    yaw = rng.uniform(-np.pi, np.pi)
    base_env.block.set_pose(
        sapien.Pose(p=[BCX, BCY, block_z], q=_yaw_to_sapien_quat(yaw))
    )

    for _ in range(20):
        angle     = rng.uniform(0.0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        support   = _obb_support(hx, hy, yaw, direction)
        wall_d    = _wall_dist(direction)
        lo = GRIPPER_RADIUS + SPAWN_CLEARANCE
        hi = min(wall_d - support - WALL_MARGIN - GRIPPER_RADIUS, SPAWN_MAX_DIST)
        if hi <= lo:
            continue
        extra = _truncated_gaussian(rng, 0.0, GRIPPER_SPAWN_STD, lo, hi)
        gx = BCX + direction[0] * (support + extra)
        gy = BCY + direction[1] * (support + extra)
        _set_gripper_xy(base_env, gx, gy)
        return yaw, gx, gy

    raise RuntimeError("custom_init: no valid spawn in 20 attempts")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    env = gym.make(
        "PushBoundary",
        obs_mode="state_dict",
        control_mode="floating_vel",
        render_mode="all" if args.render else None,
        sim_backend="auto",
        robot_uids="floating_gripper",
    )
    env = RecordEpisode(
        env,
        output_dir=args.record_dir,
        save_trajectory=True,
        save_video=args.render,
        video_fps=20,
        trajectory_name="face_push",
    )
    base_env = _unwrap(env)
    policy   = FaceApproachPushPolicy()
    rng      = np.random.default_rng(args.seed)

    obs, _ = env.reset(seed=args.seed)
    block_z = float(_get_xyz(base_env.block.pose.p)[2])
    hx      = base_env.block_dims.half_x
    hy      = base_env.block_dims.half_y

    def init_episode():
        yaw, gx, gy = custom_init(base_env, rng, block_z)
        ee_xy    = np.array([gx, gy])
        block_xy = np.array([BCX, BCY])
        policy.reset(ee_xy, block_xy, yaw, hx, hy, rng)

    init_episode()

    episode = alive_steps = total_steps = 0

    while episode < args.num_episodes:
        ee_xyz    = _get_xyz(base_env.agent.tcp.pose.p)
        block_xyz = _get_xyz(base_env.block.pose.p)
        block_yaw = float(Rotation.from_quat(
            _get_xyzw_quat(base_env.block.pose.q)
        ).as_euler("xyz")[2])

        action = policy.act(ee_xyz[:2], block_xyz[:2], block_yaw)

        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1
        alive_steps += 1

        if alive_steps >= args.max_episode_steps:
            truncated = True

        if terminated or truncated:
            episode += 1
            reason = "boundary" if terminated else "timeout"
            print(f"  ep {episode:4d}  {alive_steps:5d} steps  ({reason})")
            if episode >= args.num_episodes:
                break
            obs, _ = env.reset()
            init_episode()
            alive_steps = 0

    env.close()
    print(f"\nDone — {episode} episodes, {total_steps} total steps.")
    print(f"Saved to: {args.record_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Single-process scripted face-push collection.")
    p.add_argument("--num_episodes",      type=int, default=50)
    p.add_argument("--max_episode_steps", type=int, default=400)
    p.add_argument("--seed",              type=int, default=None)
    p.add_argument("--record_dir",        type=str, default="demos/PushBoundary/single")
    p.add_argument("--render",            action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())