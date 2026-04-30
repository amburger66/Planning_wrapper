#!/usr/bin/env python3
"""
scripts/collect_demos_batch.py

Batched face-push demo collection for PushBoundary (floating gripper).

4-face protocol
---------------
Each block configuration (random yaw) is used for exactly 4 episodes —
one per cube face (permuted randomly).  A new yaw is drawn only after all
4 faces have been visited, giving balanced push-direction coverage.

Backends
--------
  --backend gpu   native ManiSkill parallel envs (fastest, requires CUDA)
  --backend cpu   gymnasium AsyncVectorEnv (multiprocessing, works everywhere)

Usage:
    python scripts/collect_demos_batch.py --num_envs 32 --num_demos 1000 --backend gpu
    python scripts/collect_demos_batch.py --num_envs 8  --num_demos 200  --backend cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode

import envs  # noqa: F401

from envs.push_boundary import (
    BOUNDARY_CENTER_X as BCX,
    BOUNDARY_CENTER_Y as BCY,
    BOUNDARY_HALF_X   as BHX,
    BOUNDARY_HALF_Y   as BHY,
    GRIPPER_Z_FIXED,
    CUBE_Z_SPAWN,
)

from collect_demos_single import (
    FaceApproachPushPolicy,
    _get_xyz, _get_xyzw_quat, _unwrap,
    _set_gripper_xy, _yaw_to_sapien_quat,
    _obb_support, _wall_dist, _truncated_gaussian,
    GRIPPER_RADIUS, SPAWN_CLEARANCE, GRIPPER_SPAWN_STD, WALL_MARGIN,
    SPAWN_MIN_X, SPAWN_MAX_X, SPAWN_MIN_Y, SPAWN_MAX_Y,
)


# ─────────────────────────────────────────────────────────────────────────────
# 4-face rotation tracker
# ─────────────────────────────────────────────────────────────────────────────

class EnvConfig:
    """Cycles through all 4 block faces before drawing a new yaw."""

    def __init__(self):
        self._face_queue: list = []
        self.block_yaw: float  = 0.0

    def next(self, rng):
        if not self._face_queue:
            self.block_yaw   = rng.uniform(-np.pi, np.pi)
            self._face_queue = rng.permutation(4).tolist()
        return self.block_yaw, self._face_queue.pop(0)


# ─────────────────────────────────────────────────────────────────────────────
# Spawn helper
# ─────────────────────────────────────────────────────────────────────────────

def _sample_gripper_spawn(hx, hy, block_yaw, rng):
    for _ in range(100):
        angle     = rng.uniform(0.0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        support   = _obb_support(hx, hy, block_yaw, direction)
        wall_d    = _wall_dist(direction)
        lo        = GRIPPER_RADIUS + SPAWN_CLEARANCE
        hi        = wall_d - support - WALL_MARGIN - GRIPPER_RADIUS
        if hi <= lo:
            continue
        extra = _truncated_gaussian(rng, 0.0, GRIPPER_SPAWN_STD, lo, hi)
        gx = BCX + direction[0] * (support + extra)
        gy = BCY + direction[1] * (support + extra)
        return gx, gy
    raise RuntimeError("Failed to sample valid gripper spawn after 100 attempts")


def _to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CPU worker wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _make_env_fn(record_dir, worker_seed, worker_idx, hx, hy):
    """Returns a thunk for AsyncVectorEnv.  Each worker manages its own RNG."""

    def _thunk():
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        sys.path.insert(0, str(Path(__file__).resolve().parent))

        import numpy as np
        import sapien
        import gymnasium as gym
        from mani_skill.utils.wrappers.record import RecordEpisode
        import envs  # noqa: F401

        from collect_demos_single import (
            _yaw_to_sapien_quat, _set_gripper_xy, _get_xyz,
            _obb_support, _wall_dist, _truncated_gaussian,
            BCX, BCY,
            GRIPPER_RADIUS, SPAWN_CLEARANCE, GRIPPER_SPAWN_STD, WALL_MARGIN,
            GRIPPER_Z_FIXED, CUBE_Z_SPAWN,
        )

        env = gym.make(
            "PushBoundary",
            obs_mode="state_dict",
            control_mode="floating_vel",
            render_mode=None,
            sim_backend="cpu",
            num_envs=1,
            robot_uids="floating_gripper",
        )

        class _InitWrapper(gym.Wrapper):
            def __init__(self, env):
                super().__init__(env)
                self._rng        = np.random.default_rng(worker_seed)
                self._face_queue = []
                self._block_yaw  = 0.0
                self._block_z    = None

            def _base(self):
                e = self.env
                while hasattr(e, "env"):
                    e = e.env
                return e

            def _cache_z(self):
                if self._block_z is None:
                    base = self._base()
                    self._block_z = float(_get_xyz(base.block.pose.p)[2])

            def _next_config(self):
                if not self._face_queue:
                    self._block_yaw  = self._rng.uniform(-np.pi, np.pi)
                    self._face_queue = self._rng.permutation(4).tolist()
                return self._block_yaw, self._face_queue.pop(0)

            def _sample_spawn(self, block_yaw):
                for _ in range(20):
                    angle     = self._rng.uniform(0.0, 2 * np.pi)
                    direction = np.array([np.cos(angle), np.sin(angle)])
                    support   = _obb_support(hx, hy, block_yaw, direction)
                    wall_d    = _wall_dist(direction)
                    lo        = GRIPPER_RADIUS + SPAWN_CLEARANCE
                    hi        = wall_d - support - WALL_MARGIN - GRIPPER_RADIUS
                    if hi <= lo:
                        continue
                    extra = _truncated_gaussian(self._rng, 0.0, GRIPPER_SPAWN_STD, lo, hi)
                    gx = BCX + direction[0] * (support + extra)
                    gy = BCY + direction[1] * (support + extra)
                    return gx, gy
                return BCX + hx + 0.12, BCY

            def _sq(self, obs):
                if isinstance(obs, dict):
                    return {k: self._sq(v) for k, v in obs.items()}
                a = np.asarray(obs, dtype=np.float32)
                return a[0] if (a.ndim > 0 and a.shape[0] == 1) else a

            def reset(self, **kwargs):
                kwargs.pop("options", None)
                obs, info = self.env.reset(**kwargs)
                self._cache_z()

                block_yaw, face_index = self._next_config()
                gx, gy                = self._sample_spawn(block_yaw)

                base = self._base()
                base.block.set_pose(
                    sapien.Pose(p=[BCX, BCY, self._block_z],
                                q=_yaw_to_sapien_quat(block_yaw))
                )
                try:
                    base.block.set_velocity([0.0, 0.0, 0.0])
                    base.block.set_angular_velocity([0.0, 0.0, 0.0])
                except Exception:
                    pass
                _set_gripper_xy(base, gx, gy)

                raw_obs = self.env.get_obs()
                obs     = self._sq(raw_obs)
                info["init_block_yaw"]  = float(block_yaw)
                info["init_face_index"] = int(face_index)
                info["init_gx"]         = float(gx)
                info["init_gy"]         = float(gy)
                return obs, info

            def step(self, action):
                obs, r, te, tr, info = self.env.step(action)
                r  = float(np.asarray(r,  dtype=np.float32).reshape(-1)[0])
                te = bool(np.asarray(te, dtype=bool).reshape(-1)[0])
                tr = bool(np.asarray(tr, dtype=bool).reshape(-1)[0])
                return self._sq(obs), r, te, tr, info

        env = _InitWrapper(env)
        worker_record_dir = str(Path(record_dir) / f"worker_{worker_idx:03d}")
        env = RecordEpisode(
            env,
            output_dir=worker_record_dir,
            save_trajectory=True,
            save_video=False,
            trajectory_name="face_push_batch",
        )
        env.reset()
        return env

    return _thunk


# ─────────────────────────────────────────────────────────────────────────────
# GPU backend
# ─────────────────────────────────────────────────────────────────────────────

def run_gpu(args):
    import torch
    from mani_skill.utils.structs import Pose

    N          = args.num_envs
    rng_master = np.random.default_rng(args.seed)

    env = gym.make(
        "PushBoundary",
        obs_mode="state_dict",
        control_mode="floating_vel",
        render_mode=None,
        sim_backend="gpu",
        num_envs=N,
        robot_uids="floating_gripper",
    )
    env = RecordEpisode(
        env,
        output_dir=args.record_dir,
        save_trajectory=True,
        save_video=False,
        trajectory_name="face_push_batch",
    )
    base_env = _unwrap(env)
    hx, hy   = base_env.block_dims.half_x, base_env.block_dims.half_y

    obs, _ = env.reset(seed=int(rng_master.integers(2**31)))
    block_z = float(_to_np(base_env.block.pose.p)[0, 2])

    rngs     = [np.random.default_rng(rng_master.integers(2**31)) for _ in range(N)]
    configs  = [EnvConfig() for _ in range(N)]
    policies = [FaceApproachPushPolicy() for _ in range(N)]

    def init_env(i):
        yaw, face = configs[i].next(rngs[i])
        gx, gy    = _sample_gripper_spawn(hx, hy, yaw, rngs[i])

        p_all = _to_np(base_env.block.pose.p).copy()
        q_all = _to_np(base_env.block.pose.q).copy()
        p_all[i] = [BCX, BCY, block_z]
        q_all[i] = np.array(_yaw_to_sapien_quat(yaw), dtype=np.float32)
        base_env.block.set_pose(Pose.create_from_pq(
            p=torch.tensor(p_all, device=base_env.device),
            q=torch.tensor(q_all, device=base_env.device),
        ))

        # Zero block velocity
        lin_vel = _to_np(base_env.block.linear_velocity).copy()
        ang_vel = _to_np(base_env.block.angular_velocity).copy()
        lin_vel[i] = 0.0; ang_vel[i] = 0.0
        base_env.block.set_linear_velocity(torch.tensor(lin_vel, device=base_env.device))
        base_env.block.set_angular_velocity(torch.tensor(ang_vel, device=base_env.device))

        qpos = _to_np(base_env.agent.robot.get_qpos()).copy()
        qvel = _to_np(base_env.agent.robot.get_qvel()).copy()
        qpos[i, 0] = gx - BCX
        qpos[i, 1] = gy - BCY
        qvel[i]    = 0.0
        base_env.agent.robot.set_qpos(torch.tensor(qpos, device=base_env.device))
        base_env.agent.robot.set_qvel(torch.tensor(qvel, device=base_env.device))

        policies[i].reset(np.array([gx, gy]), np.array([BCX, BCY]),
                          yaw, hx, hy, rngs[i], initial_face=face)

    for i in range(N):
        init_env(i)

    completed       = np.zeros(N, dtype=int)
    alive_steps_arr = np.zeros(N, dtype=int)
    total_steps     = 0
    t0              = time.time()

    print(f"GPU batch: {N} envs  target={args.num_demos} demos")

    while completed.sum() < args.num_demos:
        ee_poses    = _to_np(base_env.agent.tcp.pose.p)
        block_poses = _to_np(base_env.block.pose.p)
        block_quats = _to_np(base_env.block.pose.q)

        actions = np.zeros((N, 2), dtype=np.float32)
        for i in range(N):
            q    = block_quats[i]
            bq   = np.array([q[1], q[2], q[3], q[0]])
            byaw = float(Rotation.from_quat(bq).as_euler("xyz")[2])
            actions[i] = policies[i].act(ee_poses[i, :2], block_poses[i, :2], byaw)

        obs, _, terminated, truncated, _ = env.step(actions)
        total_steps     += N
        alive_steps_arr += 1

        dones = (
            _to_np(terminated).reshape(N).astype(bool) |
            _to_np(truncated).reshape(N).astype(bool)  |
            (alive_steps_arr >= args.max_episode_steps)
        )

        done_envs = np.where(dones)[0]

        for i in done_envs:
            completed[i] += 1
            n_done = int(completed.sum())
            elapsed = time.time() - t0
            print(f"  [{n_done:>5d}/{args.num_demos}]  env={i}"
                  f"  steps={alive_steps_arr[i]}"
                  f"  fps={total_steps/elapsed:.0f}")
            alive_steps_arr[i] = 0

        if int(completed.sum()) >= args.num_demos:
            break

        for i in done_envs:
            env.reset(options={"env_idx": torch.tensor([i], device=base_env.device)})

        p_all = _to_np(base_env.block.pose.p).copy()
        q_all = _to_np(base_env.block.pose.q).copy()
        qpos  = _to_np(base_env.agent.robot.get_qpos()).copy()
        qvel  = _to_np(base_env.agent.robot.get_qvel()).copy()

        for i in done_envs:
            yaw, face  = configs[i].next(rngs[i])
            gx, gy     = _sample_gripper_spawn(hx, hy, yaw, rngs[i])
            p_all[i]   = [BCX, BCY, block_z]
            q_all[i]   = _yaw_to_sapien_quat(yaw)
            qpos[i, 0] = gx - BCX
            qpos[i, 1] = gy - BCY
            qvel[i, :] = 0
            policies[i].reset(np.array([gx, gy]), np.array([BCX, BCY]),
                              yaw, hx, hy, rngs[i], initial_face=face)

        base_env.block.set_pose(Pose.create_from_pq(
            p=torch.tensor(p_all, device=base_env.device),
            q=torch.tensor(q_all, device=base_env.device)))
        base_env.agent.robot.set_qpos(torch.tensor(qpos, device=base_env.device))
        base_env.agent.robot.set_qvel(torch.tensor(qvel, device=base_env.device))

    env.close()
    elapsed = time.time() - t0
    print(f"\nDone. {int(completed.sum())} demos  {elapsed:.1f}s"
          f"  ({total_steps/elapsed:.0f} steps/s)")
    print(f"Saved to: {args.record_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CPU backend
# ─────────────────────────────────────────────────────────────────────────────

def run_cpu(args):
    N          = args.num_envs
    rng_master = np.random.default_rng(args.seed)
    w_seeds    = [int(rng_master.integers(2**31)) for _ in range(N)]

    # Probe dims before forking
    _probe = gym.make(
        "PushBoundary", obs_mode="state_dict", control_mode="floating_vel",
        render_mode=None, sim_backend="cpu", num_envs=1,
        robot_uids="floating_gripper",
    )
    hx = _probe.unwrapped.block_dims.half_x
    hy = _probe.unwrapped.block_dims.half_y
    _probe.close()

    vec_env = gym.vector.AsyncVectorEnv(
        [_make_env_fn(args.record_dir, w_seeds[i], i, hx, hy) for i in range(N)],
        context="spawn",
    )

    rngs     = [np.random.default_rng(rng_master.integers(2**31)) for _ in range(N)]
    policies = [FaceApproachPushPolicy() for _ in range(N)]
    block_xy = np.array([BCX, BCY])

    obs, info = vec_env.reset(seed=w_seeds)

    def init_policies(info):
        yaws  = np.asarray(info.get("init_block_yaw",  np.zeros(N)), dtype=float)
        faces = np.asarray(info.get("init_face_index", np.zeros(N)), dtype=int)
        gxs   = np.asarray(info.get("init_gx", np.full(N, BCX + 0.12)), dtype=float)
        gys   = np.asarray(info.get("init_gy", np.full(N, BCY)),         dtype=float)
        for i in range(N):
            policies[i].reset(np.array([gxs[i], gys[i]]), block_xy,
                              float(yaws[i]), hx, hy, rngs[i],
                              initial_face=int(faces[i]))

    init_policies(info)

    completed   = 0
    alive_steps = np.zeros(N, dtype=int)
    total_steps = 0
    t0          = time.time()

    print(f"CPU batch: {N} workers  target={args.num_demos} demos")

    while completed < args.num_demos:
        tcp_poses   = np.asarray(obs["extra"]["tcp_pose"],   dtype=np.float32).reshape(N, -1)
        block_poses = np.asarray(obs["extra"]["block_pose"], dtype=np.float32).reshape(N, -1)

        actions = np.zeros((N, 2), dtype=np.float32)
        for i in range(N):
            q    = block_poses[i, 3:7]
            bq   = np.array([q[1], q[2], q[3], q[0]])
            byaw = float(Rotation.from_quat(bq).as_euler("xyz")[2])
            actions[i] = policies[i].act(tcp_poses[i, :2], block_poses[i, :2], byaw)

        obs, _, terminated, truncated, info = vec_env.step(actions)
        total_steps  += N
        alive_steps  += 1

        dones = np.asarray(terminated, bool) | np.asarray(truncated, bool)
        if dones.any():
            yaws  = np.asarray(info.get("init_block_yaw",  np.zeros(N)), dtype=float)
            faces = np.asarray(info.get("init_face_index", np.zeros(N)), dtype=int)
            gxs   = np.asarray(info.get("init_gx", np.full(N, BCX + 0.12)), dtype=float)
            gys   = np.asarray(info.get("init_gy", np.full(N, BCY)),         dtype=float)

            for i in np.where(dones)[0]:
                completed += 1
                elapsed = time.time() - t0
                print(f"  [{completed:>5d}/{args.num_demos}]  worker={i}"
                      f"  steps={alive_steps[i]}"
                      f"  fps={total_steps/elapsed:.0f}")
                alive_steps[i] = 0
                policies[i].reset(
                    np.array([gxs[i], gys[i]]), block_xy,
                    float(yaws[i]), hx, hy, rngs[i], initial_face=int(faces[i])
                )
            if completed >= args.num_demos:
                break

    vec_env.close()
    elapsed = time.time() - t0
    print(f"\nDone. {completed} demos  {elapsed:.1f}s  ({total_steps/elapsed:.0f} steps/s)")
    print(f"Saved to: {args.record_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Batched face-push demo collection.")
    p.add_argument("--num_envs",          type=int, default=16)
    p.add_argument("--num_demos",         type=int, default=1000)
    p.add_argument("--max_episode_steps", type=int, default=400)
    p.add_argument("--seed",              type=int, default=None)
    p.add_argument("--record_dir",        type=str, default="demos/PushBoundary/batch")
    p.add_argument("--backend",           type=str, default="gpu", choices=["cpu", "gpu"])
    return p.parse_args()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    args = parse_args()
    print(f"Batched face-push  backend={args.backend}  num_envs={args.num_envs}"
          f"  target={args.num_demos}  max_steps={args.max_episode_steps}")
    if args.backend == "gpu":
        run_gpu(args)
    else:
        run_cpu(args)