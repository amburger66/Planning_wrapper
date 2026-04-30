"""
envs/push_boundary.py

Push a cube block and keep it inside a rectangular boundary.
Designed exclusively for the FloatingGripper agent.

Registration: "PushBoundary"
Control mode: "floating_vel"  ->  action = [dx, dy] in metres

Observation (state mode) extra keys:
    tcp_pose   : (7,)  gripper pose [pos(3), quat_wxyz(4)]
    block_pose : (7,)  block pose
    boundary_center : (2,)
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import sapien
import sapien.render
import sapien.pysapien.physx as physx
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs  # noqa: F401
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.push_t import WhiteTableSceneBuilder
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


# ─────────────────────────────────────────────────────────────────────────────
# Geometry constants  (importable by other scripts)
# ─────────────────────────────────────────────────────────────────────────────

BOUNDARY_CENTER_X = -0.135
BOUNDARY_CENTER_Y = 0.00
BOUNDARY_HALF_X = 0.25
BOUNDARY_HALF_Y = 0.35

STRIP_THICKNESS = 0.006
STRIP_HALF_H = 1e-4
OUT_MARGIN = -0.005

GRIPPER_Z_FIXED = 0.085

# Circle block
CIRCLE_RADIUS = 0.025

CUBE_HALF = 0.025
CUBE_Z_SPAWN = CUBE_HALF + 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# Block dimension descriptor
# ─────────────────────────────────────────────────────────────────────────────


class BlockDims(NamedTuple):
    half_x: float
    half_y: float
    half_z: float


def _cube_dims() -> BlockDims:
    return BlockDims(half_x=CUBE_HALF, half_y=CUBE_HALF, half_z=CUBE_HALF)


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────


@register_env("PushBoundary", max_episode_steps=5_000)
class PushBoundaryEnv(BaseEnv):
    """
    Push-cube environment with a floating cylindrical gripper.

    Episode ends (fail) when the block escapes outside the boundary.
    There is no terminal success — the policy must keep the block inside.
    """

    SUPPORTED_ROBOTS = ["floating_gripper"]

    robot_init_qpos_noise: float = 0.00

    def __init__(
        self,
        *args,
        robot_uids: str = "floating_gripper",
        robot_init_qpos_noise: float = 0.00,
        target_xy: tuple | None = None,
        **kwargs,
    ) -> None:
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.block_dims: BlockDims = _cube_dims()
        self._target_xy = target_xy  # (x, y) or None — renders a green marker if set
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ── sim / sensor config ───────────────────────────────────────────────────

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**18,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    # ── agent ─────────────────────────────────────────────────────────────────

    def _load_agent(self, options):
        super()._load_agent(
            options,
            sapien.Pose(p=[BOUNDARY_CENTER_X, 0, GRIPPER_Z_FIXED]),
        )

    # ── scene ─────────────────────────────────────────────────────────────────

    def _load_scene(self, options: dict) -> None:
        self.table_scene = WhiteTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Main block
        self.block = self._build_cube()

        # Boundary strips (visual only)
        strip_mat = sapien.render.RenderMaterial(
            base_color=np.array([230, 80, 30, 255]) / 255
        )
        bx, by = BOUNDARY_CENTER_X, BOUNDARY_CENTER_Y
        hx, hy = BOUNDARY_HALF_X, BOUNDARY_HALF_Y
        t, sh = STRIP_THICKNESS, STRIP_HALF_H

        self.boundary_strips = []
        for name, cx, cy, hlx, hly in [
            ("bound_top", bx, by + hy, hx + t, t),
            ("bound_bot", bx, by - hy, hx + t, t),
            ("bound_left", bx - hx, by, t, hy),
            ("bound_right", bx + hx, by, t, hy),
        ]:
            bldr = self.scene.create_actor_builder()
            bldr.add_box_visual(half_size=[hlx, hly, sh], material=strip_mat)
            bldr.initial_pose = sapien.Pose(p=[cx, cy, sh])
            self.boundary_strips.append(bldr.build_kinematic(name=name))

        # ── optional target marker (green square on the floor) ────────────────
        if self._target_xy is not None:
            green = sapien.render.RenderMaterial(
                base_color=np.array([0.08, 0.85, 0.15, 0.90])
            )
            bldr = self.scene.create_actor_builder()
            bldr.add_box_visual(half_size=[0.033, 0.033, 0.0005], material=green)
            bldr.initial_pose = sapien.Pose(
                p=[float(self._target_xy[0]), float(self._target_xy[1]), 0.001]
            )
            bldr.build_static(name="target_marker")

    def _build_circle(self):
        mat = physx.PhysxMaterial(
            static_friction=1.0, dynamic_friction=0.8, restitution=0.0
        )
        blue = np.array([52, 120, 246, 255]) / 255
        bldr = self.scene.create_actor_builder()
        bldr._mass = 0.12

        # main body
        bldr.add_cylinder_collision(
            pose=sapien.Pose(
                p=[0.0, 0.0, CIRCLE_RADIUS + 1e-3], q=euler2quat(0, np.pi / 2, 0)
            ),
            radius=CIRCLE_RADIUS,
            half_length=CIRCLE_RADIUS,
            material=mat,
            density=200,
        )
        bldr.add_cylinder_visual(
            pose=sapien.Pose(
                p=[0.0, 0.0, CIRCLE_RADIUS + 1e-3], q=euler2quat(0, np.pi / 2, 0)
            ),
            radius=CIRCLE_RADIUS,
            half_length=CIRCLE_RADIUS,
            material=sapien.render.RenderMaterial(base_color=blue),
        )

        bldr.initial_pose = sapien.Pose(p=[0.0, 0.0, 0.0])
        return bldr.build(name="push_block")

    def _build_cube(self):
        mat = physx.PhysxMaterial(
            static_friction=1.5, dynamic_friction=1.2, restitution=0.0
        )
        blue = np.array([52, 120, 246, 255]) / 255
        bldr = self.scene.create_actor_builder()
        bldr.add_box_collision(
            half_size=[CUBE_HALF, CUBE_HALF, CUBE_HALF], material=mat, density=200
        )
        bldr.add_box_visual(
            half_size=[CUBE_HALF, CUBE_HALF, CUBE_HALF],
            material=sapien.render.RenderMaterial(
                base_color=blue, metallic=0.0, roughness=0.5
            ),
        )
        bldr.initial_pose = sapien.Pose(p=[0.0, 0.0, CUBE_Z_SPAWN])
        return bldr.build(name="push_block")

    # ── episode init ──────────────────────────────────────────────────────────

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Boundary strips
            bx, by = BOUNDARY_CENTER_X, BOUNDARY_CENTER_Y
            sh = STRIP_HALF_H
            hx, hy, t = BOUNDARY_HALF_X, BOUNDARY_HALF_Y, STRIP_THICKNESS
            for strip, pos in zip(
                self.boundary_strips,
                [
                    [bx, by + hy, sh],
                    [bx, by - hy, sh],
                    [bx - hx, by, sh],
                    [bx + hx, by, sh],
                ],
            ):
                p = torch.tensor([pos], dtype=torch.float32, device=self.device).expand(
                    b, -1
                )
                strip.set_pose(Pose.create_from_pq(p=p))

            # Block — random spawn inside boundary
            margin = 0.02 + CUBE_HALF
            spawn_hx = max(BOUNDARY_HALF_X - margin, 0.01)
            spawn_hy = max(BOUNDARY_HALF_Y - margin, 0.01)

            bx_r = (
                torch.rand(b, device=self.device) * 2 - 1
            ) * spawn_hx + BOUNDARY_CENTER_X
            by_r = (
                torch.rand(b, device=self.device) * 2 - 1
            ) * spawn_hy + BOUNDARY_CENTER_Y
            bz_r = torch.full((b,), CUBE_Z_SPAWN, device=self.device)

            yaw = torch.rand(b, device=self.device) * 2 * np.pi
            c, s = (yaw / 2).cos(), (yaw / 2).sin()
            bq = torch.stack([c, torch.zeros_like(c), torch.zeros_like(c), s], dim=1)
            self.block.set_pose(
                Pose.create_from_pq(p=torch.stack([bx_r, by_r, bz_r], dim=1), q=bq)
            )

    # ── evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self) -> dict:
        bpos = self.block.pose.p
        out_x = (bpos[:, 0] < BOUNDARY_CENTER_X - BOUNDARY_HALF_X - OUT_MARGIN) | (
            bpos[:, 0] > BOUNDARY_CENTER_X + BOUNDARY_HALF_X + OUT_MARGIN
        )
        out_y = (bpos[:, 1] < BOUNDARY_CENTER_Y - BOUNDARY_HALF_Y - OUT_MARGIN) | (
            bpos[:, 1] > BOUNDARY_CENTER_Y + BOUNDARY_HALF_Y + OUT_MARGIN
        )
        false_t = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return {"success": false_t, "fail": out_x | out_y}

    # ── observations ──────────────────────────────────────────────────────────

    def _get_obs_extra(self, info: dict) -> dict:
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                block_pose=self.block.pose.raw_pose,
                boundary_center=torch.tensor(
                    [BOUNDARY_CENTER_X, BOUNDARY_CENTER_Y],
                    dtype=torch.float32,
                    device=self.device,
                )
                .unsqueeze(0)
                .expand(self.num_envs, -1),
            )
        return obs

    # ── reward ────────────────────────────────────────────────────────────────

    def compute_dense_reward(self, obs: Any, action: Array, info: dict) -> torch.Tensor:
        tcp_dist = torch.linalg.norm(self.block.pose.p - self.agent.tcp.pose.p, dim=1)
        approach = (1 - torch.tanh(5 * tcp_dist)) * 0.5

        centre = torch.tensor(
            [BOUNDARY_CENTER_X, BOUNDARY_CENTER_Y, CUBE_Z_SPAWN],
            dtype=torch.float32,
            device=self.device,
        )
        block_dist = torch.linalg.norm(self.block.pose.p - centre.unsqueeze(0), dim=1)
        keep_in = (1 - torch.tanh(5 * block_dist)) * 0.5

        reward = approach + keep_in
        reward[info["fail"]] = 0.0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: Array, info: dict
    ) -> torch.Tensor:
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 1.0
