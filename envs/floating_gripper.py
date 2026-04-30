"""
envs/floating_gripper.py

A free-floating cylindrical gripper agent for ManiSkill.
Uses two prismatic joints (joint_x, joint_y) for 2D Cartesian control.
The robot geometry is defined in the URDF at `urdf_path`.

Usage:
    import envs  # triggers registration
    env = gym.make("PushBoundary", robot_uids="floating_gripper")
"""

from __future__ import annotations

import numpy as np
import sapien
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent
from transforms3d.euler import euler2quat

from envs.push_boundary import BOUNDARY_CENTER_X, GRIPPER_Z_FIXED


@register_agent()
class FloatingGripper(BaseAgent):
    """
    A single cylindrical body that floats over the table via two prismatic joints.
    Action space: [dx, dy] in metres (delta position).
    """

    uid = "floating_gripper"
    urdf_path = "floating_gripper.urdf"

    urdf_config: dict = {}

    keyframes = {
        "default": Keyframe(
            qpos=np.array([0.0, 0.0]),
            pose=sapien.Pose(p=[BOUNDARY_CENTER_X, 0.0, GRIPPER_Z_FIXED]),
        )
    }

    @property
    def _controller_configs(self):
        return {
            "floating_vel": PDJointPosControllerConfig(
                joint_names=["joint_x", "joint_y"],
                lower=-0.1,
                upper=0.1,
                stiffness=500.0,
                damping=50.0,
                force_limit=150.0,
                use_delta=True,
                normalize_action=False,
            )
        }

    @property
    def tcp(self):
        """Tool-centre-point: the gripper link itself."""
        return self.robot.links_map["gripper"]