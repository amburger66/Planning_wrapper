from typing import Optional, Any, List, Tuple
from omegaconf import DictConfig
import time
import numpy as np
from random import random
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce
import wandb
from PIL import Image
from pathlib import Path

from .df_base import DiffusionForcingBase
from utils.logging_utils import (
    make_trajectory_images,
    get_random_start_goal,
    make_convergence_animation,
    make_mpc_animation,
)
from .tree_node_fast import TreeNode

OGBENCH_ENVS = [
    "pointmaze-medium-v0",
    "pointmaze-large-v0",
    "pointmaze-giant-v0",
    "pointmaze-teleport-v0",
    "antmaze-medium-v0",
    "antmaze-large-v0",
    "antmaze-giant-v0",
    "antmaze-teleport-v0",
]


class DiffusionForcingPlanning(DiffusionForcingBase):
    def __init__(self, cfg: DictConfig):
        self.env_id = cfg.env_id
        self.dataset = cfg.dataset
        self.action_dim = len(cfg.action_mean)
        self.observation_dim = len(cfg.observation_mean)
        self.use_reward = cfg.use_reward
        self.unstacked_dim = (
            self.observation_dim + self.action_dim + int(self.use_reward)
        )
        cfg.x_shape = (self.unstacked_dim,)
        self.episode_len = cfg.episode_len
        self.n_tokens = self.episode_len // cfg.frame_stack + 1
        self.gamma = cfg.gamma
        self.reward_mean = cfg.reward_mean
        self.reward_std = cfg.reward_std
        self.observation_mean = np.array(cfg.observation_mean[: self.observation_dim])
        self.observation_std = np.array(cfg.observation_std[: self.observation_dim])
        self.action_mean = np.array(cfg.action_mean[: self.action_dim])
        self.action_std = np.array(cfg.action_std[: self.action_dim])
        self.open_loop_horizon = cfg.open_loop_horizon
        self.padding_mode = cfg.padding_mode
        self.interaction_seed = cfg.interaction_seed
        self.use_random_goals_for_interaction = cfg.use_random_goals_for_interaction
        self.task_id = cfg.task_id
        self.dql_model = cfg.dql_model
        self.val_max_steps = cfg.val_max_steps
        self.mctd = cfg.mctd
        self.mctd_guidance_scales = cfg.mctd_guidance_scales
        self.mctd_max_search_num = cfg.mctd_max_search_num
        self.mctd_num_denoising_steps = cfg.mctd_num_denoising_steps
        self.mctd_skip_level_steps = cfg.mctd_skip_level_steps
        self.jump = cfg.jump
        self.time_limit = cfg.time_limit
        self.parallel_search_num = cfg.parallel_search_num
        self.virtual_visit_weight = cfg.virtual_visit_weight
        self.warp_threshold = cfg.warp_threshold
        self.goal_threshold = cfg.get("goal_threshold", 2.0)
        self.leaf_parallelization = cfg.leaf_parallelization
        self.parallel_multiple_visits = cfg.parallel_multiple_visits
        self.early_stopping_condition = cfg.early_stopping_condition
        self.num_tries_for_bad_plans = cfg.num_tries_for_bad_plans
        self.mctd_pairwise_divergence_threshold = cfg.get(
            "mctd_pairwise_divergence_threshold", None
        )
        self.sub_goal_interval = cfg.sub_goal_interval
        self.viz_plans = cfg.viz_plans
        # MCTD batch rollout visualization (single GIF overlay).
        self.viz_mctd_batch_plans = cfg.get("viz_mctd_batch_plans", False)
        self.viz_mctd_batch_plans_alpha = float(
            cfg.get("viz_mctd_batch_plans_alpha", 0.15)
        )
        self.viz_mctd_batch_plans_max_branches = cfg.get(
            "viz_mctd_batch_plans_max_branches", None
        )
        self.viz_mctd_batch_plans_every_n_search = int(
            cfg.get("viz_mctd_batch_plans_every_n_search", 1)
        )
        self.viz_mctd_batch_plans_out_dir = Path(
            cfg.get("viz_mctd_batch_plans_out_dir", "viz")
        )
        self._pushboundary_2d_viz_block_shape = cfg.get("viz_block_shape", "square")
        self.no_sim_env = cfg.get("no_sim_env", False)
        super().__init__(cfg)
        self.plot_end_points = cfg.plot_start_goal and self.guidance_scale != 0

    def _pushboundary_2d_obstacles_xy(
        self, states_2d: np.ndarray
    ) -> List[Tuple[float, float]]:
        # Assuming states are [tcp_x, tcp_y, block_x, block_y, cos, sin, obs1_x, obs1_y...]
        # Everything from index 6 onwards are obstacle pairs.
        num_obstacles = (states_2d.shape[-1] - 6) // 2
        obstacles_xy: List[Tuple[float, float]] = []
        for i in range(num_obstacles):
            ox = float(states_2d[0, 6 + 2 * i])
            oy = float(states_2d[0, 7 + 2 * i])
            obstacles_xy.append((ox, oy))
        return obstacles_xy

    def _pushboundary_2d_limits(
        self,
        *,
        tcp_xy: np.ndarray,
        block_xy: np.ndarray,
        obstacles_xy: List[Tuple[float, float]],
        start_marker: Optional[np.ndarray],
        goal_marker: Optional[np.ndarray],
        pad: float,
        obstacle_radius: float,
    ) -> Tuple[float, float, float, float]:
        x_min = float(min(tcp_xy[:, 0].min(), block_xy[:, 0].min()) - pad)
        x_max = float(max(tcp_xy[:, 0].max(), block_xy[:, 0].max()) + pad)
        y_min = float(min(tcp_xy[:, 1].min(), block_xy[:, 1].min()) - pad)
        y_max = float(max(tcp_xy[:, 1].max(), block_xy[:, 1].max()) + pad)

        if start_marker is not None:
            x_min = min(x_min, float(start_marker[0]) - pad)
            x_max = max(x_max, float(start_marker[0]) + pad)
            y_min = min(y_min, float(start_marker[1]) - pad)
            y_max = max(y_max, float(start_marker[1]) + pad)
        if goal_marker is not None:
            x_min = min(x_min, float(goal_marker[0]) - pad)
            x_max = max(x_max, float(goal_marker[0]) + pad)
            y_min = min(y_min, float(goal_marker[1]) - pad)
            y_max = max(y_max, float(goal_marker[1]) + pad)

        for ox, oy in obstacles_xy:
            x_min = min(x_min, ox - obstacle_radius - pad)
            x_max = max(x_max, ox + obstacle_radius + pad)
            y_min = min(y_min, oy - obstacle_radius - pad)
            y_max = max(y_max, oy + obstacle_radius + pad)

        return x_min, x_max, y_min, y_max

    def _pushboundary_viz_fps(
        self,
        *,
        T: int,
        fps: Optional[float],
        min_fps: float,
        max_fps: float,
        min_duration_sec: float,
        max_gif_duration_sec: float,
    ) -> float:
        if fps is not None:
            return float(fps)
        target_duration = np.clip(T / 24.0, min_duration_sec, max_gif_duration_sec)
        return float(np.clip(T / target_duration, min_fps, max_fps))

    def _pushboundary_viz_step_indices(
        self, *, n_include: int, num_frames: Optional[int]
    ) -> List[int]:
        if num_frames is None or num_frames >= n_include:
            return list(range(n_include))
        return np.linspace(0, n_include - 1, num=num_frames, dtype=int).tolist()

    def _pushboundary_viz_out_dir(
        self, *, namespace: str, gif_out_dir: Optional[Path]
    ) -> Path:
        if gif_out_dir is not None:
            return Path(gif_out_dir)
        root_dir = Path(getattr(self.trainer, "default_root_dir", Path.cwd()))
        return (
            root_dir
            / "visualizations"
            / "pushboundary_2d"
            / namespace
            / f"global_step_{getattr(self, 'global_step', 0):07d}"
        )

    def _pushboundary_viz_save_frames(
        self,
        *,
        frames_img: List[np.ndarray],
        out_dir: Path,
        sample_idx: int,
        suffix: str,
        fps: float,
        output_format: str,
    ) -> Path:
        import os as _os

        _prev = _os.umask(0)
        try:
            out_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
        finally:
            _os.umask(_prev)

        ext = "mp4" if output_format == "mp4" else "gif"
        out_path = out_dir / f"sample_{sample_idx}_{suffix}.{ext}"

        duration_ms = int(1000.0 / fps)
        pil_frames = [Image.fromarray(f) for f in frames_img]

        # mp4 path currently falls back to GIF for parity with existing code.
        if output_format == "mp4":
            out_path_gif = out_path.with_suffix(".gif")
            pil_frames[0].save(
                str(out_path_gif),
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0,
            )
            return out_path_gif

        pil_frames[0].save(
            str(out_path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )
        return out_path

    def _log_or_save_pushboundary_pcd_gif(
        self,
        namespace: str,
        states: np.ndarray,
        actions: np.ndarray,
        sample_idx: int,
        **kwargs,
    ) -> None:
        """
        Render a low-dim PushBoundary (18D state / 9D action) trajectory as a birds-eye GIF.

        Delegates to _log_or_save_pushboundary_2d_gif with the correct index mapping for
        the 18D state layout:
          dims 0-1   gripper XY
          dims 9-10  block XY
          dims 12-17 block 6D rotation (first two rows of rotation matrix)
        Block rotation (yaw) is extracted and drawn as a rotated polygon.
        `actions` is accepted for API compatibility but is not used.
        """
        self._log_or_save_pushboundary_2d_gif(
            namespace=namespace,
            states=states,
            sample_idx=sample_idx,
            tcp_xy_indices=(0, 1),
            block_xy_indices=(9, 10),
            block_rot6d_indices=(12, 13, 14, 15, 16, 17),
            block_shape=self._pushboundary_2d_viz_block_shape,
        )

    def _log_or_save_pushboundary_2d_gif(
        self,
        namespace: str,
        states: np.ndarray,
        sample_idx: int,
        suffix: Optional[str] = None,
        *,
        tcp_xy_indices: tuple = (0, 1),
        block_xy_indices: tuple = (2, 3),
        block_yaw_indices: Optional[tuple] = (4, 5),  # Expect (cos, sin) indices
        num_frames: Optional[int] = None,
        dpi: int = 100,
        fps: Optional[float] = None,
        min_fps: float = 10.0,
        max_fps: float = 60.0,
        min_duration_sec: float = 4.0,
        max_gif_duration_sec: float = 20.0,
        gif_out_dir: Optional[Path] = None,
        start_marker: Optional[np.ndarray] = None,
        goal_marker: Optional[np.ndarray] = None,
        output_format: str = "gif",
        block_shape: str = "square",
    ) -> None:
        """
        Render a 2D PushBoundary trajectory as a GIF or MP4.
        Draws robot TCP (circle), block (rectangle/circle), and any obstacles dynamically.

        tcp_xy_indices: which two state dims are gripper x,y (default (0,1)).
        block_xy_indices: which two state dims are block x,y (default (2,3)).
        block_yaw_indices: if provided, a tuple of state dims holding (cos, sin) of the block yaw.
        """
        print("Rendering pushboundary 2d visualization...")
        min_dim = max(max(tcp_xy_indices), max(block_xy_indices)) + 1
        if states.ndim != 2 or states.shape[-1] < min_dim:
            return

        tcp_xy = states[:, list(tcp_xy_indices)]
        block_xy = states[:, list(block_xy_indices)]

        # --- YAW LOGIC ---
        if block_yaw_indices is not None:
            idx = block_yaw_indices
            block_yaws = np.arctan2(states[:, idx[1]], states[:, idx[0]])
        else:
            block_yaws = None

        obstacles_xy = self._pushboundary_2d_obstacles_xy(states)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, Polygon, Rectangle
        from PIL import Image

        # Geometry
        BLOCK_HALF = 0.025
        CIRCLE_RADIUS = 0.025
        STICK_RADIUS = 0.01
        OBSTACLE_RADIUS = 0.018

        pad = 0.02
        x_min, x_max, y_min, y_max = self._pushboundary_2d_limits(
            tcp_xy=tcp_xy,
            block_xy=block_xy,
            obstacles_xy=obstacles_xy,
            start_marker=start_marker,
            goal_marker=goal_marker,
            pad=pad,
            obstacle_radius=OBSTACLE_RADIUS,
        )

        T = states.shape[0]
        n_include = T

        fps = self._pushboundary_viz_fps(
            T=T,
            fps=fps,
            min_fps=min_fps,
            max_fps=max_fps,
            min_duration_sec=min_duration_sec,
            max_gif_duration_sec=max_gif_duration_sec,
        )
        step_indices = self._pushboundary_viz_step_indices(
            n_include=n_include, num_frames=num_frames
        )

        frames_img = []
        for s in step_indices:
            end = int(s) + 1
            fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)

            # --- Plot static obstacles ---
            for ox, oy in obstacles_xy:
                ax.add_patch(
                    Circle(
                        (ox, oy),
                        radius=OBSTACLE_RADIUS,
                        facecolor="dimgray",
                        edgecolor="black",
                        linewidth=1,
                        alpha=0.8,
                        zorder=2,
                    )
                )

            tcp_now = tcp_xy[end - 1]
            block_now = block_xy[end - 1]
            ax.add_patch(
                Circle(
                    (tcp_now[0], tcp_now[1]),
                    radius=STICK_RADIUS,
                    facecolor="#e63946",
                    edgecolor="#c1121f",
                    linewidth=1,
                    zorder=4,
                )
            )
            if block_shape == "circle":
                ax.add_patch(
                    Circle(
                        (block_now[0], block_now[1]),
                        radius=CIRCLE_RADIUS,
                        facecolor="#0066ff",
                        edgecolor="#0047ab",
                        linewidth=1,
                        zorder=3,
                    )
                )
            elif block_yaws is not None:
                yaw = block_yaws[end - 1]
                c, s_val = np.cos(yaw), np.sin(yaw)
                h = BLOCK_HALF
                local = [(-h, -h), (h, -h), (h, h), (-h, h)]
                corners = [
                    (
                        block_now[0] + c * lx - s_val * ly,
                        block_now[1] + s_val * lx + c * ly,
                    )
                    for lx, ly in local
                ]
                ax.add_patch(
                    Polygon(
                        corners,
                        closed=True,
                        facecolor="#0066ff",
                        edgecolor="#0047ab",
                        linewidth=1,
                        zorder=3,
                    )
                )
            else:
                ax.add_patch(
                    Rectangle(
                        (block_now[0] - BLOCK_HALF, block_now[1] - BLOCK_HALF),
                        width=2 * BLOCK_HALF,
                        height=2 * BLOCK_HALF,
                        facecolor="#0066ff",
                        edgecolor="#0047ab",
                        linewidth=1,
                        zorder=3,
                    )
                )
            if start_marker is not None:
                ax.plot(
                    start_marker[0],
                    start_marker[1],
                    marker="+",
                    color="#e63946",
                    markersize=14,
                    markeredgewidth=2,
                    zorder=5,
                )
            if goal_marker is not None:
                ax.plot(
                    goal_marker[0],
                    goal_marker[1],
                    marker="+",
                    color="#2d6a4f",
                    markersize=14,
                    markeredgewidth=2,
                    zorder=5,
                )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal")

            block_legend_marker = "o" if block_shape == "circle" else "s"
            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#e63946",
                    markersize=8,
                    label="tcp",
                ),
                Line2D(
                    [0],
                    [0],
                    marker=block_legend_marker,
                    color="w",
                    markerfacecolor="#0066ff",
                    markersize=8,
                    label="block",
                ),
            ]

            # --- Add Obstacle to Legend dynamically ---
            if len(obstacles_xy) > 0:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor="dimgray",
                        markersize=8,
                        label="obstacle",
                    )
                )

            if start_marker is not None:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker="+",
                        color="#e63946",
                        markersize=10,
                        label="start",
                    )
                )
            if goal_marker is not None:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker="+",
                        color="#2d6a4f",
                        markersize=10,
                        label="goal",
                    )
                )

            ax.legend(handles=legend_elements, loc="upper right", fontsize=7)
            ax.set_title(f"step {s}", fontsize=8)
            ax.set_xlabel("X", fontsize=7)
            ax.set_ylabel("Y", fontsize=7)
            fig.tight_layout(pad=0.3)
            fig.canvas.draw()
            w, h_px = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            img = buf.reshape(h_px, w, 4).copy()
            frames_img.append(img)
            plt.close(fig)

        out_dir = self._pushboundary_viz_out_dir(
            namespace=namespace, gif_out_dir=gif_out_dir
        )
        if suffix is None:
            suffix = "trajectory"
        self._pushboundary_viz_save_frames(
            frames_img=frames_img,
            out_dir=out_dir,
            sample_idx=sample_idx,
            suffix=suffix,
            fps=fps,
            output_format=output_format,
        )

        if self.logger is not None:
            try:
                for j in {0, len(frames_img) // 2, len(frames_img) - 1}:
                    if j < len(frames_img):
                        self.log_image(
                            f"training_visualization/{namespace}/sample_{sample_idx}/frame_{j}",
                            Image.fromarray(frames_img[j]),
                        )
            except Exception:
                pass

    def _log_or_save_pushboundary_2d_batch_gif(
        self,
        namespace: str,
        states: np.ndarray,
        sample_idx: int,
        suffix: Optional[str] = None,
        *,
        tcp_xy_indices: tuple = (0, 1),
        block_xy_indices: tuple = (2, 3),
        block_yaw_indices: Optional[tuple] = (4, 5),  # Expect (cos, sin) indices
        num_frames: Optional[int] = None,
        dpi: int = 100,
        fps: Optional[float] = None,
        min_fps: float = 10.0,
        max_fps: float = 60.0,
        min_duration_sec: float = 4.0,
        max_gif_duration_sec: float = 20.0,
        gif_out_dir: Optional[Path] = None,
        start_marker: Optional[np.ndarray] = None,
        goal_marker: Optional[np.ndarray] = None,
        output_format: str = "gif",
        block_shape: str = "square",
        alpha: float = 0.15,
        max_branches: Optional[int] = None,
        t_div: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        """
        Render a 2D PushBoundary *batch* of trajectories as one overlaid GIF/MP4.

        `states`: (T, B, D) array in physical units (already unnormalized), where the first
        dims follow the same layout as `_log_or_save_pushboundary_2d_gif`.
        """
        print("Rendering pushboundary 2d batch visualization...")
        min_dim = max(max(tcp_xy_indices), max(block_xy_indices)) + 1
        if states.ndim != 3 or states.shape[-1] < min_dim:
            return

        T, B, _ = states.shape
        if T < 1 or B < 1:
            return

        # Obstacles are static; infer them from branch 0.
        obstacles_xy = self._pushboundary_2d_obstacles_xy(states[:, 0, :])

        branch_ids = np.arange(B, dtype=int)
        if max_branches is not None and B > int(max_branches):
            rng = np.random.default_rng(int(seed))
            branch_ids = rng.choice(branch_ids, size=int(max_branches), replace=False)
            branch_ids = np.sort(branch_ids)

        tcp_xy = states[:, branch_ids][:, :, list(tcp_xy_indices)]  # (T, B', 2)
        block_xy = states[:, branch_ids][:, :, list(block_xy_indices)]  # (T, B', 2)

        if block_yaw_indices is not None:
            idx = block_yaw_indices
            block_yaws = np.arctan2(
                states[:, branch_ids, idx[1]], states[:, branch_ids, idx[0]]
            )  # (T, B')
        else:
            block_yaws = None

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, Polygon, Rectangle

        # Geometry (match single-trajectory viz).
        BLOCK_HALF = 0.025
        CIRCLE_RADIUS = 0.025
        STICK_RADIUS = 0.01
        OBSTACLE_RADIUS = 0.018

        pad = 0.02
        x_min, x_max, y_min, y_max = self._pushboundary_2d_limits(
            tcp_xy=tcp_xy.reshape(-1, 2),
            block_xy=block_xy.reshape(-1, 2),
            obstacles_xy=obstacles_xy,
            start_marker=start_marker,
            goal_marker=goal_marker,
            pad=pad,
            obstacle_radius=OBSTACLE_RADIUS,
        )

        fps = self._pushboundary_viz_fps(
            T=T,
            fps=fps,
            min_fps=min_fps,
            max_fps=max_fps,
            min_duration_sec=min_duration_sec,
            max_gif_duration_sec=max_gif_duration_sec,
        )
        step_indices = self._pushboundary_viz_step_indices(
            n_include=T, num_frames=num_frames
        )

        alpha = float(np.clip(alpha, 0.01, 1.0))

        frames_img: List[np.ndarray] = []
        for s in step_indices:
            fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)

            for ox, oy in obstacles_xy:
                ax.add_patch(
                    Circle(
                        (ox, oy),
                        radius=OBSTACLE_RADIUS,
                        facecolor="dimgray",
                        edgecolor="black",
                        linewidth=1,
                        alpha=0.8,
                        zorder=2,
                    )
                )

            for j in range(tcp_xy.shape[1]):
                tcp_now = tcp_xy[s, j]
                block_now = block_xy[s, j]
                ax.add_patch(
                    Circle(
                        (tcp_now[0], tcp_now[1]),
                        radius=STICK_RADIUS,
                        facecolor="#e63946",
                        edgecolor="#c1121f",
                        linewidth=0.8,
                        alpha=alpha,
                        zorder=4,
                    )
                )

                if block_shape == "circle":
                    ax.add_patch(
                        Circle(
                            (block_now[0], block_now[1]),
                            radius=CIRCLE_RADIUS,
                            facecolor="#0066ff",
                            edgecolor="#0047ab",
                            linewidth=0.8,
                            alpha=alpha,
                            zorder=3,
                        )
                    )
                elif block_yaws is not None:
                    yaw = float(block_yaws[s, j])
                    c, s_val = np.cos(yaw), np.sin(yaw)
                    h = BLOCK_HALF
                    local = [(-h, -h), (h, -h), (h, h), (-h, h)]
                    corners = [
                        (
                            block_now[0] + c * lx - s_val * ly,
                            block_now[1] + s_val * lx + c * ly,
                        )
                        for lx, ly in local
                    ]
                    ax.add_patch(
                        Polygon(
                            corners,
                            closed=True,
                            facecolor="#0066ff",
                            edgecolor="#0047ab",
                            linewidth=0.8,
                            alpha=alpha,
                            zorder=3,
                        )
                    )
                else:
                    ax.add_patch(
                        Rectangle(
                            (block_now[0] - BLOCK_HALF, block_now[1] - BLOCK_HALF),
                            width=2 * BLOCK_HALF,
                            height=2 * BLOCK_HALF,
                            facecolor="#0066ff",
                            edgecolor="#0047ab",
                            linewidth=0.8,
                            alpha=alpha,
                            zorder=3,
                        )
                    )

            if start_marker is not None:
                ax.plot(
                    start_marker[0],
                    start_marker[1],
                    marker="+",
                    color="#e63946",
                    markersize=14,
                    markeredgewidth=2,
                    zorder=5,
                )
            if goal_marker is not None:
                ax.plot(
                    goal_marker[0],
                    goal_marker[1],
                    marker="+",
                    color="#2d6a4f",
                    markersize=14,
                    markeredgewidth=2,
                    zorder=5,
                )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal")

            block_legend_marker = "o" if block_shape == "circle" else "s"
            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#e63946",
                    markersize=8,
                    label="tcp (overlay)",
                ),
                Line2D(
                    [0],
                    [0],
                    marker=block_legend_marker,
                    color="w",
                    markerfacecolor="#0066ff",
                    markersize=8,
                    label="block (overlay)",
                ),
            ]

            if len(obstacles_xy) > 0:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor="dimgray",
                        markersize=8,
                        label="obstacle",
                    )
                )
            if start_marker is not None:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker="+",
                        color="#e63946",
                        markersize=10,
                        label="start",
                    )
                )
            if goal_marker is not None:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker="+",
                        color="#2d6a4f",
                        markersize=10,
                        label="goal",
                    )
                )
            ax.legend(handles=legend_elements, loc="upper right", fontsize=7)

            title = f"step {s} (B={tcp_xy.shape[1]})"
            if t_div is not None:
                title += f", t_div={t_div}"
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("X", fontsize=7)
            ax.set_ylabel("Y", fontsize=7)

            fig.tight_layout(pad=0.3)
            fig.canvas.draw()
            w, h_px = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            img = buf.reshape(h_px, w, 4).copy()
            frames_img.append(img)
            plt.close(fig)

        out_dir = self._pushboundary_viz_out_dir(
            namespace=namespace, gif_out_dir=gif_out_dir
        )
        if suffix is None:
            suffix = "batch_overlay"
        self._pushboundary_viz_save_frames(
            frames_img=frames_img,
            out_dir=out_dir,
            sample_idx=sample_idx,
            suffix=suffix,
            fps=fps,
            output_format=output_format,
        )

    def _build_model(self):
        mean = list(self.observation_mean) + list(self.action_mean)
        std = list(self.observation_std) + list(self.action_std)
        if self.use_reward:
            mean += [self.reward_mean]
            std += [self.reward_std]
        self.cfg.data_mean = np.array(mean).tolist()
        self.cfg.data_std = np.array(std).tolist()
        super()._build_model()

    def _preprocess_batch(self, batch):
        observations, actions, rewards, nonterminals = batch
        batch_size, n_frames = observations.shape[:2]

        observations = observations[..., : self.observation_dim]
        actions = actions[..., : self.action_dim]

        if (n_frames - 1) % self.frame_stack != 0:
            raise ValueError(
                "Number of frames - 1 must be divisible by frame stack size"
            )

        nonterminals = torch.cat(
            [
                torch.ones_like(nonterminals[:, : self.frame_stack]),
                nonterminals[:, :-1],
            ],
            dim=1,
        )
        nonterminals = nonterminals.bool().permute(1, 0)
        masks = torch.cumprod(nonterminals, dim=0).contiguous()
        # masks = torch.cat([masks[:-self.frame_stack:self.jump], masks[-self.frame_stack:]], dim=0)

        rewards = rewards[:, :-1, None]
        actions = actions[:, :-1]
        init_obs, observations = torch.split(observations, [1, n_frames - 1], dim=1)
        bundles = self._normalize_x(
            self.make_bundle(observations, actions, rewards)
        )  # (b t c)
        init_bundle = self._normalize_x(self.make_bundle(init_obs[:, 0]))  # (b c)
        init_bundle[:, self.observation_dim :] = (
            0  # zero out actions and rewards after normalization
        )
        init_bundle = self.pad_init(init_bundle, batch_first=True)  # (b t c)
        bundles = torch.cat([init_bundle, bundles], dim=1)
        bundles = rearrange(bundles, "b (t fs) ... -> t b fs ...", fs=self.frame_stack)
        bundles = bundles.flatten(2, 3).contiguous()

        if self.cfg.external_cond_dim:
            raise ValueError("external_cond_dim not needed in planning")
        conditions = None
        # bundles = bundles[::self.jump]
        return bundles, conditions, masks

    def training_step(self, batch, batch_idx):
        xs, conditions, masks = self._preprocess_batch(batch)

        n_tokens, batch_size = xs.shape[:2]

        weights = masks.float()
        if not self.causal:
            # manually mask out entries to train for varying length
            random_terminal = torch.randint(
                2, n_tokens + 1, (batch_size,), device=self.device
            )
            random_terminal = nn.functional.one_hot(random_terminal, n_tokens + 1)[
                :, :n_tokens
            ].bool()
            random_terminal = repeat(
                random_terminal, "b t -> (t fs) b", fs=self.frame_stack
            )
            nonterminal_causal = torch.cumprod(~random_terminal, dim=0)
            weights *= torch.clip(nonterminal_causal.float(), min=0.05)
            masks *= nonterminal_causal.bool()

        xs_pred, loss = self.diffusion_model(
            xs, conditions, noise_levels=self._generate_noise_levels(xs, masks=masks)
        )

        loss = self.reweight_loss(loss, weights)

        if batch_idx % 100 == 0:
            self.log(
                "training/loss",
                loss,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
                prog_bar=True,
            )

        xs = self._unstack_and_unnormalize(xs)[self.frame_stack - 1 :]
        xs_pred = self._unstack_and_unnormalize(xs_pred)[self.frame_stack - 1 :]

        # PushBoundary visualization (always saved to disk; wandb is best-effort).
        if self.global_step % 10000 == 0:
            o, a, _ = self.split_bundle(xs_pred)
            o_np = o.detach().cpu().numpy()  # (t, b, obs_dim)
            a_np = a.detach().cpu().numpy()  # (t, b, action_dim)

            batch_size = o_np.shape[1]
            samples = min(1, batch_size)  # keep it lightweight for training
            for sample_idx in range(samples):
                # last observation is dummy => drop it for both states and actions
                states = o_np[:-1, sample_idx, :]
                actions = a_np[:-1, sample_idx, :]
                if self.observation_dim >= 18:
                    self._log_or_save_pushboundary_pcd_gif(
                        namespace="training",
                        states=states,
                        actions=actions,
                        sample_idx=sample_idx,
                    )
                elif self.observation_dim >= 4:
                    self._log_or_save_pushboundary_2d_gif(
                        namespace="training",
                        states=states,
                        sample_idx=sample_idx,
                        block_shape=self._pushboundary_2d_viz_block_shape,
                    )

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation"):
        xs, conditions, _ = self._preprocess_batch(batch)
        _, batch_size, *_ = xs.shape
        if self.guidance_scale == 0:
            namespace += "_no_guidance_random_walk"
        horizon = self.episode_len
        if self.no_sim_env:
            # Extract normalized start (first frame) and goal (last frame) from the batch
            # so interact() doesn't need an environment to source them.
            start_obs = xs[0, :, : self.observation_dim]  # (b, obs_dim), normalized
            goal_obs = xs[-1, :, : self.observation_dim]  # (b, obs_dim), normalized
            self.interact(
                batch_size,
                conditions,
                namespace,
                start_obs=start_obs,
                goal_obs=goal_obs,
            )
        else:
            self.interact(
                batch_size, conditions, namespace
            )  # interact if environment is installed

    def plan(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        horizon: int,
        conditions: Optional[Any] = None,
        guidance_scale: int = None,
        noise_level: Optional[torch.Tensor] = None,
        plan: Optional[torch.Tensor] = None,
    ):
        # start and goal are numpy arrays of shape (b, obs_dim)
        # start and goal are assumed to be normalized
        # returns plan history of (m, t, b, c), where the last dim of m is the fully diffused plan

        batch_size = start.shape[0]
        start = start.float()
        goal = goal.float()

        start = self.make_bundle(start)
        goal = self.make_bundle(goal)

        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        def goal_guidance(x):
            # x is a tensor of shape [t b (fs c)]
            pred = rearrange(x, "t b (fs c) -> (t fs) b c", fs=self.frame_stack)
            h_padded = (
                pred.shape[0] - self.frame_stack
            )  # include padding when horizon % frame_stack != 0

            if not self.use_reward:
                # sparse / no reward setting, guide with goal like diffuser
                target = torch.stack([start] * self.frame_stack + [goal] * (h_padded))

                # Build a boolean mask outside the autograd graph so no nan_to_num
                # appears in the gradient path. nan_to_num backward computes
                # grad_out * isfinite(input), and inf * 0 = nan in IEEE 754.
                valid_mask = ~torch.isnan(target)  # (t fs) b c, no gradient
                target_safe = target.detach().nan_to_num(0.0)  # NaN → 0, no gradient
                dist = (pred - target_safe) ** 2 * valid_mask  # (t fs) b c

                # guidance weight for observation and action
                weight = np.array(
                    [20]
                    * (self.frame_stack)  # conditoning (aka reconstruction guidance)
                    + [
                        1 for _ in range(horizon)
                    ]  # try to reach the goal at any horizon
                    # + [0 for _ in range(horizon-1)] + [1]  # Diffuer guidance
                    + [0]
                    * (
                        h_padded - horizon
                    )  # don't guide padded entries due to horizon % frame_stack != 0
                )
                # mathematically, one may also try multiplying weight by sqrt(alpha_cum)
                # this means you put higher weight to less noisy terms
                # which might be better but we haven't tried yet
                weight = torch.from_numpy(weight).float().to(self.device)

                dist_o, dist_a, _ = self.split_bundle(
                    dist
                )  # guidance observation and action with separate weights
                dist_a = torch.sum(dist_a, -1, keepdim=True).sqrt()
                # Sum over all obs dims: NaN goal dims are already zeroed by nan_to_num,
                # so only non-NaN dims contribute. eps prevents inf gradient from sqrt at 0.
                dist_o = (dist_o.sum(-1, keepdim=True) + 1e-8).sqrt()
                dist_o = torch.tanh(
                    dist_o / 2
                )  # similar to the "squashed gaussian" in RL, squash to (-1, 1)
                dist = dist_o
                weight = repeat(weight, "t -> t c", c=dist.shape[-1])
                weight[self.frame_stack :, 1:] = 8
                weight[: self.frame_stack, 1:] = 2
                weight = torch.ones_like(dist) * weight[:, None]

                episode_return = (
                    -(dist * weight).mean() * 1000 * dist.shape[1] / 16
                )  # considering the batch size
            else:
                # dense reward seeting, guide with reward
                raise NotImplementedError(
                    "reward guidance not officially supported yet, although implemented"
                )
                rewards = pred[:, :, -1]
                weight = np.array(
                    [10] * self.frame_stack
                    + [0.997**j for j in range(h)]
                    + [0] * h_padded
                )
                weight = torch.from_numpy(weight).float().to(self.device)
                episode_return = rewards * weight[:, None]

            # return self.guidance_scale * episode_return
            return guidance_scale * episode_return

        # guidance_fn = goal_guidance if self.guidance_scale else None
        guidance_fn = goal_guidance if guidance_scale else None

        plan_tokens = np.ceil(horizon / self.frame_stack).astype(int)
        pad_tokens = 0 if self.causal else self.n_tokens - plan_tokens - 1
        # pad_tokens = 0 # To be more efficient
        if noise_level is None:
            scheduling_matrix = self._generate_scheduling_matrix(plan_tokens)
        else:  # if noise_level is given, use it
            scheduling_matrix = noise_level
        if plan is None:
            chunk = torch.randn(
                (plan_tokens, batch_size, *self.x_stacked_shape), device=self.device
            )
            chunk = torch.clamp(
                chunk, -self.cfg.diffusion.clip_noise, self.cfg.diffusion.clip_noise
            )
        else:  # if plan is given, use it
            chunk = plan
            chunk = rearrange(chunk, "(t fs) b c -> t b (fs c)", fs=self.frame_stack)
        pad = torch.zeros(
            (pad_tokens, batch_size, *self.x_stacked_shape), device=self.device
        )
        init_token = rearrange(self.pad_init(start), "fs b c -> 1 b (fs c)")
        plan = torch.cat([init_token, chunk, pad], 0).float()

        plan_hist = [plan.detach().clone()[: self.n_tokens - pad_tokens]]
        stabilization = 0
        for m in range(scheduling_matrix.shape[0] - 1):
            from_noise_levels = np.concatenate(
                [
                    np.array((stabilization,), dtype=np.int64),
                    scheduling_matrix[m],
                    np.array([self.sampling_timesteps] * pad_tokens, dtype=np.int64),
                ]
            )
            to_noise_levels = np.concatenate(
                [
                    np.array((stabilization,), dtype=np.int64),
                    scheduling_matrix[m + 1],
                    np.array([self.sampling_timesteps] * pad_tokens, dtype=np.int64),
                ]
            )
            from_noise_levels = torch.from_numpy(from_noise_levels).to(self.device)
            to_noise_levels = torch.from_numpy(to_noise_levels).to(self.device)
            from_noise_levels = repeat(from_noise_levels, "t -> t b", b=batch_size)
            to_noise_levels = repeat(to_noise_levels, "t -> t b", b=batch_size)
            step_result = self.diffusion_model.sample_step(
                plan,
                conditions,
                from_noise_levels,
                to_noise_levels,
                guidance_fn=guidance_fn,
            )[1 : self.n_tokens - pad_tokens]
            plan[1 : self.n_tokens - pad_tokens] = step_result.float()
            plan_hist.append(plan.detach().clone()[: self.n_tokens - pad_tokens])

        plan_hist = torch.stack(plan_hist)
        plan_hist = rearrange(
            plan_hist, "m t b (fs c) -> m (t fs) b c", fs=self.frame_stack
        )
        plan_hist = plan_hist[:, self.frame_stack : self.frame_stack + horizon]

        return plan_hist

    def parallel_plan(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        horizon: int,
        conditions: Optional[Any] = None,
        guidance_scale: int = None,
        noise_level: Optional[torch.Tensor] = None,
        plan: Optional[torch.Tensor] = None,
    ):
        # start and goal are numpy arrays of shape (b, obs_dim)
        # start and goal are assumed to be normalized
        # returns plan history of (m, t, b, c), where the last dim of m is the fully diffused plan
        # assert batch_size == 1, "parallel planning only supports batch size 1"

        batch_size = len(plan)
        start = torch.cat([start] * batch_size, 0)
        goal = torch.cat([goal] * batch_size, 0)
        # Pad obs-only tensors to full bundle dim (obs + zero actions), matching plan().
        start = self.make_bundle(start)
        goal = self.make_bundle(goal)

        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        def goal_guidance(x):
            # x is a tensor of shape [t b (fs c)]
            pred = rearrange(x, "t b (fs c) -> (t fs) b c", fs=self.frame_stack)
            h_padded = (
                pred.shape[0] - self.frame_stack
            )  # include padding when horizon % frame_stack != 0

            if not self.use_reward:
                # sparse / no reward setting, guide with goal like diffuser
                target = torch.stack([start] * self.frame_stack + [goal] * (h_padded))
                valid_mask = ~torch.isnan(target)  # no gradient
                target_safe = target.detach().nan_to_num(0.0)  # no gradient
                dist = (pred - target_safe) ** 2 * valid_mask  # (t fs) b c

                # guidance weight for observation and action
                weight = np.array(
                    [20]
                    * (self.frame_stack)  # conditoning (aka reconstruction guidance)
                    + [
                        1 for _ in range(horizon)
                    ]  # try to reach the goal at any horizon
                    # + [0 for _ in range(horizon-1)] + [1]  # Diffuer guidance
                    + [0]
                    * (
                        h_padded - horizon
                    )  # don't guide padded entries due to horizon % frame_stack != 0
                )
                weight = torch.from_numpy(weight).float().to(self.device)

                dist_o, dist_a, _ = self.split_bundle(
                    dist
                )  # guidance observation and action with separate weights
                dist_a = torch.sum(dist_a, -1, keepdim=True).sqrt()
                # Sum over all obs dims: NaN goal dims are already zeroed by nan_to_num,
                # so only non-NaN dims contribute. eps prevents inf gradient from sqrt at 0.
                dist_o = (dist_o.sum(-1, keepdim=True) + 1e-8).sqrt()
                dist_o = torch.tanh(
                    dist_o / 2
                )  # similar to the "squashed gaussian" in RL, squash to (-1, 1)
                dist = dist_o
                weight = repeat(weight, "t -> t c", c=dist.shape[-1])
                weight[self.frame_stack :, 1:] = 8
                weight[: self.frame_stack, 1:] = 2
                weight = torch.ones_like(dist) * weight[:, None]
                episode_return = (
                    -(dist * weight).mean(dim=(0, 2)) * 1000 * dist.shape[1] / 16
                )
            else:
                raise NotImplementedError(
                    "reward guidance not officially supported yet, although implemented"
                )

            return (guidance_scale * episode_return).mean()

        guidance_fn = goal_guidance if guidance_scale is not None else None

        plan_tokens = np.ceil(horizon / self.frame_stack).astype(int)
        pad_tokens = 0 if self.causal else self.n_tokens - plan_tokens - 1
        try:
            scheduling_matrix = noise_level
        except:
            raise ValueError("noise_level is required for parallel planning")
        # if None in plan:
        #     chunk = torch.randn((plan_tokens, batch_size, *self.x_stacked_shape), device=self.device)
        #     chunk = torch.clamp(chunk, -self.cfg.diffusion.clip_noise, self.cfg.diffusion.clip_noise)
        # else:
        #     chunk = torch.stack(plan).squeeze(dim=2)
        #     chunk = rearrange(chunk, "b (t fs) c -> t b (fs c)", fs=self.frame_stack) # 5, 500, 2 =>  50, 5, 20

        chunk = []
        for i in range(batch_size):
            if plan[i] == None:
                c = torch.randn(
                    (plan_tokens, 1, *self.x_stacked_shape), device=self.device
                )
                c = torch.clamp(
                    c, -self.cfg.diffusion.clip_noise, self.cfg.diffusion.clip_noise
                )
            else:
                c = rearrange(plan[i], "(t fs) 1 c -> t 1 (fs c)", fs=self.frame_stack)
            chunk.append(c)
        chunk = torch.cat(chunk, 1)
        if len(chunk.shape) == 2:
            chunk = chunk.unsqueeze(0)
        pad = torch.zeros(
            (pad_tokens, batch_size, *self.x_stacked_shape), device=self.device
        )
        init_token = rearrange(self.pad_init(start), "fs b c -> 1 b (fs c)")
        plan = torch.cat([init_token, chunk, pad], 0)

        plan_hist = [plan.detach().clone()[: self.n_tokens - pad_tokens]]
        stabilization = 0

        for m in range(scheduling_matrix.shape[1] - 1):
            from_noise_levels = np.concatenate(
                [
                    np.full(
                        (batch_size, 1), stabilization, dtype=np.int64
                    ),  # Shape (batch_size, 1)
                    scheduling_matrix[:, m],
                    np.full(
                        (batch_size, pad_tokens),
                        self.sampling_timesteps,
                        dtype=np.int64,
                    ),  # Shape (batch_size, pad_tokens)
                ],
                axis=1,
            )
            to_noise_levels = np.concatenate(
                [
                    np.full(
                        (batch_size, 1), stabilization, dtype=np.int64
                    ),  # Shape (batch_size, 1)
                    scheduling_matrix[:, m + 1],
                    np.full(
                        (batch_size, pad_tokens),
                        self.sampling_timesteps,
                        dtype=np.int64,
                    ),  # Shape (batch_size, pad_tokens)
                ],
                axis=1,
            )
            from_noise_levels = torch.from_numpy(from_noise_levels).to(self.device)
            to_noise_levels = torch.from_numpy(to_noise_levels).to(self.device)
            from_noise_levels = rearrange(from_noise_levels, "b t -> t b", b=batch_size)
            to_noise_levels = rearrange(to_noise_levels, "b t -> t b", b=batch_size)
            plan[1 : self.n_tokens - pad_tokens] = self.diffusion_model.sample_step(
                plan,
                conditions,
                from_noise_levels,
                to_noise_levels,
                guidance_fn=guidance_fn,
            )[1 : self.n_tokens - pad_tokens]
            plan_hist.append(plan.detach().clone()[: self.n_tokens - pad_tokens])

        plan_hist = torch.stack(plan_hist)
        plan_hist = rearrange(
            plan_hist, "m t b (fs c) -> m (t fs) b c", fs=self.frame_stack
        )
        plan_hist = plan_hist[:, self.frame_stack : self.frame_stack + horizon]
        return plan_hist

    def interact(
        self,
        batch_size: int,
        conditions=None,
        namespace="validation",
        *,
        start_obs=None,
        goal_obs=None,
    ):
        if self.no_sim_env:
            # No simulation environment available yet (e.g. circle_2d before ManiSkill hookup).
            assert (
                start_obs is not None and goal_obs is not None
            ), "start_obs and goal_obs must be provided when no_sim_env=True"
            obs_mean = torch.tensor(
                self.observation_mean, device=self.device, dtype=torch.float32
            )
            obs_std = torch.tensor(
                self.observation_std, device=self.device, dtype=torch.float32
            )

            obs_normalized = start_obs.to(self.device).float()
            goal_normalized = goal_obs.to(self.device).float()
            # p_mctd_plan needs unnormalized start/goal for calculate_values
            start_np = (obs_normalized * obs_std + obs_mean).cpu().numpy()
            goal_np = (goal_normalized * obs_std + obs_mean).cpu().numpy()

            planning_start = time.time()
            with torch.no_grad():
                plan_hist = self.p_mctd_plan(
                    obs_normalized,
                    goal_normalized,
                    self.episode_len,
                    conditions,
                    start_np,
                    goal_np,
                )
            self.log(f"{namespace}/planning_time", time.time() - planning_start)

            plan_hist = self._unnormalize_x(plan_hist)
            plan = plan_hist[-1]  # (t, b, c)

            out_root = (
                Path(getattr(self.trainer, "default_root_dir", Path.cwd()))
                / "mctd_plans"
                / namespace
                / f"step_{getattr(self, 'global_step', 0):07d}"
            )
            out_root.mkdir(parents=True, exist_ok=True)

            for i in range(min(batch_size, 4)):
                obs_traj, _, _ = self.split_bundle(plan[:, i : i + 1, :])
                states = obs_traj[:, 0, :].detach().cpu().numpy()
                np.save(out_root / f"plan_{i}.npy", states)
                viz_kwargs = {}
                if self.observation_dim >= 18:
                    viz_kwargs = dict(
                        tcp_xy_indices=(0, 1),
                        block_xy_indices=(9, 10),
                        block_rot6d_indices=(12, 13, 14, 15, 16, 17),
                    )
                self._log_or_save_pushboundary_2d_gif(
                    namespace=namespace,
                    states=states,
                    sample_idx=i,
                    gif_out_dir=out_root / "gifs",
                    start_marker=start_np[i, :2],
                    goal_marker=goal_np[i, :2],
                    block_shape=self._pushboundary_2d_viz_block_shape,
                    **viz_kwargs,
                )

            # FIXME: Execute plan in ManiSkill environment (once hooked up).
            return

        try:
            import gym
            import ogbench
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError:
            print(
                "d4rl import not successful, skipping environment interaction. Check d4rl installation."
            )
            return

        print("Interacting with environment... This may take a couple minutes.")

        use_diffused_action = False

        if self.env_id in OGBENCH_ENVS:
            if "pointmaze" in self.env_id:
                envs = DummyVecEnv(
                    [
                        lambda: ogbench.locomaze.maze.make_maze_env(
                            "point", "maze", maze_type=self.env_id.split("-")[1]
                        )
                    ]
                    * batch_size
                )
                if self.action_dim == 2:
                    use_diffused_action = True
            elif "antmaze" in self.env_id:
                envs = DummyVecEnv(
                    [
                        lambda: ogbench.locomaze.maze.make_maze_env(
                            "ant", "maze", maze_type=self.env_id.split("-")[1]
                        )
                    ]
                    * batch_size
                )
                # use_diffused_action = True
                from dql.main_Antmaze import hyperparameters
                from dql.agents.ql_diffusion import Diffusion_QL as Agent

                params = hyperparameters[self.dataset]
                state_dim = envs.observation_space.shape[0]
                action_dim = envs.action_space.shape[0]
                max_action = float(envs.action_space.high[0])
                agent = Agent(
                    state_dim=state_dim * 2,
                    action_dim=action_dim,
                    max_action=max_action,
                    device=0,
                    discount=0.99,
                    tau=0.005,
                    max_q_backup=params["max_q_backup"],
                    beta_schedule="vp",
                    n_timesteps=5,
                    eta=params["eta"],
                    lr=params["lr"],
                    lr_decay=False,
                    lr_maxt=params["num_epochs"],
                    grad_norm=params["gn"],
                    goal_dim=2,
                    lcb_coef=4.0,
                )
                # pretrained agent loading
                if self.dataset == "antmaze-medium-navigate-v0":
                    dql_folder = "antmaze-medium-navigate-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|2|1.0|False|cql_antmaze|0.2|4.0|10"
                elif self.dataset == "antmaze-large-navigate-v0":
                    dql_folder = "antmaze-large-navigate-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|2|1.0|False|cql_antmaze|0.2|4.0|10"
                elif self.dataset == "antmaze-giant-navigate-v0":
                    dql_folder = "antmaze-giant-navigate-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|2|1.0|False|cql_antmaze|0.2|4.0|10"
                else:
                    raise ValueError(f"Dataset {self.dataset} not supported")

                import os

                agent.load_model(
                    os.path.join(os.getcwd(), "dql", "results", dql_folder), id=200
                )
            for i, env in enumerate(envs.envs):
                env.set_task(self.task_id + i)
                # env.set_seed(self.interaction_seed)
        else:
            envs = DummyVecEnv([lambda: gym.make(self.env_id)] * batch_size)
            envs.seed(self.interaction_seed)

        terminate = False
        obs_mean = self.data_mean[: self.observation_dim]
        obs_std = self.data_std[: self.observation_dim]
        obs = envs.reset()
        # Randomize the goal for each environment
        if (
            self.env_id in OGBENCH_ENVS
        ):  # OGBench goal setting is already done through set_task()
            pass
        else:
            if self.use_random_goals_for_interaction:
                for env in envs.envs:
                    env.set_target()

        obs = torch.from_numpy(obs).float().to(self.device)
        start = obs.detach()
        obs_normalized = (
            (obs[:, : self.observation_dim] - obs_mean[None]) / obs_std[None]
        ).detach()

        if self.env_id in OGBENCH_ENVS:  # OGBench
            goal = np.vstack(
                [envs.reset_infos[i]["goal"] for i in range(len(envs.reset_infos))]
            )
        else:
            goal = np.concatenate([[env.env._target] for env in envs.envs])
        goal = torch.Tensor(goal).float().to(self.device)
        goal = torch.cat([goal, torch.zeros_like(goal)], -1)
        goal = goal[:, : self.observation_dim]
        goal_normalized = ((goal - obs_mean[None]) / obs_std[None]).detach()

        steps = 0
        episode_reward = np.zeros(batch_size)
        episode_reward_if_stay = np.zeros(batch_size)
        reached = np.zeros(batch_size, dtype=bool)
        first_reach = np.zeros(batch_size)

        trajectory = []  # actual trajectory
        all_plan_hist = (
            []
        )  # a list of plan histories, each history is a collection of m diffusion steps

        # run mpc with diffused actions
        planning_time = []
        while not terminate and steps < self.val_max_steps:
            planning_start_time = time.time()
            if self.mctd:
                plan_hist = self.p_mctd_plan(
                    obs_normalized,
                    goal_normalized,
                    self.episode_len,
                    conditions,
                    start.cpu().numpy()[:, : self.observation_dim],
                    goal.cpu().numpy()[:, : self.observation_dim],
                )  # fake plan_hist
                plan_hist = self._unnormalize_x(plan_hist)
                plan = plan_hist[-1]  # (t b c)
            else:
                plan_hist = self.plan(
                    obs_normalized, goal_normalized, self.episode_len, conditions
                )
                plan_hist = self._unnormalize_x(plan_hist)  # (m t b c)
                plan = plan_hist[-1]  # (t b c)
            # Visualization
            start_numpy = start.cpu().numpy()[:, :2]
            goal_numpy = goal.cpu().numpy()[:, : self.observation_dim]
            image = make_trajectory_images(
                self.env_id,
                plan[:, :, :2].detach().cpu().numpy(),
                1,
                start_numpy,
                goal_numpy,
                self.plot_end_points,
            )[0]
            self.log_image(f"plan/plan_at_{steps}", Image.fromarray(image))

            planning_end_time = time.time()
            planning_time.append(planning_end_time - planning_start_time)

            # jumpy case (fill the gap)
            if self.jump > 1:
                _plan = []
                for t in range(plan.shape[0]):
                    for j in range(self.jump):
                        _plan.append(plan[t, :, :2])
                plan = torch.stack(_plan)

            all_plan_hist.append(plan_hist.cpu())

            obs_numpy = obs.detach().cpu().numpy()
            if "antmaze" in self.env_id:
                # sub_goal = plan[self.open_loop_horizon - 1, :, :2].detach().cpu().numpy()
                sub_goal = plan[self.sub_goal_interval, :, :2].detach().cpu().numpy()
                sub_goal_step = self.sub_goal_interval
            for t in range(self.open_loop_horizon):
                if use_diffused_action:
                    _, action, _ = self.split_bundle(plan[t])
                else:
                    if "antmaze" in self.env_id:
                        if np.linalg.norm(obs_numpy[0, :2] - sub_goal[0, :2]) < 1.0:
                            print(
                                f"sub_goal_step {sub_goal_step} achieved, next sub_goal_step {sub_goal_step + self.sub_goal_interval} in {plan.shape[0]} steps"
                            )
                            sub_goal_step += self.sub_goal_interval
                            if plan.shape[0] - sub_goal_step <= 0:
                                sub_goal = plan[-1, :, :2].detach().cpu().numpy()
                            else:
                                sub_goal = (
                                    plan[sub_goal_step, :, :2].detach().cpu().numpy()
                                )
                        assert (
                            obs_numpy.shape[0] == 1
                        ), f"Batch size must be 1 for AntMaze, got {obs_numpy.shape[0]}"
                        action = agent.sample_action(obs_numpy, sub_goal)
                        action = torch.from_numpy(action).float().reshape(1, -1)
                    else:
                        if t == 0:
                            plan_vel = plan[t, :, :2] - obs[:, :2]
                        else:
                            if t < plan.shape[0]:
                                plan_vel = plan[t, :, :2] - plan[t - 1, :, :2]
                            else:
                                plan_vel = 0
                        if t < plan.shape[0]:
                            action = 12.5 * (plan[t, :, :2] - obs[:, :2]) + 1.2 * (
                                plan_vel - obs[:, 2:]
                            )
                        else:
                            action = 12.5 * (plan[-1, :, :2] - obs[:, :2]) + 1.2 * (
                                plan_vel - obs[:, 2:]
                            )
                action = torch.clip(action, -1, 1).detach().cpu()
                obs_numpy, reward, done, _ = envs.step(np.nan_to_num(action.numpy()))

                reached = np.logical_or(reached, reward >= 1.0)
                episode_reward += reward
                episode_reward_if_stay += np.where(~reached, reward, 1)
                first_reach += ~reached

                if done.any():
                    terminate = True
                    break

                obs, reward, done = [
                    torch.from_numpy(item).float() for item in [obs_numpy, reward, done]
                ]
                bundle = self.make_bundle(obs, action, reward[..., None])
                trajectory.append(bundle)
                obs = obs.to(self.device)
                obs_normalized = (
                    (obs[:, : self.observation_dim] - obs_mean[None]) / obs_std[None]
                ).detach()

                steps += 1
        self.log(f"{namespace}/planning_time", np.sum(planning_time))
        self.log(f"{namespace}/episode_reward", episode_reward.mean())
        self.log(f"{namespace}/episode_reward_if_stay", episode_reward_if_stay.mean())
        self.log(f"{namespace}/first_reach", first_reach.mean())
        self.log(f"{namespace}/success_rate", sum(episode_reward >= 1.0) / batch_size)

        # Visualization
        # samples = min(16, batch_size)
        samples = min(32, batch_size)
        trajectory = torch.stack(trajectory)
        start = start[:, :2].cpu().numpy().tolist()
        goal = goal[:, :2].cpu().numpy().tolist()
        images = make_trajectory_images(
            self.env_id, trajectory, samples, start, goal, self.plot_end_points
        )

        for i, img in enumerate(images):
            self.log_image(
                f"{namespace}_interaction/sample_{i}",
                Image.fromarray(img),
            )

        if self.debug:
            samples = min(16, batch_size)
            indicies = list(range(samples))

            for i in indicies:
                filename = make_convergence_animation(
                    self.env_id,
                    all_plan_hist,
                    trajectory,
                    start,
                    goal,
                    self.open_loop_horizon,
                    namespace=namespace,
                    batch_idx=i,
                )
                self.logger.experiment.log(
                    {
                        f"convergence/{namespace}_{i}": wandb.Video(filename, fps=4),
                        f"trainer/global_step": self.global_step,
                    }
                )

                filename = make_mpc_animation(
                    self.env_id,
                    all_plan_hist,
                    trajectory,
                    start,
                    goal,
                    self.open_loop_horizon,
                    namespace=namespace,
                    batch_idx=i,
                )
                self.logger.experiment.log(
                    {
                        f"mpc/{namespace}_{i}": wandb.Video(filename, fps=24),
                        f"trainer/global_step": self.global_step,
                    }
                )

    def pad_init(self, x, batch_first=False):
        x = repeat(x, "b ... -> fs b ...", fs=self.frame_stack).clone()
        if self.padding_mode == "zero":
            x[: self.frame_stack - 1] = 0
        elif self.padding_mode != "same":
            raise ValueError("init_pad must be 'zero' or 'same'")
        if batch_first:
            x = rearrange(x, "fs b ... -> b fs ...")

        return x

    def split_bundle(self, bundle):
        if self.use_reward:
            return torch.split(bundle, [self.observation_dim, self.action_dim, 1], -1)
        else:
            o, a = torch.split(bundle, [self.observation_dim, self.action_dim], -1)
            return o, a, None

    def make_bundle(
        self,
        obs: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
    ):
        valid_value = None
        if obs is not None:
            valid_value = obs
        if action is not None and valid_value is not None:
            valid_value = action
        if reward is not None and valid_value is not None:
            valid_value = reward
        if valid_value is None:
            raise ValueError("At least one of obs, action, reward must be provided")
        batch_shape = valid_value.shape[:-1]

        if obs is None:
            obs = torch.zeros(batch_shape + (self.observation_dim,)).to(valid_value)
        if action is None:
            action = torch.zeros(batch_shape + (self.action_dim,)).to(valid_value)
        if reward is None:
            reward = torch.zeros(batch_shape + (1,)).to(valid_value)

        bundle = [obs, action]
        if self.use_reward:
            bundle += [reward]

        return torch.cat(bundle, -1)

    def _generate_noise_levels(
        self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        noise_levels = super()._generate_noise_levels(xs, masks)
        _, batch_size, *_ = xs.shape

        # first frame is almost always known, this reflect that
        if random() < 0.5:
            noise_levels[0] = torch.randint(
                0, self.timesteps // 4, (batch_size,), device=xs.device
            )

        return noise_levels

    def visualize_node_value_plans(
        self, search_num, values, names, plans, value_plans, starts, goals
    ):
        if plans.shape[1] != starts.shape[0]:
            starts = starts.repeat(plans.shape[1], axis=0)
        if plans.shape[1] != goals.shape[0]:
            goals = goals.repeat(plans.shape[1], axis=0)
        plans = self._unnormalize_x(plans)
        plan_obs, _, _ = self.split_bundle(plans)
        plan_obs = plan_obs.detach().cpu().numpy()[:-1]
        plan_images = make_trajectory_images(
            self.env_id,
            plan_obs,
            plan_obs.shape[1],
            starts,
            goals,
            self.plot_end_points,
        )
        value_plans = self._unnormalize_x(value_plans)
        value_plan_obs, _, _ = self.split_bundle(value_plans)
        value_plan_obs = value_plan_obs.detach().cpu().numpy()[:-1]
        value_plan_images = make_trajectory_images(
            self.env_id,
            value_plan_obs,
            value_plan_obs.shape[1],
            starts,
            goals,
            self.plot_end_points,
        )
        for i in range(len(plan_images)):
            plan_image = plan_images[i]
            value_plan_image = value_plan_images[i]
            img = np.concatenate([plan_image, value_plan_image], axis=0)
            self.log_image(
                f"mcts_plan/{search_num+i+1}_{names[i]}_V{values[i]}",
                Image.fromarray(img),
            )

    def calculate_values(self, plans, starts, goals):
        if plans.shape[1] != starts.shape[0]:
            starts = starts.repeat(plans.shape[1], axis=0)
        if plans.shape[1] != goals.shape[0]:
            goals = goals.repeat(plans.shape[1], axis=0)

        plans = self._unnormalize_x(plans)
        obs, _, _ = self.split_bundle(plans)
        obs = obs.detach().cpu().numpy()[:-1, :]  # last observation is dummy

        values = np.zeros(plans.shape[1])
        infos = np.array(["NotReached"] * plans.shape[1])
        achieved_ts = np.array([None] * plans.shape[1])

        for t in range(obs.shape[0]):
            if t == 0:
                pos_diff = np.linalg.norm(obs[t][:, :4] - starts[:, :4], axis=-1)
            else:
                pos_diff = np.linalg.norm(obs[t][:, :4] - obs[t - 1][:, :4], axis=-1)

            infos[(pos_diff > self.warp_threshold) * (infos == "NotReached")] = "Warp"
            values[(pos_diff > self.warp_threshold) * (infos == "NotReached")] = 0

            # --- UPDATED GOAL DISTANCE CALCULATION ---
            # Calculate squared differences (NaNs propagate here)
            squared_diff_from_goal = (obs[t][:, 2:4] - goals[:, 2:4]) ** 2

            # Sum ignoring NaNs, then take the square root
            diff_from_goal = np.sqrt(np.nansum(squared_diff_from_goal, axis=-1))
            # -----------------------------------------

            values[(diff_from_goal < self.goal_threshold) * (infos == "NotReached")] = (
                plans.shape[0] - t
            ) / plans.shape[0]

            achieved_ts[
                (diff_from_goal < self.goal_threshold) * (infos == "NotReached")
            ] = t

            infos[(diff_from_goal < self.goal_threshold) * (infos == "NotReached")] = (
                "Achieved"
            )

        return values, infos, achieved_ts

    def _first_timestep_deviation_exceeds(
        self,
        plan_last: torch.Tensor,
        threshold: float,
        *,
        obs_only: bool = True,
    ) -> Optional[int]:
        """
        plan_last: (horizon, B, C) normalized bundle, same as value_estimation_plan_hists[-1].

        Returns the first horizon index t (after dropping the final dummy step, matching
        calculate_values) where mean pairwise L2 distance across branches exceeds threshold,
        or None if B < 2 or the threshold is never exceeded.
        """
        if plan_last.shape[1] < 2:
            return None
        plans = self._unnormalize_x(plan_last)[:-1]
        if obs_only:
            x, _, _ = self.split_bundle(plans)
        else:
            x = plans
        for t in range(x.shape[0]):
            d = torch.pdist(x[t], p=2).max()
            # print("d:", d)
            if d > threshold:
                print("Found deviation at timestep", t)
                return t
        print("No deviation found, returning last timestep", x.shape[0] - 1)
        return x.shape[0] - 1

    def _decide_children_stable_prefix(self, node, t_div):
        remaining = node.plan_tokens - node.tokens_committed
        if remaining <= 0:
            return 0
        # From node.value_estimation_plan, detect contiguous low-multimodality
        # leading region and return its length.
        if t_div is None:
            return 1  # Fully autoregressive
        return t_div // self.frame_stack

    def _build_expansion_slab(self, c, s, T, plan_tokens):
        """Slab that commits tokens [c : c+s] while first c stay at 0 and rest stay at T."""
        entry = np.concatenate(
            [
                np.zeros(c, dtype=np.int32),
                np.full(plan_tokens - c, T, dtype=np.int32),
            ]
        )
        rows = [entry]
        for step in range(1, T + s):
            row = rows[-1].copy()
            for i in range(s):
                tok = c + i
                if step >= i:
                    row[tok] = max(T - (step - i), 0)
            rows.append(row)
        return np.stack(rows)

    def _build_valuation_slab(self, c, T, plan_tokens, skip):
        """Finish denoising from state (first c at 0, rest at T) all the way to 0, with stride."""
        full = self._build_expansion_slab(c, plan_tokens - c, T, plan_tokens)
        strided = full[::skip]
        if not np.array_equal(strided[-1], full[-1]):
            strided = np.concatenate([strided, full[-1:]], axis=0)
        return strided

    def p_mctd_plan(
        self, obs_normalized, goal_normalized, horizon, conditions, start, goal
    ):
        assert start.shape[0] == 1, "the batch size must be 1"
        assert (not self.leaf_parallelization) or (
            self.parallel_search_num % len(self.mctd_guidance_scales) == 0
        ), f"Parallel search num must be divisible by the number of guidance scales: {self.parallel_search_num} % {len(self.mctd_guidance_scales)} != 0"

        horizon = self.episode_len if horizon is None else horizon
        plan_tokens = np.ceil(horizon / self.frame_stack).astype(int)
        # noise_level = self._generate_scheduling_matrix(plan_tokens)
        # children_node_guidance_scales = self.mctd_guidance_scales
        # max_search_num = self.mctd_max_search_num
        # num_denoising_steps = self.mctd_num_denoising_steps
        # skip_level_steps = self.mctd_skip_level_steps
        # terminal_depth = np.ceil(
        #     (noise_level.shape[0] - 1) / num_denoising_steps
        # ).astype(int)
        children_node_guidance_scales = self.mctd_guidance_scales
        max_search_num = self.mctd_max_search_num
        skip_level_steps = self.mctd_skip_level_steps
        T = self.sampling_timesteps  # or wherever this lives in your model
        # Root Node (name, depth, parent_node, children_node_guidance_scale, plan_history)
        # root_node = TreeNode(
        #     "0",
        #     0,
        #     None,
        #     children_node_guidance_scales,
        #     [],
        #     terminal_depth=terminal_depth,
        #     virtual_visit_weight=self.virtual_visit_weight,
        # )
        # root_node.set_value(0)  # Initialize the value of the root node

        root_node = TreeNode(
            "0",
            0,
            None,
            children_node_guidance_scales,
            [],
            plan_tokens=plan_tokens,
            tokens_committed=0,
            virtual_visit_weight=self.virtual_visit_weight,
        )
        root_node.set_value(0)
        # Root decides its own children's partition. Default: full pyramid.
        # Later: replace with multimodality-based decision on a root lookahead.
        root_node.children_stable_prefix = 3

        # Search
        search_num, p_search_num, solved, achieved = 0, 0, False, False
        self._mctd_pairwise_first_divergence_timesteps = []
        achieved_plans = []  # the plans that achieved the goal through the rollout
        not_reached_plans = (
            []
        )  # the plans that did not achieve the goal, but there is no warp through the rollout
        # lists for logging time
        (
            selection_time,
            expansion_time,
            simulation_time,
            backprop_time,
            early_termination_time,
        ) = (
            [],
            [],
            [],
            [],
            [],
        )  # sum of the time for each batch
        simul_noiselevel_zero_padding_time = []
        simul_value_estimation_time = []
        simul_value_calculation_time = []
        simul_node_allocation_time = []
        while True:
            if self.time_limit is not None:
                if time.time() - self.start_time > self.time_limit:
                    break
            else:
                # if search_num >= max_search_num:
                if p_search_num >= max_search_num:
                    break

            ## For checking the virtual visit count
            # root_node.check_virtual_visit_count()

            ###############################
            # Selection
            #  When leaf parallelization is True, then the selection is done in partially parallel (the children nodes from same parent node are selected at the same time)
            #  When leaf parallelization is False, then the selection is done in fully sequential (only one node is selected at a time)
            if (
                not self.parallel_multiple_visits
            ):  # If parallel multiple visits is False, then we need to list all the nodes to expand
                expandable_node_names = root_node.get_expandable_node_names()
                # print(f"Expandable node names: {expandable_node_names}")
            selection_start_time = time.time()
            # print("============ Selection Start ============")
            psn = self.parallel_search_num
            selected_nodes, expanded_node_candidates = [], []
            while psn > 0:
                selected_node = root_node
                while (
                    (
                        not selected_node.is_expandable(
                            consider_virtually_visited=(
                                not self.parallel_multiple_visits
                            )
                        )
                    )
                    and (not selected_node.is_terminal())
                    and (selected_node.is_selectable())
                ):
                    selected_node = selected_node.select(
                        leaf_parallelization=self.leaf_parallelization
                    )
                if selected_node.is_terminal() or (
                    not selected_node.is_selectable()
                    and not selected_node.is_expandable(
                        consider_virtually_visited=(not self.parallel_multiple_visits)
                    )
                ):
                    psn -= (
                        1
                        if not self.leaf_parallelization
                        else len(children_node_guidance_scales)
                    )
                    continue
                if self.leaf_parallelization:
                    for i in range(len(children_node_guidance_scales)):
                        # when multiple visits is False, then we need to consider the virtually visited nodes to visit only once
                        expanded_node_candidate = (
                            selected_node.get_expandable_candidate(
                                index=i,
                                consider_virtually_visited=(
                                    not self.parallel_multiple_visits
                                ),
                            )
                        )
                        selected_nodes.append(selected_node)
                        expanded_node_candidates.append(expanded_node_candidate)
                        if not self.parallel_multiple_visits:
                            if (
                                not expanded_node_candidate["name"]
                                in expandable_node_names
                            ):
                                raise ValueError(
                                    f"Expanded node candidate {expanded_node_candidate['name']} is not in expandable node names"
                                )
                            expandable_node_names.remove(
                                expanded_node_candidate["name"]
                            )
                        # print(f"Expanded node candidate {expanded_node_candidate['name']} is selected")
                        psn -= 1
                else:
                    # when multiple visits is False, then we need to consider the virtually visited nodes to visit only once
                    expanded_node_candidate = selected_node.get_expandable_candidate(
                        index=None,
                        consider_virtually_visited=(not self.parallel_multiple_visits),
                    )
                    selected_nodes.append(selected_node)
                    expanded_node_candidates.append(expanded_node_candidate)
                    if not self.parallel_multiple_visits:
                        if not expanded_node_candidate["name"] in expandable_node_names:
                            raise ValueError(
                                f"Expanded node candidate {expanded_node_candidate['name']} is not in expandable node names"
                            )
                        expandable_node_names.remove(expanded_node_candidate["name"])
                    # print(f"Expanded node candidate {expanded_node_candidate['name']} is selected")
                    psn -= 1
                if not self.parallel_multiple_visits:
                    if len(expandable_node_names) == 0:
                        print("No more expandable nodes")
                        break
            if len(selected_nodes) == 0:
                print("No more selected nodes")
                break
            # print("============ Selection End ============")
            selection_end_time = time.time()
            selection_time.append(selection_end_time - selection_start_time)

            batch_expanded_node_plan_hists = []
            batch_value_estimation_plan_hists = []
            for _ in range(5):  # Run simulation multiple times to get a batch of plans
                filtered_expanded_node_plan_hists = [None] * len(
                    expanded_node_candidates
                )
                filtered_value_estimation_plan_hists = [None] * len(
                    expanded_node_candidates
                )
                for _ in range(
                    self.num_tries_for_bad_plans
                ):  # Trick used in MCTD to resample when the generated plan is terrible (e.g., not moving plans)
                    ###############################
                    # Expansion
                    expansion_start_time = time.time()
                    # print("============ Expansion Start ============")
                    expanded_node_plans = []
                    expanded_node_noise_levels = []
                    expanded_node_guidance_scales = []
                    for info in expanded_node_candidates:
                        if len(info["plan_history"]) == 0:
                            expanded_node_plans.append(None)
                        else:
                            expanded_node_plans.append(
                                info["plan_history"][-1][-1].unsqueeze(1)
                            )
                        # _noise_level = noise_level[
                        #     (info["depth"] - 1)
                        #     * num_denoising_steps : (
                        #         info["depth"] * num_denoising_steps + 1
                        #     )
                        # ]
                        # _noise_level = np.concatenate(
                        #     [_noise_level]
                        #     + [noise_level[-1:]]
                        #     * (num_denoising_steps - _noise_level.shape[0] + 1)
                        # )
                        # expanded_node_noise_levels.append(_noise_level)
                        c = (
                            info["tokens_committed"] - info["stable_prefix"]
                        )  # tokens at 0 entering this expansion
                        s = info["stable_prefix"]
                        _noise_level = self._build_expansion_slab(c, s, T, plan_tokens)
                        expanded_node_noise_levels.append(_noise_level)

                        expanded_node_guidance_scales.append(info["guidance_scale"])
                    expanded_node_guidance_scales = torch.tensor(
                        expanded_node_guidance_scales
                    ).to(
                        obs_normalized.device
                    )  # (batch_size,)

                    # After the for-loop over expanded_node_candidates, before np.array(...)
                    max_len = max(nl.shape[0] for nl in expanded_node_noise_levels)
                    for i, nl in enumerate(expanded_node_noise_levels):
                        if nl.shape[0] < max_len:
                            pad = np.tile(nl[-1:], (max_len - nl.shape[0], 1))
                            expanded_node_noise_levels[i] = np.concatenate(
                                [nl, pad], axis=0
                            )
                    expanded_node_noise_levels = np.array(
                        expanded_node_noise_levels, dtype=np.int32
                    )

                    # expanded_node_noise_levels = np.array(
                    #     expanded_node_noise_levels, dtype=np.int32
                    # )  # (batch_size, height, width)

                    expanded_node_plan_hists = self.parallel_plan(
                        obs_normalized,
                        goal_normalized,
                        horizon,
                        conditions,
                        guidance_scale=expanded_node_guidance_scales,
                        noise_level=expanded_node_noise_levels,
                        plan=expanded_node_plans,
                    )
                    # print(f"Expanded node plan hists: {expanded_node_plan_hists.shape}")
                    # print("============ Expansion End ============")
                    expansion_end_time = time.time()
                    expansion_time.append(expansion_end_time - expansion_start_time)

                    ###############################
                    # Simulation
                    #  It includes the noise level zero-padding, finding the max denoising steps, simulation, value calculation and node allocation
                    simulation_start_time = time.time()
                    # print("============ Simulation Start ============")

                    # Pad the noise levels - Sequential
                    simul_noiselevel_zero_padding_start = time.time()
                    value_estimation_plans, value_estimation_noise_levels = [], []
                    max_denoising_steps = 0
                    for i in range(
                        len(expanded_node_candidates)
                    ):  # find the max denoising steps
                        # _noise_level = np.concatenate(
                        #     [
                        #         noise_level[
                        #             (
                        #                 expanded_node_candidates[i]["depth"]
                        #                 * num_denoising_steps
                        #             ) :: skip_level_steps
                        #         ],
                        #         noise_level[-1:],
                        #     ],
                        #     axis=0,
                        # )

                        c_new = expanded_node_candidates[i][
                            "tokens_committed"
                        ]  # already includes this expansion's commit
                        _noise_level = self._build_valuation_slab(
                            c_new, T, plan_tokens, skip_level_steps
                        )

                        # update max denoising steps
                        if _noise_level.shape[0] > max_denoising_steps:
                            max_denoising_steps = _noise_level.shape[0]
                        value_estimation_noise_levels.append(_noise_level)
                        value_estimation_plans.append(
                            expanded_node_plan_hists[-1, :, i].unsqueeze(1)
                        )
                    for i in range(len(expanded_node_candidates)):  # zero-padding
                        length = value_estimation_noise_levels[i].shape[0]
                        if length < max_denoising_steps:
                            value_estimation_noise_levels[i] = np.concatenate(
                                [
                                    value_estimation_noise_levels[i],
                                    np.zeros(
                                        (
                                            max_denoising_steps - length,
                                            value_estimation_noise_levels[i].shape[1],
                                        ),
                                        dtype=np.int32,
                                    ),
                                ],
                                axis=0,
                            )  # zero-padding
                    simul_noiselevel_zero_padding_end = time.time()
                    simul_noiselevel_zero_padding_time.append(
                        simul_noiselevel_zero_padding_end
                        - simul_noiselevel_zero_padding_start
                    )

                    # Simulation - Value Estimation
                    simul_value_estimation_start = time.time()
                    value_estimation_noise_levels = np.array(
                        value_estimation_noise_levels, dtype=np.int32
                    )
                    value_estimation_plan_hists = self.parallel_plan(
                        obs_normalized,
                        goal_normalized,
                        horizon,
                        conditions,
                        guidance_scale=expanded_node_guidance_scales,
                        noise_level=value_estimation_noise_levels,
                        plan=value_estimation_plans,
                    )
                    simul_value_estimation_end = time.time()
                    # print(
                    #     f"Value estimation plan hist: {value_estimation_plan_hists.shape}"
                    # )

                    # check if any plan is good
                    plans = (
                        self._unnormalize_x(value_estimation_plan_hists[-1])[:-1]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    diffs = np.linalg.norm(
                        plans[1:] - plans[:-1], axis=-1
                    )  # (plan_len-1, N)
                    for i in range(diffs.shape[1]):
                        if filtered_expanded_node_plan_hists[i] is None and not np.all(
                            diffs[:, i] < 0.001  # FIXME: make this a parameter
                        ):
                            filtered_expanded_node_plan_hists[i] = (
                                expanded_node_plan_hists[:, :, i]
                            )
                            filtered_value_estimation_plan_hists[i] = (
                                value_estimation_plan_hists[:, :, i]
                            )
                    if None in filtered_expanded_node_plan_hists:
                        print("No good plan found, resampling")
                        simulation_end_time = time.time()
                        simulation_time.append(
                            simulation_end_time - simulation_start_time
                        )
                        continue
                    else:
                        break
                for i in range(len(filtered_expanded_node_plan_hists)):
                    if filtered_expanded_node_plan_hists[i] is None:
                        filtered_expanded_node_plan_hists[i] = expanded_node_plan_hists[
                            :, :, i
                        ]
                        filtered_value_estimation_plan_hists[i] = (
                            value_estimation_plan_hists[:, :, i]
                        )
                expanded_node_plan_hists = torch.stack(
                    filtered_expanded_node_plan_hists, dim=2
                )
                value_estimation_plan_hists = torch.stack(
                    filtered_value_estimation_plan_hists, dim=2
                )
                batch_expanded_node_plan_hists.append(expanded_node_plan_hists)
                batch_value_estimation_plan_hists.append(value_estimation_plan_hists)

            expanded_node_plan_hists = torch.cat(batch_expanded_node_plan_hists, dim=2)
            value_estimation_plan_hists = torch.cat(
                batch_value_estimation_plan_hists, dim=2
            )

            if self.mctd_pairwise_divergence_threshold is not None:
                t_div = self._first_timestep_deviation_exceeds(
                    value_estimation_plan_hists[-1],
                    float(self.mctd_pairwise_divergence_threshold),
                )
                self._mctd_pairwise_first_divergence_timesteps.append(t_div)
            else:
                t_div = None

            if (
                self.viz_mctd_batch_plans
                and self.viz_mctd_batch_plans_every_n_search > 0
                and (search_num % self.viz_mctd_batch_plans_every_n_search == 0)
            ):
                # Best-effort visualization; never fail planning due to rendering issues.
                try:
                    start_xy = np.asarray(start[0, :2], dtype=np.float64)
                    goal_xy = np.asarray(goal[0, :2], dtype=np.float64)
                    plan_phys = self._unnormalize_x(value_estimation_plan_hists[-1])
                    obs_traj, _, _ = self.split_bundle(plan_phys)
                    states_batch = obs_traj[:-1].detach().cpu().numpy()  # (T, B, D_obs)
                    max_b = self.viz_mctd_batch_plans_max_branches
                    if max_b is not None:
                        max_b = int(max_b)
                    self._log_or_save_pushboundary_2d_batch_gif(
                        namespace="mctd",
                        states=states_batch,
                        sample_idx=0,
                        suffix=f"mctd_batch_rollouts_search_{search_num}",
                        start_marker=start_xy,
                        goal_marker=goal_xy,
                        block_shape=self._pushboundary_2d_viz_block_shape,
                        gif_out_dir=self.viz_mctd_batch_plans_out_dir,
                        alpha=self.viz_mctd_batch_plans_alpha,
                        max_branches=max_b,
                        t_div=t_div,
                        seed=int(getattr(self, "interaction_seed", 0)),
                    )
                    breakpoint()
                except Exception:
                    pass

            # Value Calculation
            simul_value_calculation_start = time.time()
            values, infos, achieved_ts = self.calculate_values(
                value_estimation_plan_hists[-1], start, goal
            )  # (plan_len, N, D), (N, D), (N, D)
            for i in range(len(infos)):
                info = infos[i]
                achieved_t = achieved_ts[i]
                if info == "Achieved":
                    achieved_plans.append(
                        [value_estimation_plan_hists[-1, :achieved_t, i], values[i]]
                    )
                    achieved = True
                elif info == "NotReached":
                    not_reached_plans.append(
                        [value_estimation_plan_hists[-1, :, i], values[i]]
                    )
            print(f"Value Calculation: {values}, {infos}")
            simul_value_calculation_end = time.time()

            # Node Allocation
            simul_node_allocation_start = time.time()
            selected_nodes_for_expansion = {}
            expanded_node_infos = {}
            for i in range(len(expanded_node_candidates)):
                name = expanded_node_candidates[i]["name"]
                if name not in expanded_node_infos:
                    selected_nodes_for_expansion[name] = selected_nodes[i]
                    expanded_node_infos[name] = expanded_node_candidates[i]
                    expanded_node_infos[name]["plan_history"].append([])
                value = values[i]
                plan_hist = expanded_node_plan_hists[:, :, i]
                value_estimation_plan = value_estimation_plan_hists[-1, :, i]
                if expanded_node_infos[name]["value"] is None:
                    expanded_node_infos[name]["value"] = value
                    expanded_node_infos[name][
                        "value_estimation_plan"
                    ] = value_estimation_plan
                    expanded_node_infos[name]["plan_history"][-1] = plan_hist
                else:
                    if value > expanded_node_infos[name]["value"]:
                        expanded_node_infos[name]["value"] = value
                        expanded_node_infos[name][
                            "value_estimation_plan"
                        ] = value_estimation_plan
                        expanded_node_infos[name]["plan_history"][-1] = plan_hist
            for name in selected_nodes_for_expansion:
                child = selected_nodes_for_expansion[name].expand(
                    **expanded_node_infos[name]
                )
                child.children_stable_prefix = self._decide_children_stable_prefix(
                    child, t_div
                )
            simul_node_allocation_end = time.time()
            simul_node_allocation_time.append(
                simul_node_allocation_end - simul_node_allocation_start
            )

            # print("============ Simulation End ============")
            simulation_end_time = time.time()
            simulation_time.append(simulation_end_time - simulation_start_time)

            ######################
            # Backpropagation
            #  When leaf parallelization is True, then the backpropagation is done in partially parallel (the leafs from same parent node are backpropagated at the same time)
            #  When leaf parallelization is False, then the backpropagation is done in fully sequential (only one node is backpropagated at a time)
            backprop_start_time = time.time()
            # print("============ Backpropagation Start ============")

            distinct_selected_nodes = np.unique(selected_nodes)
            for selected_node in distinct_selected_nodes:
                selected_node.backpropagate()

            # print("============ Backpropagation End ============")
            backprop_end_time = time.time()
            backprop_time.append(backprop_end_time - backprop_start_time)

            ######################
            # Early Termination
            early_termination_start_time = time.time()
            # print("============ Early Termination Start ============")

            plans = torch.stack(
                [info["plan_history"][-1][-1] for info in expanded_node_infos.values()],
                dim=1,
            )
            _, infos, achieved_ts = self.calculate_values(
                plans, start, goal
            )  # (plan_len, N, D), (N, D), (N, D)
            print(f"Early Termination: {infos}, {achieved_ts}")
            solved = False
            for i in range(len(infos)):
                info = infos[i]
                achieved_t = achieved_ts[i]
                if info == "Achieved":
                    print("Achieved!")
                    solved = True
                    terminal_ts = achieved_t
                    solved_plan = plans[:terminal_ts, i]
                    break

            # print("============ Early Termination End ============")
            early_termination_end_time = time.time()
            early_termination_time.append(
                early_termination_end_time - early_termination_start_time
            )

            if self.viz_plans:
                self.visualize_node_value_plans(
                    search_num,
                    values,
                    [info["name"] for info in expanded_node_infos.values()],
                    expanded_node_plan_hists[-1],
                    value_estimation_plan_hists[-1],
                    start,
                    goal,
                )

            search_num += 1
            p_search_num += len(expanded_node_candidates)

            if (self.early_stopping_condition == "solved" and solved) or (
                self.early_stopping_condition == "achieved" and achieved
            ):
                break

        # goal_normalized is obs-only; make_bundle pads action dims with zeros
        # so it matches the full-bundle channel dim of the plan tensors.
        goal_bundle = self.make_bundle(obs=goal_normalized)  # (1, obs+act)
        if solved:
            # Get the final state the agent planned
            final_planned_state = solved_plan[-1]

            # Replace NaNs in the goal bundle with the agent's final planned state
            patched_goal_bundle = torch.where(
                torch.isnan(goal_bundle), final_planned_state, goal_bundle
            )

            output_plan = torch.cat(
                [solved_plan[:, None], patched_goal_bundle[None]], dim=0
            )[
                None
            ]  # (1, t, 1, c)
        else:
            if len(achieved_plans) != 0:
                print(f"Achieved plans: {len(achieved_plans)}")
                max_value = -1
                max_plan = None
                for plan, value in achieved_plans:
                    assert value >= 0, f"The value is negative: {value}"
                    if value > max_value:
                        max_value = value
                        max_plan = plan
                output_plan = torch.cat([max_plan[:, None], goal_bundle[None]], dim=0)[
                    None
                ]  # (1, t, 1, c)
            elif len(not_reached_plans) != 0:
                max_value = -1
                max_plan = None
                for plan, value in not_reached_plans:
                    assert value >= 0, f"The value is negative: {value}"
                    if value > max_value:
                        max_value = value
                        max_plan = plan
                output_plan = max_plan[None, :, None]  # (1, t, 1, c)
            else:
                print("Failed to find the plan")
                output_plan = torch.cat([obs_normalized[None]] * horizon, dim=0)[
                    None
                ]  # (1, t, 1, c) failed to find the plan

        self.log(f"validation/search_num", search_num)
        self.log(f"validation/p_search_num", p_search_num)

        self.log(f"validation_time/selection_time", np.sum(selection_time))
        self.log(f"validation_time/expansion_time", np.sum(expansion_time))
        self.log(f"validation_time/simulation_time", np.sum(simulation_time))
        self.log(f"validation_time/backprop_time", np.sum(backprop_time))
        self.log(
            f"validation_time/early_termination_time", np.sum(early_termination_time)
        )

        self.log(
            f"validation_time/simul_noiselevel_zero_padding_time",
            np.sum(simul_noiselevel_zero_padding_time),
        )
        self.log(
            f"validation_time/simul_value_estimation_time",
            np.sum(simul_value_estimation_time),
        )
        self.log(
            f"validation_time/simul_value_calculation_time",
            np.sum(simul_value_calculation_time),
        )
        self.log(
            f"validation_time/simul_node_allocation_time",
            np.sum(simul_node_allocation_time),
        )

        if self.mctd_pairwise_divergence_threshold is not None:
            self._mctd_last_pairwise_divergence_snapshot = {
                "mctd_pairwise_divergence_threshold": float(
                    self.mctd_pairwise_divergence_threshold
                ),
                "pairwise_first_divergence_timesteps": self._mctd_pairwise_first_divergence_timesteps,
                "search_num": int(search_num),
                "p_search_num": int(p_search_num),
                "global_step": getattr(self, "global_step", None),
            }
        else:
            self._mctd_last_pairwise_divergence_snapshot = None

        return solved, output_plan
