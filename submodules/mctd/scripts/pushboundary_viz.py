from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PushBoundaryVizStyle:
    block_half: float = 0.025
    circle_radius: float = 0.025
    stick_radius: float = 0.01
    obstacle_radius: float = 0.018
    pad: float = 0.02

    # Match the prior "red axes / magenta block / yellow tcp" palette seen in saved GIFs.
    tcp_face: str = "#FFD54A"
    tcp_edge: str = "#F2B705"
    block_face: str = "#FF007A"
    block_edge: str = "#C4005B"
    obstacle_face: str = "#FF7A7A"
    obstacle_edge: str = "#FF0000"
    start_color: str = "#FFD54A"
    goal_color: str = "#FF007A"
    axes_red: str = "#FF0000"


def _ffill_time_1d(col: np.ndarray) -> np.ndarray:
    x = np.asarray(col, dtype=np.float64).copy().ravel()
    for i in range(1, len(x)):
        if not np.isfinite(x[i]):
            x[i] = x[i - 1]
    for i in range(len(x) - 2, -1, -1):
        if not np.isfinite(x[i]):
            x[i] = x[i + 1]
    if not np.isfinite(x).all():
        x[~np.isfinite(x)] = 0.0
    return x


def _preprocess_states(
    *,
    states: np.ndarray,
    tcp_xy_indices: tuple[int, int],
    block_xy_indices: tuple[int, int],
    block_yaw_indices: Optional[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], list[tuple[float, float]]]:
    min_dim = max(max(tcp_xy_indices), max(block_xy_indices)) + 1
    if block_yaw_indices is not None:
        min_dim = max(min_dim, max(block_yaw_indices) + 1)
    if states.ndim != 2 or states.shape[-1] < min_dim:
        raise ValueError(f"Expected states shaped (T,D) with D>={min_dim}, got {states.shape}")

    states_viz = np.array(states, dtype=np.float64, copy=True)
    fix_cols = set(tcp_xy_indices) | set(block_xy_indices)
    if block_yaw_indices is not None:
        fix_cols |= set(block_yaw_indices)
    for c in range(6, states_viz.shape[1]):
        fix_cols.add(c)
    for c in sorted(fix_cols):
        if c < states_viz.shape[1]:
            states_viz[:, c] = _ffill_time_1d(states_viz[:, c])

    tcp_xy = states_viz[:, list(tcp_xy_indices)]
    block_xy = states_viz[:, list(block_xy_indices)]

    if block_yaw_indices is not None:
        ci, si = block_yaw_indices
        block_yaws = np.arctan2(states_viz[:, si], states_viz[:, ci])
    else:
        block_yaws = None

    num_obstacles = (states_viz.shape[-1] - 6) // 2
    obstacles_xy: list[tuple[float, float]] = []
    for i in range(num_obstacles):
        ox = float(states_viz[0, 6 + 2 * i])
        oy = float(states_viz[0, 7 + 2 * i])
        if np.isfinite(ox) and np.isfinite(oy):
            obstacles_xy.append((ox, oy))

    return states_viz, tcp_xy, block_xy, block_yaws, obstacles_xy


def _compute_limits(
    *,
    tcp_xy: np.ndarray,
    block_xy: np.ndarray,
    obstacles_xy: list[tuple[float, float]],
    style: PushBoundaryVizStyle,
    start_marker: Optional[np.ndarray],
    goal_marker: Optional[np.ndarray],
) -> tuple[float, float, float, float]:
    pad = style.pad
    x_min = min(float(np.min(tcp_xy[:, 0])), float(np.min(block_xy[:, 0]))) - pad
    x_max = max(float(np.max(tcp_xy[:, 0])), float(np.max(block_xy[:, 0]))) + pad
    y_min = min(float(np.min(tcp_xy[:, 1])), float(np.min(block_xy[:, 1]))) - pad
    y_max = max(float(np.max(tcp_xy[:, 1])), float(np.max(block_xy[:, 1]))) + pad

    if start_marker is not None and np.all(np.isfinite(start_marker)):
        x_min = min(x_min, float(start_marker[0]) - pad)
        x_max = max(x_max, float(start_marker[0]) + pad)
        y_min = min(y_min, float(start_marker[1]) - pad)
        y_max = max(y_max, float(start_marker[1]) + pad)
    if goal_marker is not None and np.all(np.isfinite(goal_marker)):
        x_min = min(x_min, float(goal_marker[0]) - pad)
        x_max = max(x_max, float(goal_marker[0]) + pad)
        y_min = min(y_min, float(goal_marker[1]) - pad)
        y_max = max(y_max, float(goal_marker[1]) + pad)

    for ox, oy in obstacles_xy:
        x_min = min(x_min, ox - style.obstacle_radius - pad)
        x_max = max(x_max, ox + style.obstacle_radius + pad)
        y_min = min(y_min, oy - style.obstacle_radius - pad)
        y_max = max(y_max, oy + style.obstacle_radius + pad)

    if (
        not all(np.isfinite(v) for v in (x_min, x_max, y_min, y_max))
        or x_min >= x_max
        or y_min >= y_max
    ):
        return -0.35, 0.35, -0.35, 0.35
    return x_min, x_max, y_min, y_max


def render_pushboundary_plan_static_overlay(
    *,
    states: np.ndarray,
    out_path: Path,
    tcp_xy_indices: tuple[int, int] = (0, 1),
    block_xy_indices: tuple[int, int] = (2, 3),
    block_yaw_indices: Optional[tuple[int, int]] = (4, 5),
    start_marker: Optional[np.ndarray] = None,
    goal_marker: Optional[np.ndarray] = None,
    block_shape: str = "square",
    opacity_min: float = 0.05,
    opacity_max: float = 0.95,
    gamma: float = 1.0,
    step_stride: int = 5,
    max_steps: Optional[int] = None,
    dpi: int = 200,
    style: PushBoundaryVizStyle = PushBoundaryVizStyle(),
) -> Path:
    """
    Render a single-image PushBoundary overlay plot.

    Overlays TCP + block poses for all (subsampled) timesteps with opacity increasing by time.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, Polygon, Rectangle
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "matplotlib is required for static visualization. Install it (e.g. `pip install matplotlib`) "
            "and re-run."
        ) from e

    _, tcp_xy, block_xy, block_yaws, obstacles_xy = _preprocess_states(
        states=states,
        tcp_xy_indices=tcp_xy_indices,
        block_xy_indices=block_xy_indices,
        block_yaw_indices=block_yaw_indices,
    )

    T = int(states.shape[0])
    if T <= 0:
        raise ValueError("states has zero timesteps")

    if step_stride is None or int(step_stride) <= 0:
        raise ValueError(f"step_stride must be >= 1, got {step_stride}")
    step_stride = int(step_stride)

    step_indices = np.arange(0, T, step_stride, dtype=int)
    if step_indices.size == 0 or step_indices[-1] != (T - 1):
        step_indices = np.unique(np.concatenate([step_indices, np.array([T - 1], dtype=int)]))

    if max_steps is not None and int(max_steps) > 0 and int(max_steps) < int(step_indices.size):
        step_indices = np.linspace(
            int(step_indices[0]),
            int(step_indices[-1]),
            num=int(max_steps),
            dtype=int,
        )

    if start_marker is None:
        start_marker = np.array([block_xy[0, 0], block_xy[0, 1]], dtype=np.float64)

    x_min, x_max, y_min, y_max = _compute_limits(
        tcp_xy=tcp_xy,
        block_xy=block_xy,
        obstacles_xy=obstacles_xy,
        style=style,
        start_marker=start_marker,
        goal_marker=goal_marker,
    )

    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)

    for ox, oy in obstacles_xy:
        ax.add_patch(
            Circle(
                (ox, oy),
                radius=style.obstacle_radius,
                facecolor=style.obstacle_face,
                edgecolor=style.obstacle_edge,
                linewidth=1,
                alpha=0.8,
                zorder=2,
            )
        )

    denom = max(1, int(len(step_indices) - 1))
    for j, t in enumerate(step_indices):
        frac = float(j) / float(denom)
        a = float(opacity_min + (opacity_max - opacity_min) * (frac**gamma))
        a = float(np.clip(a, 0.0, 1.0))

        tcp_now = tcp_xy[t]
        block_now = block_xy[t]
        ax.add_patch(
            Circle(
                (float(tcp_now[0]), float(tcp_now[1])),
                radius=style.stick_radius,
                facecolor=style.tcp_face,
                edgecolor=style.tcp_edge,
                linewidth=1,
                alpha=a,
                zorder=4,
            )
        )

        if block_shape == "circle":
            ax.add_patch(
                Circle(
                    (float(block_now[0]), float(block_now[1])),
                    radius=style.circle_radius,
                    facecolor=style.block_face,
                    edgecolor=style.block_edge,
                    linewidth=1,
                    alpha=a,
                    zorder=3,
                )
            )
        elif block_yaws is not None:
            yaw = float(block_yaws[t])
            c, s_val = float(np.cos(yaw)), float(np.sin(yaw))
            h = style.block_half
            local = [(-h, -h), (h, -h), (h, h), (-h, h)]
            corners = [
                (float(block_now[0] + c * lx - s_val * ly), float(block_now[1] + s_val * lx + c * ly))
                for lx, ly in local
            ]
            ax.add_patch(
                Polygon(
                    corners,
                    closed=True,
                    facecolor=style.block_face,
                    edgecolor=style.block_edge,
                    linewidth=1,
                    alpha=a,
                    zorder=3,
                )
            )
        else:
            ax.add_patch(
                Rectangle(
                    (float(block_now[0] - style.block_half), float(block_now[1] - style.block_half)),
                    width=2 * style.block_half,
                    height=2 * style.block_half,
                    facecolor=style.block_face,
                    edgecolor=style.block_edge,
                    linewidth=1,
                    alpha=a,
                    zorder=3,
                )
            )

    if start_marker is not None and np.all(np.isfinite(start_marker)):
        ax.plot(
            float(start_marker[0]),
            float(start_marker[1]),
            marker="+",
            color=style.start_color,
            markersize=14,
            markeredgewidth=2,
            zorder=5,
        )
    if goal_marker is not None and np.all(np.isfinite(goal_marker)):
        ax.plot(
            float(goal_marker[0]),
            float(goal_marker[1]),
            marker="+",
            color=style.goal_color,
            markersize=14,
            markeredgewidth=2,
            zorder=5,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")

    block_legend_marker = "o" if block_shape == "circle" else "s"
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=style.tcp_face, markersize=8, label="tcp"),
        Line2D([0], [0], marker=block_legend_marker, color="w", markerfacecolor=style.block_face, markersize=8, label="block"),
    ]
    if len(obstacles_xy) > 0:
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=style.obstacle_face, markersize=8, label="obstacle")
        )
    if start_marker is not None and np.all(np.isfinite(start_marker)):
        legend_elements.append(Line2D([0], [0], marker="+", color=style.start_color, markersize=10, label="start"))
    if goal_marker is not None and np.all(np.isfinite(goal_marker)):
        legend_elements.append(Line2D([0], [0], marker="+", color=style.goal_color, markersize=10, label="goal"))

    ax.legend(handles=legend_elements, loc="upper right", fontsize=7)
    ax.set_title("overlay", fontsize=8, color=style.axes_red)
    ax.set_xlabel("X", fontsize=7, color=style.axes_red)
    ax.set_ylabel("Y", fontsize=7, color=style.axes_red)
    ax.tick_params(axis="both", colors=style.axes_red, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(style.axes_red)
    fig.tight_layout(pad=0.3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi)
    plt.close(fig)
    return out_path

