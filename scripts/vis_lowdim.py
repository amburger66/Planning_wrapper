#!/usr/bin/env python3
"""
scripts/vis_lowdim.py

Visualize a low-dim NPZ trajectory (from make_lowdim_dataset.py) as 3D
point clouds.

State layout (18-dim):
  [0:3]   tcp  position
  [3:9]   tcp  6D rotation  (first two cols of rotation matrix)
  [9:12]  block position
  [12:18] block 6D rotation

Action layout (9-dim):
  [0:3]   next-step tcp position   (absolute, world frame)
  [3:9]   next-step tcp 6D rotation

Usage
-----
    python scripts/vis_lowdim.py --npz datasets/push_lowdim.npz --gif out.gif
    python scripts/vis_lowdim.py --npz datasets/push_lowdim.npz --render
    python scripts/vis_lowdim.py --npz datasets/push_lowdim.npz --render --lookahead 10
    python scripts/vis_lowdim.py --npz datasets/push_lowdim.npz --env v2 --render

Controls (--render)
-------------------
    SPACE        pause / resume
    LEFT / RIGHT step one frame (when paused)
    Q / ESC      quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Geometry constants
# ─────────────────────────────────────────────────────────────────────────────

BLOCK_HALF = 0.025

_T_box1_hw = 0.10 / 2
_T_box1_hh = 0.025 / 2
_T_com_y   = 0.0375 / 2
_T_half_t  = 0.02
T_BOXES = [
    (np.array([0.0, -_T_com_y, 0.0]),               np.array([_T_box1_hw, _T_box1_hh, _T_half_t])),
    (np.array([0.0, 4*_T_box1_hh - _T_com_y, 0.0]), np.array([_T_box1_hh, (3/4)*_T_box1_hw, _T_half_t])),
]

STICK_RADIUS           = 0.008
STICK_LENGTH           = 0.10
STICK_OFFSET_Z_FROM_TCP = -0.05   # stick center is 0.05 m behind TCP in local z

# High-contrast colors for point cloud visualization
BLOCK_COLOR = np.array([0.0, 0.35, 1.0])   # bright blue
HAND_COLOR  = np.array([1.0, 0.2, 0.0])     # bright red-orange
GHOST_COLOR = np.array([0.0, 1.0, 0.4])     # bright lime green


# ─────────────────────────────────────────────────────────────────────────────
# Rotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def rot6d_to_rotmat(r6: np.ndarray) -> np.ndarray:
    """
    (6,) 6D rotation → (3, 3) rotation matrix via Gram-Schmidt.
    r6 = [r1(3), r2(3)]  — first two columns of the rotation matrix.
    """
    r1 = r6[:3].astype(np.float64)
    r2 = r6[3:].astype(np.float64)
    a1 = r1 / np.linalg.norm(r1)
    a2 = r2 - np.dot(r2, a1) * a1
    a2 = a2 / np.linalg.norm(a2)
    a3 = np.cross(a1, a2)
    return np.stack([a1, a2, a3], axis=-1)   # columns are the axes


def apply_pose_9d(pts: np.ndarray, state_9d: np.ndarray) -> np.ndarray:
    """
    Transform local-frame point cloud using a 9D pose vector.
    state_9d: [pos(3), rot6d(6)]
    """
    pos = state_9d[:3].astype(np.float64)
    R   = rot6d_to_rotmat(state_9d[3:9])    # (3, 3)
    return (pts.astype(np.float64) @ R.T + pos).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Point cloud template samplers
# ─────────────────────────────────────────────────────────────────────────────

def _sample_box(offset: np.ndarray, half: np.ndarray, n: int,
                rng: np.random.Generator) -> np.ndarray:
    try:
        import fpsample
        use_fps = True
    except ImportError:
        use_fps = False

    n_init = n * 20
    hx, hy, hz = half
    areas  = np.array([4*hy*hz, 4*hx*hz, 4*hx*hy], dtype=np.float64)
    counts = np.round(areas / areas.sum() * n_init).astype(int)
    counts[-1] += n_init - counts.sum()
    pts = []
    if counts[0] > 0:
        s = rng.choice([-1.0, 1.0], counts[0])
        pts.append(np.stack([s*hx, rng.uniform(-hy,hy,counts[0]), rng.uniform(-hz,hz,counts[0])], 1))
    if counts[1] > 0:
        s = rng.choice([-1.0, 1.0], counts[1])
        pts.append(np.stack([rng.uniform(-hx,hx,counts[1]), s*hy, rng.uniform(-hz,hz,counts[1])], 1))
    if counts[2] > 0:
        s = rng.choice([-1.0, 1.0], counts[2])
        pts.append(np.stack([rng.uniform(-hx,hx,counts[2]), rng.uniform(-hy,hy,counts[2]), s*hz], 1))
    initial = np.concatenate(pts).astype(np.float32) + offset.astype(np.float32)
    if use_fps and len(initial) > n:
        return initial[fpsample.fps_sampling(initial, n)]
    return initial[rng.choice(len(initial), size=min(n, len(initial)), replace=False)]


def _sample_cylinder(offset_z: float, radius: float, length: float,
                     n: int, rng: np.random.Generator) -> np.ndarray:
    try:
        import fpsample
        use_fps = True
    except ImportError:
        use_fps = False

    n_init = n * 20
    th  = rng.uniform(0, 2*np.pi, n_init)
    z   = rng.uniform(-length/2, length/2, n_init) + offset_z
    pts = np.stack([radius*np.cos(th), radius*np.sin(th), z], 1).astype(np.float32)
    if use_fps and len(pts) > n:
        return pts[fpsample.fps_sampling(pts, n)]
    return pts[rng.choice(len(pts), size=min(n, len(pts)), replace=False)]


def build_templates(n_block: int, n_hand: int,
                    rng: np.random.Generator, use_T: bool) -> dict:
    if use_T:
        areas   = [2*(4*hy*hz+4*hx*hz+4*hx*hy) for _, (hx,hy,hz) in T_BOXES]
        total   = sum(areas)
        t_parts = [_sample_box(off, half, max(1, int(n_block*a/total)), rng)
                   for (off, half), a in zip(T_BOXES, areas)]
        block_tpl = np.concatenate(t_parts)
    else:
        block_tpl = _sample_box(np.zeros(3), np.full(3, BLOCK_HALF), n_block, rng)

    hand_tpl = _sample_cylinder(
        STICK_OFFSET_Z_FROM_TCP, STICK_RADIUS, STICK_LENGTH, n_hand, rng
    )
    return {"block": block_tpl, "hand": hand_tpl}


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_frame(states: np.ndarray, actions: np.ndarray,
                  step: int, future_step: int, templates: dict) -> dict:
    """
    State layout (18-dim):
      [0:9]   tcp  pose  (pos + 6D rot)
      [9:18]  block pose (pos + 6D rot)

    Action layout (9-dim):
      [0:9]   next-step tcp pose (pos + 6D rot)
    """
    s = states[step]
    tcp_9d   = s[0:9]
    block_9d = s[9:18]

    hand_now  = apply_pose_9d(templates["hand"],  tcp_9d)
    block_pts = apply_pose_9d(templates["block"], block_9d)

    # Ghost: use the stored absolute action (next TCP pose) as the ghost pose
    # This is exactly what the model will be predicting — better than just
    # indexing states[future_step] since actions ARE the prediction target.
    ghost_9d    = actions[min(step, len(actions)-1)]   # (9,) next-step tcp
    hand_future = apply_pose_9d(templates["hand"], ghost_9d)

    return {"hand_now": hand_now, "block": block_pts, "hand_future": hand_future}


# ─────────────────────────────────────────────────────────────────────────────
# Scene limits
# ─────────────────────────────────────────────────────────────────────────────

def _scene_limits(all_frames: list[dict]) -> list[tuple]:
    pts = np.concatenate(
        [f["hand_now"] for f in all_frames]
        + [f["block"]  for f in all_frames]
        + [f["hand_future"] for f in all_frames]
    )
    pad = 0.08
    c   = pts.mean(0)
    r   = max((pts.max(0) - pts.min(0)).max() / 2 + pad, 0.1)
    return [(float(c[i]-r), float(c[i]+r)) for i in range(3)]


# ─────────────────────────────────────────────────────────────────────────────
# GIF rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_frame_gif(frame: dict, step: int, elev: float, azim: float,
                     lims: list, dpi: int, ghost_alpha: float,
                     show_ghost: bool) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    ax  = fig.add_subplot(111, projection="3d")

    b = frame["block"]
    ax.scatter(b[:,0], b[:,1], b[:,2], s=2.0,
               c=[BLOCK_COLOR.tolist()], label="block",
               depthshade=True, alpha=1.0)

    h = frame["hand_now"]
    ax.scatter(h[:,0], h[:,1], h[:,2], s=3.5,
               c=[HAND_COLOR.tolist()], label="gripper (now)",
               depthshade=True, alpha=1.0)

    if show_ghost:
        g = frame["hand_future"]
        ax.scatter(g[:,0], g[:,1], g[:,2], s=3.5,
                   c=[GHOST_COLOR.tolist()], label="gripper (action target)",
                   depthshade=True, alpha=max(ghost_alpha, 0.6))

    ax.view_init(elev=elev, azim=azim)
    for i, (lo, hi) in enumerate(lims):
        [ax.set_xlim, ax.set_ylim, ax.set_zlim][i](lo, hi)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7)
    ax.set_zlabel("Z", fontsize=7); ax.tick_params(labelsize=6)
    ax.set_title(f"step {step}", fontsize=8)
    ax.legend(fontsize=7, loc="upper right", markerscale=4)
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    w, h_px = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = buf.reshape(h_px, w, 4).copy()
    plt.close(fig)
    return img


def save_gif(frames_img: list, path: str, fps: float) -> None:
    try:
        import imageio
        imageio.mimsave(path, frames_img, duration=1.0/fps, loop=0)
        return
    except ImportError:
        pass
    from PIL import Image
    pil = [Image.fromarray(f) for f in frames_img]
    pil[0].save(path, save_all=True, append_images=pil[1:],
                duration=int(1000/fps), loop=0)


# ─────────────────────────────────────────────────────────────────────────────
# Open3D interactive viewer
# ─────────────────────────────────────────────────────────────────────────────

def _make_pcd(pts: np.ndarray, color: np.ndarray):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pts), 1)))
    return pcd


def _update_pcd(pcd, pts: np.ndarray, color: np.ndarray) -> None:
    import open3d as o3d
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pts), 1)))


def run_viewer(all_frames: list[dict], fps: float, start_paused: bool,
               ghost_alpha: float, show_ghost: bool) -> None:
    import open3d as o3d

    BG = np.array([0.08, 0.08, 0.08])
    ghost_blended = ghost_alpha * GHOST_COLOR + (1 - ghost_alpha) * BG

    n   = len(all_frames)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Low-Dim Trajectory Viewer", width=900, height=900)
    opt = vis.get_render_option()
    opt.background_color = BG
    opt.point_size       = 4.0

    f0    = all_frames[0]
    geoms = {}
    geoms["block"]    = _make_pcd(f0["block"],    BLOCK_COLOR)
    geoms["hand_now"] = _make_pcd(f0["hand_now"], HAND_COLOR)
    vis.add_geometry(geoms["block"])
    vis.add_geometry(geoms["hand_now"])
    if show_ghost:
        geoms["hand_future"] = _make_pcd(f0["hand_future"], ghost_blended)
        vis.add_geometry(geoms["hand_future"])
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))

    state = {"frame": 0, "paused": start_paused, "quit": False, "last_t": time.time()}

    def _refresh(vis):
        f = all_frames[state["frame"]]
        _update_pcd(geoms["block"],    f["block"],    BLOCK_COLOR)
        _update_pcd(geoms["hand_now"], f["hand_now"], HAND_COLOR)
        vis.update_geometry(geoms["block"])
        vis.update_geometry(geoms["hand_now"])
        if show_ghost:
            _update_pcd(geoms["hand_future"], f["hand_future"], ghost_blended)
            vis.update_geometry(geoms["hand_future"])

    def on_space(vis):
        state["paused"] = not state["paused"]
        print("Paused" if state["paused"] else "Playing")

    def on_left(vis):
        if state["paused"]:
            state["frame"] = (state["frame"] - 1) % n
            _refresh(vis)

    def on_right(vis):
        if state["paused"]:
            state["frame"] = (state["frame"] + 1) % n
            _refresh(vis)

    def on_quit(vis):
        state["quit"] = True

    vis.register_key_callback(32,       on_space)
    vis.register_key_callback(263,      on_left)
    vis.register_key_callback(262,      on_right)
    vis.register_key_callback(ord("Q"), on_quit)
    vis.register_key_callback(27,       on_quit)

    frame_dt = 1.0 / fps
    print(f"\nLow-Dim Viewer: {n} frames @ {fps} fps")
    print(f"  Blue=block  Orange=gripper (now)  Green=gripper (action target)")
    print(f"  SPACE=pause/resume  ←→=step  Q/ESC=quit\n")

    while not state["quit"]:
        now = time.time()
        if not state["paused"] and (now - state["last_t"]) >= frame_dt:
            state["frame"]  = (state["frame"] + 1) % n
            state["last_t"] = now
            _refresh(vis)
        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize a low-dim NPZ trajectory as 3D point clouds."
    )
    p.add_argument("--npz",         required=True)
    p.add_argument("--traj_idx",    type=int,   default=0)
    p.add_argument("--env",         default="v1", choices=["v1", "v2"])
    p.add_argument("--every",       type=int,   default=5)
    p.add_argument("--ghost_alpha", type=float, default=0.35)
    p.add_argument("--n_block",     type=int,   default=400)
    p.add_argument("--n_hand",      type=int,   default=250)
    p.add_argument("--no_ghost",    action="store_true",
                   help="Hide the action-target ghost gripper")
    p.add_argument("--render",      action="store_true")
    p.add_argument("--paused",      action="store_true")
    p.add_argument("--gif",         default="lowdim_traj.gif")
    p.add_argument("--fps",         type=float, default=15.0)
    p.add_argument("--elev",        type=float, default=30.0)
    p.add_argument("--azim",        type=float, default=-60.0)
    p.add_argument("--dpi",         type=int,   default=100)
    return p.parse_args()


def main():
    args = parse_args()

    data    = np.load(args.npz)
    states  = data["states"]   # (N, T, 18)
    actions = data["actions"]  # (N, T,  9)
    N, T, _ = states.shape

    print(f"Loaded {args.npz}:  {N} trajectories, T={T}")
    print(f"  state_dim={states.shape[-1]}  action_dim={actions.shape[-1]}")

    if args.traj_idx >= N:
        raise ValueError(f"--traj_idx {args.traj_idx} out of range ({N} trajectories)")

    traj_states  = states[args.traj_idx]   # (T, 18)
    traj_actions = actions[args.traj_idx]  # (T,  9)
    steps        = list(range(0, T, args.every))
    show_ghost   = not args.no_ghost

    print(f"\nSanity check (traj {args.traj_idx}, step 0):")
    print(f"  tcp  pos    = {traj_states[0, 0:3]}")
    print(f"  tcp  rot6d  = {traj_states[0, 3:9]}")
    print(f"  block pos   = {traj_states[0, 9:12]}")
    print(f"  block rot6d = {traj_states[0, 12:18]}")
    print(f"  action pos  = {traj_actions[0, 0:3]}")
    print(f"  action rot6 = {traj_actions[0, 3:9]}")
    # 6D rot columns should each have norm ~1 and be ~orthogonal
    r1 = traj_states[0, 3:6];  r2 = traj_states[0, 6:9]
    print(f"  |r1|={np.linalg.norm(r1):.4f}  |r2|={np.linalg.norm(r2):.4f}  "
          f"r1·r2={np.dot(r1, r2):.4f}  (should be ~1, ~1, ~0)")
    print(f"\nVisualizing {len(steps)} frames (every={args.every})")

    rng       = np.random.default_rng(0)
    templates = build_templates(args.n_block, args.n_hand, rng, args.env == "v2")

    all_frames = [
        extract_frame(traj_states, traj_actions, s, s, templates)
        for s in steps
    ]

    if args.render:
        run_viewer(all_frames, fps=args.fps, start_paused=args.paused,
                   ghost_alpha=args.ghost_alpha, show_ghost=show_ghost)
    else:
        lims = _scene_limits(all_frames)
        frames_img = []
        for i, (frame, s) in enumerate(zip(all_frames, steps)):
            img = render_frame_gif(frame, s, args.elev, args.azim,
                                   lims, args.dpi, args.ghost_alpha, show_ghost)
            frames_img.append(img)
            if i % 20 == 0:
                print(f"  rendered {i}/{len(all_frames)} frames")
        print(f"Saving → {args.gif}")
        save_gif(frames_img, args.gif, args.fps)
        print(f"Done.")


if __name__ == "__main__":
    if "--render" in sys.argv:
        try:
            import open3d  # noqa: F401
        except ImportError:
            print("open3d required for --render:  pip install open3d")
            raise
    main()