#!/usr/bin/env python3
"""
scripts/merge_demos.py

Merge per-worker HDF5 trajectory files (from collect_demos_batch.py) into
a single HDF5 file with sequential traj_0, traj_1, ... keys.

Usage:
    python scripts/merge_demos.py --record_dir demos/PushBoundary/batch
    python scripts/merge_demos.py --record_dir demos/PushBoundary/batch \
                                  --out demos/PushBoundary/all_demos.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


def merge(record_dir: str, out_path: str) -> None:
    src_dir     = Path(record_dir)
    worker_dirs = sorted(src_dir.glob("worker_*"))

    if not worker_dirs:
        # Also handle case where record_dir contains h5 directly (single mode)
        h5_direct = sorted(src_dir.glob("*.h5"))
        if h5_direct:
            print(f"Found {len(h5_direct)} H5 file(s) directly in {src_dir}.")
            print("Nothing to merge — directory is not a batched-worker output.")
            return
        raise FileNotFoundError(
            f"No worker_* subdirectories found under '{src_dir}'."
        )

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    all_episodes = []
    demo_idx     = 0

    with h5py.File(out_file, "w") as dst:
        for worker_dir in worker_dirs:
            h5_candidates = list(worker_dir.glob("*.h5"))
            if not h5_candidates:
                print(f"  [skip] {worker_dir.name} — no .h5 file found")
                continue
            h5_path = h5_candidates[0]

            json_candidates = list(worker_dir.glob("*.json"))
            worker_episodes = []
            if json_candidates:
                with open(json_candidates[0]) as f:
                    worker_meta = json.load(f)
                worker_episodes = worker_meta.get("episodes", [])

            with h5py.File(h5_path, "r") as src:
                traj_keys = sorted(src.keys(), key=lambda k: int(k.split("_")[1]))
                for local_idx, traj_key in enumerate(traj_keys):
                    new_key = f"traj_{demo_idx}"
                    src.copy(traj_key, dst, name=new_key)

                    if local_idx < len(worker_episodes):
                        ep = dict(worker_episodes[local_idx])
                        ep["episode_id"]    = demo_idx
                        ep["source_worker"] = worker_dir.name
                        ep["source_traj"]   = traj_key
                        all_episodes.append(ep)

                    demo_idx += 1

            print(f"  {worker_dir.name}: {len(traj_keys)} demos  (total={demo_idx})")

    merged_json = {
        "env_info": {"source_dir": str(src_dir)},
        "total_demos": demo_idx,
        "episodes": all_episodes,
    }
    json_out = out_file.with_suffix(".json")
    with open(json_out, "w") as f:
        json.dump(merged_json, f, indent=2)

    print(f"\nMerged {demo_idx} demos  →  {out_file}")
    print(f"Metadata                →  {json_out}")


def parse_args():
    p = argparse.ArgumentParser(description="Merge per-worker HDF5 demo files.")
    p.add_argument("--record_dir", type=str, required=True,
                   help="Directory containing worker_000/, worker_001/, … subdirs.")
    p.add_argument("--out", type=str, default=None,
                   help="Output HDF5 path.  Defaults to <record_dir>/all_demos.h5.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out  = args.out or str(Path(args.record_dir) / "all_demos.h5")
    merge(args.record_dir, out)