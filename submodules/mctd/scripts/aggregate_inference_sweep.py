#!/usr/bin/env python3
"""
Aggregate inference sweep outputs: NPZ outcome counts + timing summaries.

Expects a sweep root produced by run_inference_sweep.sh, e.g.:
  inference_obstacles/sweep_11-17-07/<ckpt_stem>/
    guided/scale_<tag>/plan_*_{success,failed_solved,failed}.npz
    inference_times_summary.json          (from guided job)
    run_meta_guided.json
    mctd_<slug>/
      mctd/plan_*.npz
      inference_times_summary.json
      run_meta.json

Usage (from submodules/mctd):
  python3 scripts/aggregate_inference_sweep.py --sweep_root inference_obstacles/sweep_11-17-07
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def _scale_tag_to_float(tag: str) -> float:
    """Inverse of inference.py _guidance_scale_dirname (p -> ., m -> -)."""
    return float(tag.replace("m", "-").replace("p", "."))


def _count_plan_npz(plan_dir: Path) -> tuple[int, int, int]:
    n_ok = n_fs = n_fail = 0
    if not plan_dir.is_dir():
        return 0, 0, 0
    for p in plan_dir.iterdir():
        if not p.is_file() or p.suffix != ".npz":
            continue
        name = p.name
        if re.match(r"plan_\d+_success\.npz$", name):
            n_ok += 1
        elif re.match(r"plan_\d+_failed_solved\.npz$", name):
            n_fs += 1
        elif re.match(r"plan_\d+_failed\.npz$", name):
            n_fail += 1
    return n_ok, n_fs, n_fail


def _load_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _mean_from_summary(
    summary: dict[str, Any] | None,
    *,
    guided_scale: float | None = None,
    mctd: bool = False,
) -> tuple[float | None, int | None]:
    if not summary:
        return None, None
    if mctd:
        block = summary.get("by_mode", {}).get("mctd")
        if not block:
            return None, None
        return float(block["mean_sec"]), int(block.get("count", 0))
    if guided_scale is not None:
        key = f"guided/guidance_scale={float(guided_scale)}"
        # JSON may serialize 10.0 vs 10 — try exact then common float string
        by = summary.get("by_mode_and_guidance_scale", {})
        if key in by:
            b = by[key]
            return float(b["mean_sec"]), int(b.get("count", 0))
        for k, b in by.items():
            if not k.startswith("guided/guidance_scale="):
                continue
            rest = k.split("=", 1)[1]
            try:
                if abs(float(rest) - float(guided_scale)) < 1e-6:
                    return float(b["mean_sec"]), int(b.get("count", 0))
            except ValueError:
                continue
    return None, None


def _read_run_meta(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def collect_rows(sweep_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not sweep_root.is_dir():
        return rows

    for ckpt_dir in sorted(sweep_root.iterdir()):
        if not ckpt_dir.is_dir():
            continue
        stem = ckpt_dir.name
        meta_g = _read_run_meta(ckpt_dir / "run_meta_guided.json")
        ckpt_path_guided = (meta_g or {}).get("checkpoint", "")

        guided_root = ckpt_dir / "guided"
        summary_guided = _load_summary(ckpt_dir / "inference_times_summary.json")

        if guided_root.is_dir():
            for scale_dir in sorted(guided_root.iterdir()):
                if not scale_dir.is_dir() or not scale_dir.name.startswith("scale_"):
                    continue
                tag = scale_dir.name[len("scale_") :]
                try:
                    g = _scale_tag_to_float(tag)
                except ValueError:
                    g = None
                n_ok, n_fs, n_fail = _count_plan_npz(scale_dir)
                mean_t, cnt = _mean_from_summary(
                    summary_guided, guided_scale=g
                )
                rows.append(
                    {
                        "checkpoint_stem": stem,
                        "checkpoint_path": ckpt_path_guided,
                        "run_kind": "guided",
                        "subgroup": scale_dir.name,
                        "guidance_scale": g if g is not None else "",
                        "mctd_guidance_scales": "",
                        "n_success": n_ok,
                        "n_failed_solved": n_fs,
                        "n_failed": n_fail,
                        "mean_time_sec": mean_t if mean_t is not None else "",
                        "time_record_count": cnt if cnt is not None else "",
                    }
                )

        for sub in sorted(ckpt_dir.iterdir()):
            if not sub.is_dir() or not sub.name.startswith("mctd_"):
                continue
            mctd_plans = sub / "mctd"
            meta = _read_run_meta(sub / "run_meta.json")
            preset = (meta or {}).get("mctd_guidance_scales", "")
            ckpt_path_m = (meta or {}).get("checkpoint", "") or ckpt_path_guided
            summary_m = _load_summary(sub / "inference_times_summary.json")
            n_ok, n_fs, n_fail = _count_plan_npz(mctd_plans)
            mean_t, cnt = _mean_from_summary(summary_m, mctd=True)
            rows.append(
                {
                    "checkpoint_stem": stem,
                    "checkpoint_path": ckpt_path_m,
                    "run_kind": "mctd",
                    "subgroup": sub.name,
                    "guidance_scale": "",
                    "mctd_guidance_scales": preset,
                    "n_success": n_ok,
                    "n_failed_solved": n_fs,
                    "n_failed": n_fail,
                    "mean_time_sec": mean_t if mean_t is not None else "",
                    "time_record_count": cnt if cnt is not None else "",
                }
            )

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sweep_root",
        type=Path,
        required=True,
        help="Sweep directory (contains one folder per checkpoint stem)",
    )
    ap.add_argument(
        "--csv_out",
        type=Path,
        default=None,
        help="Output CSV path (default: <sweep_root>/sweep_summary.csv)",
    )
    ap.add_argument(
        "--json_out",
        type=Path,
        default=None,
        help="Explicit JSON output path (overrides --also_json name)",
    )
    ap.add_argument(
        "--also_json",
        action="store_true",
        help="Also write <sweep_root>/sweep_summary.json",
    )
    args = ap.parse_args()
    sweep_root = args.sweep_root.resolve()
    csv_out = args.csv_out or (sweep_root / "sweep_summary.csv")

    rows = collect_rows(sweep_root)
    fieldnames = [
        "checkpoint_stem",
        "checkpoint_path",
        "run_kind",
        "subgroup",
        "guidance_scale",
        "mctd_guidance_scales",
        "n_success",
        "n_failed_solved",
        "n_failed",
        "mean_time_sec",
        "time_record_count",
    ]

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    json_path = args.json_out
    if json_path is None and args.also_json:
        json_path = sweep_root / "sweep_summary.json"
    if json_path is not None:
        json_path = json_path.resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps({"sweep_root": str(sweep_root), "rows": rows}, indent=2),
            encoding="utf-8",
        )

    print(f"Wrote {len(rows)} rows to {csv_out}")
    if json_path is not None:
        print(f"Wrote JSON to {json_path}")


if __name__ == "__main__":
    main()
