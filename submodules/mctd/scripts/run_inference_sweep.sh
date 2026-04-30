#!/usr/bin/env bash
# Run guided + MCTD inference across checkpoints (Docker) with stable layout and run_meta sidecars.
#
# Edit the arrays below, then run from the host:
#   bash scripts/run_inference_sweep.sh
#
# Resume (skip jobs that already finished):
#   RESUME=1 bash scripts/run_inference_sweep.sh
# Guided is skipped if guided/guidance_sweep_manifest.json exists (all scales finished).
# MCTD preset is skipped if mctd/inference_times.jsonl has NUM_SAMPLES lines.
# If a run stopped mid-job, delete that run's folder (or the partial mctd_* dir) before
# resuming — inference appends to jsonl and would duplicate samples otherwise.
#
# Layout per checkpoint stem under SWEEP_ROOT (repo: submodules/mctd/inference_obstacles/...):
#   <stem>/guided/scale_<tag>/...
#   <stem>/mctd_<slug>/mctd/...
#   <stem>/run_meta_guided.json
#   <stem>/mctd_<slug>/run_meta.json
#
# Requires: docker, image $DOCKER_IMAGE, repo mounted at /workspace.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCTD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- host paths (edit) ---
REPO_HOST="${REPO_HOST:-/home/amli/Documents/classes/16-832/Planning_wrapper}"
SWEEP_SUBDIR="${SWEEP_SUBDIR:-inference_obstacles/sweep_autoregressive}"
SWEEP_ROOT_CONTAINER="/workspace/submodules/mctd/${SWEEP_SUBDIR}"
SWEEP_ROOT_HOST="${MCTD_ROOT}/${SWEEP_SUBDIR}"

# --- docker ---
DOCKER_IMAGE="${DOCKER_IMAGE:-fmctd:0.1}"
GPU_DEVICE="${GPU_DEVICE:-1}"

# --- inference (edit) ---
SEED="${SEED:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
EXTRA_INFERENCE_FLAGS="${EXTRA_INFERENCE_FLAGS:---scheduling_matrix autoregressive}"
# Set RESUME=1 to skip checkpoint jobs that already completed (see header comment).
RESUME="${RESUME:-1}"
export SEED NUM_SAMPLES DOCKER_IMAGE

# Checkpoint paths *inside the container* (edit list)
CHECKPOINTS=(
  "/workspace/checkpoints/11-17-07/checkpoints/epoch=3181-step=175000.ckpt"
  # "/workspace/checkpoints/11-17-07/checkpoints/epoch=5454-step=300000.ckpt"
  # "/workspace/checkpoints/11-17-07/checkpoints/epoch=7272-step=400000.ckpt"
  # "/workspace/checkpoints/11-17-07/checkpoints/epoch=9090-step=500000.ckpt"
)

# One comma-separated list for guided (edit)
GUIDED_SCALES="${GUIDED_SCALES:-0.1,0.5,1,2,5,10}"

# Several MCTD presets (edit)
MCTD_PRESETS=(
  "0,0.1,0.5,1,2"
  "0,0.5,1,2,5"
  # "0,1,2,5,10"
)

slug_mctd_preset() {
  local s="$1"
  s="${s//,/_}"
  s="${s//./p}"
  s="${s//-/m}"
  printf '%s' "$s"
}

# Returns 0 if this guided sweep already finished (manifest written last in inference.py).
_guided_sweep_complete() {
  local host_out="$1"
  [[ -f "${host_out}/guided/guidance_sweep_manifest.json" ]]
}

# Returns 0 if MCTD wrote one timing line per sample (append-only; partial runs != complete).
_mctd_run_complete() {
  local host_mctd="$1"
  local n="$2"
  local j="${host_mctd}/mctd/inference_times.jsonl"
  [[ -f "$j" ]] || return 1
  local lines
  lines="$(wc -l <"$j" | tr -d '[:space:]')"
  [[ "$lines" == "$n" ]]
}

run_in_docker() {
  local inner_cmd="$1"
  docker run --rm --gpus "device=${GPU_DEVICE}" \
    -e WANDB_API_KEY \
    -v "${REPO_HOST}:/workspace" \
    -w /workspace/submodules/mctd \
    "${DOCKER_IMAGE}" bash -lc "${inner_cmd}"
}

mkdir -p "${SWEEP_ROOT_HOST}"

for ckpt in "${CHECKPOINTS[@]}"; do
  stem=$(basename "$ckpt" .ckpt)
  out_base="${SWEEP_ROOT_CONTAINER}/${stem}"
  host_out_base="${SWEEP_ROOT_HOST}/${stem}"
  mkdir -p "${host_out_base}"

  echo "=== checkpoint ${stem} ==="

  if [[ "${RESUME}" == "1" ]] && _guided_sweep_complete "${host_out_base}"; then
    echo "  (skip guided: already complete — guided/guidance_sweep_manifest.json present)"
  else
    guided_cmd="python3 scripts/inference.py \
      --checkpoint $(printf '%q' "$ckpt") \
      --mode guided \
      --guidance_scales $(printf '%q' "$GUIDED_SCALES") \
      --num_samples ${NUM_SAMPLES} \
      --seed ${SEED} \
      --output_dir $(printf '%q' "$out_base") \
      ${EXTRA_INFERENCE_FLAGS}"
    run_in_docker "$guided_cmd"
  fi

  export SWEEP_META_CKPT="$ckpt"
  export SWEEP_META_GUIDED_SCALES="$GUIDED_SCALES"
  export SWEEP_META_OUT_DIR="$out_base"
  export SWEEP_META_HOST_JSON="${host_out_base}/run_meta_guided.json"
  python3 <<'PY'
import json, os, pathlib
path = pathlib.Path(os.environ["SWEEP_META_HOST_JSON"])
path.write_text(
    json.dumps(
        {
            "checkpoint": os.environ["SWEEP_META_CKPT"],
            "mode": "guided",
            "guidance_scales": os.environ["SWEEP_META_GUIDED_SCALES"],
            "seed": int(os.environ["SEED"]),
            "num_samples": int(os.environ["NUM_SAMPLES"]),
            "output_dir": os.environ["SWEEP_META_OUT_DIR"],
            "docker_image": os.environ["DOCKER_IMAGE"],
        },
        indent=2,
    ),
    encoding="utf-8",
)
PY

  for preset in "${MCTD_PRESETS[@]}"; do
    slug=$(slug_mctd_preset "$preset")
    mctd_out="${out_base}/mctd_${slug}"
    host_mctd="${host_out_base}/mctd_${slug}"
    mkdir -p "${host_mctd}"

    echo "--- MCTD preset ${preset} -> mctd_${slug} ---"

    if [[ "${RESUME}" == "1" ]] && _mctd_run_complete "${host_mctd}" "${NUM_SAMPLES}"; then
      echo "  (skip MCTD: already complete — ${NUM_SAMPLES} lines in mctd/inference_times.jsonl)"
    else
      mctd_cmd="python3 scripts/inference.py \
        --checkpoint $(printf '%q' "$ckpt") \
        --mode mctd \
        --mctd_guidance_scales $(printf '%q' "$preset") \
        --num_samples ${NUM_SAMPLES} \
        --seed ${SEED} \
        --output_dir $(printf '%q' "$mctd_out") \
        ${EXTRA_INFERENCE_FLAGS}"
      run_in_docker "$mctd_cmd"
    fi

    export SWEEP_META_PRESET="$preset"
    export SWEEP_META_MCTD_OUT="$mctd_out"
    export SWEEP_META_HOST_MCTD_JSON="${host_mctd}/run_meta.json"
    python3 <<'PY'
import json, os, pathlib
path = pathlib.Path(os.environ["SWEEP_META_HOST_MCTD_JSON"])
path.write_text(
    json.dumps(
        {
            "checkpoint": os.environ["SWEEP_META_CKPT"],
            "mode": "mctd",
            "mctd_guidance_scales": os.environ["SWEEP_META_PRESET"],
            "seed": int(os.environ["SEED"]),
            "num_samples": int(os.environ["NUM_SAMPLES"]),
            "output_dir": os.environ["SWEEP_META_MCTD_OUT"],
            "docker_image": os.environ["DOCKER_IMAGE"],
        },
        indent=2,
    ),
    encoding="utf-8",
)
PY
  done
done

echo "Sweep finished. Aggregate with:"
echo "  python3 ${MCTD_ROOT}/scripts/aggregate_inference_sweep.py --sweep_root ${SWEEP_ROOT_HOST}"
