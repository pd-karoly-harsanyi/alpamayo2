#!/usr/bin/env bash
# Runs ON a Nebius Ray worker, with the uploaded --working-dir (this repo) as cwd.
# Submitted by scripts/submit_nebius.py; do not run on a laptop.
#
#   bash scripts/nebius_entrypoint.sh probe              # env build + weights only, no inference
#   bash scripts/nebius_entrypoint.sh smoke [args...]    # python -m pd_alpamayo_eval.smoke args...
#   bash scripts/nebius_entrypoint.sh run --export s3://pd-nebius-ray/alpamayo-eval/NAME  # run_export
#   bash scripts/nebius_entrypoint.sh exec  cmd [args]   # arbitrary command inside the model env
#
# Layout on the node:
#   $PERSIST = /mnt/pd-general/data/alpamayo-eval   shared filesystem "pd-general" (240 GB, mounted on
#              the H100 workers; moved here from pd-fm-training-data on 2026-10-26 at the OI team's
#              request). Only what is expensive to refetch lives here: the 67 GB weights and the HF
#              cache (+ token). Never stage data here; exports and results go through the bucket.
#   $SCRATCH = /tmp/alpamayo-eval             per-pod SSD (900 GB): venv, uv/pixi caches, staged
#              exports (20 s to re-stage 1 GB from the Nebius bucket), run outputs (uploaded to
#              the bucket at the end). Lost when the pod is recycled, which is fine.
set -euo pipefail

MODE="${1:-smoke}"
shift || true

PERSIST="${ALPAMAYO_EVAL_PERSIST:-/mnt/pd-general/data/alpamayo-eval}"
PERSIST_MOUNT="$(dirname "$PERSIST")"
SCRATCH="${ALPAMAYO_EVAL_SCRATCH:-/tmp/alpamayo-eval}"
RUN_ID="${ALPAMAYO_EVAL_RUN_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)}"
MODEL_REPO="nvidia/Alpamayo2-Super"
MODEL_DIR="$PERSIST/models/alpamayo2-super"
OUT_DIR="$SCRATCH/runs/$RUN_ID"
# torch 2.8.0 from PyPI is a CUDA 12.8 / C++11-ABI build, so this is the matching prebuilt wheel.
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

export PIXI_CACHE_DIR="$SCRATCH/pixi-cache"
export UV_CACHE_DIR="$SCRATCH/uv-cache"
export UV_PYTHON_PREFERENCE=only-system  # use pixi's python; never download a managed one
export UV_PROJECT_ENVIRONMENT="$SCRATCH/venv"
export HF_HOME="$PERSIST/hf-cache"
export MPLCONFIGDIR="$SCRATCH/mpl"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
now() { date +%s; }

log "== probe: run_id=$RUN_ID mode=$MODE host=$(hostname) =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || log "nvidia-smi unavailable"
for tool in pixi uv nvcc git python3; do
  printf '  %-8s %s\n' "$tool" "$(command -v "$tool" 2>/dev/null || echo MISSING)"
done
if [ ! -d "$PERSIST_MOUNT" ]; then
  log "FATAL: $PERSIST_MOUNT is not mounted on this pod"
  exit 2
fi
echo "  uid: $(id)"
ls -ld "$PERSIST_MOUNT" "$PERSIST_MOUNT"/* 2>/dev/null | sed 's/^/  /'
mkdir -p "$PERSIST" "$SCRATCH" "$HF_HOME" "$OUT_DIR"
df -h /tmp "$PERSIST_MOUNT" | sed 's/^/  /'

# HF token precedence. The pod env already carries a cluster HF_TOKEN that can read the model
# but is NOT on the PhysicalAI-AV dataset's access list, and huggingface_hub prefers HF_TOKEN over
# the token file. So: a token forwarded by `submit_nebius.py --forward-hf-token` (HF_TOKEN plus
# the ALPAMAYO_EVAL_TOKEN_FORWARDED=1 marker, or an explicit ALPAMAYO_EVAL_HF_TOKEN via --env) is
# persisted once, overwriting whatever is there, and HF_TOKEN is always unset so the persisted
# file governs every run.
FORWARDED_TOKEN="${ALPAMAYO_EVAL_HF_TOKEN:-}"
if [ -z "$FORWARDED_TOKEN" ] && [ "${ALPAMAYO_EVAL_TOKEN_FORWARDED:-0}" = 1 ]; then
  FORWARDED_TOKEN="${HF_TOKEN:-}"
fi
if [ -n "$FORWARDED_TOKEN" ]; then
  (umask 077 && printf '%s' "$FORWARDED_TOKEN" > "$HF_HOME/token")
  log "stored forwarded HF token at $HF_HOME/token"
elif [ ! -s "$HF_HOME/token" ] && [ -n "${HF_TOKEN:-}" ]; then
  (umask 077 && printf '%s' "$HF_TOKEN" > "$HF_HOME/token")
  log "stored the pod's HF token at $HF_HOME/token (model access only)"
fi
unset HF_TOKEN ALPAMAYO_EVAL_HF_TOKEN ALPAMAYO_EVAL_TOKEN_FORWARDED FORWARDED_TOKEN

log "== env build =="
t_env=$(now)
pixi install --locked -e default
eval "$(pixi shell-hook -e default)"
log "python: $(python --version) uv: $(uv --version)"
uv sync --locked --no-install-package flash-attn
if ! "$UV_PROJECT_ENVIRONMENT/bin/python" -c 'import flash_attn' 2>/dev/null; then
  uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps "$FLASH_ATTN_WHEEL"
fi
# boto3 is not in NVIDIA's lock; the runner needs it to stage exports from / upload results to the bucket.
if ! "$UV_PROJECT_ENVIRONMENT/bin/python" -c 'import boto3' 2>/dev/null; then
  uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" boto3
fi
log "env build took $(( $(now) - t_env )) s"
# shellcheck disable=SC1091
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
python - <<'PY'
import flash_attn, torch, transformers
print(f"  torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
print(f"  transformers {transformers.__version__} flash_attn {flash_attn.__version__}")
PY

log "== weights =="
t_w=$(now)
if [ -f "$MODEL_DIR/config.json" ] && ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
  log "weights present at $MODEL_DIR"
elif [ ! -s "$HF_HOME/token" ]; then
  # Both the weights and the PhysicalAI-AV dataset are HF-gated. The token reaches the node once
  # via `submit_nebius.py --forward-hf-token`; until then only the env build can be verified.
  log "no HF token at $HF_HOME/token and no weights at $MODEL_DIR"
  if [ "$MODE" = probe ]; then
    log "probe complete (env only; weights not downloaded)"
    exit 0
  fi
  log "FATAL: smoke needs the weights; rerun once with --forward-hf-token"
  exit 3
else
  hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
  log "weights download took $(( $(now) - t_w )) s"
fi
du -sh "$MODEL_DIR" | sed 's/^/  /'

# Exports live in the Nebius bucket (s3://pd-nebius-ray/alpamayo-eval/<name>); the runner stages
# them onto the pod's local SSD (reused while the pod lives). Data never rides in the Ray package.
export ALPAMAYO_EVAL_EXPORTS="$SCRATCH/exports"
mkdir -p "$ALPAMAYO_EVAL_EXPORTS"
if [ -n "${AWS_ENDPOINT_URL_S3:-}" ]; then
  log "object store: $AWS_ENDPOINT_URL_S3 (key ${AWS_ACCESS_KEY_ID:0:4}...)"
else
  log "no object store credentials in this job (s3:// exports and result upload unavailable)"
fi

case "$MODE" in
  probe)
    log "probe complete"
    ;;
  smoke)
    log "== smoke =="
    python -m pd_alpamayo_eval.smoke --model-id "$MODEL_DIR" --out-dir "$OUT_DIR" "$@"
    ;;
  run)
    log "== run: $* =="
    python -m pd_alpamayo_eval.run_export --model-id "$MODEL_DIR" --out-dir "$OUT_DIR" --run-id "$RUN_ID" "$@"
    ;;
  exec)
    log "== exec: $* =="
    "$@"
    ;;
  *)
    log "unknown mode: $MODE"
    exit 64
    ;;
esac
