#!/usr/bin/env bash
# Launch Triton, serving the TensorRT-engine detector on GPU.
#
#   ./run_triton.sh              # foreground
#   ./run_triton.sh -d            # detached (background), name: waymo-triton
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="nvcr.io/nvidia/tritonserver:26.06-py3"
REPO="$HERE/triton_repo"
NAME="waymo-triton"

DETACH=""
if [[ "${1:-}" == "-d" ]]; then
  DETACH="-d"
fi

docker run --rm $DETACH --gpus all \
  --name "$NAME" \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$REPO:/models" \
  "$IMAGE" \
  tritonserver --model-repository=/models
