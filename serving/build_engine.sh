#!/usr/bin/env bash
# Build a TensorRT engine from the exported ONNX detector, using the *same*
# NVIDIA container image that will later serve it via Triton -- this
# guarantees the TensorRT version that builds the engine exactly matches the
# one that loads it, avoiding a common version-mismatch failure at serve time.
#
#   ./build_engine.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="$HERE/../models/detection/runs/waymo_detect_final/weights"
IMAGE="nvcr.io/nvidia/tritonserver:26.06-py3"
ONNX_FILE="best_fp16.onnx"
ENGINE_FILE="best.plan"

echo "== Pre-flight: confirming GPU is visible inside the container =="
docker run --rm --gpus all "$IMAGE" nvidia-smi

echo
echo "== Building TensorRT engine (FP16, GPU) =="
# This container's TensorRT (v11) removed the old --fp16 builder flag: ONNX
# graphs are now "strongly typed", so engine precision is read directly from
# the graph's own tensor dtypes, not a separate builder switch. That's why we
# point this at best_fp16.onnx (exported via `export.py --half`, itself
# numerically validated against the FP32 PyTorch model) rather than best.onnx.
# --memPoolSize=workspace:6144: let the builder use up to 6GB of the 8GB VRAM
#   to search a wider space of GPU kernels/tactics during optimization --
#   the more workspace TensorRT gets, the better (faster) kernels it can pick.
# --avgRuns/--duration: get a stable, quotable latency/throughput number out
#   of the same build, instead of a separate benchmarking step.
docker run --rm --gpus all \
  -v "$WEIGHTS_DIR:/workspace" \
  -w /workspace \
  "$IMAGE" \
  /usr/src/tensorrt/bin/trtexec \
    --onnx="$ONNX_FILE" \
    --saveEngine="$ENGINE_FILE" \
    --memPoolSize=workspace:6144 \
    --avgRuns=100 \
    --duration=10 \
  | tee "$HERE/build_engine.log"

echo
echo "== Done =="
ls -la "$WEIGHTS_DIR/$ENGINE_FILE"
