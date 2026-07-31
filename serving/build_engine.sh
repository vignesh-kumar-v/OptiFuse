#!/usr/bin/env bash
# Build TensorRT engines from exported ONNX graphs, using the *same* NVIDIA
# container image that will later serve them via Triton -- this guarantees the
# TensorRT version that builds an engine exactly matches the one that loads it,
# avoiding a common version-mismatch failure at serve time.
#
#   ./build_engine.sh                 # all three models
#   ./build_engine.sh detector        # just one
#   ./build_engine.sh segmentation depth
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="nvcr.io/nvidia/tritonserver:26.06-py3"

# model -> "<dir containing the onnx> <onnx filename> <engine filename>"
declare -A SPEC=(
  [detector]="$HERE/../models/detection/runs/waymo_detect_final/weights best_fp16.onnx best.plan"
  [segmentation]="$HERE/../models/segmentation/runs/seg_v1/best model_fp16.onnx model.plan"
  [depth]="$HERE/../models/depth/runs/depth_v1/best model_fp16.onnx model.plan"
)

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then MODELS=(detector segmentation depth); fi

echo "== Pre-flight: confirming GPU is visible inside the container =="
docker run --rm --gpus all "$IMAGE" nvidia-smi | sed -n '/NVIDIA-SMI/,/^$/p' | head -12

for M in "${MODELS[@]}"; do
  if [ -z "${SPEC[$M]+x}" ]; then
    echo "unknown model '$M'; expected one of: ${!SPEC[*]}" >&2
    exit 2
  fi
  read -r DIR ONNX ENGINE <<< "${SPEC[$M]}"
  if [ ! -f "$DIR/$ONNX" ]; then
    echo
    echo "!! skipping '$M': $DIR/$ONNX not found (run that model's export.py --half first)"
    continue
  fi

  echo
  echo "== Building TensorRT engine: $M  ($ONNX -> $ENGINE) =="
  # This container's TensorRT (v11) removed the old --fp16 builder flag: ONNX
  # graphs are now "strongly typed", so engine precision is read directly from
  # the graph's own tensor dtypes rather than a builder switch. Hence every
  # spec above points at an already-FP16 graph, each numerically validated
  # against its FP32 PyTorch original by the matching export.py.
  # --memPoolSize lets the builder search a wider space of kernels/tactics;
  # --avgRuns/--duration produce a stable latency number from the same run.
  docker run --rm --gpus all \
    -v "$DIR:/workspace" -w /workspace "$IMAGE" \
    /usr/src/tensorrt/bin/trtexec \
      --onnx="$ONNX" \
      --saveEngine="$ENGINE" \
      --memPoolSize=workspace:4096 \
      --avgRuns=50 \
      --duration=6 \
    > "$HERE/build_engine_$M.log" 2>&1 || {
        echo "   BUILD FAILED -- last lines of $HERE/build_engine_$M.log:"; tail -15 "$HERE/build_engine_$M.log"; exit 1; }

  echo "   $(ls -la "$DIR/$ENGINE" | awk '{print $5" bytes"}')"
  grep -E "Throughput:|^\[.*\] \[I\] Latency:" "$HERE/build_engine_$M.log" | tail -2 | sed 's/^/   /'
done

echo
echo "== Done =="
