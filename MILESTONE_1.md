# Milestone 1: Object Detection — Waymo → Fine-tuned YOLO → ONNX → TensorRT → Triton

Multi-class (vehicle / pedestrian / cyclist) object detector, fine-tuned on Waymo Open Dataset
camera imagery, exported to ONNX, compiled to a TensorRT engine, and served concurrently-capable
via NVIDIA Triton Inference Server — fed by a streaming client that reads directly from Waymo
`.tfrecord` segments. Every stage runs on GPU where it matters; zero TensorFlow or
`waymo-open-dataset` dependency anywhere in the pipeline (`waymo_lib.py` is numpy + stdlib only).

**Hardware:** NVIDIA RTX 5060 Laptop GPU (Blackwell, 8GB VRAM, CUDA compute capability 12.0).

## Pipeline

```
.tfrecord segments (waymo_lib.py)
        |
        v
data/prepare_detection_dataset.py  ->  YOLO-format dataset (images + normalized boxes)
        |
        v
models/detection/train.py          ->  fine-tuned YOLO26n (best.pt)
        |
        v
models/detection/export.py         ->  best.onnx (FP32) + best_fp16.onnx, both numerically
        |                               validated against the PyTorch model
        v
serving/build_engine.sh            ->  best.plan (FP16 TensorRT engine, built on GPU inside
        |                               the same NVIDIA container that serves it)
        v
serving/triton_repo/ + run_triton.sh -> Triton Inference Server, GPU instance
        |
        v
serving/client.py                  ->  streams tfrecord frames -> Triton -> boxes ->
                                        self-contained HTML viewer
```

## Dataset

- **Source:** 6 segments pulled from `gs://waymo_open_dataset_v_1_4_2/individual_files/validation/`
  via `gcloud storage cp`.
- **Labels:** native 2D boxes (`Frame.camera_labels(field=8)`) — genuine human-drawn boxes, not
  the looser boxes derived from projected 3D points.
- **Classes:** VEHICLE, PEDESTRIAN, CYCLIST (Waymo's SIGN class never appears in native 2D labels).
- **Cameras:** all 5 (FRONT/FRONT_LEFT/FRONT_RIGHT/SIDE_LEFT/SIDE_RIGHT), each image normalized
  against its own actual resolution (FRONT-family is 1920x1280, SIDE is 1920x886).
- **Split:** at the *segment* level, not a random per-image shuffle — frames within one segment
  are ~0.1s apart, so a random split would leak near-duplicate scenes between train and val. One
  full segment is held out entirely for validation and is also what the streaming client (step 8)
  runs against, so the demo is on genuinely unseen frames.
- **Final counts:** train 4,955 images / 60,790 boxes (vehicle 40,102 / pedestrian 19,524 /
  cyclist 1,164) before oversampling; val 990 images / 9,936 boxes (vehicle 8,506 / pedestrian
  1,083 / cyclist 347).
- **Cyclist oversampling:** images containing a cyclist were duplicated 3x on disk
  (`data/oversample_rare_classes.py`) to increase per-batch sampling frequency; train grew to
  6,195 images / 3,492 cyclist instances.

## Fine-tuning

Model: `yolo26n.pt` (Ultralytics YOLO26, NMS-free/end-to-end architecture — no NMS plugin needed
anywhere downstream in the ONNX/TensorRT graph).

```
python3 train.py --epochs 60 --imgsz 1280 --batch 12 --patience 20 --name waymo_detect_final
```

Early-stopped at epoch 22 (best epoch 9, `patience=20`).

| class      | precision | recall | mAP50  | mAP50-95 |
|------------|-----------|--------|--------|----------|
| all        | 0.441     | 0.292  | 0.331  | 0.181    |
| vehicle    | 0.670     | 0.438  | 0.503  | 0.266    |
| pedestrian | 0.583     | 0.433  | 0.457  | 0.267    |
| cyclist    | 0.068     | 0.006  | 0.033  | 0.010    |

Cyclist is intentionally left as the weak class here rather than hidden — with ~1,200 base
instances (vs 40k vehicle) it's a genuine long-tail problem; see "Debugging notes" below for
what actually moved its needle and what didn't.

**Not a from-scratch benchmark to compare against COCO leaderboards** — this measures whether the
mechanism (fine-tune -> export -> serve) works and whether the model demonstrably learns all
three classes on real, held-out driving footage. It does.

## ONNX export + validation

```
python3 export.py                # FP32 -> best.onnx
python3 export.py --half         # FP16 -> best_fp16.onnx
```

Validated by comparing raw model output tensors between PyTorch and ONNX Runtime (CPU) on 20
held-out sample images, using IoU + class matched detection comparison (not raw tensor-row
diffing — see note below) at confidence > 0.1:

| export | max box-coord diff | max confidence diff | tolerance |
|--------|--------------------|-----------------------|-----------|
| FP32   | 0.20 px            | 0.007                 | 1.0 px / 0.05 |
| FP16   | 0.56 px            | 0.021                 | 3.0 px / 0.08 |

Both pass comfortably. FP16 halves the graph size (10.3MB -> 5.0MB).

**Why IoU-matched comparison, not raw tensor diff:** the model outputs 300 detection slots per
image, of which ~280+ are near-zero-confidence padding. Sorting both backends' outputs by
confidence and diffing row-by-row initially produced apparent "1000px" mismatches — an artifact of
two backends breaking near-tied low-confidence ranks differently, not a real defect. The fix was
to threshold to confident detections, then match by (class, IoU) rather than assume positional
correspondence — real detections then agree to a fraction of a pixel.

## TensorRT engine

```
./build_engine.sh
```

Built via `trtexec` running *inside* `nvcr.io/nvidia/tritonserver:26.06-py3` — the exact same
container image that later serves it, so the TensorRT version that builds the engine always
matches the one that loads it.

**Note on precision flags:** this container ships TensorRT v11, which removed the old `--fp16`
builder flag entirely. ONNX graphs are now "strongly typed" — engine precision is read directly
from the graph's own tensor dtypes, not a separate builder switch. So the FP16 engine is built
from `best_fp16.onnx` (FP16 weights baked into the graph by `export.py --half`) with no precision
flag needed at all.

**Performance** (RTX 5060 Laptop GPU, FP16, 1280x1280 input, `trtexec` benchmark, 10s):

- Throughput: **395.5 inferences/sec**
- Latency: mean 2.53ms, median 2.49ms, p90 2.69ms, p99 3.65ms

## Triton serving

```
./run_triton.sh -d
```

`serving/triton_repo/vehicle_detector/` — `config.pbtxt` declares the exact FP16
`[1,3,1280,1280]` input / FP32 `[1,300,6]` output shapes read off the ONNX graph, `platform:
tensorrt_plan`, `instance_group: KIND_GPU`. Model loads and reports `READY` in ~1 second.

Verified: `GET /v2/health/ready` -> 200, `GET /v2/models/vehicle_detector` metadata matches the
engine's actual I/O, and one real gRPC inference round-trip through `tritonclient`.

## Streaming client

```
python3 client.py --limit 60
```

Reads frames directly from a `.tfrecord` via `waymo_lib` (no TensorFlow anywhere), letterbox-
preprocesses each image, sends it to Triton over gRPC, un-letterboxes the returned boxes back into
original image coordinates, and writes a self-contained HTML viewer (dark theme, matching the
project's existing `viewer.html`; play/pause, frame scrubbing, per-camera tabs).

**Preprocessing correctness:** since serving goes through the exported ONNX/TensorRT path only
(no live PyTorch at inference time), the client's letterbox implementation is a from-scratch
reimplementation of Ultralytics' `LetterBox` (`auto=False, scaleup=True, center=True,
padding_value=114`), verified against the real `predictor.preprocess()` output during development
— max difference 5.96e-08 (float32 rounding noise) — before the torch/ultralytics dependency was
dropped from the shipped client entirely.

**Run against the held-out validation segment** (never seen in training), 40 frames, FRONT camera:

- 268 detections above conf>0.25: 209 vehicle, 50 pedestrian, 9 cyclist
- Client round-trip latency (preprocess + network + Triton + postprocess): mean 18.5ms, p50
  16.9ms, p90 23.3ms, p99 39.2ms — the gap versus the 2.5ms GPU-only engine time above is Python
  client + gRPC + network overhead, not GPU cost.

Boxes were visually spot-checked against the source frames: vehicles parked along the curb and
pedestrians on the sidewalk are correctly localized and classified in frames the model never
trained on.

## Debugging notes (what actually moved the numbers)

- **torchvision/CUDA mismatch:** Ultralytics auto-installed a `torchvision` built against a
  different CUDA major version than our explicit `torch==...+cu128` install, causing a hard
  runtime error. Fixed with a `--force-reinstall --no-deps` pin — a plain reinstall wasn't enough
  since pip considered the mismatched version "already satisfied."
- **Learning rate:** Ultralytics' default `lr0=0.01` is tuned for training from scratch on
  COCO-scale data. Fine-tuning on this much smaller dataset with it produced noisy,
  non-converging training (best epoch 2 of 60). Fixed with `lr0=0.001` + cosine LR schedule.
- **Cyclist near-zero mAP** was the hardest problem here and took several rounds to root-cause:
  LR fix alone (0.0014), +4x oversampling (0.011), +disabled early stopping (0.011, no change) —
  all marginal. Low-confidence-threshold inference showed every cyclist prediction under 0.02
  confidence (no learned signal at all), and the largest cyclist ground-truth box in val turned
  out to be only ~75x166px in a 1920x1280 image. Root cause: training at `imgsz=640` shrank these
  already-tiny boxes below what the network could resolve. A 10-epoch test at `imgsz=1280` nearly
  doubled overall mAP and got cyclist to 0.065 — the decisive fix, later combined with a 6th,
  cyclist-rich segment and 3x oversampling for the final run.
- **TensorRT v11's strongly-typed ONNX requirement** (see above) meant the originally-planned
  `trtexec --fp16` flag simply doesn't exist anymore in this container version — precision now has
  to be baked into the ONNX export itself.

## Repository

Work lives on the `model-pipeline` branch, kept separate from the collaborator's
`feature/image-processing` branch. `data/datasets/`, `models/detection/runs/`, and `*.tfrecord`
are gitignored (generated/downloaded artifacts, not source).
