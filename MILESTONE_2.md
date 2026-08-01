# Milestone 2: Multi-Model Perception Stack

Three perception models — object detection, drivable-area segmentation, and metric monocular
depth — fine-tuned on the Waymo Open Dataset, each exported to ONNX, compiled to an FP16 TensorRT
engine, and served **concurrently from a single NVIDIA Triton instance** behind one shared
streaming preprocessing pass.

**Hardware:** NVIDIA RTX 5060 Laptop GPU (Blackwell, 8GB, compute capability 12.0).

**Status:** all three models are built, served, and verified end-to-end at *prototype* data scale.
Production-scale training on GCP is the remaining step (quota now granted — see below).

## Architecture

```
                                  .tfrecord  (waymo_lib.py — numpy + stdlib only)
                                      |
                          waymo_extract.py  ->  tar shards
                          (jpg + boxes json + seg.png + depth.npz)
                                      |
                    +-----------------+-----------------+
                    |                 |                 |
              detection          segmentation         depth
              YOLO26n            SegFormer-B0     Depth Anything V2
                    |                 |                 |
                  ONNX              ONNX              ONNX      <- each numerically
                    |                 |                 |          validated vs PyTorch
              TensorRT FP16     TensorRT FP16     TensorRT FP16
                    |                 |                 |
                    +--------- ONE Triton instance -----+
                                      |
                        serving/preprocessing.py
                    (decode JPEG once -> 3 input tensors)
                                      |
                        serving/multimodel_client.py
                  (3 requests in flight, joined on frame id)
                                      |
                          synchronized 3-panel viewer
```

## Results

### Concurrent serving — the headline

All three engines resident in one Triton instance, **690MB of 8151MB VRAM total**.

Measured by `multimodel_client.py --benchmark`, dispatching identical frames serially vs
concurrently (median over 10 frames):

| | latency |
|---|---|
| serial (3 sequential requests) | 51.2 ms |
| **concurrent (3 in flight)** | **37.1 ms** |
| speedup | **1.38x** |

Per-model round trip: detector 20.9ms, segmentation 9.9ms, depth 17.2ms.

The gain is real but well under 3x, and it is worth being precise about why: a single GPU has one
pool of SMs, so three models genuinely contend for compute rather than running free in parallel.
The concurrent wall clock (37ms) lands near the slowest model plus contention, not near the sum
(48ms). Python and gRPC overhead account for the rest — the pure engine compute times below are
far lower.

### Per-engine compute (trtexec, FP16, isolated)

| model | engine | throughput | mean latency |
|---|---|---|---|
| detection (YOLO26n, 1280x1280) | 9.7 MB | 395.5 qps | 2.53 ms |
| segmentation (SegFormer-B0, 960x640) | 11.9 MB | 250.2 qps | 3.99 ms |
| depth (Depth Anything V2 Small, 966x644) | 55.5 MB | 65.0 qps | 15.37 ms |

### Segmentation — drivable area + lane markings

SegFormer-B0 (Cityscapes-pretrained) fine-tuned on Waymo panoptic labels collapsed to 4 classes.
5 labelled segments, **segment-level** 4-train / 1-val split, 8 epochs at ~88s/epoch.

| class | IoU | share of labelled pixels |
|---|---|---|
| background | 0.909 | 75.6% |
| road | **0.838** | 19.0% |
| sidewalk | 0.395 | 4.2% |
| marking | **0.026** | 1.2% |
| **mIoU** | **0.542** | |

Road segmentation is strong and visibly correct on real frames. **Marking is not solved** — 0.026
IoU means the model essentially does not find lane markings, and that is reported here rather than
hidden inside the mIoU. Contributing factors, in order of confidence: markings are thin structures
at ~1.2% of pixels; only 5 labelled segments exist locally; and the first real run wasted 3 of 8
epochs to a scheduler bug (below) that has since been fixed but not yet re-run at length.

`LANE_MARKER(21)` is merged into `ROAD_MARKER(22)` deliberately — measured on real frames, class 21
alone is 0.00–0.12% of pixels and absent from many frames entirely, which is unlearnable in
isolation.

**Two plausible explanations for the marking failure were tested and ruled out**, recorded here so
they are not re-investigated:

| hypothesis | measured | verdict |
|---|---|---|
| thin paint is destroyed downscaling 1920x1280 → 960x640 | 99.9% of marking pixels retained, 0/25 frames lose it entirely | ruled out |
| thin paint cannot be represented at SegFormer's 1/4-resolution decode head (240x160) | 98.1% retained, ~607 head cells per frame | ruled out |

Marking is therefore fully representable on the model's own output grid, so the cause is
optimization/data rather than geometry. The known contributor is the LR-schedule bug below, which
froze 3 of the 8 epochs at lr≈0; that fix has not yet been re-run at length.

### Depth — metric, supervised on sparse lidar

Depth Anything V2 (Metric, Outdoor) fine-tuned against lidar projected into the camera. Only
~0.1–1% of pixels carry a lidar return, so **every loss and metric is masked to valid pixels** —
computing them densely would average over ~99% zeros and yield a model that looks fine by the loss
and predicts nothing useful.

| | AbsRel | RMSE | δ<1.25 |
|---|---|---|---|
| pretrained baseline (zero-shot on Waymo) | 0.2778 | 10.30 m | 0.622 |
| **after fine-tuning** | **0.1705** | **8.50 m** | **0.739** |

Measured over 489,166 real lidar points. Reporting the pretrained baseline matters: it shows the
fine-tuning is contributing (-39% AbsRel) rather than the pretrained model doing all the work.

Best result was **epoch 1**, with validation degrading over epochs 2–4 while training loss kept
falling — overfitting on limited data, the same pattern the detector showed in Milestone 1.

Known artifact: sky is predicted at mid-range rather than far. This is a standard monocular-depth
failure on textureless, unbounded regions and is visible in the viewer.

### Fusion: distance to the object ahead

The three models are combined into a single perception output rather than three
independent overlays. For each detection, depth is sampled inside the box, and the camera
calibration converts (pixel, depth) into a position in the **vehicle's own frame** — forward
distance and lateral offset. Anything within ±1.8 m of centre counts as in-lane; the nearest such
object is the lead object, which is what an in-car display actually shows.

Geometry follows Waymo's camera convention (x = forward optical axis, y = left, z = up — not
OpenCV's z-forward). The depth model's training target is the x-component in camera frame, so a
prediction is already distance along the optical axis and back-projects directly:

```
y = -(u - c_u) * depth / f_u        then  p_vehicle = extrinsic @ [x, y, z, 1]
z = -(v - c_v) * depth / f_v
```

Depth per box is a low percentile of an **inner** patch: a box's outer margin usually contains
background (road behind a car, sky beside a pole), and including it biases the estimate long.

**Validated against 3D lidar labels** (`python3 serving/distance.py`), projecting ground-truth
boxes into the image and comparing to our estimate at those pixels — 34 vehicles, 3–70 m, ≥30
lidar returns each:

| metric | value |
|---|---|
| mean relative error | **5.5%** |
| median absolute error | **1.48 m** |
| within 15% relative | **88.2%** |
| mean signed error | −2.21 m (we slightly under-read, consistent with sampling the near face) |

Estimates are also temporally stable without any tracking or smoothing — a lead vehicle reads
66.5–67.3 m across nine consecutive frames.

A useful negative result: on the residential val segment the system reports **no lead object in any
frame**. That is correct, not a bug — every detection there sits 5–18 m laterally (cars parked along
both curbs, pedestrians on the sidewalk) and the road ahead is genuinely empty. Sign conventions
check out independently: objects right of frame centre get negative lateral offset, left positive.

### Detection quality by lighting condition

Measured across our segments — detections found at conf 0.20 vs ground-truth objects per frame:

| segment | condition | GT/frame | found | ratio |
|---|---|---|---|---|
| 10203656… | Day / phx | 28.6 | 10.0 | 0.35 |
| 10243601… | Day / sf | 13.4 | 9.2 | 0.69 |
| 10689101… | Dawn/Dusk / phx | 8.2 | 6.6 | 0.80 |
| 20946813… | Day / phx | 28.1 | 15.5 | 0.55 |
| 18024188… | **Night** / phx | 10.6 | **0.4** | **0.04** |
| 14300007… | **Night** / sf | 26.0 | **0.6** | **0.02** |

Daylight performance is moderate (0.35–0.80); night is a **~20x collapse**. It is not a threshold
artifact — at conf 0.10 the night segments still yield only 0.9–2.9 detections against 10–26
ground-truth objects, i.e. there is no signal to recover. The Milestone 1 detector was trained on
4 Day + 1 Dawn/Dusk segments and zero Night, and this is that gap measured directly.

### Night regression test

Running the full stack on a held-out **Night** segment (`14300007604205869133`, `time_of_day:
Night`) is the explicit regression test for the domain gap found in Milestone 1, and it produced a
clean contrast:

- **Segmentation generalizes well to night.** Road is accurately covered around parked cars and
  the yellow centre line is picked up (road 31.8% of pixels). Its training set *included* a Night
  segment.
- **Detection returns zero detections** on clearly visible parked cars. Its training set contained
  **zero** Night segments.

Same scene, same frame, same server. This is about as direct a demonstration as one could ask for
that the Milestone 1 detector's weakness is a data-coverage problem, not an architecture problem —
and it is why the production run below is stratified by condition.

## Data findings

`data/probe_segments.py` classifies GCS segments by reading only the first ~20MB via HTTP range
requests — **~40x faster** than streaming with `gcloud storage cat` (13s for 8 segments vs 67s
each), which makes scanning the whole dataset practical.

Two findings that change the production plan:

1. **Camera segmentation labels are concentrated in the validation split.** 19 of 202 validation
   segments carry them; **0 of the first 80 training segments** do. If segmentation were uniformly
   distributed at the validation rate, seeing 0 in 80 would happen ~0.05% of the time. Segmentation
   training therefore has to draw from the validation split, with a segment-level split inside it.
   Within a labelled segment, coverage is dense: ~50% of frames (every odd frame), all 5 cameras.
2. **Condition coverage of the validation split:** Day 160, Dawn/Dusk 23, Night 19, rain 1. Night
   data exists in useful quantity; Milestone 1 simply never sampled it. Rain is genuinely scarce.

## Bugs found and fixed

- **Depth ONNX had symbolic output dims.** `squeeze(1)` left the graph with
  `Squeezedepth_dim_0` instead of a fixed shape, which TensorRT will not bind to a static engine
  without an optimization profile. Reshaping to literal ints pins the output at `[1, 644, 966]`.
- **Segmentation LR schedule exhausted early.** A per-batch `OneCycleLR` was sized from a guessed
  200 steps/epoch when the real figure is 327, so the schedule ran out and froze training at
  lr≈0 — epochs 6, 7 and 8 of the first real run were byte-identical. Replaced with a per-epoch
  cosine schedule, which needs no step count (the dataset is an `IterableDataset` with no length).
- **Segmentation was missing ImageNet normalization.** The checkpoint's processor has
  `do_normalize=True`, but training fed plain 0-1 pixels, shifting the input distribution away
  from what the pretrained Cityscapes encoder expects. Both new models now bake normalization
  **into the ONNX graph**, so every engine takes plain 0-1 RGB and the constants cannot drift
  between trainer and client.
- **FP16 depth output broke the viewer.** `cv2.resize` has no float16 kernel; widened before resize.

## Engineering notes

- **One letterbox implementation.** `serving/preprocessing.py` owns it; `client.py` delegates.
  Verified at promotion time to still match Ultralytics' `predictor.preprocess()` to **5.96e-08**
  and to be bit-identical to the previous inline copy, and `client.py` reproduced its exact prior
  output (268 detections: 209/50/9) afterwards.
- **Segmentation emits a finished class map, not logits.** The graph bundles upsample + argmax, so
  a frame costs 614KB on the wire instead of 9.8MB of float32 logits — 16x less at streaming rates.
- **Task-appropriate validation.** Discrete class maps are checked as pixel agreement (worst
  99.956% FP16), depth in metres (worst 0.31m FP16), detections by IoU+class matching. A single
  generic float tolerance would be meaningless across all three.
- **Reused rather than rebuilt.** `waymo_extract.py` / `waymo_dataset.py` come from the
  collaborator's `feature/image-processing` branch and supply the tar-shard data layer for both new
  models. Their `_selfcheck()` gates run as part of this milestone's verification.
- **Visualization.** Class colours come from a validated categorical palette (all-pairs on the dark
  surface: worst CVD ΔE 8.4, normal-vision 19.8, all ≥3:1 contrast), and every class ships a swatch
  *and* a label so identity is never colour-alone. Depth uses a monotonic-lightness ramp with a
  metre scale legend — not JET, whose non-monotonic lightness invents banding that is not in the
  data.

## Remaining: production training on GCP

Everything above is prototype scale (5 segmentation segments, 11 depth segments). The limiting
factor is data, not the pipeline.

**GPU quota is granted** — `GPUS_ALL_REGIONS` went 0 → 1 on project `vl-waymo-2026` (NVIDIA L4
requested; regional L4 quota was already 1, the global cap was the actual blocker).

Planned:
1. GCS bucket for extracted shards; run extraction **on a GCP VM** — the Waymo bucket is public
   GCS, so same-cloud reads are fast and egress-free, avoiding a 100GB+ home download.
2. Segment selection **stratified by `Frame.stats`** (~50% Day / 25% Night / 25% Dawn-Dusk, mixed
   sf/phx, rain where available) — directly correcting the gap the night test exposes.
3. Detection trains from the **training** split and validates on the **validation** split; the
   detector backbone scales n → s (or m). Segmentation must draw from validation-split segments per
   the finding above, and that constraint gets stated wherever its numbers are quoted.
4. Re-export → rebuild engines → re-serve → re-run the confidence sweep, per-class IoU and depth
   metrics, and record whatever the real numbers turn out to be.

Cost control: spot instances, VMs shut down after each run, only trained weights retained.
