# Waymo Open Dataset Perception v1.4.3 — TFRecord Deep Dive

Everything below was **measured** from a real file in this repo, not recalled from docs. Decoded with
pure-Python protobuf wire parsing (stdlib + numpy), no TensorFlow, no `waymo-open-dataset` package.

Reference file:

```
waymo_perception/training/segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord
```

Schema source: `dataset.proto`, `label.proto`, `map.proto`, `segmentation.proto` from
`waymo-research/waymo-open-dataset` @ master.

---

## Table of contents

1. [What one file is](#1-what-one-file-is)
   - [1.1 Filename anatomy](#11-filename-anatomy)
   - [1.2 The run id](#12-the-run-id--10017090168044687777)
   - [1.3 The window](#13-the-window--6380_000_6400_000)
   - [1.4 Filename ↔ context.name](#14-filename--contextname)
   - [1.5 `_with_camera_labels`](#15-_with_camera_labels)
   - [1.6 The five id namespaces](#16-the-five-id-namespaces)
2. [Container layer: TFRecord](#2-container-layer-tfrecord)
3. [Frame proto — top level](#3-frame-proto--top-level)
4. [Context (field 1) — calibration block](#4-context-field-1--the-calibration-block)
5. [Coordinate frames](#5-coordinate-frames--four-of-them)
6. [Timing](#6-timing-field-2--per-image)
7. [Cameras (field 4)](#7-cameras-field-4)
8. [LiDAR (field 5)](#8-lidar-field-5--the-66-of-the-file)
9. [Range image → point cloud (verified)](#9-range-image--point-cloud-verified-recipe)
10. [Labels (fields 6, 7, 8, 9)](#10-labels)
11. [Segmentation](#11-segmentation-the-v14-bonus-most-people-miss)
12. [Map (field 10)](#12-map-field-10--frame-0-only)
13. [Segment-level statistics](#13-segment-level-all-198-frames)
14. [Gotchas ranked](#14-gotchas-ranked-by-how-much-theyll-cost-you)
15. [Storage math](#15-storage-math)
16. [Scripts](#16-scripts)

---

## 1. What one file is

One **segment** = one continuous 20 s driving run from one Waymo car.

| property | value |
|---|---|
| size | 1,113,225,482 B (1.04 GiB) |
| records | **198 frames** |
| rate | 100.01 ms mean Δt → **10.00 Hz** (min 99.79, max 100.17) |
| duration | 19.702 s |
| frame size | min 5.38 MB, max 6.18 MB, mean 5.62 MB |
| filename `10017090168044687777` | run id — uint64, see [1.2](#12-the-run-id--10017090168044687777) |
| filename `6380_000_6400_000` | window inside the parent drive, see [1.3](#13-the-window--6380_000_6400_000) |
| filename `_with_camera_labels` | file also carries native 2D camera boxes (field 8) |

20 files × ~1 GB = the 20 GB set. Each file is fully independent; nothing is shared across files
(object ids included).

### 1.1 Filename anatomy

```
segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord
        └────────┬─────────┘ └──┬───┘ └──┬───┘ └────────┬────────┘
             run id          start      end         has 2D labels
```

Everything below was checked against all 21 local files, not assumed.

### 1.2 The run id — `10017090168044687777`

An opaque **uint64 handle for one physical drive** (one car, one session). Not a timestamp, not
sequential, carries no meaning — purely a key.

```
range         1005081002024129653 .. 10868756386479184868
unique        21 / 21 files
all < 2^64    True
any > 2^63    True
```

> **Gotcha:** `10017090168044687777` exceeds 2^63 (9.22e18), so it **overflows int64**. Postgres
> `BIGINT`, Java `long`, `numpy.int64` and `struct`'s `q` all wrap or raise. Length also varies
> (19 or 20 digits). **Keep it a string.** Never parse it as a signed integer.

All 21 ids being distinct means no two local files come from the same drive — the dataset spreads
segments across many separate runs. The local set spans 2017-10 → 2019-05, locations
`location_sf` / `location_phx` / `location_other`, and time-of-day `Day` / `Night` / `Dawn/Dusk`
(that last value contains a slash — do not split on `/`).

### 1.3 The window — `6380_000_6400_000`

`6380.000 s → 6400.000 s`, **elapsed seconds inside the parent drive**, formatted
`<seconds>_<milliseconds>`. So this segment is minute 106 of a drive that ran at least 1 h 47 m.

The parent drive is **not in the dataset** — only these 20 s slices are. Offsets across the local
files range from 472 s to 7625 s, i.e. arbitrary cut points in long drives. The `_000` is not always
zero: `1372_870_1392_870` and `5313_150_5333_150` both occur, so cut points are not snapped to whole
seconds.

The nominal span is always exactly 20.000 s. The contents are not:

| run id | window | frames | measured |
|---|---|---|---|
| 10017090168044687777 | 6380.0→6400.0 | 198 | **19.702 s** |
| 10023947602400723454 | 1120.0→1140.0 | 199 | 19.800 s |
| 1005081002024129653 | 5313.15→5333.15 | 199 | 19.800 s |
| 10061305430875486848 | 1080.0→1100.0 | 198 | 19.693 s |
| 10072140764565668044 | 4060.0→4080.0 | 198 | 19.716 s |
| 10072231702153043603 | 5725.0→5745.0 | 198 | 19.700 s |
| 10075870402459732738 | 1060.0→1080.0 | 199 | 19.775 s |
| 10082223140073588526 | 6140.0→6160.0 | **197** | 19.600 s |

Never 200 frames, never exactly 20 s. **Do not hardcode 200** — count records.

### 1.4 Filename ↔ `context.name`

Verified on all 21 files: **`context.name` equals the filename minus the `segment-` prefix and the
`_with_camera_labels.tfrecord` suffix.**

```
file:         segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord
context.name:          10017090168044687777_6380_000_6400_000
```

So `context.name` is the join key that maps frames back to their segment without touching the
filesystem. Note it is the **full** string (run id *and* window) that identifies a segment — the run
id alone does not, since one drive can yield many segments.

### 1.5 `_with_camera_labels`

The file carries `Frame.camera_labels` (field 8) — human-drawn 2D boxes. Files without the suffix
still carry `projected_lidar_labels` (field 9, derived from the 3D boxes); they just lack native 2D
annotation. All 21 local files have the suffix.

### 1.6 The five id namespaces

Mixing these is a common bug. They are mutually unrelated:

| id | example | scope |
|---|---|---|
| segment / run id | `10017090168044687777` | the file; uint64 |
| 3D object id | `Ne2AaddGwrcxS2iKCX3xKw` | 22 chars; stable **within one segment only** |
| 2D camera label id | `12ba81ee-9cee-4134-94ca-df9ed01e0ff1` | UUID; **does not match the 3D id** |
| projected label id | `Ne2AaddGwrcxS2iKCX3xKw_FRONT` | 3D id + `_` + camera name |
| camera-seg `sequence_id` | `17766991336812920378` | also a 20-digit uint64 — **but not the segment id** |

That last row is the trap: in frame 23 of this segment, the segment id is `10017090168044687777`
while the panoptic label's `sequence_id` is `17766991336812920378`. Same shape, unrelated value.

Object ids are **not** valid across files. The same physical parked car appearing in two segments
receives two unrelated ids; the dataset contains no cross-segment identity of any kind.
See [section 10](#10-labels) for how the label id namespaces interact.

---

## 2. Container layer: TFRecord

Not a Waymo invention — TensorFlow framing. Per record, exactly:

```
uint64 length                      little endian
uint32 crc32c_masked(length_bytes)
byte   data[length]                <- serialized waymo.open_dataset.Frame proto
uint32 crc32c_masked(data)
```

masked crc = `((crc >> 15) | (crc << 17)) + 0xa282ead8`.

* Overhead: 16 B/record = **3,168 B total = 0.0003%** of the file.
* **No container-level compression.** Compression lives inside the payload (zlib per range image, JPEG per image).
* **No index, no random access.** Frame N requires walking N records. Cheap if you `seek` past
  payloads instead of reading them — 198 frames scanned in 0.8 s.

Reading needs no TensorFlow:

```python
import struct

def records(path):
    with open(path, "rb") as f:
        while (h := f.read(8)):
            n, = struct.unpack("<Q", h)
            f.read(4)                # length crc
            data = f.read(n)
            f.read(4)                # data crc
            yield data
```

---

## 3. Frame proto — top level

proto2 message `waymo.open_dataset.Frame`. Frame 0 measured byte budget:

| # | field | count | bytes | % of frame |
|---|---|---|---|---|
| 1 | `context` | 1 | 2,716 | 0.04% |
| 2 | `timestamp_micros` | 1 | 8 | — |
| 3 | `pose` (4×4) | 1 | 144 | — |
| 4 | `images` | 5 | 1.80 MB | **29.2%** |
| 5 | `lasers` | 5 | 4.06 MB | **65.8%** |
| 6 | `laser_labels` | 16 | ~4 KB | 0.1% |
| 7 | `no_label_zones` | 0 | 0 | — |
| 8 | `camera_labels` | 5 | tiny | ~0 |
| 9 | `projected_lidar_labels` | 5 | tiny | ~0 |
| 10 | `map_features` | 257 | 300 KB | 4.9% |
| 11 | `map_pose_offset` | 1 | ~26 | — |

**Field 10 appears on frame 0 only** — the map is identical for the whole segment and is stored once.
The other 197 frames carry 0 map features.

Field numbers 1000+ are reserved for third-party extensions.

---

## 4. Context (field 1) — the calibration block

`context.name = "10017090168044687777_6380_000_6400_000"`, identical in every frame → segment key.

Camera and laser calibration sub-blobs are **byte-identical across frames** (verified on 3 frames).
Read once, cache, skip thereafter.

### 4.1 `stats` — the trap

```
time_of_day = "Day"    location = "location_sf"    weather = "sunny"
laser_object_counts  {VEHICLE: 7, SIGN: 9}
camera_object_counts {VEHICLE: 7}
```

The proto comment claims these are "the number of unique objects with the type **in the segment**".
**That is false in this data.** Measured frame by frame:

```
frame 0: 16 labels   stats.laser {VEHICLE:7, SIGN:9}    stats.camera {VEHICLE:7}
frame 1: 16 labels   stats.laser {VEHICLE:7, SIGN:9}    stats.camera {VEHICLE:7}
frame 2: 17 labels   stats.laser {VEHICLE:7, SIGN:10}   stats.camera {VEHICLE:8}
frame 3: 17 labels   stats.laser {VEHICLE:7, SIGN:10}   stats.camera {VEHICLE:9}
frame 4: 19 labels   stats.laser {VEHICLE:9, SIGN:10}   stats.camera {VEHICLE:10}
frame 5: 19 labels   stats.laser {VEHICLE:9, SIGN:10}   stats.camera {VEHICLE:10}
```

They track the **per-frame** label count. The segment actually contains **82 unique object ids**.
Never use `context.stats` for dataset-level counting.

Consequence: the serialized `context` blob is *not* byte-identical across frames (the stats differ),
even though the calibrations inside it are.

### 4.2 `camera_calibrations` — 5 cameras

| name | enum | resolution | f_u = f_v | c_u, c_v | k1 | k2 | p1 | p2 | k3 |
|---|---|---|---|---|---|---|---|---|---|
| FRONT | 1 | 1920×1280 | 2059.61 | 952.41, 634.59 | 0.03545 | −0.33830 | 1.9e-5 | 7.1e-4 | 0 |
| FRONT_LEFT | 2 | 1920×1280 | 2046.63 | 975.06, 640.91 | 0.03050 | −0.31017 | 2.6e-3 | 8.9e-4 | 0 |
| FRONT_RIGHT | 3 | 1920×1280 | 2053.55 | 944.36, 630.65 | 0.03332 | −0.30189 | 8.6e-6 | −1.9e-4 | 0 |
| SIDE_LEFT | 4 | **1920×886** | 2050.25 | 970.31, 248.14 | 0.03347 | −0.33634 | 2.2e-4 | 2.5e-3 | 0 |
| SIDE_RIGHT | 5 | **1920×886** | 2053.62 | 970.55, 235.62 | 0.02637 | −0.28801 | −2.2e-5 | −8.3e-4 | 0 |

* `intrinsic` is a flat 9-element array `[f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3]` — **not** a 3×3
  matrix. Distortion follows the OpenCV convention (radial k1,k2,k3; tangential p1,p2).
* Side cameras are **886 px tall with principal point near y≈240** — cropped hard from the top.
  Never assume 1280.
* Enum values REAR_LEFT=6, REAR=7, REAR_RIGHT=8 exist but are **not present in v1.4.3 perception**.
  5 cameras only. Combined FOV ≈ 252°, **rear is blind**.
* `rolling_shutter_direction = LEFT_TO_RIGHT` (enum 2) on **all five**. The image is exposed
  column by column, not row by row.
* `extrinsic` = 4×4 row-major **camera → vehicle**. Measured translations (m, vehicle frame):

```
FRONT       t = [ 1.539, -0.024, 2.116]   yaw   -0.2°
FRONT_LEFT  t = [ 1.494,  0.092, 2.116]   yaw  +44.1°
FRONT_RIGHT t = [ 1.490, -0.094, 2.116]   yaw  -44.7°
SIDE_LEFT   t = [ 1.432,  0.111, 2.115]   yaw  +89.4°
SIDE_RIGHT  t = [ 1.428, -0.111, 2.116]   yaw  -90.1°
```

All roof-mounted at z ≈ 2.12 m.

* Camera frame axes: **x forward (optical axis), y left, z up** — *not* OpenCV's z-forward.
  Projection requires the axis swap (see §7).

### 4.3 `laser_calibrations` — 5 lidars

| name | enum | inc_min | inc_max | beam_inclinations | extrinsic t (m) | yaw |
|---|---|---|---|---|---|---|
| TOP | 1 | −0.31450 (−18.02°) | +0.03989 (+2.29°) | **64 explicit values** | [1.430, 0.000, 2.184] | +148.0° |
| FRONT | 2 | −1.5708 (−90°) | +0.5236 (+30°) | empty → uniform | [4.070, 0.000, 0.689] | −0.2° |
| SIDE_LEFT | 3 | −1.5708 | +0.5236 | empty | [3.245, +1.025, 0.979] | +89.2° |
| SIDE_RIGHT | 4 | −1.5708 | +0.5236 | empty | [3.245, −1.025, 0.979] | −91.5° |
| REAR | 5 | −1.5708 | +0.5236 | empty | [−1.155, 0.000, 0.464] | +179.2° |

* TOP beams are **non-uniform**: spacing varies 0.150° → 0.606°, dense near the horizon, sparse
  looking down. Values ascend −0.30926 → +0.03849 rad.
* **Range image row 0 = highest beam**, so `inclination_per_row = beam_inclinations[::-1]`.
* The 4 short-range lidars have empty `beam_inclinations` → uniform:
  `inc[row] = inc_min + (H - 1 - row + 0.5)/H * (inc_max - inc_min)`.
* TOP extrinsic yaw **+148°** is not a bug — it is the physical mount azimuth offset. Range image
  column 0 is *not* straight ahead. `az_correction` removes it (§9).

---

## 5. Coordinate frames — four of them

1. **Global / world** — ENU-like, meters, arbitrary per-run origin. Frame 0 vehicle sits at
   `x = −1257.18, y = 10546.04, z = 22.45`.
2. **Vehicle (SDC)** — **x forward, y left, z up**, origin on the ground under the rear axle.
   All 3D labels live here.
3. **Sensor** — one per lidar/camera, related to vehicle by `extrinsic` (sensor → vehicle).
4. **Image** — pixels, `(0,0)` = left edge of the first pixel.

`frame.pose` (field 3) = 4×4 row-major **vehicle → global**:

```
[  0.9483  -0.2350   0.2134   -1257.18 ]
[  0.2331   0.9719   0.0342   10546.04 ]
[ -0.2154   0.0173   0.9764      22.45 ]
[  0        0        0            1    ]
```

yaw 13.81°, **pitch 12.4°** — a real San Francisco hill. This is why reconstructed points reach
z = −17.7 m at 70 m range.

> **Critical mismatch, stated in the proto itself:** `frame.pose` does **not** correspond to
> `timestamp_micros`. The timestamp is the start of the top-lidar spin; the pose roughly corresponds
> to the *middle* of the frame. The 3D label boxes are defined in *this* pose's frame.

---

## 6. Timing (field 2 + per-image)

`timestamp_micros = 1550083467346370` → unix 1550083467.346370 (2019-02-13). int64 **microseconds**.
Defined as the **frame start = first TOP lidar scan**.

Every camera carries its own clock (float64 seconds since epoch). Frame 0:

| cam | trigger − frame_t0 | shutter | readout span | pose_timestamp |
|---|---|---|---|---|
| SIDE_LEFT | **−1.60 ms** | 9.992 ms | 54.19 ms | 1550083467.371079 |
| FRONT_LEFT | +10.97 ms | 9.992 ms | 54.17 ms | .381080 |
| FRONT | +23.26 ms | 9.992 ms | 54.25 ms | .401080 |
| FRONT_RIGHT | +35.61 ms | 9.992 ms | 54.23 ms | .411081 |
| SIDE_RIGHT | +48.20 ms | 9.992 ms | 54.15 ms | .421080 |

Cameras fire **sequentially**, ~12 ms apart, sweeping left → right to chase the lidar spin. Total
spread ≈ 50 ms = half a frame. At 6 m/s that is 30 cm of ego motion between SIDE_LEFT and SIDE_RIGHT.

> **Cameras are not synchronized.** Treating all five as one instant is the single largest source of
> silent projection error.

Per-image `pose` (field 3) is a **separate** 4×4 for that camera's own moment, and `velocity`
(field 4) allows extrapolation between them:

```
FRONT: v = (5.765, 1.371, -1.329) m/s      w = (-0.01002, -0.01492, +0.10547) rad/s
```

`v` is expressed in the **global** frame; `w` is the body angular rate.
`shutter` = per-column exposure time. `camera_readout_done_time − camera_trigger_time` = 54 ms =
exposure + full sensor readout.

---

## 7. Cameras (field 4)

`CameraImage` fields: `1 name`, `2 image` (JPEG bytes), `3 pose`, `4 velocity`, `5 pose_timestamp`,
`6 shutter`, `7 camera_trigger_time`, `8 camera_readout_done_time`, `10 camera_segmentation_label`.

Frame 0 JPEG payload sizes:

```
FRONT       412,699 B      FRONT_LEFT  389,779 B      FRONT_RIGHT 433,372 B
SIDE_LEFT   285,466 B      SIDE_RIGHT  281,027 B
```

Baseline JFIF (`ff d8 ff e0`), decodes to RGB 1920×1280 (verified with PIL).

> **Images appear in file order FRONT, FRONT_LEFT, SIDE_LEFT, FRONT_RIGHT, SIDE_RIGHT — not enum
> order.** Always index by `name`, never by list position.

### Projection vehicle → pixel

```python
pc = inv(extrinsic) @ p_vehicle            # vehicle -> camera frame
if pc.x <= 0: reject                       # behind camera (x is the optical axis)
xn, yn = -pc.y / pc.x, -pc.z / pc.x        # axis swap: cam is x-fwd / y-left / z-up
r2  = xn*xn + yn*yn
rad = 1 + k1*r2 + k2*r2**2 + k3*r2**3
xd  = xn*rad + 2*p1*xn*yn + p2*(r2 + 2*xn**2)
yd  = yn*rad + p1*(r2 + 2*yn**2) + 2*p2*xn*yn
u, v = f_u*xd + c_u, f_v*yd + c_v
```

**Validated against the dataset's own stored projections** (19,712 TOP lidar points → FRONT camera):

```
pixel error:  median 3.44    p90 12.26    max 31.8
```

The residual is exactly the rolling shutter + per-pixel ego motion deliberately not modelled above.

> Use the **stored** `camera_projection` channels (§8) whenever they exist — they are
> rolling-shutter-correct, a hand-rolled projection is not.

---

## 8. LiDAR (field 5) — the 66% of the file

`Laser` fields: `1 name`, `2 ri_return1`, `3 ri_return2`. Both returns present on all 5 sensors.

A range image is a **2D rasterization of one spin**: rows = beam pitch, columns = azimuth.

* TOP → **[64, 2650, 4]**
* all four short-range lidars → **[200, 600, 4]**

Decoding path for every `*_compressed` field:

```
zlib.decompress(blob) -> parse as MatrixFloat / MatrixInt32
  MatrixFloat  { repeated float data = 1 [packed]; MatrixShape shape = 2; }
  MatrixInt32  { repeated int32 data = 1 [packed]; MatrixShape shape = 2; }
  MatrixShape  { repeated int32 dims = 1; }
reshape row-major to dims
```

### 8.1 `range_image_compressed` (field 2) — [H, W, 4] MatrixFloat

| ch | meaning | measured on TOP return1 |
|---|---|---|
| 0 | range, meters | 153,830 / 169,600 valid (**90.7%**), 1.99 → 73.88 m |
| 1 | intensity | 0.0005 → 29.50, mean 0.125 (unbounded, long tail — clip before training) |
| 2 | elongation | 0.0000 → 1.493, mean 0.089 (pulse stretch; high = fog / foliage / grazing incidence) |
| 3 | in no-label-zone | all **−1.0** here (consistent with 0 NLZs). `1` = inside an NLZ |

> **Empty pixels are `−1.0`** — not 0, not NaN. Valid mask is `range > 0`.

### 8.2 Per-sensor fill and compression, frame 0

| sensor | ret | shape | zlib B | raw B | ratio | valid | max range |
|---|---|---|---|---|---|---|---|
| TOP | 1 | 64×2650×4 | 952,474 | 2,713,600 | 2.8× | 90.7% | 73.88 m |
| TOP | 2 | 64×2650×4 | 109,437 | 2,713,600 | 24.8× | 6.7% | 74.71 m |
| FRONT | 1 | 200×600×4 | 36,942 | 1,920,000 | 52× | 3.2% | 17.76 m |
| FRONT | 2 | 200×600×4 | 3,447 | 1,920,000 | 557× | 0.03% | 14.35 m |
| SIDE_LEFT | 1 | 200×600×4 | 35,697 | 1,920,000 | 54× | 3.2% | 18.93 m |
| SIDE_LEFT | 2 | 200×600×4 | 4,559 | 1,920,000 | 421× | 0.1% | 19.97 m |
| SIDE_RIGHT | 1 | 200×600×4 | 44,808 | 1,920,000 | 43× | 4.0% | 19.45 m |
| SIDE_RIGHT | 2 | 200×600×4 | 4,868 | 1,920,000 | 394× | 0.1% | 12.83 m |
| REAR | 1 | 200×600×4 | 47,532 | 1,920,000 | 40× | 4.4% | 19.38 m |
| REAR | 2 | 200×600×4 | 8,741 | 1,920,000 | 220× | 0.4% | 19.37 m |

The short-range lidars are ~97% empty (their ±90°/+30° vertical FOV mostly hits nothing) and are
capped around 20 m. TOP is the real sensor.

### 8.3 `camera_projection_compressed` (field 3) — [H, W, 6] MatrixInt32

Channels: `[cam1_name, x1, y1, cam2_name, x2, y2]`, value 0 = no projection.
A point may hit multiple cameras; the first two are kept in the priority order
FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT.

TOP return1, frame 0:

```
no projection 70,003    FRONT 19,712    FRONT_LEFT 20,022    FRONT_RIGHT 19,180
SIDE_LEFT     20,340    SIDE_RIGHT 20,343
points with a 2nd projection: 6,566
```

**41% of lidar points hit no camera at all** (rear, plus above/below camera FOV). This field is a
free, exact, rolling-shutter-correct lidar↔pixel association — do not recompute it.

### 8.4 `range_image_pose_compressed` (field 4) — [H, W, 6] MatrixFloat

**TOP return1 only.** Channels `[roll, pitch, yaw, x, y, z]` = a full vehicle → global pose **for
every single pixel**, because the spin takes 100 ms and the car moves throughout.

* 3-2-1 Euler convention: rotating navigation → vehicle is yaw, then pitch, then roll about z, y, x.
  Right-hand rule, positive counter-clockwise.
* Zeros on invalid pixels.
* **2,464,588 B = 40% of the entire frame.** The single largest item in the file.
* Return 2 has no pose field — the proto states it is identical to return 1.

Reconstruction of the 4×4 from `[r, p, y, x, y, z]`:

```python
R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
T = [[R, t], [0, 0, 0, 1]]
```

### 8.5 `range_image_flow_compressed` (5) and `segmentation_label_compressed` (6)

* Field 5 (scene flow, `[vx, vy, vz, class]`, class −1 = unlabeled) is **absent** in this segment.
* Field 6 (lidar panoptic) **is present** on 30 frames — see §11.

---

## 9. Range image → point cloud (verified recipe)

```python
inclination   = beam_inclinations[::-1]              # row 0 = top beam
az_correction = atan2(extrinsic[1,0], extrinsic[0,0])
ratios        = (W - col - 0.5) / W                  # DESCENDING. col 0 -> ~1.0
azimuth       = (ratios*2 - 1)*pi - az_correction

x = r*cos(incl)*cos(az)
y = r*cos(incl)*sin(az)
z = r*sin(incl)

p = R_extrinsic @ p + t_extrinsic     # -> vehicle frame at that pixel's time
p = P_pixel     @ p                   # -> global        (TOP only, per-pixel pose)
p = inv(frame.pose) @ p               # -> frame vehicle frame
keep where r > 0
```

> The **descending column order** is the classic trap. Ascending `col` mirrors the whole cloud and
> the result still *looks* plausible. Measured cost of getting it wrong: **950 px** median
> reprojection error vs **3.44 px** when correct.

Frame 0 output:

```
TOP        ret1 153,830    ret2  11,340
FRONT      ret1   3,844    ret2      40
SIDE_LEFT  ret1   3,894    ret2     131
SIDE_RIGHT ret1   4,794    ret2     117
REAR       ret1   5,244    ret2     446
return1 total 171,606      both returns 183,680
bounds  x[-71.4, 71.8]   y[-9.7, 23.2]   z[-17.7, 4.4]
```

### 9.1 Ground-truth verification

Counted reconstructed points inside each labeled 3D box and compared to the dataset's own
`num_lidar_points_in_box`:

| pipeline | exact matches | per-box error |
|---|---|---|
| pose-compensated, **both returns** | **7 / 16** | `[-6,0,-15,0,-3,-1,-10,-1,0,0,0,0,-1,0,-1,-5]` — always ≤ 0, ≤ 5% |
| pose-compensated, return 1 only | 5 / 16 | much worse on large boxes |
| **no** per-pixel pose compensation | 3 / 16 | errors to −41 |

Box padding was tested and makes it **worse** (0.05 m → 4/16, 0.1 m → 3/16), so the gap is not a
tolerance. Three conclusions:

1. **Per-pixel pose compensation is mandatory.** Skipping it costs up to 41 points on one box (−30%).
2. `num_lidar_points_in_box` counts **both returns**, all 5 lidars.
3. The reconstruction is never exactly equal, always slightly under. Cause: the range image is a
   **grid**; several raw laser shots can land in one `(row, col)` cell and only one survives.
   Waymo's counter ran on pre-rasterization points. **The range image is lossy with respect to the
   raw sensor** — the label counts are the evidence.

---

## 10. Labels

### 10.1 `laser_labels` (field 6) — 3D, vehicle frame

Frame 0: 16 labels (7 VEHICLE, 9 SIGN). Fields present across those 16: `1,2,3,4,5,6,7,11,12,13`.
Note `5`/`6` (difficulty) appear on only **1 of 16** — absent means default LEVEL_1.

```
id=0QXtLAoMcF26x6k0m-7gVQ  VEHICLE  c=(+31.96,-2.35,+3.81)  lwh=(6.14,2.37,3.20)
                           hdg=-0.214  npts=309/top309  mostvis=FRONT  v=(-0.00,+0.00,-0.03)
```

**`Label.Box` field numbers are out of order** — a classic hand-parsing bug:

| field # | name | meaning |
|---|---|---|
| 1, 2, 3 | `center_x/y/z` | box **center** (not ground contact) |
| **4** | `width` | y extent |
| **5** | `length` | x extent |
| **6** | `height` | z extent |
| 7 | `heading` | radians, ∈ [−π, π), rotation of +x onto the box front-face normal |

Other `Label` fields:

* `type` (3): 1 VEHICLE, 2 PEDESTRIAN, 3 SIGN, 4 CYCLIST.
  **No PEDESTRIAN or CYCLIST anywhere in this segment.**
* `id` (4): 22-char base64-ish string, **stable across frames** → free tracking ground truth.
* `detection_difficulty_level` (5), `tracking_difficulty_level` (6): LEVEL_1 / LEVEL_2.
* `num_lidar_points_in_box` (7), `num_top_lidar_points_in_box` (13). Occasionally `top > all`
  (e.g. 210 vs 208) — the two were computed by separate passes and are mildly inconsistent.
  Zero-point boxes exist (distant signs).
* `metadata` (2): `speed_x/y/z` (1,2,5) and `accel_x/y/z` (3,4,6) — ground-truth object velocity,
  no frame differencing needed.
* `most_visible_camera_name` (11): e.g. `FRONT`, or the **empty string** when the object is behind
  the car (rear camera gap).
* `camera_synced_box` (12): present on all 16. The box shifted to the moment the most-visible camera
  actually captured the object center, correcting for rolling shutter *and* object motion.
  **This is the ground truth for the camera-only 3D challenge.** Use field 12 for camera work,
  field 1 for lidar work.
* `laser_keypoints` (8) / `camera_keypoints` (9) / `association` (10): absent in this segment.

### 10.2 `camera_labels` (field 8) — native 2D

Human-drawn on the images, **independent of lidar**. Frame 0: FRONT 5, FRONT_LEFT 2, others 0
(all VEHICLE).

The same `Box` message is reused for 2D: `center_x/y` = pixel center, `length` = width in px,
`width` = height in px, `heading` = 0. Always axis-aligned.

IDs are **UUIDs**: `12ba81ee-9cee-4134-94ca-df9ed01e0ff1` — a completely different namespace from
the 3D ids.

> **2D and 3D boxes do not correspond.** Only pedestrian labels ever carry
> `association.laser_object_id`, and this segment has none.

A camera with an entry in this field but an empty `labels` list means *it was labeled and nothing is
present* — that is information, not missing data.

### 10.3 `projected_lidar_labels` (field 9) — derived 2D

3D boxes flattened to the smallest axis-aligned rectangle covering all projected box points.
Frame 0: FRONT 5, FRONT_LEFT 4, SIDE_LEFT 1, others 0.

ID = **3D id + `_` + camera name**: `0QXtLAoMcF26x6k0m-7gVQ_FRONT`. That suffix is the cross-modal
join key. Includes SIGNs (which the native 2D labels ignore). Clamped at image borders; dropped
entirely if the projection falls fully outside.

### 10.4 `no_label_zones` (field 7)

0 in this segment. `Polygon2dProto`: `x` (1) and `y` (2) as repeated double, `id` (3) — expressed in
the **global** frame. When present, detections inside must be ignored during evaluation, and range
image channel 3 flips to `1` for affected points.

---

## 11. Segmentation (the v1.4 bonus most people miss)

Scanned all 198 frames.

### 11.1 LiDAR panoptic — `RangeImage.segmentation_label_compressed` (field 6)

* **TOP lidar only**, **30 of 198 frames** (roughly every 6th, e.g. frame 24). Both returns labeled.
* Shape `[64, 2650, 2]` MatrixInt32 = `[instance_id, semantic_class]`, ~20 KB zlib.

Frame 24, return 1, over valid pixels:

```
15 VEGETATION    71,825  (46.50%)      18 ROAD          31,501  (20.40%)
14 BUILDING      25,396  (16.44%)       1 CAR            7,832  ( 5.07%)
22 SIDEWALK       6,782  ( 4.39%)      21 WALKABLE       5,901  ( 3.82%)
10 POLE           1,762  ( 1.14%)      17 CURB           1,166  ( 0.75%)
 0 UNDEFINED        789  ( 0.51%)      16 TREE_TRUNK       575  ( 0.37%)
 8 SIGN             361  ( 0.23%)      19 LANE_MARKER      349  ( 0.23%)
20 OTHER_GROUND     211  ( 0.14%)
```

14 unique instance ids, range −1..52. Class 0 = `TYPE_UNDEFINED` = an unlabeled point inside an
otherwise labeled frame.

Full 23-class enum: `UNDEFINED, CAR, TRUCK, BUS, OTHER_VEHICLE, MOTORCYCLIST, BICYCLIST, PEDESTRIAN,
SIGN, TRAFFIC_LIGHT, POLE, CONSTRUCTION_CONE, BICYCLE, MOTORCYCLE, BUILDING, VEGETATION, TREE_TRUNK,
CURB, ROAD, LANE_MARKER, OTHER_GROUND, WALKABLE, SIDEWALK` (0–22).

### 11.2 Camera panoptic — `CameraImage.camera_segmentation_label` (field 10)

**100 images = 20 frames × 5 cameras.** Frame 23, FRONT:

```
panoptic_label:  uint16 PNG, mode I;16, 1920x1280, 27,536 B
panoptic_label_divisor = 1000
    semantic = pixel // 1000
    instance = pixel %  1000        (instance 0 = no instance)
sequence_id = 17766991336812920378
instance_id_to_global_id_mapping: 4 entries, all is_tracked = true
num_cameras_covered: uint8 PNG, values {1, 2}
```

Local instance ids are valid **within one image only**. `instance_id_to_global_id_mapping`
(`local_instance_id`, `global_instance_id`, `is_tracked`) provides cross-camera and cross-time
identity. These global ids are **not** the same namespace as bounding-box ids.
`num_cameras_covered` is the per-pixel weight for the wSTQ metric.

---

## 12. Map (field 10) — frame 0 only

257 features, coordinates in the **global** frame:

```
driveway 135    lane 73    road_edge 31    road_line 7    stop_sign 7    crosswalk 2    speed_bump 2
```

`MapFeature` = `id` (1, int64) + a oneof:
`lane` (3), `road_line` (4), `road_edge` (5), `stop_sign` (7), `crosswalk` (8), `speed_bump` (9),
`driveway` (10).

**The polyline/polygon field number differs per type** — parse per type, not generically:

| type | geometry field | notes |
|---|---|---|
| `LaneCenter` | `polyline` = **8** | plus `speed_limit_mph` (1), `type` (2), `interpolating` (3), `entry_lanes` (9) / `exit_lanes` (10) packed int64, `left/right_neighbors` (11,12), `left/right_boundaries` (13,14) |
| `RoadEdge` | `polyline` = **2** | `type` (1): BOUNDARY / MEDIAN |
| `RoadLine` | `polyline` = **2** | `type` (1): 9 values, broken/solid × single/double × white/yellow |
| `Crosswalk` | `polygon` = **1** | verified: id 149 → 4 corners |
| `SpeedBump` | `polygon` = **1** | |
| `Driveway` | `polygon` = **1** | |
| `StopSign` | `position` = **2** | plus `lane` (1) = associated lane ids; single point |

`MapPoint` = `x` (1), `y` (2), `z` (3) doubles. Samples:

```
road_edge id=4    55 pts   first (-1369.54, 10587.35, 18.35)
road_line id=6    28 pts   first (-1297.41, 10554.88, 31.29)
lane      id=60   12 pts   first (-1370.38, 10590.74, 17.98)   subfields [1,2,3,8,10,11,14]
stop_sign id=153   1 pt    (-1300.43, 10547.18, 35.37)
```

`map_pose_offset` (field 11) = `[0.0, 0.0, 0.0]` here. When non-zero it **must be added to lidar
points** before comparing against map features — it compensates pose drift.

---

## 13. Segment-level (all 198 frames)

```
laser_labels / frame          16 .. 50    mean 35.4
camera_labels / frame          7 .. 51    mean 21.8
projected_lidar_labels/frame  10 .. 49    mean 27.5
no_label_zones, total          0
unique 3D object ids          82
label instances            7,013   (VEHICLE 4,337, SIGN 2,676)
track length             1 .. 198 frames, mean 85.5, median 70
tracks with gaps              0 / 82
map_features                  1 frame carries data (257); rest 0
lidar segmentation           30 frames
camera segmentation         100 images (20 frames x 5 cams)
```

**Zero tracks have gaps** — every object id occupies a contiguous frame range. Labeling is dense and
interpolated; there is no re-identification problem *inside* a segment. Object ids are **not** valid
across files.

---

## 14. Gotchas, ranked by how much they'll cost you

1. **Azimuth column order is descending** — `(W - col - 0.5)/W`. Reversing it mirrors the entire
   cloud and still *looks* plausible. Measured cost: 950 px reprojection error.
2. **Reverse `beam_inclinations`** — range image row 0 is the top beam.
3. **Per-pixel pose compensation is mandatory** for TOP — and it is 40% of the file size.
4. **Empty range pixels are `−1.0`**, not 0, not NaN.
5. **Cameras are not synchronized** — 50 ms spread, each with its own `pose` + `velocity`.
   Use those, not `frame.pose`.
6. **Camera frame is x-forward / y-left / z-up**, not OpenCV. Swap axes or everything is sideways.
7. **`context.stats` is per-frame**, despite the proto comment claiming per-segment.
8. **`camera_labels` (UUID) and `laser_labels` (22-char) do not correspond.** Only
   `projected_lidar_labels` join, via `<3d_id>_<CAMERA>`.
9. **`Label.Box` field numbers are out of order**: width = 4, length = 5, height = 6.
10. **Side cameras are 886 px tall**, principal point near y ≈ 240.
11. **Images are stored in file order, not enum order.** Index by `name`.
12. **`frame.pose` does not match `timestamp_micros`** — pose ≈ mid-frame, timestamp = frame start.
13. **No random access** in TFRecord. Convert to your own format if you will iterate more than twice.
14. **The range image is lossy vs raw points** — proven by the systematic undercount against
    `num_lidar_points_in_box`.
15. **`map_features` exists only on frame 0** of each segment.
16. **The segment id overflows int64** — `10017090168044687777` > 2^63. Keep it a string; never
    parse it into a signed 64-bit column or dtype.
17. **Frame count is not 200 and duration is not exactly 20 s**, despite what the filename window
    says. Observed 197–199 frames, 19.60–19.80 s. Count records instead of hardcoding.
18. `waymo-open-dataset` on PyPI requires **Python ≤ 3.11 + TensorFlow**. The `.venv` in this repo is
    Python 3.14 — it will not install. Everything here was decoded without it.

---

## 15. Storage math

Per frame: lidar 3.7 MB (of which the per-pixel pose alone is 2.46 MB) + JPEG 1.8 MB + labels ~4 KB.

* Dropping `range_image_pose` and return 2 → ~1.2 MB/frame, **~5× smaller**, at the accuracy cost
  quantified in §9.1.
* Extracting to xyz + intensity float32, return 1 only: 171,606 × 16 B = **2.7 MB/frame** —
  *larger* than the compressed range image. The range image representation **is** the compression.
* Whole set: 20 files × ~1 GB ≈ 20 GB for ~4,000 frames ≈ 66 s of driving per file.

---

## 16. Scripts

Written for this analysis; stdlib + numpy + PIL only, no TensorFlow, no `waymo-open-dataset`.

In this repo:

| file | purpose |
|---|---|
| `waymo_lib.py` | The reader. TFRecord framing, protobuf wire decoding, zlib matrices, range image → point cloud, `Frame` accessors for calibration / images / points / labels |
| `waymo_viz.py` | Exports one segment to a single self-contained HTML viewer. `--restamp` re-applies the template to an already-exported viewer without re-decoding |
| `viewer.html` | Viewer template — WebGL point cloud, 3D boxes, camera panel with 2D boxes, timeline, click-to-measure. Opening it directly shows a "this is the template" notice |

```bash
python3 waymo_viz.py waymo_perception/training/segment-….tfrecord     # -> …_viewer.html
python3 waymo_viz.py --restamp waymo_perception/*/*_viewer.html        # re-skin, seconds
```

Output is ~43 MB per segment: 9,000 points/frame (int16 cm, 12-byte records) plus 480 px JPEGs, all
base64'd into one blob. Tune with `--points`, `--stride`, `--img-width`, `--cams`, `--frames`.

Reference schemas: `dataset.proto`, `label.proto`, `map.proto`, `segmentation.proto`.
