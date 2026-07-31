#!/usr/bin/env python3
"""Read the tar shards written by waymo_extract.py into training samples.

    from waymo_dataset import WaymoImages
    ds = WaymoImages("cache/training", task="detection", size=(960, 640))
    for s in ds:
        s["image"]    # uint8 [H,W,3], letterboxed
        s["boxes"]    # float32 [N,4] xyxy in resized pixels
        s["labels"]   # int64 [N]

Tasks: "detection" (all samples), "segmentation" (only samples carrying a
panoptic label), "depth" (samples carrying lidar depth).

Everything in the cache is at native resolution because `camera_projection` uv
is in full-res pixels. Resizing happens here so image, boxes, mask and uv are
transformed by one scale factor and cannot drift apart.

Torch is optional -- the core yields numpy. `torch_dataset()` wraps it.
"""
import glob
import io
import json
import os
import tarfile

import numpy as np
from PIL import Image

# Native 2D labels carry only these three; SIGN exists solely in the
# projected-3D labels, which waymo_extract.py does not write.
DET_CLASSES = {1: 0, 2: 1, 4: 2}                    # VEHICLE, PEDESTRIAN, CYCLIST
DET_NAMES = ["VEHICLE", "PEDESTRIAN", "CYCLIST"]

CAMERAS = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]

PANOPTIC_DIVISOR = 1000
IGNORE_INDEX = 255                                  # letterbox padding in seg targets

TASK_REQUIRES = {"detection": "json", "segmentation": "seg.png", "depth": "depth.npz"}


def iter_tar(path):
    """Yield (key, {ext: bytes}) per sample. Members of one sample are adjacent."""
    with tarfile.open(path) as tar:
        cur, parts = None, {}
        for m in tar:
            if not m.isfile():
                continue
            key, ext = m.name.split(".", 1)
            if key != cur:
                if cur is not None and parts:
                    yield cur, parts
                cur, parts = key, {}
            parts[ext] = tar.extractfile(m).read()
        if cur is not None and parts:
            yield cur, parts


def letterbox_params(w, h, size):
    """(scale, pad_x, pad_y) fitting w*h inside size without distorting aspect."""
    tw, th = size
    s = min(tw / w, th / h)
    return s, int((tw - round(w * s)) // 2), int((th - round(h * s)) // 2)


def letterbox_image(im, size, fill=0):
    tw, th = size
    s, px, py = letterbox_params(im.width, im.height, size)
    r = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
    out = Image.new(im.mode, (tw, th), fill)
    out.paste(r, (px, py))
    return out


def decode(key, parts, task, size):
    """One cache sample -> one training sample. `size` None keeps native resolution."""
    meta = json.loads(parts["json"])
    im = Image.open(io.BytesIO(parts["jpg"])).convert("RGB")
    w, h = im.width, im.height

    if size is None:
        s, px, py = 1.0, 0, 0
    else:
        s, px, py = letterbox_params(w, h, size)
        im = letterbox_image(im, size)

    out = {"key": key, "camera": meta["camera"], "image": np.asarray(im),
           "scale": s, "pad": (px, py)}

    if task == "detection":
        boxes, labels = [], []
        for b in meta["boxes"]:
            if b["type"] not in DET_CLASSES:
                continue
            boxes.append([(b["cx"] - b["w"] / 2) * s + px, (b["cy"] - b["h"] / 2) * s + py,
                          (b["cx"] + b["w"] / 2) * s + px, (b["cy"] + b["h"] / 2) * s + py])
            labels.append(DET_CLASSES[b["type"]])
        out["boxes"] = np.array(boxes, np.float32).reshape(-1, 4)
        out["labels"] = np.array(labels, np.int64)

    elif task == "segmentation":
        pan = np.array(Image.open(io.BytesIO(parts["seg.png"])))
        sem = (pan // PANOPTIC_DIVISOR).astype(np.uint8)
        inst = (pan % PANOPTIC_DIVISOR).astype(np.uint16)
        if size is not None:
            # nearest only -- interpolating label ids invents classes that do
            # not exist, and averaging instance ids is meaningless.
            sem = _letterbox_nearest(sem, size, IGNORE_INDEX)
            inst = _letterbox_nearest(inst, size, 0)
        out["semantic"] = sem
        out["instance"] = inst

    elif task == "depth":
        d = np.load(io.BytesIO(parts["depth.npz"]))
        uv = d["uv"].astype(np.float32) * s + np.array([px, py], np.float32)
        z = d["z_cm"].astype(np.float32) / 100.0
        tw, th = (size if size is not None else (w, h))
        u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        keep = (u >= 0) & (u < tw) & (v >= 0) & (v < th)
        depth = np.zeros((th, tw), np.float32)
        # Nearest surface wins when several points land on one pixel; painting
        # in far-to-near order leaves the closest value on top.
        order = np.argsort(-z[keep])
        depth[v[keep][order], u[keep][order]] = z[keep][order]
        out["depth"] = depth
        out["depth_mask"] = depth > 0

    else:
        raise ValueError(f"unknown task {task!r}")
    return out


def _letterbox_nearest(arr, size, fill):
    """Letterbox a label array with NEAREST resampling, padding with `fill`."""
    tw, th = size
    h, w = arr.shape
    s, px, py = letterbox_params(w, h, size)
    nw, nh = round(w * s), round(h * s)
    r = np.asarray(Image.fromarray(arr).resize((nw, nh), Image.NEAREST))
    out = np.full((th, tw), fill, arr.dtype)
    out[py:py + nh, px:px + nw] = r
    return out


class WaymoImages:
    """Iterate cache shards. Streams tars sequentially -- no random access."""

    def __init__(self, root, task="detection", size=(960, 640), cameras=None, shards=None):
        if task not in TASK_REQUIRES:
            raise ValueError(f"task must be one of {sorted(TASK_REQUIRES)}")
        if cameras:
            # Checked before touching the filesystem: a typo would otherwise
            # yield zero samples silently, and a partial name like "LEFT" would
            # match both SIDE_LEFT and FRONT_LEFT via the endswith test.
            bad = set(cameras) - set(CAMERAS)
            if bad:
                raise ValueError(f"unknown camera(s) {sorted(bad)}; expected {CAMERAS}")
        self.shards = sorted(shards or glob.glob(os.path.join(root, "*.tar")))
        if not self.shards:
            raise FileNotFoundError(f"no .tar shards under {root!r}")
        self.task, self.size = task, size
        self.cameras = set(cameras) if cameras else None
        self.required = TASK_REQUIRES[task]

    def _wanted(self, key, parts):
        if self.required not in parts or "json" not in parts:
            return False
        # Keys are {segment}_{frame:04d}_{CAM} and BOTH segment names and camera
        # names contain underscores, so the camera cannot be split off by
        # position -- rsplit("_", 1) turns SIDE_LEFT into "LEFT" and matches
        # nothing. Test against the known names instead.
        if self.cameras and not any(key.endswith("_" + c) for c in self.cameras):
            return False
        return True

    def _stream(self, shards):
        for shard in shards:
            for key, parts in iter_tar(shard):
                if self._wanted(key, parts):
                    yield decode(key, parts, self.task, self.size)

    def __iter__(self):
        return self._stream(self.shards)

    def torch_dataset(self):
        """Wrap as a torch IterableDataset, sharded across dataloader workers."""
        import torch

        outer = self

        class _DS(torch.utils.data.IterableDataset):
            def __iter__(self):
                info = torch.utils.data.get_worker_info()
                shards = outer.shards
                if info is not None:
                    shards = shards[info.id::info.num_workers]
                return outer._stream(shards)

        return _DS()


def _selfcheck():
    """Geometry check: boxes, mask and depth must land where the image does."""
    W, H, SZ = 1920, 886, (960, 640)              # a SIDE camera: needs real padding
    s, px, py = letterbox_params(W, H, SZ)
    assert abs(s - 0.5) < 1e-9, s
    assert px == 0 and py == (640 - 443) // 2, (px, py)

    parts = {}
    img = Image.new("RGB", (W, H), (10, 20, 30))
    b = io.BytesIO(); img.save(b, "JPEG"); parts["jpg"] = b.getvalue()
    parts["json"] = json.dumps({"camera": "SIDE_LEFT", "width": W, "height": H,
                                "boxes": [{"cx": 960.0, "cy": 443.0, "w": 100.0,
                                           "h": 50.0, "type": 1, "id": "x"}]}).encode()
    d = decode("k_SIDE_LEFT", parts, "detection", SZ)
    assert d["image"].shape == (640, 960, 3), d["image"].shape
    # Source centre maps to the centre of the *pasted region*, not of the
    # canvas: 886*0.5 = 443 is odd, so centring it in 640 costs half a pixel.
    cx = (d["boxes"][0][0] + d["boxes"][0][2]) / 2
    cy = (d["boxes"][0][1] + d["boxes"][0][3]) / 2
    assert abs(cx - (px + W * s / 2)) < 1e-4, (cx, px + W * s / 2)
    assert abs(cy - (py + H * s / 2)) < 1e-4, (cy, py + H * s / 2)
    assert abs((d["boxes"][0][2] - d["boxes"][0][0]) - 50.0) < 1e-4    # 100 * 0.5

    # padding rows must be ignore, not a real class
    pan = np.full((H, W), 20 * PANOPTIC_DIVISOR + 7, np.uint16)
    b = io.BytesIO(); Image.fromarray(pan).save(b, "PNG"); parts["seg.png"] = b.getvalue()
    d = decode("k_SIDE_LEFT", parts, "segmentation", SZ)
    assert d["semantic"].shape == (640, 960)
    assert d["semantic"][0, 0] == IGNORE_INDEX, d["semantic"][0, 0]
    assert d["semantic"][320, 480] == 20 and d["instance"][320, 480] == 7

    # a lidar point at source (1000, 400) must land at (500, 298) after 0.5x + pad
    b = io.BytesIO()
    np.savez_compressed(b, uv=np.array([[1000, 400]], np.int16),
                        z_cm=np.array([1234], np.uint16))
    parts["depth.npz"] = b.getvalue()
    d = decode("k_SIDE_LEFT", parts, "depth", SZ)
    assert d["depth_mask"].sum() == 1
    v, u = np.argwhere(d["depth_mask"])[0]
    assert (u, v) == (500, 200 + py), (u, v, py)
    assert abs(d["depth"][v, u] - 12.34) < 1e-3, d["depth"][v, u]

    # Camera filtering: multi-word names must not be split by position.
    class _Fake(WaymoImages):
        def __init__(self, cams):
            self.task, self.size, self.required = "detection", None, "json"
            self.cameras, self.shards = set(cams), ["x"]
    k = "10017090168044687777_6380_000_6400_000_0023_SIDE_LEFT"
    assert _Fake({"SIDE_LEFT"})._wanted(k, {"json": b""})
    assert not _Fake({"FRONT"})._wanted(k, {"json": b""})
    try:                       # validated before any filesystem access
        WaymoImages("/nonexistent", cameras={"LEFT"})
    except ValueError:
        pass
    else:
        raise AssertionError("partial camera name 'LEFT' should be rejected")

    print("selfcheck OK: letterbox, boxes, panoptic ignore-pad, depth uv, camera filter")


def _smoke(root, n=60):
    """Read real shards and assert every task's targets are self-consistent."""
    print(f"\nsmoke test on {root}")
    for task in ("detection", "segmentation", "depth"):
        ds = WaymoImages(root, task=task, size=(960, 640))
        seen = 0
        for s in ds:
            H, W = s["image"].shape[:2]
            assert (H, W) == (640, 960), s["image"].shape
            if task == "detection":
                b = s["boxes"]
                assert b.shape[1:] == (4,) and len(b) == len(s["labels"])
                if len(b):
                    assert (b[:, 2] >= b[:, 0]).all() and (b[:, 3] >= b[:, 1]).all(), "x1<x0"
                    assert b.min() >= -1 and b[:, 0].max() <= W + 1, "box outside canvas"
                    assert set(np.unique(s["labels"])) <= {0, 1, 2}
            elif task == "segmentation":
                sem = s["semantic"]
                assert sem.shape == (H, W) and s["instance"].shape == (H, W)
                real = sem[sem != IGNORE_INDEX]
                assert real.max() <= 28, real.max()
                # padding must be ignore, never a real class
                py = s["pad"][1]
                if py:
                    assert (sem[:py] == IGNORE_INDEX).all(), "pad row leaked a class"
            else:
                d, m = s["depth"], s["depth_mask"]
                assert d.shape == (H, W) and m.shape == (H, W)
                assert (d[m] > 0).all() and (d[~m] == 0).all()
                assert d[m].max() < 120, f"implausible depth {d[m].max()}"
            seen += 1
            if seen >= n:
                break
        assert seen, f"no samples for task={task}"
        print(f"  {task:13} {seen:3} samples OK")
    print("smoke test OK")


if __name__ == "__main__":
    import sys
    _selfcheck()
    for root in sys.argv[1:]:
        _smoke(root)
