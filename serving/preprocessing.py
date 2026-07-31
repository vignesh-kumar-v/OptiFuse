#!/usr/bin/env python3
"""Shared streaming preprocessing for all three perception models.

One camera frame feeds three engines that each want a different input geometry:

    detector      1280x1280  FP16   (YOLO26, square letterbox)
    segmentation   960x640   FP16   (SegFormer)
    depth          966x644   FP16   (Depth Anything; dims must be multiples of 14
                                     for the DINOv2 patch grid)

The JPEG is decoded **once** and reused for all three, which is the whole point
of having this module rather than three independent clients: decode is the
expensive per-frame CPU step and it is identical for every consumer.

`letterbox()` here is the single source of truth for the detector's geometry. It
reimplements ultralytics.data.augment.LetterBox(auto=False, scaleup=True,
center=True, padding_value=114) and was verified against the real
predictor.preprocess() output to 5.96e-08 -- float32 rounding noise. Serving goes
through ONNX/TensorRT only, with no live PyTorch, so this must stay exact:
silently mismatched preprocessing degrades accuracy without raising anything.

Pixel normalization beyond /255 is deliberately NOT done here. The segmentation
and depth ONNX graphs bake their own ImageNet mean/std in, so every engine takes
plain 0-1 RGB and the constants cannot drift between trainer and client.
"""
from dataclasses import dataclass

import cv2
import numpy as np

DETECTOR_SIZE = (1280, 1280)   # (w, h)
SEG_SIZE = (960, 640)
DEPTH_SIZE = (966, 644)

PAD_VALUE = 114


@dataclass(frozen=True)
class LetterboxInfo:
    """Everything needed to map model-space coordinates back to the source image."""
    ratio: float
    pad_x: int
    pad_y: int
    orig_w: int
    orig_h: int

    def to_original(self, xyxy):
        """Undo letterbox for an (N,4) xyxy array, clipped to the source image."""
        out = np.asarray(xyxy, np.float32).copy()
        out[:, [0, 2]] = np.clip((out[:, [0, 2]] - self.pad_x) / self.ratio, 0, self.orig_w)
        out[:, [1, 3]] = np.clip((out[:, [1, 3]] - self.pad_y) / self.ratio, 0, self.orig_h)
        return out


def letterbox(im_bgr, size, pad_value=PAD_VALUE):
    """Aspect-preserving resize + centered pad. See module docstring on exactness.

    `size` is (width, height). Returns (padded_image, LetterboxInfo).
    """
    h, w = im_bgr.shape[:2]
    tw, th = size
    r = min(tw / w, th / h)
    new_w, new_h = round(w * r), round(h * r)
    dw, dh = (tw - new_w) / 2, (th - new_h) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    resized = cv2.resize(im_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(pad_value,) * 3)
    return padded, LetterboxInfo(r, left, top, w, h)


def to_tensor(padded_bgr, dtype=np.float16):
    """BGR HWC uint8 -> RGB CHW batch of 1, scaled to 0-1."""
    rgb = padded_bgr[..., ::-1]
    chw = rgb.transpose(2, 0, 1).astype(dtype) / dtype(255.0)
    return np.ascontiguousarray(chw[None])


def preprocess(im_bgr, size, dtype=np.float16):
    """Single-model convenience: letterbox + tensor in one call."""
    padded, info = letterbox(im_bgr, size)
    return to_tensor(padded, dtype), info


def decode(jpeg_bytes):
    """JPEG bytes -> BGR HWC uint8. Done once per frame, shared by all models."""
    return cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)


class FramePreprocessor:
    """Decode one frame once, emit the input tensor each engine expects.

        pre = FramePreprocessor()
        batch = pre(jpeg_bytes)                 # all three
        batch = pre(jpeg_bytes, ("detector",))  # just one
    """

    SIZES = {"detector": DETECTOR_SIZE, "segmentation": SEG_SIZE, "depth": DEPTH_SIZE}

    def __init__(self, dtype=np.float16, models=None):
        self.dtype = dtype
        self.models = tuple(models or self.SIZES)
        unknown = set(self.models) - set(self.SIZES)
        if unknown:
            raise ValueError(f"unknown model(s) {sorted(unknown)}; expected {sorted(self.SIZES)}")

    def __call__(self, jpeg_bytes, models=None):
        im = decode(jpeg_bytes)
        if im is None:
            raise ValueError("could not decode JPEG")
        return self.from_image(im, models)

    def from_image(self, im_bgr, models=None):
        """Same as __call__ but for an already-decoded frame."""
        out = {}
        for name in (models or self.models):
            padded, info = letterbox(im_bgr, self.SIZES[name])
            out[name] = (to_tensor(padded, self.dtype), info)
        return out


def _selfcheck():
    """Geometry assertions with hand-computed answers, mirroring the style of
    waymo_dataset._selfcheck so both data paths are checked the same way."""
    # A FRONT camera frame: 1920x1280 into the detector's 1280x1280 square.
    im = np.zeros((1280, 1920, 3), np.uint8)
    padded, info = letterbox(im, DETECTOR_SIZE)
    assert padded.shape == (1280, 1280, 3), padded.shape
    assert abs(info.ratio - 1280 / 1920) < 1e-9, info.ratio
    # 1280*(2/3)=853.33 -> 853 tall, so 427 total vertical pad, split 213/214.
    assert (info.pad_x, info.pad_y) == (0, 213), (info.pad_x, info.pad_y)

    # Round-trip: a box on the padded canvas maps back to source pixels.
    box = np.array([[info.pad_x, info.pad_y, info.pad_x + 1920 * info.ratio,
                     info.pad_y + 1280 * info.ratio]], np.float32)
    back = info.to_original(box)[0]
    assert np.allclose(back, [0, 0, 1920, 1280], atol=1e-3), back

    # A SIDE camera (1920x886) needs horizontal-only padding at seg size.
    im2 = np.zeros((886, 1920, 3), np.uint8)
    _, i2 = letterbox(im2, SEG_SIZE)
    assert abs(i2.ratio - 0.5) < 1e-9, i2.ratio           # min(960/1920, 640/886)
    assert i2.pad_x == 0 and i2.pad_y == (640 - 443) // 2, (i2.pad_x, i2.pad_y)

    # Depth dims must stay on the DINOv2 patch grid.
    assert DEPTH_SIZE[0] % 14 == 0 and DEPTH_SIZE[1] % 14 == 0, DEPTH_SIZE

    # One decode, three tensors, each the shape its engine declares.
    pre = FramePreprocessor()
    got = pre.from_image(im)
    assert got["detector"][0].shape == (1, 3, 1280, 1280), got["detector"][0].shape
    assert got["segmentation"][0].shape == (1, 3, 640, 960), got["segmentation"][0].shape
    assert got["depth"][0].shape == (1, 3, 644, 966), got["depth"][0].shape
    assert got["detector"][0].dtype == np.float16

    # 0-1 range, and padding really is 114/255 in every channel.
    t = got["detector"][0]
    assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0, (t.min(), t.max())
    assert abs(float(t[0, 0, 0, 0]) - 114 / 255) < 1e-3, float(t[0, 0, 0, 0])

    try:
        FramePreprocessor(models=["detector", "nope"])
    except ValueError:
        pass
    else:
        raise AssertionError("unknown model name should be rejected")

    print("selfcheck OK: letterbox geometry, coord round-trip, per-model shapes, pad value")


if __name__ == "__main__":
    _selfcheck()
