#!/usr/bin/env python3
"""Waymo Perception v1.4.x reader. stdlib + numpy only, no TensorFlow.

Decodes the TFRecord container and the Frame protobuf by hand, because the
official `waymo-open-dataset` package needs Python <= 3.11 + TF.

See WAYMO_TFRECORD_DEEP_DIVE.md for the format itself.
"""
import struct
import zlib
from collections import defaultdict

import numpy as np

CAM = {0: "UNKNOWN", 1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT",
       5: "SIDE_RIGHT", 6: "REAR_LEFT", 7: "REAR", 8: "REAR_RIGHT"}
LAS = {0: "UNKNOWN", 1: "TOP", 2: "FRONT", 3: "SIDE_LEFT", 4: "SIDE_RIGHT", 5: "REAR"}
OBJ = {0: "UNKNOWN", 1: "VEHICLE", 2: "PEDESTRIAN", 3: "SIGN", 4: "CYCLIST"}


# --------------------------------------------------------------------------- io
def records(path, limit=None, skip=0, stride=1):
    """Yield (index, payload_bytes) per TFRecord record.

    Layout per record: uint64 len | uint32 crc(len) | data | uint32 crc(data).
    CRCs are not checked; seek past payloads we don't want.
    """
    with open(path, "rb") as f:
        i = n_out = 0
        while True:
            head = f.read(8)
            if len(head) < 8:
                return
            (n,) = struct.unpack("<Q", head)
            want = i >= skip and (i - skip) % stride == 0
            f.seek(4, 1)
            if want:
                payload = f.read(n)
            else:
                f.seek(n, 1)
            f.seek(4, 1)
            if want:
                yield i, payload
                n_out += 1
                if limit and n_out >= limit:
                    return
            i += 1


def count_records(path):
    n = 0
    with open(path, "rb") as f:
        while (head := f.read(8)) and len(head) == 8:
            (ln,) = struct.unpack("<Q", head)
            f.seek(4 + ln + 4, 1)
            n += 1
    return n


# ---------------------------------------------------------------- protobuf wire
def _varint(buf, p):
    r = s = 0
    while True:
        b = buf[p]
        p += 1
        r |= (b & 0x7F) << s
        if not b & 0x80:
            return r, p
        s += 7


def fields(buf):
    """Yield (field_no, wire_type, value) for one serialized message."""
    p, end = 0, len(buf)
    while p < end:
        key, p = _varint(buf, p)
        fno, wt = key >> 3, key & 7
        if wt == 0:
            v, p = _varint(buf, p)
        elif wt == 1:
            v = struct.unpack_from("<d", buf, p)[0]
            p += 8
        elif wt == 2:
            ln, p = _varint(buf, p)
            v = buf[p:p + ln]
            p += ln
        elif wt == 5:
            v = struct.unpack_from("<f", buf, p)[0]
            p += 4
        else:
            raise ValueError(f"bad wire type {wt} at {p}")
        yield fno, wt, v


def msg(buf):
    """{field_no: [values]} for one message. Repeated fields keep all values."""
    d = defaultdict(list)
    for fno, _wt, v in fields(buf):
        d[fno].append(v)
    return d


def _packed_varints(b):
    """Decode a packed stream of protobuf varints (LEB128) into int64.

    Vectorized: the byte-at-a-time loop this replaced spent ~75% of total
    export time here alone (profiled on real camera_projection streams,
    which are the only int32 packed field this code decodes). Grouping is
    inherently serial (continuation bits chain byte-to-byte), but decoding
    all groups at once is not -- so this computes per-byte group ids and
    per-byte shift amounts with cumsum/arange, then reduces per group with
    bincount instead of iterating in Python.

    bincount requires float64 weights; each 32-bit half of the accumulated
    uint64 is summed separately to stay under 2**53 exactly (max per-term
    contribution is ~2.7e11 for the worst-case 10-byte varint, see the
    correctness check in the fix's commit), so no precision is lost.
    """
    n = len(b)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    buf = np.frombuffer(b, dtype=np.uint8)
    stop = (buf & 0x80) == 0                       # last byte of each varint
    end_idx = np.flatnonzero(stop)
    if len(end_idx) == 0 or end_idx[-1] != n - 1:
        raise ValueError("truncated varint stream")

    starts_new = np.empty(n, dtype=bool)
    starts_new[0] = False
    starts_new[1:] = stop[:-1]
    group_id = np.cumsum(starts_new, dtype=np.int64)

    start_idx = np.empty(len(end_idx), dtype=np.int64)
    start_idx[0] = 0
    start_idx[1:] = end_idx[:-1] + 1
    pos_in_group = np.arange(n, dtype=np.int64) - start_idx[group_id]

    payload = (buf & 0x7F).astype(np.uint64)
    shifted = payload << (pos_in_group.astype(np.uint64) * np.uint64(7))

    m = len(end_idx)
    lo = np.bincount(group_id, weights=(shifted & np.uint64(0xFFFFFFFF)).astype(np.float64), minlength=m)
    hi = np.bincount(group_id, weights=(shifted >> np.uint64(32)).astype(np.float64), minlength=m)
    out = lo.astype(np.uint64) | (hi.astype(np.uint64) << np.uint64(32))
    return out.view(np.int64)


def matrix(blob, int32=False):
    """zlib blob -> MatrixFloat/MatrixInt32 -> reshaped ndarray."""
    m = msg(zlib.decompress(blob))
    dims = list(msg(m[2][0])[1])
    if dims and isinstance(dims[0], (bytes, bytearray)):
        dims = list(_packed_varints(dims[0]))
    raw = b"".join(m[1])
    arr = _packed_varints(raw) if int32 else np.frombuffer(raw, dtype="<f4")
    return arr.reshape([int(d) for d in dims])


def transform(vals):
    return np.array(vals, dtype=np.float64).reshape(4, 4)


# ------------------------------------------------------------------- geometry
def pose_from_rpy_xyz(rp):
    """[...,6] (roll,pitch,yaw,x,y,z) -> [...,4,4]. 3-2-1 Euler, vehicle->global."""
    r, p, y = rp[..., 0], rp[..., 1], rp[..., 2]
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    T = np.zeros(rp.shape[:-1] + (4, 4))
    T[..., 0, 0] = cy * cp; T[..., 0, 1] = cy * sp * sr - sy * cr; T[..., 0, 2] = cy * sp * cr + sy * sr
    T[..., 1, 0] = sy * cp; T[..., 1, 1] = sy * sp * sr + cy * cr; T[..., 1, 2] = sy * sp * cr - cy * sr
    T[..., 2, 0] = -sp;     T[..., 2, 1] = cp * sr;                T[..., 2, 2] = cp * cr
    T[..., :3, 3] = rp[..., 3:6]
    T[..., 3, 3] = 1.0
    return T


def range_image_to_points(ri, calib, frame_pose, pixel_pose=None):
    """[H,W,>=4] range image -> (xyz [N,3] in frame-vehicle coords, valid mask).

    Column order is DESCENDING -- (W - col - 0.5)/W. Reversing it silently
    mirrors the entire cloud; see the deep-dive doc, section 9.
    """
    H, W = ri.shape[:2]
    ext = calib["extrinsic"]
    inc = calib["beam_inclinations"]
    if inc:
        inclination = np.asarray(inc)[::-1]                 # row 0 = top beam
    else:
        lo, hi = calib["inc_min"], calib["inc_max"]
        inclination = lo + (np.arange(H)[::-1] + 0.5) / H * (hi - lo)
    az_corr = np.arctan2(ext[1, 0], ext[0, 0])
    ratios = (W - np.arange(W) - 0.5) / W
    azimuth = ((ratios * 2.0 - 1.0) * np.pi - az_corr)[None, :]
    incl = inclination[:, None]

    rng = ri[..., 0]
    ci, si = np.cos(incl), np.sin(incl)
    pts = np.stack([rng * ci * np.cos(azimuth),
                    rng * ci * np.sin(azimuth),
                    rng * si * np.ones_like(azimuth)], -1)

    pts = pts @ ext[:3, :3].T + ext[:3, 3]                  # -> vehicle @ pixel time
    if pixel_pose is not None:
        P = pose_from_rpy_xyz(pixel_pose)                   # -> global
        pts = np.einsum("hwij,hwj->hwi", P[..., :3, :3], pts) + P[..., :3, 3]
        inv = np.linalg.inv(frame_pose)                     # -> frame vehicle frame
        pts = pts @ inv[:3, :3].T + inv[:3, 3]
    mask = rng > 0
    return pts[mask], mask


# ---------------------------------------------------------------------- frame
class Frame:
    """Lazy accessors over one decoded Frame proto."""

    def __init__(self, payload):
        self.f = msg(payload)
        self.nbytes = len(payload)

    # -- scalars
    @property
    def timestamp(self):
        return self.f[2][0] / 1e6

    @property
    def pose(self):
        return transform(msg(self.f[3][0])[1])

    @property
    def context_name(self):
        return msg(self.f[1][0])[1][0].decode()

    @property
    def stats(self):
        s = msg(msg(self.f[1][0])[4][0])
        return {"time_of_day": s[2][0].decode(), "location": s[3][0].decode(),
                "weather": s[4][0].decode()}

    # -- calibration (identical every frame; read once and cache upstream)
    def camera_calibrations(self):
        out = {}
        for cc in msg(self.f[1][0])[2]:
            c = msg(cc)
            out[CAM[c[1][0]]] = {
                "intrinsic": list(c[2]),
                "extrinsic": transform(msg(c[3][0])[1]),
                "width": c[4][0], "height": c[5][0],
                "rolling_shutter": c[6][0],
            }
        return out

    def laser_calibrations(self):
        out = {}
        for lc in msg(self.f[1][0])[3]:
            l = msg(lc)
            out[LAS[l[1][0]]] = {
                "extrinsic": transform(msg(l[5][0])[1]),
                "beam_inclinations": list(l.get(2, [])),
                "inc_min": l[3][0], "inc_max": l[4][0],
            }
        return out

    # -- sensors
    def images(self):
        """[{name, jpeg, pose, velocity, trigger, shutter, readout}]"""
        out = []
        for im in self.f.get(4, []):
            i = msg(im)
            v = msg(i[4][0]) if 4 in i else {}
            out.append({
                "name": CAM[i[1][0]],
                "jpeg": i[2][0],
                "pose": transform(msg(i[3][0])[1]),
                "velocity": [v[k][0] for k in (1, 2, 3)] if v else [0, 0, 0],
                "pose_timestamp": i[5][0],
                "shutter": i[6][0],
                "trigger": i[7][0],
                "readout_done": i[8][0],
            })
        return out

    def points(self, returns=(2, 3), lasers=None):
        """Merged point cloud in the frame vehicle frame.

        Returns dict of parallel arrays: xyz [N,3], intensity, elongation,
        cam [N] (CameraName enum of first projection, 0 = none), uv [N,2].
        """
        calibs = self.laser_calibrations()
        fp = self.pose
        xyz, inten, elong, cam, uv = [], [], [], [], []
        for la in self.f.get(5, []):
            L = msg(la)
            name = LAS[L[1][0]]
            if lasers and name not in lasers:
                continue
            r1 = msg(L[2][0])
            ppose = matrix(r1[4][0]) if 4 in r1 else None   # TOP only
            for rf in returns:
                if rf not in L:
                    continue
                R = msg(L[rf][0])
                ri = matrix(R[2][0])
                p, mask = range_image_to_points(ri, calibs[name], fp, ppose)
                xyz.append(p)
                inten.append(ri[..., 1][mask])
                elong.append(ri[..., 2][mask])
                if 3 in R:
                    cp = matrix(R[3][0], int32=True)[mask]
                    cam.append(cp[:, 0])
                    uv.append(cp[:, 1:3])
                else:
                    cam.append(np.zeros(len(p), np.int64))
                    uv.append(np.zeros((len(p), 2), np.int64))
        if not xyz:
            return {k: np.empty((0, 3) if k in ("xyz", "uv") else 0)
                    for k in ("xyz", "intensity", "elongation", "cam", "uv")}
        return {"xyz": np.concatenate(xyz),
                "intensity": np.concatenate(inten),
                "elongation": np.concatenate(elong),
                "cam": np.concatenate(cam),
                "uv": np.concatenate(uv)}

    # -- labels
    def laser_labels(self):
        """3D boxes in the vehicle frame. Box field order: w=4, l=5, h=6."""
        out = []
        for b in self.f.get(6, []):
            L = msg(b)
            B = msg(L[1][0])
            g = lambda k: B[k][0] if k in B else 0.0
            md = msg(L[2][0]) if 2 in L else {}
            vel = [md[k][0] if k in md else 0.0 for k in (1, 2, 5)]   # speed x,y,z
            out.append({
                "id": L[4][0].decode(),
                "type": L[3][0],
                "center": [g(1), g(2), g(3)],
                "size": [g(5), g(4), g(6)],          # length(x), width(y), height(z)
                "heading": g(7),
                "num_points": L[7][0] if 7 in L else 0,
                "most_visible_camera": L[11][0].decode() if 11 in L else "",
                "velocity": vel,
                "speed": float(np.hypot(vel[0], vel[1])),
            })
        return out

    def camera_labels(self, field=8):
        """field 8 = native 2D labels, field 9 = projected 3D labels."""
        out = {}
        for cl in self.f.get(field, []):
            C = msg(cl)
            boxes = []
            for b in C.get(2, []):
                L = msg(b)
                B = msg(L[1][0])
                g = lambda k: B[k][0] if k in B else 0.0
                boxes.append({"id": L[4][0].decode(), "type": L[3][0],
                              "cx": g(1), "cy": g(2), "w": g(5), "h": g(4)})
            out[CAM[C[1][0]]] = boxes
        return out

    def has_map(self):
        return len(self.f.get(10, [])) > 0
