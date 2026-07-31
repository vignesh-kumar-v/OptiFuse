#!/usr/bin/env python3
"""Per-object distance and ego-lane assignment by fusing detection + depth + calibration.

This is the layer that makes three separate models into one perception system:
the detector says *where* an object is in the image, the depth model says how far
that pixel is, and the camera calibration turns the pair into a position in the
vehicle's own frame -- which is what an in-car display actually shows ("lead
vehicle, 23 m").

Geometry follows Waymo's camera convention: the camera frame is x=forward (the
optical axis), y=left, z=up -- NOT OpenCV's z-forward. The depth model is trained
against `camera_depth()` in waymo_extract.py, whose target is the x-component in
camera frame, so a predicted value is already distance along the optical axis
rather than Euclidean range. Projection is therefore

    u = c_u - f_u * (y / x)        and so, inverted, given u and depth x:
    v = c_v - f_v * (z / x)            y = -(u - c_u) * x / f_u
                                       z = -(v - c_v) * x / f_v

Run this file directly to validate the whole chain against 3D laser labels:

    python3 distance.py --tfrecord ../../waymo_data/<seg>.tfrecord
"""
import argparse
import os
import sys

import numpy as np

# Half-width of the ego lane. US highway lanes are ~3.7m; 1.8m each side keeps an
# adjacent-lane vehicle out while tolerating some lateral error in the estimate.
EGO_LANE_HALF_WIDTH = 1.8


def backproject(u, v, depth, intrinsic):
    """Pixel + depth-along-optical-axis -> camera-frame (x fwd, y left, z up)."""
    f_u, f_v, c_u, c_v = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
    x = np.asarray(depth, np.float64)
    y = -(np.asarray(u, np.float64) - c_u) * x / f_u
    z = -(np.asarray(v, np.float64) - c_v) * x / f_v
    return np.stack([x, y, z], -1)


def camera_to_vehicle(pts_cam, extrinsic):
    """Camera frame -> vehicle frame using the calibration's 4x4 camera->vehicle."""
    p = np.asarray(pts_cam, np.float64).reshape(-1, 3)
    out = p @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    return out.reshape(np.shape(pts_cam))


def vehicle_to_camera(pts_veh, extrinsic):
    inv = np.linalg.inv(extrinsic)
    p = np.asarray(pts_veh, np.float64).reshape(-1, 3)
    out = p @ inv[:3, :3].T + inv[:3, 3]
    return out.reshape(np.shape(pts_veh))


def project(pts_cam, intrinsic):
    """Camera frame -> pixel (u, v). Points at or behind the image plane give NaN."""
    f_u, f_v, c_u, c_v = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
    p = np.asarray(pts_cam, np.float64).reshape(-1, 3)
    x = p[:, 0]
    safe = x > 1e-6
    u = np.full_like(x, np.nan)
    v = np.full_like(x, np.nan)
    u[safe] = c_u - f_u * (p[safe, 1] / x[safe])
    v[safe] = c_v - f_v * (p[safe, 2] / x[safe])
    return np.stack([u, v], -1)


def sample_box_depth(depth_map, box_xyxy, percentile=30.0, shrink=0.25):
    """Robust depth for one detection, in metres.

    Samples an inner patch rather than the whole box: a box's outer margin
    usually contains background (road behind a car, sky beside a pole), and
    including it drags the estimate long. A low percentile of the inner patch
    then favours the nearest real surface, which is the quantity a following
    distance actually cares about.

    Returns NaN when the box has no usable depth (e.g. fully outside the map).
    """
    h, w = depth_map.shape
    x1, y1, x2, y2 = box_xyxy
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return float("nan")
    ix1 = int(round(x1 + bw * shrink / 2))
    ix2 = int(round(x2 - bw * shrink / 2))
    iy1 = int(round(y1 + bh * shrink / 2))
    iy2 = int(round(y2 - bh * shrink / 2))
    ix1, ix2 = max(0, min(ix1, w - 1)), max(1, min(ix2, w))
    iy1, iy2 = max(0, min(iy1, h - 1)), max(1, min(iy2, h))
    if ix2 <= ix1 or iy2 <= iy1:
        return float("nan")
    patch = np.asarray(depth_map[iy1:iy2, ix1:ix2], np.float32)
    patch = patch[np.isfinite(patch) & (patch > 0)]
    if patch.size == 0:
        return float("nan")
    return float(np.percentile(patch, percentile))


def object_positions(dets_xyxy, depth_map_full, intrinsic, extrinsic, **kw):
    """For each detection: (forward_m, lateral_m, in_ego_lane).

    `depth_map_full` must already be un-letterboxed to original image pixels so
    that box coordinates and depth pixels share one coordinate system.
    """
    out = []
    for box in np.asarray(dets_xyxy, np.float64).reshape(-1, 4):
        d = sample_box_depth(depth_map_full, box, **kw)
        if not np.isfinite(d):
            out.append((float("nan"), float("nan"), False))
            continue
        # Bottom-centre: where the object meets the ground, the most stable
        # point for lateral placement (a box's centre floats with object height).
        u = (box[0] + box[2]) / 2.0
        v = box[3]
        cam = backproject(u, v, d, intrinsic)
        veh = camera_to_vehicle(cam, extrinsic)
        fwd, lat = float(veh[0]), float(veh[1])
        out.append((fwd, lat, abs(lat) < EGO_LANE_HALF_WIDTH and fwd > 0))
    return out


def lead_object(positions):
    """Index of the closest in-ego-lane object ahead, or None."""
    best, best_d = None, float("inf")
    for i, (fwd, _lat, in_lane) in enumerate(positions):
        if in_lane and np.isfinite(fwd) and 0 < fwd < best_d:
            best, best_d = i, fwd
    return best


# --------------------------------------------------------------------------- #
# Validation against 3D laser labels
# --------------------------------------------------------------------------- #

def _validate(tfrecord, n_frames, camera, url, conf_thres):
    """Project ground-truth 3D boxes into the image, read our predicted depth at
    those pixels, and compare to the label's true distance. This checks the whole
    chain -- convention, intrinsics, extrinsics and the depth model together."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import preprocessing
    import tritonclient.grpc as grpcclient
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import waymo_lib as wl

    client = grpcclient.InferenceServerClient(url=url)
    if not client.is_model_ready("depth_estimator"):
        raise SystemExit("depth_estimator not READY on Triton")

    errs, rel, n_obj = [], [], 0
    for i, payload in wl.records(tfrecord):
        if n_frames and i >= n_frames:
            break
        F = wl.Frame(payload)
        calib = F.camera_calibrations()[camera]
        intr, extr = calib["intrinsic"], calib["extrinsic"]

        img = next((im for im in F.images() if im["name"] == camera), None)
        if img is None:
            continue
        im_bgr = preprocessing.decode(img["jpeg"])
        tensor, info = preprocessing.preprocess(im_bgr, preprocessing.DEPTH_SIZE, np.float16)

        inp = grpcclient.InferInput("pixel_values", tensor.shape, "FP16")
        inp.set_data_from_numpy(tensor)
        res = client.infer("depth_estimator", inputs=[inp],
                           outputs=[grpcclient.InferRequestedOutput("depth")])
        dmap = res.as_numpy("depth")[0].astype(np.float32)

        # un-letterbox back to source pixels
        h, w = dmap.shape
        inner = dmap[info.pad_y:h - info.pad_y or None, info.pad_x:w - info.pad_x or None]
        import cv2
        full = cv2.resize(inner, (info.orig_w, info.orig_h), interpolation=cv2.INTER_LINEAR)

        for lb in F.laser_labels():
            if lb["type"] != 1 or lb["num_points"] < 30:
                continue  # vehicles with enough returns to trust the label
            centre_veh = np.array(lb["center"], np.float64)
            cam = vehicle_to_camera(centre_veh, extr)
            if cam[0] < 3 or cam[0] > 70:
                continue
            uv = project(cam, intr).reshape(2)
            u, v = uv
            if not (np.isfinite(u) and 0 <= u < info.orig_w and 0 <= v < info.orig_h):
                continue
            pred_axis = float(full[int(v), int(u)])
            if not np.isfinite(pred_axis) or pred_axis <= 0:
                continue
            # label centre is the box centre, our depth is the visible near face,
            # so add half the object's length back before comparing
            true_axis = float(cam[0]) - float(lb["size"][0]) / 2.0
            if true_axis <= 0:
                continue
            errs.append(pred_axis - true_axis)
            rel.append(abs(pred_axis - true_axis) / true_axis)
            n_obj += 1

    if not errs:
        raise SystemExit("no comparable objects found")
    e = np.array(errs)
    r = np.array(rel)
    print(f"\nValidated on {n_obj} ground-truth vehicles (3-70m, >=30 lidar points):")
    print(f"  mean signed error : {e.mean():+.2f} m   (positive = we over-estimate)")
    print(f"  median |error|    : {np.median(np.abs(e)):.2f} m")
    print(f"  mean rel. error   : {r.mean():.1%}")
    print(f"  within 2m         : {(np.abs(e) < 2).mean():.1%}")
    print(f"  within 15% rel.   : {(r < 0.15).mean():.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tfrecord", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "waymo_data",
        "segment-10689101165701914459_2072_300_2092_300_with_camera_labels.tfrecord"))
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--camera", default="FRONT")
    ap.add_argument("--url", default="localhost:8001")
    ap.add_argument("--conf-thres", type=float, default=0.2)
    args = ap.parse_args()
    _validate(args.tfrecord, args.frames, args.camera, args.url, args.conf_thres)


if __name__ == "__main__":
    main()
