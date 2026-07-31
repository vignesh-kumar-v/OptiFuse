#!/usr/bin/env python3
"""Classify Waymo segments in GCS without downloading them whole.

Each segment is ~1GB, but everything we need to *choose* segments lives in the
first couple of records: `Frame.stats` (time_of_day / weather / location) is in
frame 0, and camera segmentation labels -- when a segment has them at all --
appear from frame 1 onward. So we stream only the first few MB per segment.

Two uses:
  1. Find segments that carry camera segmentation labels (only a subset do).
  2. Select a condition-stratified training set. Our Milestone 1 detector was
     trained on 4x Day + 1x Dawn/Dusk with zero Night, and measurably failed on
     a night scene -- this exists so that gap is chosen away, not stumbled into.

    python3 probe_segments.py --split training --limit 40
    python3 probe_segments.py --split training --limit 200 --need-segmentation \
        --manifest seg_segments.json
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import waymo_lib as wl

BUCKET_NAME = "waymo_open_dataset_v_1_4_2"
BUCKET = f"gs://{BUCKET_NAME}/individual_files"
# ~5.6MB/frame and segmentation first appears on frame 1, so 20MB covers frames
# 0-1 with margin -- ~2% of a full segment.
DEFAULT_BYTES = 20_000_000


def list_segments(split):
    out = subprocess.run(["gcloud", "storage", "ls", f"{BUCKET}/{split}/"],
                         capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        raise SystemExit(f"gcloud storage ls failed: {out.stderr.strip()}")
    return [l.strip() for l in out.stdout.splitlines() if l.strip().endswith(".tfrecord")]


def access_token():
    out = subprocess.run(["gcloud", "auth", "print-access-token"],
                         capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise SystemExit("could not get an access token; run `gcloud auth login`")
    return out.stdout.strip()


def fetch_head(uri, n_bytes, token, dest):
    """True HTTP range request for the first n_bytes of a GCS object.

    `gcloud storage cat ... | head -c` looks equivalent but is ~4x slower here:
    it keeps streaming well past what head consumes. A Range header makes the
    server send only what we ask for (HTTP 206).
    """
    obj = uri[len(f"gs://{BUCKET_NAME}/"):]
    url = (f"https://storage.googleapis.com/storage/v1/b/{BUCKET_NAME}/o/"
           f"{urllib.parse.quote(obj, safe='')}?alt=media")
    r = subprocess.run(["curl", "-s", "--fail",
                        "-H", f"Authorization: Bearer {token}",
                        "-H", f"Range: bytes=0-{n_bytes - 1}",
                        url, "-o", dest], timeout=600)
    return r.returncode == 0 and os.path.getsize(dest) > 0


def probe(uri, n_bytes, token):
    """Read the head of one segment and report its stats + segmentation flag."""
    info = {"uri": uri, "name": os.path.basename(uri), "stats": None,
            "has_segmentation": False, "frames_read": 0, "error": None}
    with tempfile.NamedTemporaryFile(suffix=".tfrecord", delete=True) as tmp:
        try:
            if not fetch_head(uri, n_bytes, token, tmp.name):
                info["error"] = "range fetch failed"
                return info
        except Exception as e:
            info["error"] = f"fetch: {type(e).__name__}"
            return info
        try:
            for i, payload in wl.records(tmp.name, limit=2):
                F = wl.Frame(payload)
                if info["stats"] is None:
                    info["stats"] = F.stats
                for raw in F.f.get(4, []):
                    if 10 in wl.msg(raw):
                        info["has_segmentation"] = True
                        break
                info["frames_read"] = i + 1
        except Exception as e:
            # A truncated trailing payload is normal -- we keep whatever parsed.
            if info["stats"] is None:
                info["error"] = f"{type(e).__name__}: {e}"
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="training", choices=["training", "validation"])
    ap.add_argument("--limit", type=int, default=40, help="how many segments to probe")
    ap.add_argument("--offset", type=int, default=0, help="skip the first N segments")
    ap.add_argument("--bytes", type=int, default=DEFAULT_BYTES)
    ap.add_argument("--need-segmentation", action="store_true",
                    help="stop early once --want segments with segmentation are found")
    ap.add_argument("--want", type=int, default=4,
                    help="with --need-segmentation, how many hits to collect [4]")
    ap.add_argument("--manifest", default=None, help="write results to this JSON file")
    ap.add_argument("-j", "--jobs", type=int, default=8, help="parallel probes [8]")
    args = ap.parse_args()

    segs = list_segments(args.split)
    todo = segs[args.offset:args.offset + args.limit]
    print(f"{args.split}: {len(segs)} segments total; probing {len(todo)} "
          f"from offset {args.offset} ({args.jobs} parallel)\n")

    token = access_token()
    results, hits = [], 0

    # Batched rather than one big map so --need-segmentation can stop early
    # without waiting for every remaining probe to finish.
    with ThreadPoolExecutor(args.jobs) as ex:
        for start in range(0, len(todo), args.jobs):
            batch = todo[start:start + args.jobs]
            for r in ex.map(lambda u: probe(u, args.bytes, token), batch):
                results.append(r)
                s = r["stats"] or {}
                tag = "SEG" if r["has_segmentation"] else "   "
                short = r["name"].replace("segment-", "")[:22]
                print(f"  [{tag}] {short:24} {s.get('time_of_day','?'):10} "
                      f"{s.get('weather','?'):7} {s.get('location','?'):13}"
                      f"{'  ERR: ' + r['error'] if r['error'] else ''}")
                hits += bool(r["has_segmentation"])
            if args.need_segmentation and hits >= args.want:
                print(f"\nfound {hits} segmentation segments, stopping early")
                break

    ok = [r for r in results if r["stats"]]
    print(f"\n{len(ok)}/{len(results)} probed cleanly | "
          f"{sum(r['has_segmentation'] for r in results)} with segmentation")
    for key in ("time_of_day", "weather"):
        counts = {}
        for r in ok:
            counts[r["stats"].get(key, "?")] = counts.get(r["stats"].get(key, "?"), 0) + 1
        print(f"  {key:12} " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    if args.manifest:
        with open(args.manifest, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.manifest}")


if __name__ == "__main__":
    main()
