#!/usr/bin/env python3
"""Select condition-stratified Waymo segments and mirror them into our bucket.

    python3 stage_segments.py --manifest ../data/probe_training_300.json --n 30

Why this runs locally rather than on the training VM: Waymo gates the dataset on
per-Google-account licence acceptance, so a VM's service account gets
`403 storage.objects.get denied` no matter its IAM roles -- a service account
cannot accept Waymo's terms. Running under the human account here and copying
into a bucket we own means the VM only ever reads its own project's storage.

The copy is a GCS server-side rewrite (`gcloud storage cp gs://... gs://...`),
measured at ~52 MiB/s, so a 1GB segment moves in ~25s without touching the local
uplink. Downloading 30GB and re-uploading it from a home connection would take
hours.

Selection targets ~50% Day / 25% Night / 25% Dawn-Dusk. Milestone 1 trained on
4 Day + 1 Dawn/Dusk with zero Night and measurably collapsed at night (0.02
detections per ground-truth object); this exists so that gap is chosen away.
"""
import argparse
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict

DEFAULT_TARGET = {"Day": 0.50, "Night": 0.25, "Dawn/Dusk": 0.25}


def select(records, n, target=DEFAULT_TARGET, seed=0):
    by = defaultdict(list)
    for r in records:
        if r.get("stats"):
            by[r["stats"]["time_of_day"]].append(r)
    rng = random.Random(seed)
    for k in by:
        rng.shuffle(by[k])

    picked, used = [], set()
    for cond, frac in sorted(target.items(), key=lambda kv: -kv[1]):
        want = round(n * frac)
        for r in by.get(cond, [])[:want]:
            picked.append(r)
            used.add(r["uri"])
    # Top up if a condition was short of its quota (Night usually is).
    if len(picked) < n:
        rest = [r for r in records if r.get("stats") and r["uri"] not in used]
        rng.shuffle(rest)
        for r in rest[:n - len(picked)]:
            picked.append(r)
            used.add(r["uri"])
    return picked[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--manifest", default=os.path.join(here, "..", "data", "probe_training_300.json"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--bucket", default="gs://vl-waymo-2026-optifuse")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    records = json.load(open(os.path.normpath(args.manifest)))
    ok = [r for r in records if r.get("stats")]
    print(f"manifest: {len(ok)}/{len(records)} probed cleanly")
    print(f"  available: {dict(Counter(r['stats']['time_of_day'] for r in ok))}")

    picked = select(ok, args.n)
    counts = Counter(r["stats"]["time_of_day"] for r in picked)
    print(f"\nselected {len(picked)}: {dict(counts)}")
    print(f"  locations: {dict(Counter(r['stats']['location'] for r in picked))}")
    print(f"  weather:   {dict(Counter(r['stats']['weather'] for r in picked))}")
    if not counts.get("Night"):
        print("\n  WARNING: no Night segment selected -- the night gap will NOT be fixed.")

    raw = f"{args.bucket}/raw"
    if args.dry_run:
        print(f"\n[dry run] would mirror {len(picked)} segments to {raw}")
        for r in picked:
            print(f"    {r['stats']['time_of_day']:10} {r['name'][:52]}")
        return

    # Written next to the data so the VM can pick val/holdout by condition
    # without needing Waymo access to re-probe.
    man = os.path.join(here, "manifest_selected.json")
    with open(man, "w") as f:
        json.dump(picked, f, indent=2)
    subprocess.run(["gcloud", "storage", "cp", man, f"{raw}/manifest_selected.json"],
                   check=True, capture_output=True)

    print(f"\nmirroring to {raw} (server-side copy)...")
    uris = "\n".join(r["uri"] for r in picked)
    p = subprocess.run(["gcloud", "storage", "cp", "-I", f"{raw}/"],
                       input=uris, text=True)
    if p.returncode != 0:
        sys.exit(f"mirror failed with exit {p.returncode}")

    out = subprocess.run(["gcloud", "storage", "ls", f"{raw}/"],
                         capture_output=True, text=True)
    n_there = sum(1 for l in out.stdout.splitlines() if l.endswith(".tfrecord"))
    print(f"done: {n_there} .tfrecord objects now in {raw}")


if __name__ == "__main__":
    main()
