#!/usr/bin/env python3
"""Download a size-capped subset of the Waymo Open Dataset Perception v1.4.3.

For each requested split (training, validation, ...) the script lists
gs://waymo_open_dataset_v_1_4_3/individual_files/<split>, then downloads files one at
a time while that split's cumulative size stays under the cap. The cap applies
per split, so `--splits training validation --cap-gb 10` pulls up to 10 GiB of each.

`gcloud storage`'s own per-file transfer meter streams live; an overall progress bar
is printed between files. Failed downloads are skipped and reported instead of
aborting the run. Ctrl+C stops cleanly and still prints the summary.

Uses `gcloud storage` rather than `gsutil`: gsutil's bundled Python segfaults
(EXC_BAD_ACCESS) on macOS when forking workers for sliced downloads.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

BUCKET_ROOT = "gs://waymo_open_dataset_v_1_4_3/individual_files"
KNOWN_SPLITS = ("training", "validation", "testing", "testing_3d_camera_only_detection")
DEFAULT_DEST = "./waymo_perception"
GIB = 1024**3

# ANSI colors, disabled when stdout is not a TTY (e.g. piped to tee/a log file).
if sys.stdout.isatty():
    C_DIM, C_GRN, C_RED, C_YLW, C_CYN, C_BLD, C_RST = (
        "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[1m", "\033[0m",
    )
else:
    C_DIM = C_GRN = C_RED = C_YLW = C_CYN = C_BLD = C_RST = ""


def human(nbytes):
    n = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.0f} B" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024


def hms(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def bar(fraction, width=32):
    fraction = min(1.0, max(0.0, fraction))
    filled = int(round(fraction * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def print_overall(split, done_files, planned_files, done_bytes, planned_bytes, cap, started_at):
    """Progress within the current split: files, bytes, cap headroom, rate, ETA."""
    frac = done_bytes / planned_bytes if planned_bytes else 0.0
    elapsed = time.time() - started_at
    rate = done_bytes / elapsed if elapsed > 0 and done_bytes else 0
    eta = (planned_bytes - done_bytes) / rate if rate > 0 else 0

    print(
        f"{C_CYN}{bar(frac)}{C_RST} {frac * 100:5.1f}%  {split}  "
        f"files {done_files}/{planned_files}  "
        f"{human(done_bytes)} / {human(planned_bytes)}  "
        f"cap {human(cap)} ({human(cap - done_bytes)} left)"
    )
    print(
        f"{C_DIM}         elapsed {hms(elapsed)}  "
        f"avg {human(rate)}/s  "
        f"eta {hms(eta) if rate else '--'}{C_RST}\n"
    )


def cleanup_partial(local):
    """Remove a half-written file and the sidecar temp file gcloud storage leaves behind."""
    for path in (local, local + "_.gstmp"):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as exc:
            print(f"{C_YLW}  could not remove {path}: {exc}{C_RST}")


def require_gcloud():
    if shutil.which("gcloud") is None:
        sys.exit(
            "gcloud not found. Install the Google Cloud SDK first:\n"
            "  brew install --cask google-cloud-sdk      # macOS\n"
            "  curl https://sdk.cloud.google.com | bash  # other platforms"
        )
    probe = subprocess.run(["gcloud", "storage", "--help"], capture_output=True, text=True)
    if probe.returncode != 0:
        sys.exit(
            "`gcloud storage` is unavailable in this SDK install.\n"
            "Update the SDK: gcloud components update"
        )


def active_account():
    out = subprocess.run(
        ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def require_auth():
    account = active_account()
    if not account:
        print("No active gcloud account. Launching `gcloud auth login` ...")
        if subprocess.run(["gcloud", "auth", "login"]).returncode != 0:
            sys.exit("gcloud auth login failed. Authenticate manually and re-run.")
        account = active_account()
        if not account:
            sys.exit("Still no active account after login. Aborting.")
    print(f"Authenticated as: {C_BLD}{account}{C_RST}")


def list_objects(split):
    """Return [(size_bytes, gs_uri), ...] for one split, sorted by filename."""
    prefix = f"{BUCKET_ROOT}/{split}"
    print(f"Listing {prefix}/ ... (~30s)", flush=True)
    proc = subprocess.run(
        ["gcloud", "storage", "ls", "-l", f"{prefix}/*.tfrecord"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"{C_RED}Failed to list {split}:{C_RST}\n{proc.stderr.strip()}")
        return []

    objects = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        # Data lines look like: <size> <iso-date> <gs://...>; the trailing
        # "TOTAL: N objects, N bytes" line fails the isdigit check and is dropped.
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        uri = parts[-1]
        if uri.endswith("/") or not uri.endswith(".tfrecord"):
            continue
        objects.append((int(parts[0]), uri))

    return sorted(objects, key=lambda o: o[1])


def plan(objects, cap):
    """Pick the leading run of files that fits under the cap. Returns (selected, stopper)."""
    selected, total = [], 0
    for size, uri in objects:
        if total + size > cap:
            return selected, (size, uri)
        selected.append((size, uri))
        total += size
    return selected, None


def download_split(split, cap, dest_root, dry_run):
    """Download one split under its own cap. Returns a result dict for the final summary."""
    dest = os.path.join(dest_root, split)
    started_at = time.time()
    result = {
        "split": split, "dest": dest, "downloaded": [], "failed": [],
        "total": 0, "cap": cap, "interrupted": False,
    }

    print(f"\n{C_BLD}{'=' * 72}\nSPLIT: {split}\n{'=' * 72}{C_RST}")

    objects = list_objects(split)
    if not objects:
        print(f"{C_YLW}No .tfrecord objects found for {split}; skipping.{C_RST}")
        result["failed"].append((split, "no objects listed"))
        return result

    selected, stopper = plan(objects, cap)
    planned_bytes = sum(s for s, _ in selected)

    print(
        f"Found {len(objects)} files. Plan: {C_BLD}{len(selected)} files{C_RST}, "
        f"{C_BLD}{human(planned_bytes)}{C_RST} (cap {human(cap)})."
    )
    if stopper:
        print(f"{C_DIM}Next file would exceed cap: {stopper[1].rsplit('/', 1)[-1]} "
              f"({human(stopper[0])}){C_RST}")
    print(f"Destination: {dest}\n")

    if dry_run:
        for i, (size, uri) in enumerate(selected, 1):
            print(f"  {i:>2}. {uri.rsplit('/', 1)[-1]}  ({human(size)})")
        result["total"] = planned_bytes
        result["downloaded"] = [(u.rsplit("/", 1)[-1], s) for s, u in selected]
        return result

    os.makedirs(dest, exist_ok=True)
    print_overall(split, 0, len(selected), 0, planned_bytes, cap, started_at)

    for idx, (size, uri) in enumerate(selected, 1):
        name = uri.rsplit("/", 1)[-1]
        local = os.path.join(dest, name)
        tag = f"[{split} {idx}/{len(selected)}]"

        if os.path.exists(local) and os.path.getsize(local) == size:
            result["total"] += size
            result["downloaded"].append((name, size))
            print(f"{C_GRN}{tag} CACHED{C_RST} {name}  {human(size)}")
            print_overall(split, len(result["downloaded"]), len(selected),
                          result["total"], planned_bytes, cap, started_at)
            continue

        print(f"{C_BLD}{tag} downloading{C_RST} {name}  ({human(size)})")

        try:
            # No capture: gcloud storage's own byte-level progress meter streams
            # straight to the terminal. Failures are caught by exit code + size check.
            rc = subprocess.run(["gcloud", "storage", "cp", uri, local]).returncode
        except KeyboardInterrupt:
            print(f"\n{C_YLW}Interrupted. Cleaning up partial file for {name} ...{C_RST}")
            cleanup_partial(local)
            result["interrupted"] = True
            break

        if rc != 0:
            result["failed"].append((name, f"gcloud storage exited {rc} (see output above)"))
            print(f"{C_RED}{tag} FAILED{C_RST} {name}: {result['failed'][-1][1]}")
            cleanup_partial(local)  # drop partial so it is neither counted nor reused
        else:
            actual = os.path.getsize(local) if os.path.exists(local) else 0
            if actual != size:
                result["failed"].append((name, f"size mismatch: expected {size}, got {actual}"))
                print(f"{C_RED}{tag} FAILED{C_RST} {name}: {result['failed'][-1][1]}")
                cleanup_partial(local)
            else:
                result["total"] += actual
                result["downloaded"].append((name, actual))
                print(f"{C_GRN}{tag} OK{C_RST} {name}  {human(actual)}")

        print_overall(split, len(result["downloaded"]), len(selected),
                      result["total"], planned_bytes, cap, started_at)

    result["elapsed"] = time.time() - started_at
    return result


def final_summary(results, dest_root, started_at):
    print("\n" + "=" * 72)
    print(f"{C_BLD}SUMMARY{C_RST}")
    print("=" * 72)

    grand_files = grand_bytes = 0
    all_under = True

    for r in results:
        under = r["total"] <= r["cap"]
        all_under = all_under and under
        grand_files += len(r["downloaded"])
        grand_bytes += r["total"]

        flag = f"  {C_YLW}(interrupted){C_RST}" if r["interrupted"] else ""
        print(f"\n{C_BLD}{r['split']}{flag}{C_RST}")
        print(f"  Files downloaded : {len(r['downloaded'])}")
        print(f"  Total size       : {human(r['total'])}  ({r['total']} bytes)")
        print(f"  Cap              : {human(r['cap'])}  ({r['cap']} bytes)")
        print(f"  Under cap        : {(C_GRN + 'YES') if under else (C_RED + 'NO')}{C_RST}")
        print(f"  Destination      : {r['dest']}")

        if r["failed"]:
            print(f"  {C_RED}Failed ({len(r['failed'])}):{C_RST}")
            for name, err in r["failed"]:
                print(f"    - {name}: {err}")

        print("  Filenames:")
        if r["downloaded"]:
            for i, (name, size) in enumerate(r["downloaded"], 1):
                print(f"    {i:>2}. {name}  ({human(size)})")
        else:
            print("    (none)")

    print("\n" + "-" * 72)
    print(f"{C_BLD}TOTAL{C_RST}: {grand_files} files, {human(grand_bytes)} "
          f"({grand_bytes} bytes) across {len(results)} split(s)")
    print(f"All splits under cap : {(C_GRN + 'YES') if all_under else (C_RED + 'NO')}{C_RST}")
    print(f"Elapsed              : {hms(time.time() - started_at)}")
    print(f"Root destination     : {os.path.abspath(dest_root)}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--splits", nargs="+", default=["training", "validation"],
        help=f"splits to download (default: training validation). Known: {', '.join(KNOWN_SPLITS)}",
    )
    ap.add_argument("--dest", default=DEFAULT_DEST,
                    help=f"root destination dir; each split gets a subdir (default: {DEFAULT_DEST})")
    ap.add_argument("--cap-gb", type=float, default=10.0,
                    help="cap in GiB applied PER SPLIT (default: 10)")
    ap.add_argument("--dry-run", action="store_true", help="show the plan, download nothing")
    args = ap.parse_args()

    for split in args.splits:
        if split not in KNOWN_SPLITS:
            print(f"{C_YLW}Warning: '{split}' is not a known split "
                  f"({', '.join(KNOWN_SPLITS)}); trying anyway.{C_RST}")

    cap = int(args.cap_gb * GIB)
    dest_root = os.path.abspath(args.dest)
    started_at = time.time()

    require_gcloud()
    require_auth()

    print(f"Splits: {C_BLD}{', '.join(args.splits)}{C_RST}  "
          f"Cap per split: {C_BLD}{human(cap)}{C_RST}  "
          f"Max total: {C_BLD}{human(cap * len(args.splits))}{C_RST}")

    results = []
    for split in args.splits:
        r = download_split(split, cap, dest_root, args.dry_run)
        results.append(r)
        if r["interrupted"]:
            print(f"{C_YLW}Stopping: interrupted during {split}.{C_RST}")
            break

    final_summary(results, dest_root, started_at)

    any_ok = any(r["downloaded"] for r in results)
    return 0 if any_ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)