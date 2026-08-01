#!/usr/bin/env bash
# VM startup script: condition-stratified detection training on a spot L4.
#
# Runs unattended on a GCP VM and does the whole job -- select segments, pull
# them, build the YOLO dataset, train, upload. Reading Waymo's public bucket
# from inside GCP is same-region, so segments arrive far faster than over a home
# connection and cost nothing in egress. That is the entire reason extraction
# happens here rather than locally.
#
# Progress and the best checkpoint are pushed to GCS continuously, because this
# runs on a *spot* instance and can be preempted at any moment -- a run that
# only uploads at the end would lose everything.
set -uo pipefail

BUCKET="${BUCKET:-gs://vl-waymo-2026-optifuse}"
RUN="${RUN:-det_cloud_v1}"
N_SEG="${N_SEG:-30}"
PROBE="${PROBE:-260}"
EPOCHS="${EPOCHS:-24}"
MODEL="${MODEL:-yolo26s.pt}"
IMGSZ="${IMGSZ:-1280}"
BATCH="${BATCH:-24}"
GCS="$BUCKET/$RUN"

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
sync_log(){ gcloud storage cp /var/log/optifuse.log "$GCS/train.log" >/dev/null 2>&1 || true; }
exec > >(tee -a /var/log/optifuse.log) 2>&1

log "=== OptiFuse cloud detection run: $RUN ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---------------------------------------------------------------- environment
log "installing deps"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git python3-venv >/dev/null 2>&1
cd /opt
git clone -q --branch model-pipeline --depth 1 \
  https://github.com/vignesh-kumar-v/OptiFuse.git optifuse || { log "clone FAILED"; exit 1; }
cd /opt/optifuse
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
# cu128 index: the same Blackwell-capable wheels used locally. The L4 (Ada) is
# happy with them too, so one pin covers both machines.
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -q ultralytics onnx opencv-python-headless pillow
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
sync_log

# ------------------------------------------------------- stratified selection
# Correcting Milestone 1's actual defect: it trained on 4 Day + 1 Dawn/Dusk and
# zero Night, and measurably collapsed at night. Selection is by condition here,
# not by whatever sorts first.
log "probing $PROBE training segments for condition mix"
python3 data/probe_segments.py --split training --limit "$PROBE" -j 16 \
  --manifest /opt/manifest.json | tail -5
sync_log

python3 - <<'PY' > /opt/selected.txt
import json, random
from collections import defaultdict
import os
n_target = int(os.environ.get("N_SEG", "30"))
recs = [r for r in json.load(open("/opt/manifest.json")) if r.get("stats")]
by = defaultdict(list)
for r in recs:
    by[r["stats"]["time_of_day"]].append(r)
random.Random(0).shuffle(recs)
for k in by:
    random.Random(0).shuffle(by[k])
# ~50% Day / 25% Night / 25% Dawn-Dusk, falling back to whatever exists
want = {"Day": int(n_target*0.5), "Night": int(n_target*0.25), "Dawn/Dusk": int(n_target*0.25)}
picked, seen = [], set()
for cond, k in want.items():
    for r in by.get(cond, [])[:k]:
        picked.append(r); seen.add(r["uri"])
for r in recs:                      # top up if a condition was short
    if len(picked) >= n_target: break
    if r["uri"] not in seen:
        picked.append(r); seen.add(r["uri"])
from collections import Counter
c = Counter(r["stats"]["time_of_day"] for r in picked)
print("\n".join(r["uri"] for r in picked[:n_target]))
import sys; print(f"# selected {len(picked[:n_target])}: {dict(c)}", file=sys.stderr)
PY
log "selected: $(grep -vc '^#' /opt/selected.txt) segments"
sync_log

# ------------------------------------------------------------------ pull data
log "downloading segments (same-region, no egress cost)"
mkdir -p /opt/waymo_data
grep -v '^#' /opt/selected.txt | gcloud storage cp -I /opt/waymo_data/ 2>&1 | tail -2
log "have $(ls /opt/waymo_data/*.tfrecord 2>/dev/null | wc -l) segments, $(du -sh /opt/waymo_data | cut -f1)"
sync_log

# --------------------------------------------------------------- build labels
# Hold out one Night segment entirely so "it works at night" stays a
# generalization claim rather than a memorized one.
HOLDOUT=$(python3 - <<'PY'
import json
recs=[r for r in json.load(open("/opt/manifest.json")) if r.get("stats")]
sel=set(l.strip() for l in open("/opt/selected.txt") if l.strip() and not l.startswith("#"))
night=[r for r in recs if r["uri"] in sel and r["stats"]["time_of_day"]=="Night"]
print(night[0]["name"].replace("segment-","").split("_with")[0] if night else "")
PY
)
log "holding out (unseen test): ${HOLDOUT:-<none>}"

VAL=$(ls /opt/waymo_data/*.tfrecord | grep -v "${HOLDOUT:-__none__}" | head -1 | xargs basename)
log "val segment: $VAL"

cd /opt/optifuse/data
python3 prepare_detection_dataset.py --segments-dir /opt/waymo_data \
  --out-dir /opt/dataset --val-segment "$VAL" \
  ${HOLDOUT:+--exclude "$HOLDOUT"} 2>&1 | tail -8
python3 oversample_rare_classes.py --dataset-dir /opt/dataset --class-idx 2 --factor 4 2>&1 | tail -2
log "train images: $(ls /opt/dataset/images/train | wc -l)"
sync_log

# -------------------------------------------------------------------- training
# Push best.pt to GCS in the background: on a spot VM the run can vanish
# mid-epoch, and a checkpoint on a dead instance is worth nothing.
( while true; do
    sleep 300
    [ -f /opt/optifuse/models/detection/runs/$RUN/weights/best.pt ] && \
      gcloud storage cp /opt/optifuse/models/detection/runs/$RUN/weights/best.pt \
        "$GCS/best.pt" >/dev/null 2>&1
    sync_log
  done ) &
KEEPER=$!

log "training $MODEL, imgsz=$IMGSZ batch=$BATCH epochs=$EPOCHS"
cd /opt/optifuse/models/detection
python3 train.py --model "$MODEL" --data /opt/dataset/dataset.yaml \
  --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH" --patience 12 --name "$RUN" 2>&1 \
  | grep -vE "^\s*$" | tail -400
STATUS=$?
kill $KEEPER 2>/dev/null

# --------------------------------------------------------------------- upload
log "uploading results (exit=$STATUS)"
gcloud storage cp -r "runs/$RUN" "$GCS/" >/dev/null 2>&1
gcloud storage cp /opt/dataset/dataset.yaml "$GCS/" >/dev/null 2>&1
echo "$STATUS" > /opt/status.txt && gcloud storage cp /opt/status.txt "$GCS/STATUS" >/dev/null 2>&1
sync_log
log "=== DONE (exit $STATUS) -- results at $GCS ==="

# Stop, don't delete: the disk keeps the run for inspection, and a stopped
# instance bills only for that disk (pennies) instead of the GPU.
shutdown -h now
