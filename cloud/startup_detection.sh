#!/usr/bin/env bash
# VM startup script: detection training on a spot L4.
#
# Reads its segments from OUR bucket, not Waymo's. Waymo gates its dataset on
# per-Google-account licence acceptance, and a VM's service account cannot
# accept those terms -- the first attempt at this died on
# "403 ... does not have storage.objects.get access". So segment selection and
# mirroring happen locally under the human account (see cloud/stage_segments.py,
# which uses a server-side bucket-to-bucket copy, ~25s/GB, never touching the
# local uplink), and the VM only ever touches a bucket its own service account
# owns.
#
# Progress and the best checkpoint go to GCS continuously: this is a *spot*
# instance and can be preempted at any moment, so a run that only uploaded at
# the end would lose everything.
set -uo pipefail

BUCKET="${BUCKET:-gs://vl-waymo-2026-optifuse}"
RUN="${RUN:-det_cloud_v1}"
EPOCHS="${EPOCHS:-24}"
MODEL="${MODEL:-yolo26s.pt}"
IMGSZ="${IMGSZ:-1280}"
BATCH="${BATCH:-24}"
GCS="$BUCKET/$RUN"
RAW="$BUCKET/raw"

exec > >(tee -a /var/log/optifuse.log) 2>&1
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
sync_log(){ gcloud storage cp /var/log/optifuse.log "$GCS/train.log" >/dev/null 2>&1 || true; }
# Fail loudly and stop. The first run plowed through a failed download and
# produced a confusing cascade of empty-dataset errors; better to halt at the
# real cause, upload the log, and release the GPU.
die(){ log "FATAL: $*"; echo "1" > /opt/status.txt
       gcloud storage cp /opt/status.txt "$GCS/STATUS" >/dev/null 2>&1 || true
       sync_log; shutdown -h now; exit 1; }

log "=== OptiFuse cloud detection run: $RUN ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---------------------------------------------------------------- environment
log "installing deps"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
# libgl1/libglib2.0-0: ultralytics pulls in plain opencv-python (not the
# headless build) as a dependency, and that needs libGL even on a machine with
# no display. The first run died here with "ImportError: libGL.so.1".
apt-get install -y -qq git python3-venv libgl1 libglib2.0-0 >/dev/null 2>&1

cd /opt
git clone -q --branch model-pipeline --depth 1 \
  https://github.com/vignesh-kumar-v/OptiFuse.git optifuse || die "git clone failed"
cd /opt/optifuse
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128 || die "torch install failed"
pip install -q ultralytics onnx pillow || die "ultralytics install failed"
python3 -c "import torch,cv2;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))" \
  || die "torch/cv2 import failed"
sync_log

# ------------------------------------------------------------------ pull data
log "downloading staged segments from $RAW"
mkdir -p /opt/waymo_data
gcloud storage cp "$RAW/*.tfrecord" /opt/waymo_data/ 2>&1 | tail -2
N=$(ls /opt/waymo_data/*.tfrecord 2>/dev/null | wc -l)
[ "$N" -ge 5 ] || die "only $N segments downloaded from $RAW (expected >=5)"
log "have $N segments, $(du -sh /opt/waymo_data | cut -f1)"
sync_log

# --------------------------------------------------------------- build labels
# stage_segments.py writes these alongside the data so the VM does not need to
# re-derive the condition of each segment (which would need Waymo access).
gcloud storage cp "$RAW/manifest_selected.json" /opt/ >/dev/null 2>&1 || die "no manifest"
read -r VAL HOLDOUT <<< "$(python3 - <<'PY'
import json
m = json.load(open("/opt/manifest_selected.json"))
# Hold out a Night segment entirely so "works at night" is a generalization
# claim; validate on a different one so val still steers training.
night = [r for r in m if r["stats"]["time_of_day"] == "Night"]
other = [r for r in m if r["stats"]["time_of_day"] != "Night"]
hold = night[0]["name"] if night else ""
val = (other or night)[0]["name"]
print(val, hold.replace("segment-", "").split("_with")[0] if hold else "")
PY
)"
log "val segment: $VAL"
log "held out (unseen test): ${HOLDOUT:-<none>}"

cd /opt/optifuse/data
python3 prepare_detection_dataset.py --segments-dir /opt/waymo_data \
  --out-dir /opt/dataset --val-segment "$VAL" \
  ${HOLDOUT:+--exclude "$HOLDOUT"} 2>&1 | tail -8 || die "dataset prep failed"
python3 oversample_rare_classes.py --dataset-dir /opt/dataset --class-idx 2 --factor 4 2>&1 | tail -2
NTRAIN=$(ls /opt/dataset/images/train 2>/dev/null | wc -l)
[ "$NTRAIN" -ge 1000 ] || die "only $NTRAIN train images built"
log "train images: $NTRAIN"
sync_log

# -------------------------------------------------------------------- training
( while true; do
    sleep 300
    [ -f "/opt/optifuse/models/detection/runs/$RUN/weights/best.pt" ] && \
      gcloud storage cp "/opt/optifuse/models/detection/runs/$RUN/weights/best.pt" \
        "$GCS/best.pt" >/dev/null 2>&1
    sync_log
  done ) &
KEEPER=$!

log "training $MODEL imgsz=$IMGSZ batch=$BATCH epochs=$EPOCHS on $NTRAIN images"
cd /opt/optifuse/models/detection
python3 train.py --model "$MODEL" --data /opt/dataset/dataset.yaml \
  --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH" --patience 12 --name "$RUN" 2>&1 \
  | grep -viE "it/s\]|^\s*$" | tail -300
STATUS=${PIPESTATUS[0]}
kill $KEEPER 2>/dev/null

# --------------------------------------------------------------------- upload
log "uploading results (exit=$STATUS)"
gcloud storage cp -r "runs/$RUN" "$GCS/" >/dev/null 2>&1
gcloud storage cp /opt/dataset/dataset.yaml "$GCS/" >/dev/null 2>&1
echo "$STATUS" > /opt/status.txt
gcloud storage cp /opt/status.txt "$GCS/STATUS" >/dev/null 2>&1
sync_log
log "=== DONE (exit $STATUS) -- results at $GCS ==="
shutdown -h now
