#!/usr/bin/env bash
set -uo pipefail
cd ~/leisaac
export HDF5_USE_FILE_LOCKING=FALSE

# ---------------- config ----------------
HF_REPO="ArunMurugesan/cube_to_tray_carton"
N_BATCH=30            # 30 batches x 10 episodes = 300
PER=10
STEP_HZ=1000          # 60 = real-time viewing (slow)
FPS=30
CONVERTER="isaaclab2lerobotv3.py"
TASK=""               # leave empty to auto-detect from the task's __init__.py
WORK=~/leisaac/datasets/_work
STATE=~/leisaac/datasets/_done
export HF_LEROBOT_HOME="$WORK/lerobot_home"
# ----------------------------------------
mkdir -p "$WORK" "$STATE" "$HF_LEROBOT_HOME"
HF=$(command -v hf || command -v huggingface-cli)
[ -z "$TASK" ] && TASK=$(grep -ohE 'id="[^"]+"' source/leisaac/leisaac/tasks/cube_to_tray/__init__.py | head -n1 | cut -d'"' -f2)
echo "task id: $TASK | repo: $HF_REPO | batches: $N_BATCH x $PER"
[ -z "$TASK" ] && { echo "[stop] could not detect task id, set TASK= in the script"; exit 1; }

record() {   # window ON (no --headless)
  pkill -9 -f generate_cube_to_tray.py 2>/dev/null; sleep 3
  timeout -k 30 2400 python scripts/datagen/state_machine/generate_cube_to_tray.py \
    --enable_cameras --cam_view off --record \
    --dataset_file "$1" --num_demos $PER --max_attempts $((PER*4)) --step_hz $STEP_HZ \
    --grip_close 0.14 --clearance 0.003 --damping 3 --grip_damping 10
}
count() {    # prints "episodes action_dim" or "0 0"
  python - "$1" 2>/dev/null <<'PY' || echo "0 0"
import sys, h5py
d = h5py.File(sys.argv[1], "r")["data"]; k = list(d)[0]
print(len(d), d[k]["actions"].shape[-1])
PY
}

run_batch() {
  local tag=$1 H5="$WORK/$1.hdf5" OUT="$HF_LEROBOT_HOME/local/$1" n=0 dim=0
  rm -rf "$OUT"

  [ -f "$H5" ] && read -r n dim < <(count "$H5")
  if [ "$n" -lt "$PER" ]; then
    echo "[$tag] recording $PER episodes..."
    rm -f "$H5"
    record "$H5" > "/tmp/rec_$tag.log" 2>&1
    [ -f "$H5" ] || { echo "[$tag] no hdf5 produced, see /tmp/rec_$tag.log"; return 1; }
  fi

  python ~/leisaac/fix_actions.py "$H5" > "/tmp/fix_$tag.log" 2>&1 || { echo "[$tag] fix_actions failed"; return 1; }
  read -r n dim < <(count "$H5")
  echo "[$tag] episodes=$n action_dim=$dim"
  [ "$n" -ge "$PER" ] && [ "$dim" -eq 6 ] || { echo "[$tag] bad data (need >=$PER episodes, dim 6)"; return 1; }

  echo "[$tag] converting..."
  python scripts/convert/$CONVERTER --headless --enable_cameras \
    --task_name "$TASK" --repo_id "local/$tag" \
    --hdf5_root "$WORK" --hdf5_files "$tag.hdf5" --fps $FPS \
    > "/tmp/conv_$tag.log" 2>&1 &
  cpid=$!
  for _ in $(seq 1 360); do
    sleep 5
    kill -0 $cpid 2>/dev/null || break
    grep -q "Finished converting" "/tmp/conv_$tag.log" && { sleep 5; break; }
    grep -q "Traceback" "/tmp/conv_$tag.log" && break
  done
  kill $cpid 2>/dev/null; sleep 3; kill -9 $cpid 2>/dev/null; wait $cpid 2>/dev/null
  [ -f "$OUT/meta/info.json" ] || { echo "[$tag] convert produced nothing, see /tmp/conv_$tag.log"; return 1; }
  echo "[$tag] dataset size: $(du -sh "$OUT" | cut -f1)"
  python - "$OUT" <<'PY' 2>/dev/null || true
import sys, glob, pandas as pd, numpy as np
fs = sorted(glob.glob(sys.argv[1] + "/data/**/*.parquet", recursive=True))
df = pd.concat([pd.read_parquet(f) for f in fs])
a = np.stack(df["action"].values); s = np.stack(df["observation.state"].values)
print("   frames:", len(df), "| action min", a.min(0).round(1), "max", a.max(0).round(1))
print("   state  min", s.min(0).round(1), "max", s.max(0).round(1))
PY

  echo "[$tag] uploading..."
  for try in 1 2 3; do
    "$HF" upload "$HF_REPO" "$OUT" "batches/$tag" --repo-type dataset \
        --commit-message "$tag ($PER episodes)" && break
    echo "[$tag] upload try $try failed, retrying in 30 s"; sleep 30
    [ "$try" -eq 3 ] && { echo "[$tag] upload failed, local files kept"; return 1; }
  done

  rm -rf "$H5" "$OUT"
  touch "$STATE/$tag"
  echo "[$tag] DONE, local copy deleted"
}

fails=0
for b in $(seq 1 $N_BATCH); do
  tag=$(printf "b%02d" $b)
  [ -f "$STATE/$tag" ] && { echo "[$tag] already uploaded, skipping"; continue; }
  free_gb=$(df --output=avail -BG ~ | tail -n 1 | tr -dc 0-9)
  [ "$free_gb" -lt 6 ] && { echo "[stop] only ${free_gb} GB free"; exit 1; }
  ok=0
  for attempt in 1 2 3; do
    if run_batch "$tag"; then ok=1; break; fi
    echo "[$tag] attempt $attempt failed, cleaning up and retrying"
    pkill -9 -f generate_cube_to_tray.py 2>/dev/null; pkill -9 -f isaaclab2lerobot 2>/dev/null
    rm -f "$WORK/$tag.hdf5"; sleep 20
  done
  [ "$ok" -eq 1 ] || { echo "[stop] $tag failed 3 times, check /tmp/rec_$tag.log and /tmp/conv_$tag.log"; exit 1; }
done
echo "ALL $N_BATCH BATCHES DONE ($((N_BATCH*PER)) episodes)"
