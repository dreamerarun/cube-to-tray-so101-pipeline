# SO101 Cube-to-Tray Data Pipeline (LeIsaac / Isaac Sim 5.1)

Automated generation of robot-learning demonstrations for an **SO101 arm** that picks up a cube and places it in a **cardboard tray**.

A scripted state machine drives the arm in Isaac Sim / Isaac Lab (via [LeIsaac](https://github.com/LightwheelAI/leisaac)), the demos are recorded with three cameras, converted to the **LeRobot dataset format**, and uploaded to the Hugging Face Hub in small batches, so the whole run never needs much local disk.

```
 state-machine generator ──► HDF5 (10 episodes) ──► fix_actions.py ──► LeRobot converter ──► Hugging Face Hub
 (Isaac Sim window open)      8D IK actions          6D joint targets     v3 dataset + MP4      batches/b01 … b30
                                                                                                  │
                                                          local copy deleted after upload ◄───────┘
```

One batch is 10 episodes. The default run is 30 batches = **300 episodes**.

---

## Contents

1. [What is in this repo](#1-what-is-in-this-repo)
2. [Requirements](#2-requirements)
3. [Installation](#3-installation)
4. [Check your setup with one demo](#4-check-your-setup-with-one-demo)
5. [Running the full collection](#5-running-the-full-collection)
6. [Configuration reference](#6-configuration-reference)
7. [How each stage works](#7-how-each-stage-works)
8. [Dataset format](#8-dataset-format)
9. [Merging the batches](#9-merging-the-batches)
10. [Disk space and time](#10-disk-space-and-time)
11. [Troubleshooting](#11-troubleshooting)
12. [Credits](#12-credits)

---

## 1. What is in this repo

| Path | Purpose |
|---|---|
| `scripts/datagen/state_machine/generate_cube_to_tray.py` | State-machine demo generator: approach, align, descend, grasp, verify, transport, place, verify. Records episodes to HDF5. |
| `source/leisaac/leisaac/tasks/cube_to_tray/` | Task and environment config: SO101 arm, cube, cardboard tray, three cameras (`front`, `wrist`, `right`, 320x240). |
| `fix_actions.py` | Post-processes a recorded HDF5: replaces the 8D IK-pose actions with 6D joint targets, clipped to the SO101 joint limits. Originals are kept as `actions_ik`. |
| `collect_all.sh` | The driver: for each batch it records, fixes actions, converts to LeRobot, uploads to the Hub, and deletes the local files. Resumable, with retries and a watchdog. |
| `leisaac_local_changes.patch` *(optional)* | Any other edits made to existing LeIsaac files, if present. |

The repo contains only the files you add on top of LeIsaac. It is **not** a full LeIsaac checkout: copy the files into one (see [Installation](#3-installation)).

---

## 2. Requirements

### Hardware

| | Minimum used during development |
|---|---|
| GPU | NVIDIA RTX-class GPU with RT cores (developed on an RTX 4050 laptop GPU, 6 GB VRAM) |
| RAM | ~16 GB |
| Free disk | **at least 6 GB** while running (about 1.3 GB for one raw batch plus the converted batch). `collect_all.sh` stops when less than 6 GB is free. |
| Internet | Needed for Hugging Face uploads, and on the first run to stream the tray's cardboard material from NVIDIA's asset server (see [Troubleshooting](#11-troubleshooting)) |

A display is required for the default mode, because recording runs with the Isaac Sim window **open** so you can watch the arm. Conversion runs headless.

### Operating system

Linux (developed on Ubuntu). `collect_all.sh` is a Bash script and uses `systemd-inhibit` (optional, but recommended) to keep the machine awake.

### Software

| Component | Version / notes |
|---|---|
| NVIDIA driver | A driver supported by Isaac Sim 5.1 (see NVIDIA's Isaac Sim requirements page) |
| Python | 3.11 (the version Isaac Sim 5.x expects) |
| Conda | Miniconda or Anaconda, with an environment named `leisaac` (the scripts assume `conda activate leisaac`) |
| Isaac Sim | 5.1.0 |
| Isaac Lab | The release that supports Isaac Sim 5.1 |
| LeIsaac | Recent `main`, installed in editable mode. It provides the SO101 robot, the task framework and the `isaaclab2lerobot*.py` converters. |
| LeRobot | A recent release that uses **dataset format v3** (the repo uses `isaaclab2lerobotv3.py`). Installed in the same environment. |
| PyAV (`av`) | `>=15,<16`. Newer major versions removed an API that LeRobot still uses (see [Troubleshooting](#11-troubleshooting)). |
| `h5py`, `numpy`, `pandas` | Used by `fix_actions.py` and the sanity checks. Normally installed with the packages above. |
| `huggingface_hub` (CLI `hf`) | For `hf auth login` and `hf upload`. Installed with LeRobot. |
| `rsync`, `git`, `curl` | Standard Linux tools used in the setup steps |

You also need a **Hugging Face account** and a token with **write** access, plus an (empty) dataset repository to upload into.

---

## 3. Installation

> The exact Isaac Sim, Isaac Lab and PyTorch versions that go together change between releases. Follow the official install guides for those, and use the commands below as an outline. If the official docs differ from anything here, the official docs win.

### 3.1 Create the environment

```bash
conda create -n leisaac python=3.11 -y
conda activate leisaac
```

Every command below, and every run of `collect_all.sh`, must happen in a terminal where `(leisaac)` is active. If you see `python: command not found`, this is the reason.

### 3.2 Install Isaac Sim 5.1, Isaac Lab and LeIsaac

Follow the Isaac Lab pip-installation guide for Isaac Sim 5.1 (installs Isaac Sim from PyPI, the matching PyTorch build, then Isaac Lab), then the LeIsaac install guide. In outline:

```bash
# Isaac Sim 5.1 from PyPI (see the Isaac Lab docs for the exact command and torch version)
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && ./isaaclab.sh --install && cd ..

# LeIsaac
git clone https://github.com/LightwheelAI/leisaac.git
cd leisaac
pip install -e source/leisaac
```

The first time Isaac Sim starts, it asks you to accept the NVIDIA EULA and compiles shaders, which can take several minutes.

### 3.3 Install LeRobot (needed by the converter)

```bash
pip install lerobot
pip install "av>=15.0.0,<16.0.0"
python -c "import lerobot, av; print('lerobot OK, av', av.__version__)"
```

If LeIsaac's docs pin a specific LeRobot version, use that one.

### 3.4 Log in to Hugging Face and create the dataset repo

```bash
hf auth login            # paste a token with *write* access
hf auth whoami
hf repos create <your-hf-username>/cube_to_tray_carton --repo-type dataset --private
```

Never put the token in a script, a notebook cell, or a screenshot. If a token is ever exposed, delete it at huggingface.co/settings/tokens and create a new one.

### 3.5 Add this repo's files to LeIsaac

```bash
git clone https://github.com/dreamerarun/cube-to-tray-so101-pipeline.git
cd cube-to-tray-so101-pipeline
rsync -a --exclude='.git' --exclude='README.md' ./ ~/leisaac/
chmod +x ~/leisaac/collect_all.sh
```

If your LeIsaac checkout lives elsewhere, adjust the destination. The scripts assume `~/leisaac` (see `cd ~/leisaac` at the top of `collect_all.sh`, and the `WORK`/`STATE` paths in its config block). If the repo contains `leisaac_local_changes.patch`, apply it from the LeIsaac root with `git apply leisaac_local_changes.patch`.

### 3.6 Verify the install

```bash
cd ~/leisaac
# the task is registered
grep -ohE 'id="[^"]+"' source/leisaac/leisaac/tasks/cube_to_tray/__init__.py
# the converters exist
ls scripts/convert/
# conversion dependencies import
python -c "import h5py, pandas, numpy; print('python deps OK')"
```

The first command should print `LeIsaac-SO101-CubeToTray-v0`.

---

## 4. Check your setup with one demo

Before a long run, record one episode with the viewer and camera image dumps on:

```bash
cd ~/leisaac && conda activate leisaac
python scripts/datagen/state_machine/generate_cube_to_tray.py \
  --enable_cameras --cam_view file --num_demos 1 --max_attempts 1 \
  --grip_close 0.14 --clearance 0.003 --damping 3 --grip_damping 10
```

You should see the arm approach the cube, grasp it and place it in the tray. With `--cam_view file`, camera frames are written to `/tmp/leisaac_cams/` (`front`, `wrist`, `right`), which is the quickest way to check the camera views and the tray material.

Useful generator options:

| Option | Meaning |
|---|---|
| `--enable_cameras` | Render the three cameras (required for recording images) |
| `--cam_view file` / `off` | Dump camera frames to `/tmp/leisaac_cams/`, or disable that |
| `--record` | Write episodes to an HDF5 file |
| `--dataset_file PATH` | Output HDF5 path |
| `--num_demos N` | Successful episodes to record |
| `--max_attempts N` | Give up after N attempts (failed attempts are not saved) |
| `--step_hz N` | Environment stepping rate. `60` is roughly real time; `1000` runs as fast as rendering allows |
| `--grip_close`, `--clearance`, `--damping`, `--grip_damping` | State-machine tuning values (defaults used by `collect_all.sh`: `0.14`, `0.003`, `3`, `10`) |
| `--headless` | Run without a window (not used by default in `collect_all.sh`) |

Typical results: about 60 to 65 % of attempts succeed; episodes are roughly 700 to 1,600 steps long.

---

## 5. Running the full collection

```bash
cd ~/leisaac && conda activate leisaac
systemd-inhibit --what=sleep:idle --why="leisaac data collection" \
  ./collect_all.sh 2>&1 | tee -a /tmp/collect_all.log
```

Keep the laptop plugged in and the lid open. The Isaac Sim window opens and closes for every batch; this is expected, because each batch is a fresh process.

For every batch (`b01`, `b02`, ...) the script does:

1. **Record** 10 successful episodes to `datasets/_work/bNN.hdf5` (window visible, 40-minute timeout).
2. **Fix actions** (`fix_actions.py`): 8D IK poses become 6D joint targets.
3. **Check** that the file has 10 episodes and 6D actions.
4. **Convert** to a LeRobot v3 dataset (headless, with a watchdog that stops the converter when it finishes or crashes).
5. **Upload** to the Hub under `batches/bNN` (3 tries).
6. **Delete** the local HDF5 and converted folder, and write a marker file in `datasets/_done/`.

At the end it prints `ALL 30 BATCHES DONE (300 episodes)`.

### Progress and monitoring

```bash
tail -n 20 /tmp/collect_all.log        # overall progress
ls ~/leisaac/datasets/_done            # finished batches (one marker each)
df -h ~ | tail -n 1                    # free disk
tail -n 5 /tmp/rec_b05.log             # recording log of a batch
tail -n 5 /tmp/conv_b05.log            # conversion log of a batch
```

### Stopping and resuming

Press `Ctrl+C`, then make sure nothing is left holding the GPU:

```bash
pkill -9 -f collect_all.sh; pkill -9 -f generate_cube_to_tray; pkill -9 -f isaaclab2lerobot
pgrep -af "isaac|generate_cube|isaaclab2lerobot" || echo "all closed"
```

Run the same command again to resume. Batches that already have a marker in `datasets/_done/` are skipped, and the half-written batch is discarded and redone.

### Failure handling

Each batch gets up to three attempts. Between attempts the script kills stray Isaac Sim processes, deletes the partial HDF5 and waits 20 seconds. If a batch fails three times, the script stops and names the log to read in `/tmp`.

---

## 6. Configuration reference

All settings are at the top of `collect_all.sh`.

| Variable | Default | Meaning |
|---|---|---|
| `HF_REPO` | *(your dataset repo)* | Hub dataset repo that receives the batches. **Change this to your own.** |
| `N_BATCH` | `30` | Number of batches. Use `1` for a test run. |
| `PER` | `10` | Episodes per batch |
| `STEP_HZ` | `1000` | Stepping rate for recording. Use `60` for real-time viewing (much slower). |
| `FPS` | `30` | Frame rate written to the LeRobot dataset |
| `CONVERTER` | `isaaclab2lerobotv3.py` | Converter script in `scripts/convert/` |
| `TASK` | auto-detected | Gym task id, read from the task's `__init__.py` |
| `WORK` | `~/leisaac/datasets/_work` | Scratch folder for HDF5 files and the temporary LeRobot home |
| `STATE` | `~/leisaac/datasets/_done` | Marker files, one per uploaded batch |

To change a value without opening an editor:

```bash
sed -i 's/^N_BATCH=.*/N_BATCH=1            # test run/' ~/leisaac/collect_all.sh
grep -n "^N_BATCH" ~/leisaac/collect_all.sh
```

Tray appearance is controlled in `cube_to_tray_env_cfg.py`: `CARDBOARD_MDL` selects the cardboard material streamed from NVIDIA's asset server; set it to `None` for a plain brown colour.

---

## 7. How each stage works

### 7.1 Demo generator

`generate_cube_to_tray.py` runs a state machine on the SO101: move above the cube, align, descend, close the gripper, check that the grasp holds, lift, move over the tray, lower, release, and verify that the cube rests in the tray. Only episodes where both the grasp and the placement succeed are saved. The cube pose is randomised at each reset.

### 7.2 Action fix (`fix_actions.py`)

The generator is driven by end-effector IK, so the recorded `actions` are 8D (position, quaternion, gripper). A joint-space policy needs 6D joint commands instead. `fix_actions.py`:

- takes `obs/joint_pos_target` (the commanded joint targets),
- clips them to the SO101 USD joint limits (degrees: shoulder pan ±110, shoulder lift ±100, elbow −100 to 90, wrist flex ±95, wrist roll ±160, gripper −10 to 100),
- shifts them by one step, so `action[t]` is the target applied at step `t`,
- renames the original array to `actions_ik` and writes the new 6D `actions`.

The clipping matters: the IK solver sometimes asks for wrist-flex angles past the joint limit (up to about 143°), where the simulator stops the joint at 95°. Without the clip, the dataset would contain commands the arm can never reach. The script is idempotent, so running it twice is safe, and it prints how many frames were clipped.

### 7.3 Conversion

`isaaclab2lerobotv3.py` replays each successful episode and writes a LeRobot v3 dataset: a parquet table with state, action, timestamps and indices, plus one MP4 video per camera. The converter outputs joint values in **degrees**, in the same units for `observation.state` and `action`. Episodes shorter than 10 frames and unsuccessful episodes are skipped.

> The older `isaaclab2lerobot.py` converter targets dataset format v2 and fails with recent LeRobot versions (`add_frame() got an unexpected keyword argument 'task'`). Use the v3 converter.

### 7.4 Upload

Each converted batch is uploaded as a folder (`batches/bNN`) in one Hub dataset repo. The local copy is deleted only after the upload succeeds.

---

## 8. Dataset format

| Field | Shape | Notes |
|---|---|---|
| `observation.state` | 6 | Joint positions in degrees: shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, gripper |
| `action` | 6 | Joint targets in degrees, same order and units as the state |
| `observation.images.front` | 240x320x3 (MP4) | Front camera |
| `observation.images.wrist` | 240x320x3 (MP4) | Wrist camera |
| `observation.images.right` | 240x320x3 (MP4) | Right camera |
| `fps` | 30 | |

Typical numbers: about 400 to 1,500 frames per episode; a 10-episode batch converts to roughly 40 to 55 MB.

---

## 9. Merging the batches

The Hub repo holds 30 separate datasets (`batches/b01` … `batches/b30`). Merge them into one dataset before using them for anything that expects a single LeRobot dataset:

```python
import json, shutil
from pathlib import Path
from huggingface_hub import snapshot_download
from lerobot.datasets.aggregate import aggregate_datasets

SRC = snapshot_download("<your-hf-username>/cube_to_tray_carton", repo_type="dataset",
                        local_dir="./cube_to_tray_src")
batches = sorted(p.name for p in Path(SRC, "batches").iterdir() if p.is_dir())

OUT = Path("./cube_to_tray_merged")
shutil.rmtree(OUT, ignore_errors=True)       # the target folder must not exist
aggregate_datasets(
    repo_ids=[f"local/{b}" for b in batches],
    aggr_repo_id="<your-hf-username>/cube_to_tray_carton_merged",
    roots=[Path(SRC, "batches", b) for b in batches],
    aggr_root=OUT,
)

info = json.load(open(OUT / "meta" / "info.json"))
print(info["total_episodes"], "episodes |", info["total_frames"], "frames | fps", info["fps"])
```

A full 300-episode collection produced **297 episodes, about 382,000 frames, 1.5 GB** after merging (a few episodes are dropped by the converter's filters). The function's signature can change between LeRobot versions, so check it with `inspect.signature(aggregate_datasets)` if the call fails.

---

## 10. Disk space and time

| | Value |
|---|---|
| One raw 10-episode HDF5 | about 1.3 GB (compressed) |
| One converted batch | about 40 to 55 MB |
| Peak local usage | about 2 to 3 GB |
| Whole 300-episode dataset on the Hub | about 1.3 to 1.5 GB |
| Stop threshold | `collect_all.sh` exits if less than 6 GB is free |

Recording speed depends on `STEP_HZ` and the GPU. Per batch, expect the recording (a few minutes at `STEP_HZ=1000`), about two minutes for conversion and a short upload.

---

## 11. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `python: command not found` or `timeout: failed to run command 'python'` | The conda environment is not active. Run `conda activate leisaac` first. |
| `[stop] action_dim=8, expected 6` | `fix_actions.py` did not run or failed. Check `/tmp/fix_bNN.log`. |
| Converter crashes with `add_frame() got an unexpected keyword argument 'task'` | You are using the v2 converter with a recent LeRobot. Set `CONVERTER="isaaclab2lerobotv3.py"`. |
| `ImportError: 'av' is required` or `AttributeError: module 'av' has no attribute 'option'` | Wrong `av` version. Install `pip install "av>=15.0.0,<16.0.0"`. |
| The converter finishes but the process never exits | A known Isaac Sim shutdown hang. `collect_all.sh` already watches the converter log and stops the process itself. For manual runs, press `Ctrl+C` after `Finished converting`. |
| A batch hangs during recording with `cudaMemcpy failed ... device -2` in the Isaac Sim status bar | A CUDA error that leaves the process alive. The 40-minute timeout and the retry logic recover from it. Make sure no stale Isaac Sim processes still hold the GPU (`nvidia-smi`). |
| Tray walls show no cardboard texture, or a material warning appears | The material is streamed over HTTPS from NVIDIA's asset server, so the first run needs internet. Set `CARDBOARD_MDL = None` in the env config to use a plain colour. Warnings about `project_uvw` or `texture_scale` not being available are harmless. |
| `403 Forbidden ... You don't have the rights to create a dataset under the namespace` | You tried to create the repo under a name that is not yours. Use your own Hugging Face username. |
| `N_BATCH` still `1` after you tried to change it | A `sed` pattern did not match because of a trailing comment. Use `sed -i 's/^N_BATCH=.*/N_BATCH=30/'`. |
| Run stops after one batch | `N_BATCH=1` is set. Change it to `30`. |
| Upload fails | Check `hf auth whoami` and that the token has write access. Local files are kept, and rerunning the script retries. |
| Machine goes to sleep mid-run | Start the script with `systemd-inhibit` as shown above, keep it plugged in, and keep the lid open. |

---

## 12. Credits

Built on top of [LeIsaac](https://github.com/LightwheelAI/leisaac), [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim), with datasets in the [LeRobot](https://github.com/huggingface/lerobot) format and hosted on the [Hugging Face Hub](https://huggingface.co). Check the licences of those projects before redistributing files that derive from them.
