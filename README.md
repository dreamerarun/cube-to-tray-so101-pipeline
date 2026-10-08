# SO101 cube-to-tray data pipeline (LeIsaac / Isaac Sim 5.1)

Scripted demos of an SO101 arm picking a cube into a cardboard tray, recorded
in Isaac Lab (LeIsaac), converted to LeRobot format, and pushed to Hugging Face.

## Files
- `scripts/datagen/state_machine/generate_cube_to_tray.py`: state-machine demo generator (approach, align, descend, grasp, verify, place)
- `source/leisaac/leisaac/tasks/cube_to_tray/`: task and environment config (SO101, tray, cameras: front, wrist, right)
- `fix_actions.py`: rewrites each episode's actions from 8D IK poses to 6D joint targets, clipped to the SO101 USD joint limits (originals kept as `actions_ik`)
- `collect_all.sh`: loop of 30 batches x 10 episodes: record, fix actions, convert (`isaaclab2lerobotv3.py`), upload to the Hub, delete local files; resumable, with retries and a watchdog

## Use
Copy the files into a LeIsaac checkout (same paths), then:

    conda activate leisaac
    hf auth login
    # edit HF_REPO, N_BATCH, STEP_HZ at the top of collect_all.sh
    ./collect_all.sh

Each batch lands in the dataset repo under `batches/bNN` and has to be merged before training.
