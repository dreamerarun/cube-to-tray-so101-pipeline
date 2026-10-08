import sys, h5py, numpy as np
# SO101 USD joint limits (deg): pan, lift, elbow, wrist_flex, wrist_roll, gripper
LIM = np.deg2rad(np.array([[-110, 110], [-100, 100], [-100, 90], [-95, 95], [-160, 160], [-10, 100]], dtype=np.float32))
path = sys.argv[1]
with h5py.File(path, "r+") as f:
    d = f["data"]; clipped = 0; total = 0; gap = []
    for k in d:
        ep = d[k]
        if "actions_ik" not in ep:
            ep.move("actions", "actions_ik")           # keep the original 8D IK actions
        elif "actions" in ep:
            del ep["actions"]                          # recompute (idempotent)
        tgt = np.asarray(ep["obs/joint_pos_target"], dtype=np.float32)
        pos = np.asarray(ep["obs/joint_pos"], dtype=np.float32)
        c = np.clip(tgt, LIM[:, 0], LIM[:, 1])
        clipped += int((c != tgt).any(1).sum()); total += len(tgt)
        new = np.concatenate([c[1:], c[-1:]], axis=0)  # action[t] = target seen at t+1
        ep.create_dataset("actions", data=new)
        gap.append(np.abs(new - pos))
    g = np.rad2deg(np.concatenate(gap))
    print(f"fixed {len(d)} episodes | frames with clipped target: {clipped}/{total}")
    print("max |action - joint_pos| per joint (deg):", g.max(0).round(0))
    print("95th percentile gap (deg):               ", np.percentile(g, 95, axis=0).round(0))
