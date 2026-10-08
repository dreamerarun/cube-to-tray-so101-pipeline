"""Scripted data generation for LeIsaac-SO101-CubeToTray-v0 (no teleoperation).

A state machine reads the cube's pose from the simulator, drives the SO101 gripper to it with
differential IK, GRASPS it only after the real jaw geometry is verified, carries it to the tray,
releases it and retreats.

GRASP GEOMETRY (the whole point of this version)
-------------------------------------------------
The previous versions drove a *guessed* constant point (URDF `gripper_frame` / `tcp_local`, or the
hand-typed `jaw + (-0.021,-0.070,0.02)` offset) onto the cube centre and declared success when that
single point was within a few millimetres.  A point can be numerically aligned while BOTH jaws are
still outside the cube, so the arm closed on air or shoved the cube sideways.

This version measures the two real jaw locations from the simulation every step:

    FIXED_JAW  = gripper body origin + R_gripper @ --fixed_jaw_local   (fixed jaw inner face)
    MOVING_JAW = jaw     body origin + R_jaw     @ --jaw_tip_local      (moving jaw face)

and derives everything from them:

    CLOSING_AXIS = the gripper-frame axis the two jaws actually separate along (dominant axis of
                   MOVING_JAW - FIXED_JAW with the along-jaw component removed) -> sign measured,
                   not guessed.
    JAW_GAP      = dot(MOVING_JAW - FIXED_JAW, CLOSING_AXIS)            (real opening, in metres)
    JAW_MIDPOINT = FIXED_JAW + 0.5 * JAW_GAP * CLOSING_AXIS
    grasp target = cube centre pushed along CLOSING_AXIS by cube/2 + clearance, so the FIXED jaw
                   face lands `clearance` away from the near cube face, and the moving jaw face is
                   JAW_GAP away on the other side.

Nothing closes unless the printed GRASP WINDOW says ALL_OK=True, and there is a hard safety gate
in get_action() that keeps the gripper OPEN if that flag is somehow false.

State machine
    approach -> descend (target XY frozen) -> GRASP WINDOW -> close -> verify (small lift)
        -> lift -> transport -> lower -> release -> settle -> verify placement
    any failure: open, rise, recompute the jaw geometry, retry (max --max_retries).

Success requires GRASP_SUCCESS and PLACE_SUCCESS.

Run a single safe episode first:

    python scripts/datagen/state_machine/generate_cube_to_tray.py \
        --enable_cameras --num_demos 1 --cam_view off --grasp_debug
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import os
import signal
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted data generation for the CubeToTray task.")
parser.add_argument("--task", type=str, default="LeIsaac-SO101-CubeToTray-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--record", action="store_true", help="Record episodes to --dataset_file (HDF5).")
parser.add_argument("--dataset_file", type=str, default="./datasets/cube_to_tray_auto.hdf5")
parser.add_argument("--resume", action="store_true", help="Append to an existing dataset file.")
parser.add_argument("--num_demos", type=int, default=300, help="Successful demos to record (0 = infinite).")
parser.add_argument("--max_attempts", type=int, default=1500, help="Stop after this many episodes even if short.")
parser.add_argument("--step_hz", type=int, default=60, help="Env stepping rate. Use a big number (e.g. 1000) for max speed.")
parser.add_argument("--quality", action="store_true", help="Quality render mode.")
parser.add_argument("--grip_close", type=float, default=0.05, help="Gripper joint target (rad) when closing.")
parser.add_argument("--standoff", type=float, default=0.06, help="Hover gap: grasp point height above cube centre before descending (m).")
parser.add_argument("--grasp_dz", type=float, default=0.0,
                    help="Trim added to the DERIVED grasp height (m). 0 keeps the jaw tips "
                         "--jaw_tip_clear above the table.")
parser.add_argument("--gain", type=float, default=1.0, help="Closed-loop gain on the grasp-point error.")
parser.add_argument("--max_step", type=float, default=0.15, help="Max IK target jump per step (m).")
parser.add_argument("--damping", type=float, default=10.0, help="Joint damping written each step (shipped state machine uses 10).")
parser.add_argument("--ik_mode", type=str, default="pose", choices=["pose", "position"],
                    help="pose: top-down gripper yawed to the cube yaw. position: xyz only.")
parser.add_argument("--no_yaw_align", action="store_true", help="Do not rotate the gripper to the cube yaw.")
parser.add_argument("--cam_view", type=str, default="window", choices=["off", "window", "file"],
                    help="live camera views: OpenCV windows (every camera in the scene), or jpg files in /tmp/leisaac_cams")
# ---------------- jaw geometry -------------------------------------------------------------
parser.add_argument("--fixed_jaw_local", type=float, nargs=3, default=[-0.0079, -0.0002, -0.0981],
                    help="Fixed-jaw inner face / gripper_frame point, in the GRIPPER body frame (m).")
parser.add_argument("--jaw_tip_local", type=float, nargs=3, default=[-0.021, -0.070, 0.020],
                    help="Moving-jaw face point, in the JAW body frame (m). Same point LeIsaac's own "
                         "ee_frame target 1 uses, so the window matches the env's grasp notion.")
parser.add_argument("--clearance", type=float, default=0.008,
                    help="Gap (m) between the fixed jaw face and the near cube face when closing.")
parser.add_argument("--jaw_tip_clear", type=float, default=0.006,
                    help="Jaw tips above the table when grasping (m).")
# ---------------- grasp window thresholds --------------------------------------------------
parser.add_argument("--lat_tol", type=float, default=0.008, help="Max |lateral| cube error vs jaw midpoint (m).")
parser.add_argument("--face_lo", type=float, default=-0.002, help="Min fixed-jaw-face to cube-face distance (m).")
parser.add_argument("--face_hi", type=float, default=0.020, help="Max fixed-jaw-face to cube-face distance (m).")
parser.add_argument("--yaw_tol", type=float, default=0.087, help="Max jaw yaw error (rad) ~ 5 deg.")
parser.add_argument("--tilt_tol", type=float, default=5.0, help="Max gripper tilt from vertical (deg).")
parser.add_argument("--bump_tol", type=float, default=0.010, help="Abort the descent if the cube moves this far (m).")
parser.add_argument("--max_retries", type=int, default=5, help="Grasp attempts per episode.")
parser.add_argument("--robot_yaw_deg", type=float, default=None,
                    help="Override the robot base yaw (deg, about world z) at spawn.")
parser.add_argument("--reach_r", type=float, nargs=2, default=[0.17, 0.27],
                    help="Cube spawn radius range from the robot base (m).")
parser.add_argument("--reach_az_deg", type=float, nargs=2, default=[-35.0, 35.0],
                    help="Cube spawn azimuth range about the robot forward axis (deg).")
parser.add_argument("--min_tray_dist", type=float, default=0.10, help="Min cube-to-tray-centre distance (m).")
parser.add_argument("--no_cube_respawn", action="store_true", help="Keep the env's own cube randomisation.")
parser.add_argument("--grip_stiffness", type=float, default=None,
                    help="Override the gripper joint stiffness at runtime (shipped value 17.8).")
parser.add_argument("--grip_damping", type=float, default=10.0,
                    help="Gripper joint damping, applied after --damping (which hits all joints).")
parser.add_argument("--carry_h", type=float, default=0.09,
                    help="Cube-centre height above the table while carrying (m).")
parser.add_argument("--place_clear", type=float, default=0.008,
                    help="Cube-centre height above the tray floor at release, minus half the cube (m).")
parser.add_argument("--cube_fwd_rng", type=float, nargs=2, default=[0.16, 0.24],
                    help="Cube spawn forward range from the robot base (m), cfg layout frame.")
parser.add_argument("--cube_right_rng", type=float, nargs=2, default=[-0.05, 0.03],
                    help="Cube spawn sideways range (m), cfg layout frame (tray is at +0.14).")
parser.add_argument("--min_tray_d", type=float, default=0.10, help="Min cube-to-tray-centre distance (m).")
parser.add_argument("--max_tray_d", type=float, default=0.18, help="Max cube-to-tray-centre distance (m).")
parser.add_argument("--snap", type=int, default=1, help="1: save annotated snapshots to /tmp/leisaac_cams/snap.")
parser.add_argument("--snap_max", type=int, default=80, help="Max snapshot events per run.")
parser.add_argument("--grasp_debug", action="store_true", help="Print every jaw/transform calculation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import leisaac.tasks  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab.envs import mdp  # noqa: E402
from isaaclab.managers import DatasetExportMode, SceneEntityCfg, TerminationTermCfg  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_inv, quat_mul  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from leisaac.enhance.managers import EnhanceDatasetExportMode, StreamingRecorderManager  # noqa: E402
from leisaac.tasks.cube_to_tray.cube_to_tray_env_cfg import (  # noqa: E402
    _TRAY_XY,
    CUBE_SIZE,
    TRAY_BASE_T,
    TRAY_INNER,
)
from leisaac.tasks.cube_to_tray.mdp import cube_in_tray  # noqa: E402

OPEN, CLOSE = 1.0, -1.0  # BinaryJointPositionAction: value < 0 -> close
SNAP_DIR = "/tmp/leisaac_cams/snap"
_ep_counter = 0
_snap_count = 0

# Each phase runs for at least `min` steps and then WAITS until the measured error is small enough
# (or `max` steps pass).  kind "move": grasp point within `tol` metres of the target.
#            kind "grip": gripper joint has stopped moving.   kind "wait": always ready.
PHASES = [
    dict(name="approach", kind="move", min=90, max=300, tol=0.015, grip=OPEN),
    dict(name="descend", kind="move", min=90, max=300, tol=0.008, grip=OPEN),
    dict(name="close", kind="grip", min=40, max=200, tol=0.0, grip=CLOSE),
    dict(name="verify", kind="move", min=40, max=160, tol=0.020, grip=CLOSE),
    dict(name="lift", kind="move", min=60, max=250, tol=0.015, grip=CLOSE),
    dict(name="transport", kind="move", min=120, max=400, tol=0.012, grip=CLOSE),
    dict(name="lower", kind="move", min=80, max=300, tol=0.010, grip=CLOSE),
    dict(name="release", kind="grip", min=30, max=150, tol=0.0, grip=OPEN),
    dict(name="retreat", kind="move", min=50, max=200, tol=0.020, grip=OPEN),
    dict(name="settle", kind="wait", min=40, max=40, tol=0.0, grip=OPEN),
]
IX = {p["name"]: i for i, p in enumerate(PHASES)}
CARRY_H = args_cli.carry_h          # grasp-point height above the table while carrying
VERIFY_LIFT = 0.03      # small lift used to verify the grasp (m)
MIN_CUBE_RISE = 0.012   # cube must rise at least this much to count as held (m)
TIP_BELOW_TCP = 0.0073  # m: jaw tip below --fixed_jaw_local (SO101 meshes: -0.1054 vs -0.0981)


class CubeToTrayStateMachine:
    """Pick-and-place driven by the MEASURED jaw geometry, never by a guessed offset."""

    def __init__(self, standoff, grasp_dz, gain, yaw_align, grip_close, max_step, use_quat):
        self.standoff, self.grasp_dz, self.gain = standoff, grasp_dz, gain
        self.yaw_align, self.grip_close, self.max_step, self.use_quat = yaw_align, grip_close, max_step, use_quat
        self.cube_size = float(CUBE_SIZE)
        # how far the fixed-jaw face point must sit BELOW the cube centre so the jaw TIPS end up
        # --jaw_tip_clear above the table.  Derived, not guessed.
        self.finger_drop = max(0.0, self.cube_size / 2.0 - (args_cli.jaw_tip_clear + TIP_BELOW_TCP)
                               + args_cli.grasp_dz)
        self.reset()

    # ---- setup --------------------------------------------------------------------------------
    def setup(self, env):
        assert env.num_envs == 1, "this script supports --num_envs 1"
        self.env = env
        robot = env.scene["robot"]
        self.robot = robot
        self.ee = env.scene["ee_frame"]
        self.ee_idx = robot.find_bodies("gripper")[0][0]
        try:
            self.jaw_idx = robot.find_bodies("jaw")[0][0]
        except Exception:  # noqa: BLE001
            self.jaw_idx = self.ee_idx
        self.grip_ids = robot.find_joints("gripper")[0]
        dev = env.device
        self.f_local = torch.tensor(args_cli.fixed_jaw_local, device=dev, dtype=torch.float32)
        self.m_local = torch.tensor(args_cli.jaw_tip_local, device=dev, dtype=torch.float32)
        self.z_axis = torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=torch.float32)
        self.info_printed = False
        print(f"[SM] robot bodies = {robot.body_names}")
        print(f"[SM] jaw geometry sources: FIXED_JAW = body '{robot.body_names[self.ee_idx]}' + "
              f"{args_cli.fixed_jaw_local} | MOVING_JAW = body '{robot.body_names[self.jaw_idx]}' + "
              f"{args_cli.jaw_tip_local}")
        print(f"[SM] clearance={args_cli.clearance * 1000:.0f} mm, jaw tip clear="
              f"{args_cli.jaw_tip_clear * 1000:.0f} mm, fixed-jaw drop below cube centre="
              f"{self.finger_drop * 1000:+.1f} mm, max_retries={args_cli.max_retries}")
        self._geometry_selfcheck(env)

    def _geometry_selfcheck(self, env):
        """Print the REAL jaw transforms once and refuse to look plausible if they do not.

        This is the whole point of the rewrite: the grasp target is derived from the two jaw
        bodies, so if those offsets are wrong nothing downstream can be trusted.  We also
        cross-check our MOVING_JAW point against LeIsaac's own `ee_frame` target 1 (the point
        `object_grasped` uses), which is the project's canonical jaw-tip location.
        """
        g = self._jaw(env)
        m_g = quat_apply(quat_inv(g["ee_q"]), g["m_w"] - g["ee"])   # MOVING_JAW in the gripper frame
        f_g = self.f_local
        print("[SM] GEOMETRY SELF-CHECK from the live body transforms "
              f"(gripper joint = {g['gj']:.3f} rad):")
        print(f"[SM]   FIXED_JAW  (gripper frame) = ({f_g[0]:+.4f},{f_g[1]:+.4f},{f_g[2]:+.4f}) m")
        print(f"[SM]   MOVING_JAW (gripper frame) = ({m_g[0, 0]:+.4f},{m_g[0, 1]:+.4f},{m_g[0, 2]:+.4f}) m")
        print(f"[SM]   JAW_GAP = {g['gap'] * 1000:.1f} mm | CLOSING_AXIS (gripper) = "
              f"({g['c_g'][0]:+.0f},{g['c_g'][1]:+.0f},{g['c_g'][2]:+.0f}) | jaw body idx = "
              f"{self.jaw_idx} of {len(self.robot.body_names)}")
        warn = []
        if not (0.05 <= abs(float(f_g[2])) <= 0.15):
            warn.append(f"--fixed_jaw_local z={float(f_g[2]):.4f} is not in the 0.05..0.15 m fingertip band")
        if not (0.05 <= abs(float(m_g[0, 2])) <= 0.15):
            warn.append(f"MOVING_JAW depth {float(m_g[0, 2]):.4f} is not in the 0.05..0.15 m fingertip band")
        if not (0.005 <= g["gap"] <= 0.15):
            warn.append(f"JAW_GAP {g['gap'] * 1000:.1f} mm at joint {g['gj']:.2f} rad is implausible")
        if warn:
            print("[SM] *** GEOMETRY SELF-CHECK FAILED ***")
            for w in warn:
                print(f"[SM]     - {w}")
            print("[SM]     the jaw offsets are wrong -> the GRASP WINDOW would be meaningless.")
            print("[SM]     set --fixed_jaw_local / --jaw_tip_local from the URDF gripper_frame "
                  "and the jaw link origin.")
        else:
            print("[SM]   geometry self-check OK")
        # cross-check against LeIsaac's own ee_frame target 1 (used by object_grasped)
        try:
            ref = self.ee.data.target_pos_w[:, 1, :]
            d = float(torch.linalg.norm(ref - g["m_w"]))
            print(f"[SM]   MOVING_JAW vs LeIsaac ee_frame target 1: {d * 1000:.2f} mm "
                  f"({'MATCH' if d < 1e-4 else 'MISMATCH -- check --jaw_tip_local'})")
        except Exception as e:  # noqa: BLE001
            print(f"[SM]   could not cross-check ee_frame target 1: {e}")

    def reset(self):
        self._idx = 0
        self._pstep = 0
        self._done = False
        self._aborted = False
        self._retry = False
        self._seg_start = None
        self._cube_hold = None
        self._cube_ref = None
        self._alpha = None
        self._alpha_hold = None
        self._ready = False
        self._m = {}
        self._printed = False
        self._window_ok = False
        self._grasp_ok = False
        self._place_ok = False
        self._grasp_target = None   # MOVING_JAW wanted world position, frozen at the end of descend
        self._verify_anchor = None  # control point where the small verification lift starts
        self._cube0z = 0.0
        self._ee0z = 0.0
        self._cube0xy = None
        self._held_local = None
        self._grip_before = None
        self._retries = 0
        global _ep_counter
        _ep_counter += 1

    @property
    def is_episode_done(self):
        return self._done

    def pre_step(self, env):
        pass

    def check_success(self, env):
        in_tray = cube_in_tray(env, cube_cfg=SceneEntityCfg("cube"), tray_xy=_TRAY_XY,
                               inner_half=TRAY_INNER / 2.0)
        c = env.scene["cube"].data.root_pos_w[0] - env.scene.env_origins[0]
        print(f"[SM] final cube-to-tray offset: dx={c[0] - _TRAY_XY[0]:+.3f} dy={c[1] - _TRAY_XY[1]:+.3f} "
              f"z={c[2]:.3f} | grasp_retries={self._retries} grasp_ok={self._grasp_ok} "
              f"place_ok={self._place_ok} cube_in_tray={bool(in_tray.all().item())}")
        ok = bool(in_tray.all().item()) and self._grasp_ok and self._place_ok
        print(f"[SM] Episode {'SUCCESS' if ok else 'FAILED'} "
              f"(GRASP_SUCCESS={self._grasp_ok} PLACE_SUCCESS={self._place_ok and bool(in_tray.all().item())})")
        return ok

    # ---- measured jaw geometry ----------------------------------------------------------------
    def _jaw(self, env):
        """Everything about the real jaws, measured from the simulation's body transforms."""
        robot = self.robot
        n = env.num_envs
        ee = robot.data.body_pos_w[:, self.ee_idx, :]
        ee_q = robot.data.body_quat_w[:, self.ee_idx, :]
        jaw = robot.data.body_pos_w[:, self.jaw_idx, :]
        jaw_q = robot.data.body_quat_w[:, self.jaw_idx, :]
        cube = env.scene["cube"].data.root_pos_w
        base_q = robot.data.root_quat_w
        binv = quat_inv(base_q)

        # --- the two actual jaw locations ---
        f_w = ee + quat_apply(ee_q, self.f_local.unsqueeze(0).repeat(n, 1))   # fixed jaw inner face
        m_w = jaw + quat_apply(jaw_q, self.m_local.unsqueeze(0).repeat(n, 1))  # moving jaw face

        # --- closing axis, in the GRIPPER frame: which gripper axis do the jaws separate along? ---
        m_g = quat_apply(quat_inv(ee_q), m_w - ee)[0]
        v = (m_g - self.f_local).clone()
        v[2] = 0.0                                   # drop the along-jaw (gripper z) component
        axis = 0 if abs(float(v[0])) >= abs(float(v[1])) else 1
        c_g = torch.zeros(3, device=v.device)
        c_g[axis] = 1.0 if float(v[axis]) >= 0.0 else -1.0   # measured sign: fixed jaw -> moving jaw
        c_w = quat_apply(ee_q, c_g.unsqueeze(0).repeat(n, 1))              # world closing axis
        # finger direction (gripper -z) and the jaw thickness axis (perpendicular to both)
        d_w = quat_apply(ee_q, torch.tensor([[0.0, 0.0, -1.0]], device=v.device).repeat(n, 1))
        l_w = torch.cross(c_w, d_w, dim=-1)
        l_w = l_w / (torch.linalg.vector_norm(l_w, dim=-1, keepdim=True) + 1e-9)

        gap_vec = m_w - f_w
        gap = float(torch.dot(gap_vec[0], c_w[0]))     # real face-to-face opening (m)
        mid_w = f_w + 0.5 * gap * c_w                  # real jaw midpoint

        # --- cube / jaw relations, all in the robot base frame (fwd,left,up) ---
        def rel(x):
            return quat_apply(binv, x - cube)[0]

        f_rel, m_rel, mid_rel = rel(f_w), rel(m_w), rel(mid_w)
        s_face = float(torch.dot(cube[0] - f_w[0], c_w[0]))   # cube centre past the fixed jaw face
        half = self.cube_size / 2.0
        lat = float(torch.dot(cube[0] - mid_w[0], l_w[0]))    # lateral cube error vs jaw midpoint
        axial = float(torch.dot(cube[0] - mid_w[0], c_w[0]))  # axial cube error vs jaw midpoint
        zf = float(torch.dot(cube[0] - f_w[0], d_w[0]))       # + = below the fixed jaw tip
        face_dist = s_face - half                             # fixed jaw face -> near cube face

        eq = quat_mul(binv, ee_q)
        cube_q = quat_mul(binv, env.scene["cube"].data.root_quat_w)[0]
        cube_yaw = float(torch.atan2(2.0 * (cube_q[0] * cube_q[3] + cube_q[1] * cube_q[2]),
                                     1.0 - 2.0 * (cube_q[2] ** 2 + cube_q[3] ** 2)))
        c_b = quat_apply(eq, c_g.unsqueeze(0).repeat(n, 1))[0]
        az = float(torch.atan2(c_b[1], c_b[0]))
        d = cube_yaw - az
        yaw_err = float(np.remainder(d + np.pi / 4.0, np.pi / 2.0) - np.pi / 4.0)
        zb = float(quat_apply(eq, self.z_axis.unsqueeze(0).repeat(n, 1))[0][2])
        tilt = float(np.degrees(np.arccos(np.clip(zb, -1.0, 1.0))))

        gj = float(robot.data.joint_pos[0, self.grip_ids[0]])
        return dict(
            f_w=f_w, m_w=m_w, mid_w=mid_w, c_g=c_g, c_w=c_w, d_w=d_w, l_w=l_w,
            gap=gap, s_face=s_face, face_dist=face_dist, lat=lat, axial=axial, zf=zf,
            yaw_err=yaw_err, cube_yaw=cube_yaw, az=az, tilt=tilt, gj=gj,
            f_rel=f_rel * 100.0, m_rel=m_rel * 100.0, mid_rel=mid_rel * 100.0,
            ee=ee, ee_q=ee_q, cube=cube,
        )

    def _want_moving_jaw(self, g, cube):
        """World position the MOVING jaw face should occupy.

        Built from the two measured jaws, not from a guessed gripper-body offset:

            FIXED_JAW  = cube - (cube/2 + clearance) * CLOSING_AXIS + finger_drop * DOWN
                         (c runs fixed jaw -> moving jaw, so the cube centre ends up
                          (cube/2 + clearance) PAST the fixed jaw face: the jaws straddle it)
            MOVING_JAW = FIXED_JAW + (current jaw gap vector)

        The gap vector keeps the target valid whatever the gripper is currently open to, so
        driving the moving-jaw face onto this point lands the fixed jaw face exactly `clearance`
        away from the near cube face.  Raising it by --standoff gives the hover pose, hence the
        descent is a pure vertical drop on a frozen XY.
        """
        c_w, d_w = g["c_w"], g["d_w"]
        f_want = (cube
                  - (self.cube_size / 2.0 + args_cli.clearance) * c_w
                  + self.finger_drop * d_w)
        gap_vec = g["m_w"] - g["f_w"]
        return f_want + gap_vec

    # ---- grasp window -------------------------------------------------------------------------
    def _grasp_window(self, g, cube):
        """All of these must be True before the gripper is allowed to close."""
        half = self.cube_size / 2.0
        cube_moved = 0.0
        if self._cube_ref is not None:
            cube_moved = float(torch.linalg.norm((cube - self._cube_ref)[0, :2]))
        gj = g["gj"]
        pos_ok = abs(g["lat"]) <= args_cli.lat_tol
        jaw_ok = (args_cli.face_lo <= g["face_dist"] <= args_cli.face_hi
                  and abs(g["axial"]) + half <= 0.5 * g["gap"] + 1e-6
                  and -0.025 <= g["zf"] <= 0.010      # cube vertically inside the jaw contact region
                  and g["gap"] > self.cube_size + 0.005)
        yaw_ok = abs(g["yaw_err"]) <= args_cli.yaw_tol if self.yaw_align else True
        tilt_ok = g["tilt"] <= args_cli.tilt_tol
        open_ok = gj > 0.6
        static_ok = cube_moved <= args_cli.bump_tol
        all_ok = bool(pos_ok and jaw_ok and yaw_ok and tilt_ok and open_ok and static_ok)
        return dict(position_ok=bool(pos_ok), jaw_geometry_ok=bool(jaw_ok), yaw_ok=bool(yaw_ok),
                    tilt_ok=bool(tilt_ok), gripper_open=bool(open_ok), cube_static=bool(static_ok),
                    cube_moved=cube_moved, ALL_OK=all_ok)

    def _print_jaw(self, g, tag=""):
        pre = "[SM] " if not tag else f"[SM] {tag}: "
        print(f"{pre}FIXED_JAW relative cube (fwd,left,up) cm = "
              f"({g['f_rel'][0]:+.1f},{g['f_rel'][1]:+.1f},{g['f_rel'][2]:+.1f})")
        print(f"{pre}MOVING_JAW relative cube (fwd,left,up) cm = "
              f"({g['m_rel'][0]:+.1f},{g['m_rel'][1]:+.1f},{g['m_rel'][2]:+.1f})")
        print(f"{pre}JAW_GAP relative cube = {g['gap'] * 1000:.1f} mm (opening), "
              f"cube half = {self.cube_size / 2 * 1000:.1f} mm")
        print(f"{pre}CUBE_CENTER relative jaw_midpoint = "
              f"(axial {g['axial'] * 1000:+.1f}, lateral {g['lat'] * 1000:+.1f}, "
              f"below-tip {g['zf'] * 1000:+.1f}) mm")
        print(f"{pre}CLOSING_AXIS (gripper frame) = "
              f"({g['c_g'][0]:+.0f},{g['c_g'][1]:+.0f},{g['c_g'][2]:+.0f}) | "
              f"in base (fwd,left,up) = ({g['c_w'][0, 0]:+.2f},{g['c_w'][0, 1]:+.2f},{g['c_w'][0, 2]:+.2f})")
        print(f"{pre}JAW_FACE_DISTANCE = {g['face_dist'] * 1000:+.1f} mm "
              f"(fixed jaw face -> near cube face; want {args_cli.face_lo * 1000:+.0f}.."
              f"{args_cli.face_hi * 1000:+.0f} mm) | jaw yaw err = "
              f"{np.degrees(g['yaw_err']):+.1f} deg | tilt = {g['tilt']:.1f} deg | gripper = {g['gj']:.2f} rad")

    # ---- camera diagnostics -------------------------------------------------------------------
    def _cam_img(self, sensor):
        out = getattr(getattr(sensor, "data", None), "output", None)
        if out is None or not hasattr(out, "keys") or "rgb" not in out:
            return None
        img = out["rgb"][0].detach().cpu().numpy()
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        return np.ascontiguousarray(img[..., :3])

    def _project(self, sensor, pts):
        """World (N,3) -> pixel (N,2); nan where the point is behind the camera."""
        try:
            d = sensor.data
            K = d.intrinsic_matrices[0]
            pos, q = d.pos_w[0], d.quat_w_ros[0]
            pc = quat_apply(quat_inv(q).unsqueeze(0).repeat(pts.shape[0], 1), pts - pos.unsqueeze(0))
            uv = (K.unsqueeze(0) @ pc.unsqueeze(-1)).squeeze(-1)
            uv = uv[:, :2] / uv[:, 2:3]
            uv[pc[:, 2] <= 0.0] = float("nan")
            return uv.cpu().numpy(), pc[:, 2].detach().cpu().numpy()
        except Exception as e:  # noqa: BLE001
            if not getattr(self, "_proj_warned", False):
                print(f"[SM] camera projection failed: {e}")
                self._proj_warned = True
            return None, None

    def _snap(self, env, tag, g=None, want=None):
        """Save every RGB camera with cube / jaws / midpoint / TCP / target / gripper annotated."""
        global _snap_count
        if not args_cli.snap or _snap_count >= args_cli.snap_max:
            return
        try:
            if g is None:
                g = self._jaw(env)
            pts = {
                "cube": g["cube"][0],
                "cube_c": g["cube"][0],
                "fixJAW": g["f_w"][0],
                "movJAW": g["m_w"][0],
                "mid": g["mid_w"][0],
                "TCP": g["m_w"][0],
                "grip": g["ee"][0],
            }
            if want is not None:
                pts["target"] = want[0]
            cols = {"cube": (255, 0, 255), "cube_c": (255, 0, 255), "fixJAW": (255, 0, 0),
                    "movJAW": (0, 128, 255), "mid": (0, 255, 255), "TCP": (0, 255, 0),
                    "grip": (0, 160, 255), "target": (255, 255, 0)}
            os.makedirs(SNAP_DIR, exist_ok=True)
            keys = list(pts.keys())
            P = torch.stack([pts[k] for k in keys])
            for name, sensor in env.scene.sensors.items():
                img = self._cam_img(sensor)
                if img is None:
                    print(f"[SM] snap {tag} | cam {name}: no RGB output")
                    continue
                uv, depth = self._project(sensor, P)
                if uv is None:
                    print(f"[SM] snap {tag} | cam {name}: projection unavailable")
                    continue
                bad = [k for k, (u, v) in zip(keys, uv) if not (np.isfinite(u) and np.isfinite(v))]
                if bad:
                    why = "behind camera" if depth is not None and np.all(depth[list(keys.index(b) for b in bad)] <= 0) \
                        else "non-finite projection"
                    print(f"[SM] snap {tag} | cam {name}: NaN for {bad} -> {why}"
                          + ("  <-- wrist camera is probably too close (clipping_range starts at 0.01 m)"
                             if name.lower().find("wrist") >= 0 else ""))
                for k, (u, v) in zip(keys, uv):
                    if not (np.isfinite(u) and np.isfinite(v)):
                        continue
                    col = cols[k]
                    cv2.circle(img, (int(u), int(v)), 5, col, 1)
                    cv2.putText(img, k, (int(u) + 6, int(v) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
                def px(k):
                    i = keys.index(k)
                    return uv[i]
                if np.all(np.isfinite(px("cube"))) and np.all(np.isfinite(px("fixJAW"))):
                    print(f"[SM] snap {tag} | cam {name}: cube px=({px('cube')[0]:.0f},{px('cube')[1]:.0f}) "
                          f"fixJaw px=({px('fixJAW')[0]:.0f},{px('fixJAW')[1]:.0f}) -> "
                          f"cube-fixJaw = ({px('cube')[0] - px('fixJAW')[0]:+.0f},"
                          f"{px('cube')[1] - px('fixJAW')[1]:+.0f}) px | "
                          f"cube-movJaw = ({px('cube')[0] - px('movJAW')[0]:+.0f},"
                          f"{px('cube')[1] - px('movJAW')[1]:+.0f}) px | "
                          f"cube-mid = ({px('cube')[0] - px('mid')[0]:+.0f},"
                          f"{px('cube')[1] - px('mid')[1]:+.0f}) px")
                img2 = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                path = f"{SNAP_DIR}/ep{_ep_counter:03d}_{_snap_count:02d}_{tag}_{name}.png"
                cv2.imwrite(path, cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
            _snap_count += 1
        except Exception as e:  # noqa: BLE001
            print(f"[SM] snapshot failed: {e}")

    # ---- phase bookkeeping --------------------------------------------------------------------
    def advance(self):
        if self._done:
            return
        ph = PHASES[self._idx]
        self._pstep += 1
        finished = self._pstep >= ph["min"] and self._ready
        timed_out = self._pstep >= ph["max"] and not finished
        if timed_out:
            m = self._m
            extra = f" err={m.get('err', 0) * 100:.1f} cm" if ph["kind"] == "move" else f" gripper vel={m.get('gv', 0):.3f}"
            print(f"[SM] phase '{ph['name']}' timed out after {self._pstep} steps;{extra}")
        if not (finished or timed_out):
            return
        self._retry = False
        self._on_phase_end(ph["name"])
        if self._aborted:
            self._done = True
            return
        if self._retry:
            # _fail() already re-initialised the state back to 'approach'
            return
        self._idx += 1
        self._pstep = 0
        self._ready = False
        if self._idx >= len(PHASES):
            self._done = True

    def _fail(self, why):
        """Open, rise, recompute the jaw geometry, retry.  Jumps the phase machine immediately so a
        bump mid-descent cannot leave the arm running the old trajectory for another 100 steps."""
        self._retries += 1
        print(f"[SM] {why} -> retry {self._retries}/{args_cli.max_retries}")
        if self._retries > args_cli.max_retries:
            print("[SM] giving up on this episode")
            self._aborted = True
            self._retry = False
            self._done = True        # end the episode now, even if _fail() came from get_action()
            return
        self._retry = True
        self._idx = IX["approach"]
        self._pstep = 0
        self._ready = False
        self._seg_start = None
        self._cube_hold = None      # reacquire the cube from scratch
        self._cube_ref = None
        self._alpha = None
        self._alpha_hold = None
        self._window_ok = False
        self._grasp_target = None
        self._verify_anchor = None
        self._held_local = None

    def _on_phase_end(self, name):
        m = self._m
        _lim = {"approach": 0.02, "lift": 0.05, "transport": 0.05, "lower": 0.05}
        if name in _lim and m.get("err", 0.0) > _lim[name]:
            print(f"[SM] STUCK in '{name}' (err={m['err'] * 100:.1f} cm) -> cube/tray not reachable "
                  f"from this pose, skipping episode")
            self._aborted = True
            return
        if name == "approach":
            self._snap(self.env, "approach", m.get("jaw"), m.get("want"))
            print("[SM] approach done -> descending on a FROZEN target XY")
        elif name == "descend":
            g, w = m["jaw"], m.get("window")
            self._print_jaw(g, "end of descend")
            if w is None:
                w = self._grasp_window(g, m["cube"])
            print(f"[SM] GRASP WINDOW: position_ok={w['position_ok']} jaw_geometry_ok={w['jaw_geometry_ok']} "
                  f"yaw_ok={w['yaw_ok']} tilt_ok={w['tilt_ok']} gripper_open={w['gripper_open']} "
                  f"cube_static={w['cube_static']} (cube moved {w['cube_moved'] * 1000:.1f} mm) "
                  f"ALL_OK={w['ALL_OK']}")
            if not w["ALL_OK"]:
                self._snap(self.env, "grasp_window_fail", g, m.get("want"))
                self._fail("GRASP WINDOW failed -> NOT closing")
                return
            self._window_ok = True
            self._grasp_target = m.get("want")
            if self._grasp_target is None:
                self._grasp_target = self._want_moving_jaw(g, m["cube"])
            self._snap(self.env, "grasp_window", g, self._grasp_target)
            print("[SM] calculated jaw target "
                  f"= ({float(self._grasp_target[0, 0]):+.4f},{float(self._grasp_target[0, 1]):+.4f},"
                  f"{float(self._grasp_target[0, 2]):+.4f}) world | grasp command: OPEN -> CLOSE")
        elif name == "close":
            self._snap(self.env, "grasp_closed", m.get("jaw"), m.get("want"))
            g = m["jaw"]
            held = self._window_ok and g["gj"] > max(self.grip_close, -0.09) + 0.05
            print(f"[SM] close check: stall angle={g['gj']:.3f} rad, jaw gap={g['gap'] * 1000:.1f} mm, held={held}")
            before = m.get("grip_before")
            before = float("nan") if before is None else float(before)
            print(f"[SM] gripper before close = {before:.2f} rad | "
                  f"gripper after close = {g['gj']:.2f} rad (target {self.grip_close}) | "
                  f"gripper settled (vel) = {m.get('gv', 0):.3f} | "
                  f"jaw contact state = {'CONTACT (joint stalled on the cube)' if held else 'NO CONTACT'}")
            if not held:
                self._snap(self.env, "grasp_failed", g, m.get("want"))
                self._fail("closed on nothing -> opening, rising, recomputing jaw geometry")
                return
            print(f"[SM] jaw gap after close = {g['gap'] * 1000:.1f} mm vs cube "
                  f"{self.cube_size * 1000:.0f} mm -> small lift to VERIFY")
        elif name == "verify":
            g = m["jaw"]
            cube_rise = m["cube_z"] - self._cube0z
            ee_rise = m["ee_z"] - self._ee0z
            dxy = float(torch.linalg.norm((m["cube"][0, :2] - self._cube0xy[0, :2])))
            _v0 = getattr(self, "_v_local0", None)
            _loc = quat_apply(quat_inv(g["ee_q"]), m["cube"] - g["ee"])[0]
            drift = float(torch.linalg.norm(_loc - _v0)) if _v0 is not None else 1.0
            between = drift < 0.015
            success = bool(ee_rise > 0.008 and cube_rise > 0.5 * ee_rise and between)
            print(f"[SM] VERIFY detail: cube moved {drift * 1000:.1f} mm inside the gripper frame "
                  f"during the lift (want < 15 mm) -> held={between}")
            print(f"[SM] VERIFY: gripper_lift={ee_rise * 1000:.1f} mm cube_lift={cube_rise * 1000:.1f} mm "
                  f"cube_gripper_relative={dxy * 1000:.1f} mm cube_between_jaws={between} "
                  f"grasp_success={success}")
            if not success:
                self._snap(self.env, "grasp_failed", g, m.get("want"))
                self._fail("cube did not follow the gripper -> GRASP FAILED")
                return
            self._grasp_ok = True
            self._snap(self.env, "successful_lift", g, m.get("want"))
            print("[SM] grasp SUCCESS -> transport to tray")
            # cube relative to the CONTROL POINT (moving jaw face), in the gripper frame
            self._held_local = quat_apply(quat_inv(g["ee_q"]), m["cube"] - g["m_w"])[0].clone()
        elif name == "lower":
            _c = self.env.scene["cube"].data.root_pos_w[0] - self.env.scene.env_origins[0]
            print(f"[SM] LOWERED: cube-to-tray-centre dx={float(_c[0]) - _TRAY_XY[0]:+.3f} "
                  f"dy={float(_c[1]) - _TRAY_XY[1]:+.3f} z={float(_c[2]):.3f} "
                  f"(tray inner half-width {TRAY_INNER / 2:.3f} m; want |dx|,|dy| < 0.03)")
        elif name == "release":
            print("[SM] released -> settling, then PLACE VERIFY")
        elif name == "settle":
            in_tray = cube_in_tray(self.env, cube_cfg=SceneEntityCfg("cube"), tray_xy=_TRAY_XY,
                                   inner_half=TRAY_INNER / 2.0)
            c = self.env.scene["cube"].data.root_pos_w[0] - self.env.scene.env_origins[0]
            tray_floor = TRAY_BASE_T + 0.001
            resting = float(c[2]) < tray_floor + self.cube_size + 0.006
            self._place_ok = bool(in_tray.all().item()) and resting
            print(f"[SM] PLACE VERIFY: cube_in_tray={bool(in_tray.all().item())} resting_on_tray={resting} "
                  f"cube_z={float(c[2]):.3f} -> PLACE_SUCCESS={self._place_ok}")
            self._snap(self.env, "place", m.get("jaw"), m.get("want"))

    # ---- control -------------------------------------------------------------------------------
    def get_action(self, env):
        robot = self.robot
        robot.write_joint_damping_to_sim(damping=args_cli.damping)
        robot.write_joint_damping_to_sim(args_cli.grip_damping, joint_ids=self.grip_ids)
        if args_cli.grip_stiffness is not None:
            robot.write_joint_stiffness_to_sim(args_cli.grip_stiffness, joint_ids=self.grip_ids)
        dev = env.device
        origin = env.scene.env_origins
        base_pos, base_quat = robot.data.root_pos_w, robot.data.root_quat_w
        ph = PHASES[self._idx]
        name = ph["name"]

        grip_pos = robot.data.body_pos_w[:, self.ee_idx, :]
        cube_obj = env.scene["cube"]
        cube = cube_obj.data.root_pos_w.clone()

        g = self._jaw(env)
        tcp = g["m_w"]                                   # control point = measured moving jaw face
        ee_q = g["ee_q"]
        n = env.num_envs
        up = torch.tensor([[0.0, 0.0, 1.0]], device=dev)

        def pack(local, alpha, gcmd):
            if not self.use_quat:
                return torch.cat([local, gcmd], dim=-1)
            q_world = torch.zeros(1, 4, device=dev)
            q_world[:, 0] = torch.cos(alpha / 2)
            q_world[:, 3] = torch.sin(alpha / 2)
            q_local = quat_mul(quat_inv(base_quat), q_world)
            return torch.cat([local, q_local, gcmd], dim=-1)

        # ---- cube pose tracking: live ONLY while the gripper is high above it ----
        if name == "approach":
            self._cube_hold = cube.clone()
            yaw = euler_xyz_from_quat(cube_obj.data.root_quat_w)[2]
            a = torch.remainder(yaw + torch.pi / 4, torch.pi / 2) - torch.pi / 4
            self._alpha = a if self.yaw_align else torch.zeros_like(a)
        elif name == "descend" and self._pstep == 0:
            # FREEZE: from here on the arm must never chase a cube that got bumped
            self._cube_ref = cube.clone()
            self._alpha_hold = self._alpha.clone() if self._alpha is not None else torch.zeros(1, device=dev)
            print("[SM] calculated jaw target from the measured jaws:")
            self._print_jaw(g, "pre_grasp")
            self._snap(env, "pre_grasp", g)
        elif name == "verify" and self._pstep == 0:
            self._cube0z = float(cube[0, 2])
            self._ee0z = float(grip_pos[0, 2])
            self._cube0xy = cube.clone()
            self._verify_anchor = tcp.clone()
            self._v_local0 = quat_apply(quat_inv(ee_q), cube - grip_pos)[0].clone()
            print("[SM] jaws closed on the cube -> snapshot before the verification lift")
            self._snap(env, "verify", g, self._grasp_target)
        hold = self._cube_hold if self._cube_hold is not None else cube
        if name == "approach":
            alpha = self._alpha
        else:
            alpha = self._alpha_hold if self._alpha_hold is not None else self._alpha
        if alpha is None:
            alpha = torch.zeros(1, device=dev)

        # ---- bump detection while lowering: abort immediately, never push the cube sideways ----
        if name == "descend" and self._cube_ref is not None:
            moved = float(torch.linalg.norm((cube - self._cube_ref)[0, :2]))
            if moved > args_cli.bump_tol:
                self._print_jaw(g, "bump_detected")
                self._snap(env, "bump_detected", g)
                self._fail(f"bumped the cube {moved * 1000:.1f} mm while lowering -> aborting the grasp")
                target_w = grip_pos + 0.03 * up          # rise straight up, jaws opening
                local = quat_apply(quat_inv(base_quat), target_w - base_pos)
                return pack(local, alpha, torch.full((1, 1), OPEN, device=dev))

        tray = origin.clone()
        tray[:, 0] += _TRAY_XY[0]
        tray[:, 1] += _TRAY_XY[1]
        z0 = origin[:, 2]

        # carry yaw: when the base swings toward the tray, swing the gripper yaw with it so
        # wrist_roll does not have to cancel the whole swing and run out of range
        if name in ("transport", "lower", "release", "retreat", "settle") and self._cube_ref is not None:
            _bxy = base_pos[0, :2]
            _vc, _vt = self._cube_ref[0, :2] - _bxy, tray[0, :2] - _bxy
            _d = torch.atan2(_vt[1], _vt[0]) - torch.atan2(_vc[1], _vc[0])
            _d = torch.atan2(torch.sin(_d), torch.cos(_d))
            if name == "transport":
                _s = min(1.0, (self._pstep + 1) / ph["min"])
                _s = _s * _s * (3.0 - 2.0 * _s)
            else:
                _s = 1.0
            alpha = alpha + _s * _d

        def at(src, z):
            p = src.clone()
            p[:, 2] = z
            return p

        def cp(desired_cube_w):
            """Control-point (= moving jaw face) position that puts the CUBE at desired_cube_w."""
            if self._held_local is not None:
                return desired_cube_w - quat_apply(ee_q, self._held_local.unsqueeze(0).repeat(n, 1))
            return desired_cube_w - (tcp - cube)

        place_z = z0 + TRAY_BASE_T + CUBE_SIZE / 2.0 + args_cli.place_clear
        want_m = self._want_moving_jaw(g, hold)          # geometry-driven grasp pose
        end = {
            "approach": want_m + self.standoff * up,     # hover: the grasp pose raised vertically
            "descend": want_m,                           # vertical drop onto the frozen grasp pose
            "close": None,                               # hold the arm completely still
            "verify": None,                              # anchor + VERIFY_LIFT, set below
            "lift": cp(at(hold, z0 + CARRY_H)),
            "transport": cp(at(tray, z0 + CARRY_H)),
            "lower": cp(at(tray, place_z)),
            "release": None,
            "retreat": cp(at(tray, z0 + CARRY_H)),
            "settle": cp(at(tray, z0 + CARRY_H)),
        }[name]
        if name == "verify":
            if self._verify_anchor is None:
                self._verify_anchor = tcp.clone()
            end = self._verify_anchor + VERIFY_LIFT * up

        # ---- measurements used by the phase logic / diagnostics ----
        gj, gv = g["gj"], float(robot.data.joint_vel[0, self.grip_ids[0]])
        if name == "close" and self._pstep == 0:
            self._grip_before = gj
            print(f"[SM] grasp command: OPEN -> CLOSE | gripper before close = {gj:.2f} rad")
        if end is None:                                  # hold the control point exactly where it is
            end = tcp
        err_vec = end - tcp
        win = self._grasp_window(g, cube) if name in ("descend", "close") else None
        self._m = dict(
            gj=gj, gv=gv, err=float(torch.linalg.norm(err_vec)),
            tcp_minus_cube=(tcp - cube)[0].clone(), cube=cube.clone(),
            dist=float(torch.linalg.norm(tcp - cube)),
            cube_z=float(cube[0, 2]), ee_z=float(grip_pos[0, 2]),
            jaw=g, want=want_m, window=win, grip_before=self._grip_before,
        )

        if args_cli.grasp_debug and name in ("approach", "descend"):
            self._print_jaw(g, "debug")
        if ph["kind"] == "move":
            self._ready = self._m["err"] < ph["tol"]
        elif ph["kind"] == "grip":
            self._ready = abs(gv) < 0.02
        else:
            self._ready = True
        # The descent does not end until the GRASP WINDOW itself is satisfied: this makes the
        # window the convergence criterion rather than a post-hoc report.
        if name == "descend" and win is not None:
            self._ready = self._ready and win["ALL_OK"]
            if self._pstep % 50 == 0 and self._pstep > 0:
                print(f"[SM] descend step {self._pstep}: err={self._m['err'] * 1000:.1f} mm "
                      f"face_dist={g['face_dist'] * 1000:+.1f} mm lat={g['lat'] * 1000:+.1f} mm "
                      f"zf={g['zf'] * 1000:+.1f} mm yaw={np.degrees(g['yaw_err']):+.1f} deg "
                      f"tilt={g['tilt']:.1f} deg ALL_OK={win['ALL_OK']}")

        if not self._printed:
            c_local = quat_apply(quat_inv(base_quat), cube - base_pos)[0]
            t_local = quat_apply(quat_inv(base_quat), tray - base_pos)[0]
            reach = float(torch.hypot(c_local[0], c_local[1]))
            fw = quat_apply(base_quat, torch.tensor([[1.0, 0.0, 0.0]], device=dev))[0]
            print(f"[SM] cube (fwd,left)=({c_local[0]:.3f},{c_local[1]:.3f}) reach={reach:.3f} m | "
                  f"tray (fwd,left)=({t_local[0]:.3f},{t_local[1]:.3f}) | cube yaw="
                  f"{np.degrees(g['cube_yaw']):+.0f} deg | closing axis azimuth={np.degrees(g['az']):+.0f} deg | "
                  f"robot forward in world xy=({float(fw[0]):+.2f},{float(fw[1]):+.2f})")
            if c_local[0] < 0.05 or reach > 0.34:
                print("[SM]   <-- cube may be out of reach / behind the arm (check ARM_FWD)")
            import math as _m
            _ac = _m.degrees(_m.atan2(float(c_local[1]), float(c_local[0])))
            _at = _m.degrees(_m.atan2(float(t_local[1]), float(t_local[0])))
            print(f"[SM] YAW FIX: cube is {_ac:+.1f} deg and tray {_at:+.1f} deg off the robot's forward axis "
                  f"-> set ROBOT_YAW_OFFSET_DEG += {(_ac + _at) / 2:+.1f}")
            self._printed = True

        # ---- waypoint interpolation in grasp-point space ----
        if self._pstep == 0:
            self._seg_start = tcp.clone()
        if name in ("close", "release"):
            want = tcp                                   # never move the arm while the jaws act
        else:
            s = min(1.0, (self._pstep + 1) / ph["min"])
            a = s * s * (3.0 - 2.0 * s)                  # smoothstep
            want = self._seg_start + (end - self._seg_start) * a

        # ---- closed loop: move the gripper body so the measured jaw face lands on `want` ----
        err = torch.clamp(want - tcp, -self.max_step, self.max_step)
        target_w = grip_pos + self.gain * err
        local = quat_apply(quat_inv(base_quat), target_w - base_pos)
        gcmd = torch.full((1, 1), ph["grip"], device=dev)

        # ---- HARD SAFETY GATE: no phase may close the jaws unless the GRASP WINDOW passed ----
        if name == "close" and not self._window_ok:
            print("[SM] SAFETY: GRASP WINDOW not satisfied -> gripper forced OPEN, arm held")
            self._fail("safety gate tripped")
            return pack(quat_apply(quat_inv(base_quat), grip_pos - base_pos), alpha,
                        torch.full((1, 1), OPEN, device=dev))

        return pack(local, alpha, gcmd)


# -------------------------------------------------------------------------------------------------
_cam_count = 0


def show_cameras(env, mode):
    """Show every RGB camera in the scene (front + wrist + side for this task)."""
    global _cam_count
    _cam_count += 1
    if _cam_count % 2:  # every 2nd step is plenty
        return
    for name, sensor in env.scene.sensors.items():
        out = getattr(getattr(sensor, "data", None), "output", None)
        if out is None or not hasattr(out, "keys") or "rgb" not in out:
            continue
        img = out["rgb"][0].detach().cpu().numpy()
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
        if mode == "window":
            cv2.imshow(name, bgr)
        else:
            os.makedirs("/tmp/leisaac_cams", exist_ok=True)
            tmp = f"/tmp/leisaac_cams/{name}.tmp.jpg"
            cv2.imwrite(tmp, bgr)
            os.replace(tmp, f"/tmp/leisaac_cams/{name}.jpg")
    if mode == "window":
        cv2.waitKey(1)


class RateLimiter:
    def __init__(self, hz):
        self.sleep_duration = 1.0 / hz
        self.last_time = time.time()
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        next_wakeup = self.last_time + self.sleep_duration
        while time.time() < next_wakeup:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time += self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def place_cube(env):
    """Random cube pose inside the reachable box, in the cfg's own (fwd, right) layout frame."""
    if args_cli.no_cube_respawn:
        return
    import math
    from leisaac.tasks.cube_to_tray.cube_to_tray_env_cfg import ARM_FWD, _RIGHT
    with torch.inference_mode():
        cube = env.scene["cube"]
        o = env.scene.env_origins[0]
        for _ in range(500):
            f = float(np.random.uniform(*args_cli.cube_fwd_rng))
            r = float(np.random.uniform(*args_cli.cube_right_rng))
            x = f * ARM_FWD[0] + r * _RIGHT[0]
            y = f * ARM_FWD[1] + r * _RIGHT[1]
            d = math.hypot(x - _TRAY_XY[0], y - _TRAY_XY[1])
            if args_cli.min_tray_d <= d <= args_cli.max_tray_d:
                break
        cyaw = float(np.random.uniform(-0.4, 0.4))
        pose = torch.zeros(1, 7, device=env.device)
        pose[0, 0], pose[0, 1], pose[0, 2] = x + float(o[0]), y + float(o[1]), CUBE_SIZE / 2.0 + 0.001 + float(o[2])
        pose[0, 3], pose[0, 6] = math.cos(cyaw / 2), math.sin(cyaw / 2)
        ids = torch.tensor([0], device=env.device)
        cube.write_root_pose_to_sim(pose, env_ids=ids)
        cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=ids)
        print(f"[SM] cube placed: fwd={f:.3f} right={r:.3f} tray_dist={d:.3f} m yaw={math.degrees(cyaw):+.0f} deg")


def auto_terminate(env, success: bool):
    fn = (lambda e: torch.ones(e.num_envs, dtype=torch.bool, device=e.device)) if success else (
        lambda e: torch.zeros(e.num_envs, dtype=torch.bool, device=e.device)
    )
    env.termination_manager.set_term_cfg("success", TerminationTermCfg(func=fn))
    env.termination_manager.compute()


def main():
    out_dir = os.path.dirname(args_cli.dataset_file)
    out_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    if args_cli.robot_yaw_deg is not None:
        import math
        h = math.radians(args_cli.robot_yaw_deg) / 2.0
        env_cfg.scene.robot.init_state.rot = (math.cos(h), 0.0, 0.0, math.sin(h))  # (w,x,y,z)
        print(f"[SM] robot yaw overridden to {args_cli.robot_yaw_deg} deg")
    if args_cli.quality:
        env_cfg.sim.render.antialiasing_mode = "FXAA"
        env_cfg.sim.render.rendering_mode = "quality"

    # IK-pose actions (7D pose in the base frame) + binary gripper: 8D action
    env_cfg.task_type = "so101_state_machine"
    env_cfg.actions.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        body_name="gripper",
        controller=mdp.DifferentialIKControllerCfg(
            command_type=args_cli.ik_mode, ik_method="dls", ik_params={"lambda_val": 0.04}
        ),
    )
    env_cfg.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        open_command_expr={"gripper": 1.0},
        close_command_expr={"gripper": args_cli.grip_close},
    )
    env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True

    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    env_cfg.terminations.success = None

    if args_cli.record:
        if args_cli.resume:
            assert os.path.exists(args_cli.dataset_file), "--resume needs an existing dataset file"
            env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_ALL_RESUME
        else:
            assert not os.path.exists(args_cli.dataset_file), "dataset exists; use --resume or another --dataset_file"
            env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
        env_cfg.recorders.dataset_export_dir_path = out_dir
        env_cfg.recorders.dataset_filename = out_name
        env_cfg.terminations.success = TerminationTermCfg(
            func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        )
    else:
        env_cfg.recorders = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # no gravity on any robot link (matches the shipped state-machine generator)
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    for prim in omni.usd.get_context().get_stage().Traverse():
        if "Robot" in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)

    if args_cli.record:
        del env.recorder_manager
        env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
        env.recorder_manager.flush_steps = 100
        env.recorder_manager.compression = "lzf"

    limiter = RateLimiter(args_cli.step_hz)
    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()
    place_cube(env)
    sm = CubeToTrayStateMachine(
        args_cli.standoff, args_cli.grasp_dz, args_cli.gain, not args_cli.no_yaw_align,
        args_cli.grip_close, args_cli.max_step, use_quat=(args_cli.ik_mode == "pose"),
    )
    sm.setup(env)
    sm.reset()

    resumed = 0
    if args_cli.record and args_cli.resume:
        resumed = env.recorder_manager._dataset_file_handler.get_num_episodes()
        print(f"Resuming with {resumed} episodes already in the file.")

    successes, attempts = 0, 0
    interrupted = False

    def on_sigint(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] Ctrl+C: finishing up...")

    old_handler = signal.signal(signal.SIGINT, on_sigint)
    try:
        while simulation_app.is_running() and not simulation_app.is_exiting() and not interrupted:
            with torch.inference_mode():
                if sm.is_episode_done:
                    ok = sm.check_success(env)
                    attempts += 1
                    successes += int(ok)
                    print(f"Episode {attempts}: {'SUCCESS' if ok else 'failed'}  ({successes}/{attempts} ok)")
                    if args_cli.record:
                        auto_terminate(env, ok)
                    env.reset()  # the recorder exports the finished episode here
                    place_cube(env)
                    sm.reset()
                    if args_cli.record:
                        auto_terminate(env, False)
                    if args_cli.num_demos > 0 and successes >= args_cli.num_demos:
                        print(f"Collected {successes} successful demos. Done.")
                        break
                    if attempts >= args_cli.max_attempts:
                        print(f"Reached --max_attempts={args_cli.max_attempts}. Stopping.")
                        break
                else:
                    sm.pre_step(env)
                    env.step(sm.get_action(env))
                    sm.advance()
                limiter.sleep(env)
                if args_cli.cam_view != "off":
                    show_cameras(env, args_cli.cam_view)
    except Exception as e:  # noqa: BLE001
        import traceback

        print(f"\n[ERROR] {e}\n")
        traceback.print_exc()
    finally:
        signal.signal(signal.SIGINT, old_handler)
        if args_cli.record and hasattr(env.recorder_manager, "finalize"):
            env.recorder_manager.finalize()
        if args_cli.cam_view == "window":
            cv2.destroyAllWindows()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
