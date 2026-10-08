import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs.mdp import reset_root_state_uniform
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from ..template import (
    SingleArmObservationsCfg,
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
)
from ..template import mdp as tmdp
from .mdp import cube_in_tray

# ----------------------------------------------------------------------------
# Layout. The robot base sits at the world origin; table top is at z = 0.
# "fwd" = distance from the base in the direction the arm reaches,
# "right" = the robot's right-hand side. All metres.
# If the arm points the other way in the viewport, change ARM_FWD to (0.0, 1.0).
# ----------------------------------------------------------------------------
ROBOT_YAW_DEG = 270.0   # the ONLY orientation setting: robot, cube, tray and cameras all follow it
ARM_FWD = (round(float(np.cos(np.radians(ROBOT_YAW_DEG))), 6), round(float(np.sin(np.radians(ROBOT_YAW_DEG))), 6))
_RIGHT = (ARM_FWD[1], -ARM_FWD[0])

# Robot base yaw so the arm faces ARM_FWD (assumes the robot's native forward is world +X).
# If it still faces the wrong way, set ROBOT_YAW_OFFSET_DEG = 180.0 (or +/-90.0 if it faces sideways).
# (ROBOT_YAW_OFFSET_DEG removed: use ROBOT_YAW_DEG above)
# Turns ONLY the robot (cube, tray and cameras stay where ROBOT_YAW_DEG puts them).
ROBOT_FACE_FIX_DEG = 90.0
ROBOT_YAW = float(np.radians(ROBOT_YAW_DEG + ROBOT_FACE_FIX_DEG))

CUBE_FWD, CUBE_RIGHT = 0.20, -0.07          # cube centre of the random region
CUBE_RANGE_FWD, CUBE_RANGE_RIGHT = 0.03, 0.04  # +/- random offset each episode
CUBE_YAW_RANGE = 0.4                        # +/- radians
CUBE_SIZE = 0.03

TRAY_FWD, TRAY_RIGHT = 0.20, 0.14           # fixed tray position
TRAY_INNER = 0.09
TRAY_WALL_T, TRAY_WALL_H, TRAY_BASE_T = 0.005, 0.035, 0.004   # wall thickness, wall HEIGHT, floor

CAM_W, CAM_H = 320, 240

CARDBOARD_MDL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Materials/MI_CardBoxB_05.mdl"   # set to None for the plain brown colour
CARDBOARD_TEX_SCALE = (1.0, 1.0)

# NVIDIA carton asset. Set CARTON_USD = None to keep the plain brown tray.
# The carton is used as the VISUAL; the invisible tray cuboids keep acting as its collision walls,
# so the box must be open-topped and about TRAY_INNER (0.09 m) wide inside.
CARTON_USD = None   # the NVIDIA carton has closed flaps; the open box below is used instead
CARTON_SCALE = (0.25, 0.25, 0.25)
CARTON_OFFSET = (0.0, 0.0, 0.0)    # shift if the asset origin is not its base centre


def _w(fwd: float, right: float):
    """robot-local (fwd, right) -> world xy"""
    return (fwd * ARM_FWD[0] + right * _RIGHT[0], fwd * ARM_FWD[1] + right * _RIGHT[1])


def _look_at_quat_ros(eye, target):
    """Quaternion (w, x, y, z) for a ROS-convention camera at `eye` looking at `target`."""
    f = np.array(target, float) - np.array(eye, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f], axis=1)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return (float(w), float(x), float(y), float(z))


def _static_box(name, size, pos, color):
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=(
                sim_utils.MdlFileCfg(mdl_path=CARDBOARD_MDL)
                if (CARDBOARD_MDL and name.startswith('Tray'))
                else sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.8)
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


_TRAY_XY = _w(TRAY_FWD, TRAY_RIGHT)
_CUBE_XY = _w(CUBE_FWD, CUBE_RIGHT)
_TABLE_XY = _w(0.25, 0.0)
_WALL_OFF = TRAY_INNER / 2.0 + TRAY_WALL_T / 2.0
_WALL_LEN = TRAY_INNER + 2.0 * TRAY_WALL_T
_WALL_Z = TRAY_BASE_T + TRAY_WALL_H / 2.0
_TRAY_COLOR = (0.76, 0.60, 0.42)

_FRONT_EYE = (*_w(0.50, 0.0), 0.45)
_FRONT_TGT = (*_w(0.17, 0.0), 0.02)

_RIGHT_EYE = (*_w(0.14, 0.42), 0.20)  # right-hand side of the robot, looking across the workspace
_RIGHT_TGT = (*_w(0.14, 0.0), 0.05)


@configclass
class CubeToTraySceneCfg(SingleArmTaskSceneCfg):
    """White table, yellow cube, fixed tray, SO-101, front + wrist cameras."""

    # (the template declares `scene`; here it is the white table)
    scene: AssetBaseCfg = _static_box("Table", (0.9, 0.9, 0.05), (_TABLE_XY[0], _TABLE_XY[1], -0.025), (0.95, 0.95, 0.95))

    tray_base: AssetBaseCfg = _static_box(
        "TrayBase", (_WALL_LEN, _WALL_LEN, TRAY_BASE_T), (_TRAY_XY[0], _TRAY_XY[1], TRAY_BASE_T / 2.0), _TRAY_COLOR
    )
    tray_wall_a: AssetBaseCfg = _static_box(
        "TrayWallA", (TRAY_WALL_T, _WALL_LEN, TRAY_WALL_H), (_TRAY_XY[0] + _WALL_OFF, _TRAY_XY[1], _WALL_Z), _TRAY_COLOR
    )
    tray_wall_b: AssetBaseCfg = _static_box(
        "TrayWallB", (TRAY_WALL_T, _WALL_LEN, TRAY_WALL_H), (_TRAY_XY[0] - _WALL_OFF, _TRAY_XY[1], _WALL_Z), _TRAY_COLOR
    )
    tray_wall_c: AssetBaseCfg = _static_box(
        "TrayWallC", (_WALL_LEN, TRAY_WALL_T, TRAY_WALL_H), (_TRAY_XY[0], _TRAY_XY[1] + _WALL_OFF, _WALL_Z), _TRAY_COLOR
    )
    tray_wall_d: AssetBaseCfg = _static_box(
        "TrayWallD", (_WALL_LEN, TRAY_WALL_T, TRAY_WALL_H), (_TRAY_XY[0], _TRAY_XY[1] - _WALL_OFF, _WALL_Z), _TRAY_COLOR
    )

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0), roughness=0.6),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(_CUBE_XY[0], _CUBE_XY[1], CUBE_SIZE / 2.0 + 0.001)),
    )

    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04), rot=(-0.404379, -0.912179, -0.0451242, 0.0486914), convention="ros"
        ),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=CAM_W,
        height=CAM_H,
        update_period=1 / 30.0,
    )

    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/front_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=_FRONT_EYE, rot=_look_at_quat_ros(_FRONT_EYE, _FRONT_TGT), convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=CAM_W,
        height=CAM_H,
        update_period=1 / 30.0,
    )


    right: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/right_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=_RIGHT_EYE, rot=_look_at_quat_ros(_RIGHT_EYE, _RIGHT_TGT), convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=CAM_W,
        height=CAM_H,
        update_period=1 / 30.0,
    )


@configclass
class CubeToTrayObservationsCfg(SingleArmObservationsCfg):

    @configclass
    class PolicyCfg(SingleArmObservationsCfg.PolicyCfg):
        wrist = ObsTerm(
            func=tmdp.image, params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False}
        )

        right = ObsTerm(
            func=tmdp.image, params={"sensor_cfg": SceneEntityCfg("right"), "data_type": "rgb", "normalize": False}
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class CubeToTrayTerminationsCfg(SingleArmTerminationsCfg):

    success = DoneTerm(
        func=cube_in_tray,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "tray_xy": _TRAY_XY,
            "inner_half": TRAY_INNER / 2.0,
        },
    )


@configclass
class CubeToTrayEnvCfg(SingleArmTaskEnvCfg):
    """Pick the yellow cube and put it in the tray; the cube moves every reset, the tray never does."""

    scene: CubeToTraySceneCfg = CubeToTraySceneCfg(env_spacing=8.0)
    observations: CubeToTrayObservationsCfg = CubeToTrayObservationsCfg()
    terminations: CubeToTrayTerminationsCfg = CubeToTrayTerminationsCfg()

    task_description: str = "Pick up the yellow cube and place it in the tray."

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (*_w(0.60, 0.30), 0.55)
        self.viewer.lookat = (*_w(0.17, 0.0), 0.0)

        self.scene.robot.init_state.pos = (0.0, 0.0, 0.0)

        # black robot
        self.scene.robot.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.03, 0.03, 0.03), roughness=0.45
        )

        # NVIDIA carton in place of the tray
        if CARTON_USD:
            self.scene.carton = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Carton",
                spawn=sim_utils.UsdFileCfg(usd_path=CARTON_USD, scale=CARTON_SCALE),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(_TRAY_XY[0] + CARTON_OFFSET[0], _TRAY_XY[1] + CARTON_OFFSET[1], CARTON_OFFSET[2])
                ),
            )
            for _n in ("tray_base", "tray_wall_a", "tray_wall_b", "tray_wall_c", "tray_wall_d"):
                getattr(self.scene, _n).spawn.visible = False
        # face the arm toward the cube / tray (quaternion w,x,y,z for a yaw about world Z)
        self.scene.robot.init_state.rot = (float(np.cos(ROBOT_YAW / 2.0)), 0.0, 0.0, float(np.sin(ROBOT_YAW / 2.0)))

        # new random cube pose on every reset (offsets are added to the cube's default pose)
        rx = abs(ARM_FWD[0]) * CUBE_RANGE_FWD + abs(_RIGHT[0]) * CUBE_RANGE_RIGHT
        ry = abs(ARM_FWD[1]) * CUBE_RANGE_FWD + abs(_RIGHT[1]) * CUBE_RANGE_RIGHT
        self.events.randomize_cube = EventTerm(
            func=reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-rx, rx), "y": (-ry, ry), "z": (0.0, 0.0), "yaw": (-CUBE_YAW_RANGE, CUBE_YAW_RANGE)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("cube"),
            },
        )
