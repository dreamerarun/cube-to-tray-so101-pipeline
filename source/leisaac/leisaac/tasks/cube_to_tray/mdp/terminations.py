from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg


def cube_in_tray(env, cube_cfg: SceneEntityCfg, tray_xy, inner_half: float,
                 max_height: float = 0.035, max_speed: float = 0.05) -> torch.Tensor:
    """True when the cube rests inside the tray footprint (env-frame xy), low and nearly still."""
    cube: RigidObject = env.scene[cube_cfg.name]
    pos = cube.data.root_pos_w - env.scene.env_origins
    dx = (pos[:, 0] - tray_xy[0]).abs()
    dy = (pos[:, 1] - tray_xy[1]).abs()
    inside = (dx < inner_half) & (dy < inner_half)
    low = (pos[:, 2] < max_height) & (pos[:, 2] > 0.005)
    slow = torch.linalg.norm(cube.data.root_lin_vel_w, dim=1) < max_speed
    return inside & low & slow
