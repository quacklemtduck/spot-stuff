from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_obs(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    position = ee_frame.data.target_pos_w.squeeze(1) - env.scene.env_origins[:]
    return position
    
