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

    position = ee_frame.data.target_pos_source.squeeze(1)
    quat = ee_frame.data.target_quat_w.squeeze(1)
    return torch.cat((position, quat), -1)
    
