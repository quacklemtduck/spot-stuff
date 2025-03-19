from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def catchy_increase(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor | None:
    bc = env.termination_manager.get_term("body_contact")
    val = torch.sum(bc.float()) / env.num_envs
    if val < 0.02:
        catchy = env.reward_manager.get_term_cfg("catchy_points")
        catchy.weight = -0.4
        vel1 = env.reward_manager.get_term_cfg("base_angular_velocity")
        vel1.weight = 2.0
        vel2 = env.reward_manager.get_term_cfg("base_linear_velocity")
        vel2.weight = 2.0
    else:
        catchy = env.reward_manager.get_term_cfg("catchy_points")
        catchy.weight = -0.2
        vel1 = env.reward_manager.get_term_cfg("base_angular_velocity")
        vel1.weight = 5.0
        vel2 = env.reward_manager.get_term_cfg("base_linear_velocity")
        vel2.weight = 5.0

    return val

# RewardTermCfg(func=<function catch_box at 0x7fa7ec0b2830>, 
# params={'robot_cfg': SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), 
# fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=['arm0_link_fngr'], body_ids=[23], 
# object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False)}, weight=-0.01)