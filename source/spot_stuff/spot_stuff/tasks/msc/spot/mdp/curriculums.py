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
        if catchy.weight > -0.1:
            print("Setting weight to -0.1")
            catchy.weight = -0.2
            print("Set weight")
            env.reward_manager.set_term_cfg("catchy_points", catchy)
            print("we have Set the weight")
    else:
        catchy = env.reward_manager.get_term_cfg("catchy_points")
        if catchy.weight < -0.02:
            print("Setting weight to -0.01")
            catchy.weight = -0.01
            print("Set weight back")
            env.reward_manager.set_term_cfg("catchy_points", catchy)
            print("we have Set the weight back")
    return val

# RewardTermCfg(func=<function catch_box at 0x7fa7ec0b2830>, 
# params={'robot_cfg': SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), 
# fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=['arm0_link_fngr'], body_ids=[23], 
# object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False)}, weight=-0.01)