from __future__ import annotations
import math

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

target = 800000

def catchy_increase(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], num_steps: int
) -> torch.Tensor | None:
    global target
    bc = env.termination_manager.get_term("body_contact")
    val = torch.sum(bc.float()) / env.num_envs
    if val < 0.02 and env.common_step_counter > 100 and target == 800000:
        target = env.common_step_counter + num_steps

    if env.common_step_counter > target:
        catchy = env.reward_manager.get_term_cfg("catchy_points")
        catchy.weight = -2.0
        env.reward_manager.set_term_cfg("catchy_points", catchy)
        catchy_tanh = env.reward_manager.get_term_cfg("catchy_points_tanh")
        catchy_tanh.weight = 1.0
        env.reward_manager.set_term_cfg("catchy_points_tanh", catchy_tanh)
        oritentation = env.reward_manager.get_term_cfg("end_effector_orientation_tracking")
        oritentation.weight = -0.5
        env.reward_manager.set_term_cfg("end_effector_orientation_tracking", oritentation)

        # if env.common_step_counter > target + 8500:
        #     body_orientation = env.reward_manager.get_term_cfg("body_orentation")
        #     body_orientation.weight = -0.5
        #     env.reward_manager.set_term_cfg("body_orentation", body_orientation)
        

    return val

# RewardTermCfg(func=<function catch_box at 0x7fa7ec0b2830>, 
# params={'robot_cfg': SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), 
# fixed_tendon_names=None, fixed_tendon_ids=slice(None, None, None), body_names=['arm0_link_fngr'], body_ids=[23], 
# object_collection_names=None, object_collection_ids=slice(None, None, None), preserve_order=False)}, weight=-0.01)