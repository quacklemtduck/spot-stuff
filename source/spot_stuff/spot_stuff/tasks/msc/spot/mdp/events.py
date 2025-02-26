# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""This sub-module contains the functions that can be used to enable Spot randomizations.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the randomization introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_joints_around_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints in the interval around the default position and velocity by the given ranges.

    This function samples random values from the given ranges around the default joint positions and velocities.
    The ranges are clipped to fit inside the soft joint limits. The sampled values are then set into the physics
    simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get default joint state
    joint_min_pos = asset.data.default_joint_pos[env_ids] + position_range[0]
    joint_max_pos = asset.data.default_joint_pos[env_ids] + position_range[1]
    joint_min_vel = asset.data.default_joint_vel[env_ids] + velocity_range[0]
    joint_max_vel = asset.data.default_joint_vel[env_ids] + velocity_range[1]
    # clip pos to range
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids, ...]
    joint_min_pos = torch.clamp(joint_min_pos, min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1])
    joint_max_pos = torch.clamp(joint_max_pos, min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1])
    # clip vel to range
    joint_vel_abs_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_min_vel = torch.clamp(joint_min_vel, min=-joint_vel_abs_limits, max=joint_vel_abs_limits)
    joint_max_vel = torch.clamp(joint_max_vel, min=-joint_vel_abs_limits, max=joint_vel_abs_limits)
    # sample these values randomly
    joint_pos = sample_uniform(joint_min_pos, joint_max_pos, joint_min_pos.shape, joint_min_pos.device)
    joint_vel = sample_uniform(joint_min_vel, joint_max_vel, joint_min_vel.shape, joint_min_vel.device)
    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

  
def reset_marker_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    position_range: dict = {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.0, 0.5)},
) -> None:
    """Reset the position of a marker or simple prim.
    
    Args:
        env: The environment.
        env_ids: The environment IDs to reset.
        asset_cfg: The asset configuration for the marker.
        position_range: Range for random position.
    """
    # Get the marker handle
    marker = env.scene[asset_cfg.name]
    print("----NEW----")
    print(vars(marker))
    
    # Get current poses
    current_pos, current_rot = marker.get_world_poses(indices=env_ids)
    print(current_pos)
    
    # Generate random positions within the specified ranges
    num_envs = len(env_ids)
    device = env_ids.device
    
    # Create new positions based on ranges
    new_pos = current_pos.clone()
    
    if "x" in position_range:
        x_min, x_max = position_range["x"]
        new_pos[0, 0] = torch.rand(1, device=device) * (x_max - x_min) + x_min
    
    if "y" in position_range:
        y_min, y_max = position_range["y"]
        new_pos[0, 1] = torch.rand(1, device=device) * (y_max - y_min) + y_min
    
    if "z" in position_range:
        z_min, z_max = position_range["z"]
        new_pos[0, 2] = torch.rand(1, device=device) * (z_max - z_min) + z_min
    print(new_pos)
    # Set the new poses
    marker.set_world_poses(new_pos, current_rot, env_ids)


"""
Robot methods: [
    'actuators', 'body_names', 'cfg', 'data', 'device', 'find_bodies', 'find_fixed_tendons', 'find_joints', 'fixed_tendon_names', 
    'has_debug_vis_implementation', 'has_external_wrench', 'is_fixed_base', 'is_initialized', 'joint_names', 'num_bodies', 'num_fixed_tendons', 
    'num_instances', 'num_joints', 'reset', 'root_physx_view', 'set_debug_vis', 'set_external_force_and_torque', 'set_fixed_tendon_damping', 
    'set_fixed_tendon_limit', 'set_fixed_tendon_limit_stiffness', 'set_fixed_tendon_offset', 'set_fixed_tendon_rest_length', 'set_fixed_tendon_stiffness', 
    'set_joint_effort_target', 'set_joint_position_target', 'set_joint_velocity_target', 'update', 'write_data_to_sim', 'write_fixed_tendon_properties_to_sim', 
    'write_joint_armature_to_sim', 'write_joint_damping_to_sim', 'write_joint_effort_limit_to_sim', 'write_joint_friction_to_sim', 
    'write_joint_limits_to_sim', 'write_joint_state_to_sim', 'write_joint_stiffness_to_sim', 'write_joint_velocity_limit_to_sim', 
    'write_root_com_pose_to_sim', 'write_root_com_state_to_sim', 'write_root_com_velocity_to_sim', 'write_root_link_pose_to_sim', 
    'write_root_link_state_to_sim', 'write_root_link_velocity_to_sim', 'write_root_pose_to_sim', 'write_root_state_to_sim', 'write_root_velocity_to_sim'
]
Marker methods: [
    'apply_visual_materials', 'count', 'get_applied_visual_materials', 'get_default_state', 'get_local_poses', 'get_local_scales', 'get_visibilities', 
    'get_world_poses', 'get_world_scales', 'initialize', 'initialized', 'is_non_root_articulation_link', 'is_valid', 'is_visual_material_applied', 
    'name', 'post_reset', 'prim_paths', 'prims', 'set_default_state', 'set_local_poses', 'set_local_scales', 'set_visibilities', 'set_world_poses'
]
"""