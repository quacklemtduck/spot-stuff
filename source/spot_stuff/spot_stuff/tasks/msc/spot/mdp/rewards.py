# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the reward functions that can be used for Spot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.sensors import FrameTransformer
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import wrap_to_pi
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, euler_xyz_from_quat
##
# Task Rewards
##


def air_time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Reward longer feet air and contact time."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1).unsqueeze(dim=1).expand(-1, 4)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
    reward = torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        torch.where(t_max < mode_time, t_min, 0),
        stance_cmd_reward,
    )
    return torch.sum(reward, dim=1)


def base_angular_velocity_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, 2]
    ang_vel_error = torch.linalg.norm((target - asset.data.root_ang_vel_b[:, 2]).unsqueeze(1), dim=1)
    return torch.exp(-ang_vel_error / std)


def base_linear_velocity_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float, ramp_at_vel: float = 1.0, ramp_rate: float = 0.5
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, :2]
    lin_vel_error = torch.linalg.norm((target - asset.data.root_lin_vel_b[:, :2]), dim=1) #TODO: look into changing this
    # fixed 1.0 multiple for tracking below the ramp_at_vel value, then scale by the rate above
    vel_cmd_magnitude = torch.linalg.norm(target, dim=1)
    velocity_scaling_multiple = torch.clamp(1.0 + ramp_rate * (vel_cmd_magnitude - ramp_at_vel), min=1.0)
    return torch.exp(-lin_vel_error / std) * velocity_scaling_multiple


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_lin_vel_b[:, :2], dim=1)
        return torch.where(
            torch.logical_or(cmd > 0.0, body_vel > self.velocity_threshold), sync_reward * async_reward, 0.0
        )

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


##
# Regularization Penalties
##


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    #print(torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action),dim=1))
    return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


# ! look into simplifying the kernel here; it's a little oddly complex
def base_motion_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize base vertical and roll/pitch velocity"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return 0.8 * torch.square(asset.data.root_lin_vel_b[:, 2]) + 0.2 * torch.sum(
        torch.abs(asset.data.root_ang_vel_b[:, :2]), dim=1
    )


def base_orientation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize non-flat base orientation

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.projected_gravity_b[:, :2]), dim=1)


def foot_slip_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)

    reward = is_contact * foot_planar_velocity
    return torch.sum(reward, dim=1)


def joint_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


def joint_torques_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint torques on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)

def arm_velocity_penalty(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    #print(robot.data.joint_vel.shape)
    #print(robot.data.joint_vel[0])
    return torch.linalg.norm((robot.data.joint_vel), dim=1)


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)



def good_boy_points(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Reward function for encouraging a robot to stay at its initial position.

    Args:
        env: The Isaac Lab environment.
        asset_cfg: Configuration for the asset (robot).

    Returns:
        A torch.Tensor containing the reward for each robot instance.
    """
    # Extract the asset
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get current position of the robot
    current_positions = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]

    # Get the initial position of the robot (assuming it's stored in the environment)
    initial_positions = asset.data.default_root_state[:, :2]

    # Calculate the distance from the initial position
    distances = torch.linalg.norm(current_positions - initial_positions, dim=1)

    # Calculate the reward using an exponential decay based on distance
    reward = torch.exp(distances) - 1

    # Optional: Add a small penalty for excessive joint movements (example)
    # joint_velocities = asset.data.joint_vel  # type: ignore
    # joint_movement_penalty = torch.sum(torch.abs(joint_velocities), dim=1) * 0.01
    # reward -= joint_movement_penalty

    return reward

def catch_box(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Reward function for encouraging a robot to move arm to box.

    Args:
        env: The Isaac Lab environment.
        asset_cfg: Configuration for the asset (robot).
        box_cfg: Configuration for the box .

    Returns:
        A torch.Tensor containing the reward for each robot instance.
    """
    # Extract the asset
    #asset: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Get current position of the robot
    #current_positions =  asset.data.body_pos_w[:, robot_cfg.body_ids].squeeze(1) - env.scene.env_origins[:]
    current_positions = ee_frame.data.target_pos_source.squeeze(1)
    #print("Current" ,current_positions[0])
    # Get the initial position of the robot (assuming it's stored in the environment)
    target = env.command_manager.get_command("goal_command")[:, :3]
    #print(target[0])
    # Calculate the distance from the initial position
    distances = torch.linalg.norm(current_positions - target, dim=1)
    #print(distances[0])
    return distances
    # # Define parameters for the reward function
    # max_distance = 10.0  # Maximum distance to consider for reward
    # reward_scale = 1.0  # Scaling factor for the reward

    # # Calculate the reward using an exponential decay based on distance
    # reward = torch.exp(-distances**2 / (2 * (max_distance / 3) ** 2)) * reward_scale

    # # Optional: Add a small penalty for excessive joint movements (example)
    # # joint_velocities = asset.data.joint_vel  # type: ignore
    # # joint_movement_penalty = torch.sum(torch.abs(joint_velocities), dim=1) * 0.01
    # # reward -= joint_movement_penalty

    # return reward

def catch_box_tanh(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """
    Reward function for encouraging a robot to move arm to box.

    Args:
        env: The Isaac Lab environment.
        asset_cfg: Configuration for the asset (robot).
        box_cfg: Configuration for the box .

    Returns:
        A torch.Tensor containing the reward for each robot instance.
    """
    # Extract the asset
    # Extract the asset
    #asset: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Get current position of the robot
    #current_positions =  asset.data.body_pos_w[:, robot_cfg.body_ids].squeeze(1) - env.scene.env_origins[:]
    current_positions = ee_frame.data.target_pos_source.squeeze(1)
    # Get the initial position of the robot (assuming it's stored in the environment)
    target = env.command_manager.get_command("goal_command")[:, :3]
    
    # Calculate the distance from the initial position
    distances = torch.linalg.norm(current_positions - target, dim=1)
    return 1 - torch.tanh(distances / std)

def feet_on_ground(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type: ignore
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_magnitudes = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1) # type: ignore
    #print(contact_magnitudes.shape)
    #is_contact = contact_magnitudes > 1.0
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.0
    # Calculate contact reward - all feet should be in contact
    num_feet = len(sensor_cfg.body_ids) # type: ignore
    num_feet_in_contact = torch.sum(is_contact, dim=1)
    contact_reward = num_feet_in_contact / num_feet
    return contact_reward


def catch_box_move(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Reward function for encouraging a robot to move arm to box.

    Args:
        env: The Isaac Lab environment.
        asset_cfg: Configuration for the asset (robot).
        box_cfg: Configuration for the box .

    Returns:
        A torch.Tensor containing the reward for each robot instance.
    """
    # Extract the asset
    asset: Articulation = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Get current position of the robot
    #current_positions =  asset.data.body_pos_w[:, robot_cfg.body_ids].squeeze(1) - env.scene.env_origins[:]
    current_positions = ee_frame.data.source_pos_w - env.scene.env_origins
    # Get the initial position of the robot (assuming it's stored in the environment)
    target = env.command_manager.get_command("goal_command")[:, :3]
    
    # Calculate the distance from the initial position
    distances = torch.linalg.norm(current_positions - target, dim=1)
    joint_vels = asset.data.joint_vel[:, asset_cfg.joint_ids]
    #print("WEEEEEEEEEE")
    #print(joint_vels[0])
    joint_vels = torch.linalg.norm(joint_vels, dim=1)
    #print(distances[0])
    #print(joint_vels[0])
    #print(distances / (joint_vels + 1e-6))
    return distances / (joint_vels + 1e-6)

def catch_box_move_towards(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Reward function for encouraging a robot to move arm to box.

    Args:
        env: The Isaac Lab environment.
        asset_cfg: Configuration for the asset (robot).
        box_cfg: Configuration for the box .

    Returns:
        A torch.Tensor containing the reward for each robot instance.
    """
    # Extract the asset
    asset: Articulation = env.scene[asset_cfg.name]
    asset.data.body_vel_w
    # Get current position of the robot
    current_positions =  asset.data.body_pos_w[:, asset_cfg.body_ids].squeeze(1) - env.scene.env_origins[:]
    #current_positions = ee_frame.data.target_pos_w.squeeze(1) - env.scene.env_origins
    # Get the initial position of the robot (assuming it's stored in the environment)
    target = env.command_manager.get_command("goal_command")[:, :3]

    direction_vectors = target - current_positions
    direction_unit_vectors = direction_vectors / torch.linalg.norm(direction_vectors, dim=1, keepdim=True)

    velocities = asset.data.body_vel_w[:, asset_cfg.body_ids].squeeze(1)

    dot_product = torch.sum(velocities[:, :3] * direction_unit_vectors, dim=1)
    reward = torch.clamp(dot_product, min=0)
    #print(reward)
    return reward

def catch_box_old(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Reward function for encouraging a robot to move arm to box.

    Args:
        env: The Isaac Lab environment.
        asset_cfg: Configuration for the asset (robot).
        box_cfg: Configuration for the box .

    Returns:
        A torch.Tensor containing the reward for each robot instance.
    """
    # Extract the asset
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Get current position of the robot
    #current_positions =  asset.data.body_pos_w[:, robot_cfg.body_ids].squeeze(1) - env.scene.env_origins[:]
    current_positions = ee_frame.data.target_pos_w.squeeze(1) - env.scene.env_origins
    # Get the initial position of the robot (assuming it's stored in the environment)
    target = env.command_manager.get_command("goal_command")[:, :3]
    # Calculate the distance from the initial position
    #print("Distance:",(current_positions - target)[0])
    distances = torch.linalg.norm(current_positions - target, dim=1)
    #print("Norm:", distances[0])
    # Define parameters for the reward function
    max_distance = 10.0  # Maximum distance to consider for reward
    reward_scale = 1.0  # Scaling factor for the reward

    # Calculate the reward using an exponential decay based on distance
    reward = torch.exp(-distances**2 / (2 * (max_distance / 3) ** 2)) * reward_scale
    #print("Reward:", reward[0])
    # Optional: Add a small penalty for excessive joint movements (example)
    # joint_velocities = asset.data.joint_vel  # type: ignore
    # joint_movement_penalty = torch.sum(torch.abs(joint_velocities), dim=1) * 0.01
    # reward -= joint_movement_penalty

    return reward


def orientation_command_error(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Get current position of the robot
    #current_positions =  asset.data.body_pos_w[:, robot_cfg.body_ids].squeeze(1) - env.scene.env_origins[:]
    # obtain the desired and current orientations
    command = env.command_manager.get_command("goal_command")
    des_quat = command[:,3:7]
    curr_quat = ee_frame.data.target_quat_w.squeeze(1)

    return quat_error_magnitude(des_quat, curr_quat)


def body_orientation_penalty_exp(env: ManagerBasedRLEnv,
                                 asset_cfg: SceneEntityCfg,
                                 alpha: float = 5.0
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    rotation = robot.data.body_quat_w[:, asset_cfg.body_ids].squeeze(1)
    yaw = euler_xyz_from_quat(rotation)[2]
    
    # error = -yaw  # we want 0 − yaw
    # e = torch.abs(error)

    #convert from [0, 2π) to [−π, π] so small negative angles aren’t huge positives
    yaw = wrap_to_pi(yaw)
    #print("YAW:", yaw)
    # your “error” from zero heading
    e = torch.abs(yaw)

    # exponential penalty that is 0 at e=0, and grows for large e
    return torch.exp(alpha * e) - 1.0
