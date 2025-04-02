from dataclasses import MISSING
import torch
from isaaclab.managers import CommandTermCfg, CommandTerm
from isaaclab.utils import configclass
from typing import TYPE_CHECKING, Sequence
from isaaclab.assets import Articulation
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import WorldPoseCommandCfg


@configclass
class WorldPoseCommand(CommandTerm):
    cfg: "WorldPoseCommandCfg" = MISSING # type: ignore

    def __init__(self, cfg: "WorldPoseCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env) # type: ignore
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_w[:, 3] = 1.0
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

        self.origins = torch.cat((env.scene.env_origins, torch.zeros(self.num_envs, 4, device=self.device)), 1)
    def __str__(self) -> str:
        msg = "WorldPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg
    
    @property
    def command(self) -> torch.Tensor:
        # Extract the first 3 elements for subtraction
        result = self.pose_command_w[:, :3] - (
            self.robot.data.body_link_state_w[:, self.body_idx, :3] - self.origins[:, :3]
        )

        # Concatenate the remaining elements of body_link_state_w[:, self.body_idx] - origins
        remaining = self.robot.data.body_link_state_w[:, self.body_idx, 3:7] - self.origins[:, 3:]
        # Combine the first 3 elements (result) with the remaining elements
        return torch.cat((result, remaining), dim=-1)

    
    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3] + self.origins[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_link_state_w[:, self.body_idx, :3],
            self.robot.data.body_link_state_w[:, self.body_idx, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- position
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_w[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_w[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_w[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_w[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        self.pose_command_w[env_ids, 3:] = quat

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3] + self.origins[:,:3], self.pose_command_w[:, 3:])