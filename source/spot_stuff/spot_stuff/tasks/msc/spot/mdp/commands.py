from dataclasses import MISSING
import datetime
import torch
from isaaclab.managers import CommandTermCfg, CommandTerm
from isaaclab.utils import configclass
from typing import TYPE_CHECKING, Sequence
from isaaclab.assets import Articulation
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .commands_cfg import WorldPoseCommandCfg


@configclass
class WorldPoseCommand(CommandTerm):
    cfg: "WorldPoseCommandCfg" = MISSING # type: ignore

    def __init__(self, cfg: "WorldPoseCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env) # type: ignore
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        self.ee_frame: FrameTransformer = env.scene[cfg.ee_name]
        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_w[:, 3] = 1.0
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

        self.origins = torch.cat((env.scene.env_origins, torch.zeros(self.num_envs, 4, device=self.device)), 1)
        if self.cfg.print_metrics:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            metrics_filename = f'metrics_log_{timestamp}.csv'
            self.metrics_file = open(metrics_filename, 'w')
            # Optional: write header
            self.metrics_file.write('step, command_counter, position_error, orientation_error\n')

    def __del__(self):
        super().__del__()
        if hasattr(self, 'metrics_file'):
            self.metrics_file.close()
    

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
        remaining = self.pose_command_w[:, 3:]
        # Combine the first 3 elements (result) with the remaining elements
        return torch.cat((result, remaining), dim=-1)

    
    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3] + self.origins[:, :3],
            self.pose_command_w[:, 3:],
            self.ee_frame.data.target_pos_w[:, 0],
            self.ee_frame.data.target_quat_w[:, 0],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)
        if self.cfg.print_metrics:
            #print(f'{self._env.common_step_counter}, {(self.command_counter - 1)[0]}, {self.metrics["position_error"].mean()}, {self.metrics["orientation_error"].mean()}') # type: ignore
            metrics_line = f'{self._env.common_step_counter}, {(self.command_counter - 1)[0] % len(self.cfg.positions)}, {self.metrics["position_error"].mean()}, {self.metrics["orientation_error"].mean()}\n' # type: ignore
            self.metrics_file.write(metrics_line)
            # Optional: flush to ensure data is written
            self.metrics_file.flush()
            if (self.command_counter)[0] > len(self.cfg.positions) * 2:
                print("DONE")
                self.cfg.print_metrics = False
                self.metrics_file.close()
                exit(0)
        

    def _resample_command(self, env_ids: Sequence[int]):
        if self.cfg.positions is not None and len(self.cfg.positions) > 0:
            # Convert positions list to a tensor for easy indexing
            # (Do this conversion once if possible, not in _resample_command)
            # For demonstration, converting here:
            positions_tensor = torch.tensor(self.cfg.positions, dtype=torch.float32, device=self.device)
            num_available_positions = positions_tensor.shape[0]

            # Get the current command counters for the selected environments
            current_counters = (self.command_counter[env_ids] - 1) % len(self.cfg.positions)

            # Create a mask for environments where the counter is a valid index
            valid_mask = current_counters < num_available_positions

            # Get the environment IDs from env_ids where the counter is valid
            valid_env_ids = torch.tensor(env_ids, device=self.command_counter.device)[valid_mask]

            # Get the counters that are valid indices
            valid_counters = current_counters[valid_mask]

            # If there are any valid commands to set
            if valid_env_ids.numel() > 0:
                # Get the new commands by indexing the positions tensor
                new_commands = positions_tensor[valid_counters]

                # Assign the new commands to the selected environments
                self.pose_command_w[valid_env_ids] = new_commands
            return
        # sample new pose targets
        # -- position
        
        cube_centers = self.robot.data.body_link_state_w[:, self.body_idx, :3] - self.origins[:, :3] # Center position of the cube deadzone [x, y, z]
        cube_width = 0.9  # Width of the cube (x-dimension)
        cube_height = 0.5  # Height of the cube (y-dimension)
        cube_depth = 0.3  # Depth of the cube (z-dimension)
        
        # Calculate cube bounds
        # x_min = cube_center[:, 0] - cube_width/2
        # x_max = cube_center[:, 0] + cube_width/2
        # y_min = cube_center[:, 1] - cube_height/2
        # y_max = cube_center[:, 1] + cube_height/2
        # z_min = cube_center[:, 2] - 5
        # z_max = cube_center[:, 2] + cube_depth/2
        half_width = torch.tensor([cube_width/2, cube_height/2, cube_depth/2], device=self.device)
        cube_min = cube_centers - half_width
        cube_max = cube_centers + half_width
        cube_min[:, 2] -= 5
        # Sample positions
        valid_positions = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        max_attempts = 10  # Avoid infinite loops
        if isinstance(env_ids, torch.Tensor):
        # If already a tensor, ensure it's on the correct device
            env_ids_tensor = env_ids.to(device=self.device)
        else:
            # If it's a sequence (list, tuple, etc.), convert to tensor
            env_ids_tensor = torch.tensor(env_ids, device=self.device)
        for attempt in range(max_attempts):
            # Find which environments still need valid positions
            invalid_mask = ~valid_positions
            n_invalid = invalid_mask.sum().item()
            
            if n_invalid == 0:
                break
                
            # Get the indices of invalid environments
            invalid_indices = torch.where(invalid_mask)[0]
            # print("Resample", attempt)
            # print(env_ids)
            # print(invalid_indices)
            #invalid_env_ids = torch.tensor([env_ids[i] for i in invalid_indices], device=self.device) # type: ignore
            invalid_env_ids = env_ids_tensor[invalid_mask]
            # Sample new positions for invalid environments
            self.pose_command_w[invalid_env_ids, 0] = torch.empty(n_invalid, device=self.device).uniform_(*self.cfg.ranges.pos_x) # type: ignore
            self.pose_command_w[invalid_env_ids, 1] = torch.empty(n_invalid, device=self.device).uniform_(*self.cfg.ranges.pos_y) # type: ignore
            self.pose_command_w[invalid_env_ids, 2] = torch.empty(n_invalid, device=self.device).uniform_(*self.cfg.ranges.pos_z) # type: ignore
            # Check if positions are outside their respective cubes
            # Check if positions are outside their respective cubes
            # Note: We need to index the cube bounds by the original env_ids
            sampled_positions = self.pose_command_w[env_ids_tensor, :3]
            cube_min_for_envs = cube_min[env_ids_tensor]
            cube_max_for_envs = cube_max[env_ids_tensor]
            
            outside_min = (sampled_positions < cube_min_for_envs)
            outside_max = (sampled_positions > cube_max_for_envs)
            outside_any_dim = outside_min | outside_max
            valid_positions = outside_any_dim.any(dim=1)
            # Check if positions are outside the cube
            # for i, env_id in enumerate(env_ids):
            #     pos = self.pose_command_w[env_id, :3]
            #     # A point is outside the cube if any coordinate is outside the cube's bounds
            #     outside_x = (pos[0] < x_min) or (pos[0] > x_max)
            #     outside_y = (pos[1] < y_min) or (pos[1] > y_max)
            #     outside_z = (pos[2] < z_min) or (pos[2] > z_max)
            #     valid_positions[i] = outside_x or outside_y or outside_z
    
    # If we still have invalid positions after max attempts, place them outside the cube
        # For any remaining invalid positions, place them outside the cube
        invalid_mask = ~valid_positions
        if invalid_mask.any():
            invalid_indices = torch.where(invalid_mask)[0]
            invalid_env_ids = torch.tensor([env_ids[i] for i in invalid_indices], device=self.device) # type: ignore
            n_invalid = len(invalid_indices)
            
            # Randomly select a face (0-5) for each invalid position
            faces = torch.randint(0, 6, (n_invalid,), device=self.device)
            
            # Create a tensor to hold the new positions
            new_positions = torch.zeros((n_invalid, 3), device=self.device)
            
            # Get relevant cube bounds for invalid environments
            invalid_centers = cube_centers[invalid_indices]
            invalid_mins = cube_min[invalid_indices]
            invalid_maxs = cube_max[invalid_indices]
            
            # Helper for random positions on a face
            rand_x = torch.rand(n_invalid, device=self.device) * cube_width - cube_width/2 + invalid_centers[:, 0]
            rand_y = torch.rand(n_invalid, device=self.device) * cube_height - cube_height/2 + invalid_centers[:, 1]
            rand_z = torch.rand(n_invalid, device=self.device) * cube_depth - cube_depth/2 + invalid_centers[:, 2]
            
            # +X face
            mask = (faces == 0)
            if mask.any():
                new_positions[mask, 0] = invalid_maxs[mask, 0]
                new_positions[mask, 1] = rand_y[mask]
                new_positions[mask, 2] = rand_z[mask]
            
            # -X face
            mask = (faces == 1)
            if mask.any():
                new_positions[mask, 0] = invalid_mins[mask, 0]
                new_positions[mask, 1] = rand_y[mask]
                new_positions[mask, 2] = rand_z[mask]
            
            # +Y face
            mask = (faces == 2)
            if mask.any():
                new_positions[mask, 0] = rand_x[mask]
                new_positions[mask, 1] = invalid_maxs[mask, 1]
                new_positions[mask, 2] = rand_z[mask]
            
            # -Y face
            mask = (faces == 3)
            if mask.any():
                new_positions[mask, 0] = rand_x[mask]
                new_positions[mask, 1] = invalid_mins[mask, 1]
                new_positions[mask, 2] = rand_z[mask]
            
            # +Z face
            mask = (faces == 4)
            if mask.any():
                new_positions[mask, 0] = rand_x[mask]
                new_positions[mask, 1] = rand_y[mask]
                new_positions[mask, 2] = invalid_maxs[mask, 2]
            
            # -Z face
            mask = (faces == 5)
            if mask.any():
                new_positions[mask, 0] = rand_x[mask]
                new_positions[mask, 1] = rand_y[mask]
                new_positions[mask, 2] = invalid_mins[mask, 2]
            
            # Update positions
            self.pose_command_w[invalid_env_ids, :3] = new_positions
        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_w[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        # Calculate yaw to face the robot base
        robot_positions = cube_centers  # Shape: [len(env_ids), 3]
        robot_positions[:, 0] += 0.29
        for i, env_id in enumerate(env_ids):
            # Vector from target position to robot base
            direction = robot_positions[i] - self.pose_command_w[env_id, :3]
            
            # Calculate yaw angle (in xy-plane)
            # atan2(y, x) gives the angle in the xy-plane
            yaw = torch.atan2(direction[1], direction[0])
            
            # Flip the angle by 180 degrees so it faces toward the robot
            # This assumes the forward direction of the end effector is along the x-axis
            euler_angles[i, 2] = yaw + torch.pi
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