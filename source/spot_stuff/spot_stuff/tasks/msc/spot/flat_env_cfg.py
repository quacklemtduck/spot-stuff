# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab.assets.asset_base_cfg import AssetBaseCfg
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.scene.interactive_scene_cfg import InteractiveSceneCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
import isaaclab.sim as sim_utils
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
import math
from dataclasses import MISSING
from . import mdp as spot_mdp
# import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab.envs.mdp as mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas import modify_articulation_root_properties
##
# Pre-defined configs
##
from .spot_arm import SPOT_ARM_CFG


@configclass
class SpotActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
    #joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["arm0_el.*","arm0_sh.*"], scale=0.5, use_default_offset=True)

    # joint_pos = mdp.JointPositionActionCfg(
    #     asset_name="robot", 
    #     joint_names=["(?!arm0_).*"], # All joints except arm joints
    #     scale=0.2, 
    #     use_default_offset=True
    # )
    
    # # Arm joints with reduced scale
    # arm_joint_pos = mdp.JointPositionActionCfg(
    #     asset_name="robot", 
    #     joint_names=["arm0_.*"], 
    #     scale=0.05, # Reduced scale for arm joints
    #     use_default_offset=True
    # )


@configclass
class SpotCommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
            # lin_vel_x=(-2.0, 3.0), lin_vel_y=(-1.5, 1.5), ang_vel_z=(-2.0, 2.0)
        ),
    )

    goal_command = spot_mdp.WorldPoseCommandCfg(
        asset_name="robot",
        body_name="body", # type: ignore
        ee_name="ee_frame",
        resampling_time_range=(2.0, 4.0),
        debug_vis=True,
        ranges=spot_mdp.WorldPoseCommandCfg.Ranges(
            pos_x=(-0.4, 0.8),
            pos_y=(-0.4, 0.4),
            pos_z=(0.1, 0.8),
            roll=(0.0, 0.0),
            pitch=(math.pi / 4, math.pi / 2),  # depends on end-effector axis
            yaw=(0, 0),
        )
    )

@configclass
class BenchmarkCommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
            # lin_vel_x=(-2.0, 3.0), lin_vel_y=(-1.5, 1.5), ang_vel_z=(-2.0, 2.0)
        ),
    )

    goal_command = spot_mdp.WorldPoseCommandCfg(
        asset_name="robot",
        body_name="body", # type: ignore
        ee_name="ee_frame",
        resampling_time_range=(2.0, 2.0),
        positions=[
            (0.5, 0.0, 0.5, 0.819, 0.0, 0.574, 0.0),
            (-0.3, 0.151, 0.8, 0.179, -0.191, 0.031, 0.9646),
            (0.8, 0.0, 0.5, 0.819, 0.0, 0.574, 0.0),
            (0.5, 0.0, 0.1, 0.819, 0.0, 0.574, 0.0),
            (0.5, 0.0, 0.8, 0.819, 0.0, 0.574, 0.0),
            (0.5, 0.4, 0.5, 0.819, 0.0, 0.574, 0.0),
            (0.5, 0.4, 0.8, 0.819, 0.0, 0.574, 0.0),
            (0.5, -0.4, 0.5, 0.819, 0.0, 0.574, 0.0),
            (0.5, -0.4, 0.8, 0.819, 0.0, 0.574, 0.0),
            (0.8, -0.4, 0.8, 0.819, 0.0, 0.574, 0.0),
            
        ],
        print_metrics=True,
        debug_vis=True,
        ranges=spot_mdp.WorldPoseCommandCfg.Ranges(
            pos_x=(0.5, 0.8),
            pos_y=(-0.4, 0.4),
            pos_z=(0.1, 0.8),
            roll=(0.0, 0.0),
            pitch=(math.pi / 4, math.pi / 2),  # depends on end-effector axis
            yaw=(0, 0),
        )
    )


@configclass
class SpotObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        #velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )

        arm_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_command"})
        
        finger_obs = ObsTerm(func=spot_mdp.object_obs)
        body_obs = ObsTerm(func=spot_mdp.body_obs)
        # arm_vel = ObsTerm(
        #     func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names="arm0_.*")}, noise=Unoise(n_min=-0.5, n_max=0.5)
        # )

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class SpotEventCfg:
    """Configuration for randomization."""

    # startup
    # physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,  # type: ignore
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.3, 1.0),
    #         "dynamic_friction_range": (0.3, 0.8),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="body"),
    #         "mass_distribution_params": (-2.5, 2.5),
    #         "operation": "add",
    #     },
    # )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-0.0, 0.0)},
            "velocity_range": {},
        },
    )

    reset_robot_joints = EventTerm(
        func=spot_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
    #     },
    # )


@configclass
class SpotRewardsCfg:

    base_angular_velocity = RewardTermCfg(
        func=spot_mdp.base_angular_velocity_reward,
        weight=2.0,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_linear_velocity = RewardTermCfg(
        func=spot_mdp.base_linear_velocity_reward,
        weight=2.0,
        params={"std": 1.0, "ramp_rate": 0.5, "ramp_at_vel": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )

    #Arm reward
    catchy_points = RewardTermCfg(
        func=spot_mdp.catch_box,
        weight=-0.00001,
        params={"ee_frame_cfg": SceneEntityCfg("ee_frame")}
    )

    #Arm reward
    catchy_points_tanh = RewardTermCfg(
        func=spot_mdp.catch_box_tanh,
        weight=0.00001,
        params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "std": 0.1}
    )

    #Arm reward
    end_effector_orientation_tracking = RewardTermCfg(
        func=spot_mdp.orientation_command_error,
        weight=-0.00001,
        params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )

    good_boy_points = RewardTermCfg(
        func=spot_mdp.good_boy_points,
        weight= -0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="body")}
    )
   

    # -- penalties
    #action_smoothness = RewardTermCfg(func=spot_mdp.action_smoothness_penalty, weight=-1.0)

    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.0001)
    joint_arm_vel = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="arm.*")},
    )

    foot_slip = RewardTermCfg(
        func=spot_mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )

    base_motion = RewardTermCfg(
        func=spot_mdp.base_motion_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    base_orientation = RewardTermCfg(
        func=spot_mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    joint_acc = RewardTermCfg(
        func=spot_mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
    )

    joint_legs_pos = RewardTermCfg(
        func=spot_mdp.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_h[xy]", ".*_kn"]),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
        },
    )
    joint_torques = RewardTermCfg(
        func=spot_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_h[xy]", ".*_kn"])},
    )
    joint_vel = RewardTermCfg(
        func=spot_mdp.joint_velocity_penalty,
        weight=-1.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
    )

    # body_orentation = RewardTermCfg(
    #     func=spot_mdp.body_orientation_penalty_exp,
    #     weight=-0.0001,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="body")},
    # )
    # joint_torques = RewardTermCfg(
    #     func=spot_mdp.joint_torques_penalty,
    #     weight=-5.0e-3,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    # )
    # (1) Constant running reward
    alive = RewardTermCfg(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewardTermCfg(func=mdp.is_terminated, weight=-2.0)


@configclass
class SpotTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*leg"]), "threshold": 1.0},
    )

    # bad_orientation = DoneTerm(
    #     func=mdp.bad_orientation,
    #     params={"limit_angle": 0.25}
    # )

@configclass
class BenchmarkTerminationsCfg:
    """Termination terms for the MDP."""

    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*leg"]), "threshold": 1.0},
    )

    # bad_orientation = DoneTerm(
    #     func=mdp.bad_orientation,
    #     params={"limit_angle": 0.25}
    # )

@configclass
class SpotSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    robot: ArticulationCfg = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
        debug_vis=True,
        target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/arm0_link_fngr")]
    )

@configclass
class CurriculumCfg:
    #catchy_curriculum = CurrTerm(func=spot_mdp.catchy_increase) # type: ignore
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 18500}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_arm_vel", "weight": -0.001, "num_steps": 18500}
    )

    catchy_increase = CurrTerm(
        func=spot_mdp.catchy_increase, params={"num_steps": 8500} # type: ignore
    )

@configclass
class MscEnvCfg(ManagerBasedRLEnvCfg):
    
    scene: SpotSceneCfg = SpotSceneCfg(num_envs=1024, env_spacing=2.5)
    # Basic settings'
    observations: SpotObservationsCfg = SpotObservationsCfg()
    actions: SpotActionsCfg = SpotActionsCfg()
    commands: SpotCommandsCfg | BenchmarkCommandsCfg = SpotCommandsCfg()

    # MDP setting
    rewards: SpotRewardsCfg = SpotRewardsCfg()
    terminations: SpotTerminationsCfg | BenchmarkTerminationsCfg = SpotTerminationsCfg()
    events: SpotEventCfg = SpotEventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Viewer
    viewer = ViewerCfg(eye=(10.5, 10.5, 0.3), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        # post init of parent

        # general settings
        self.decimation = 10  # 50 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.002  # 500 Hz
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.ee_frame.visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1) # type: ignore
        # switch robot to Spot-d


class SpotFlatEnvCfg_PLAY(MscEnvCfg):


    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None

class SpotFlatEnv_Benchmark(MscEnvCfg):

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        self.terminations = BenchmarkTerminationsCfg()
        self.commands = BenchmarkCommandsCfg()

        self.episode_length_s = 1000
        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None