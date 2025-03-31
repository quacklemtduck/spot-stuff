from isaaclab.managers import CommandTermCfg, CommandTerm
from isaaclab.utils import configclass
from dataclasses import MISSING
from .commands import WorldPoseCommand, OpenCloseCommand

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

@configclass
class WorldPoseCommandCfg(CommandTermCfg):
    class_type: type = WorldPoseCommand
    asset_name: str = MISSING # type: ignore
    body_name: str = MISSING # type: ignore
    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        pos_x: tuple[float, float] = MISSING # type: ignore
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = MISSING # type: ignore
        """Range for the y position (in m)."""

        pos_z: tuple[float, float] = MISSING # type: ignore
        """Range for the z position (in m)."""
    
    ranges: Ranges = MISSING # type: ignore

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose") # type: ignore
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace( # type: ignore
        prim_path="/Visuals/Command/body_pose"
    )
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1) # type: ignore
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1) # type: ignore


@configclass
class OpenCloseCommandCfg(CommandTermCfg):
    class_type: type = OpenCloseCommand

