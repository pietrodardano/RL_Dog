import math
import torch

from isaaclab.envs      import ManagerBasedEnv, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.assets    import Articulation, RigidObject, AssetBaseCfg
from isaaclab.assets    import RigidObjectCfg, ArticulationCfg
from isaaclab.sensors   import ContactSensor, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.scene     import InteractiveSceneCfg
from isaaclab.terrains  import TerrainImporterCfg

from isaaclab.managers  import ObservationGroupCfg as ObsGroup
from isaaclab.managers  import ObservationTermCfg as ObsTerm
from isaaclab.managers  import RewardTermCfg as RewTerm
from isaaclab.managers  import EventTermCfg as EventTerm
from isaaclab.managers  import TerminationTermCfg as DoneTerm
from isaaclab.managers  import SceneEntityCfg

from isaaclab.utils       import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_assets.robots.unitree    import AliengoCFG_Color, AliengoCFG_Black

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
# import isaaclab.envs.mdp.commands.velocity_command

########### SCENE ###########

@configclass
class SceneCfg(InteractiveSceneCfg):
    """
    Scene configuration for the AlienGo environment.
    """
    # Terrain
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    
    # Robot
    robot: ArticulationCfg = AliengoCFG_Black.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))


########### MDP - RL ###########

### ACTIONS ###
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)

### OBSERVATIONS ###
@configclass
class ObsCfg(ObsGroup):
    """
    Observation configuration for the AlienGo environment.
    """
    class PolicyCfg(ObsGroup):
        ### Robot State
        base_lin_pos  = ObsTerm(func=mdp.root_pos_w,     noise=Unoise(n_min=-0.01, n_max=0.01))    # [m]
        base_quat_pos = ObsTerm(func=mdp.root_quat_w,    noise=Unoise(n_min=-0.02, n_max=0.02))    # [quaternion]
        base_lin_vel  = ObsTerm(func=mdp.root_lin_vel_w, noise=Unoise(n_min=-0.1,  n_max=0.1))      # [m/s]
        base_ang_vel  = ObsTerm(func=mdp.root_ang_vel_w, noise=Unoise(n_min=-0.08, n_max=0.08))    # [rad/s]
            
        ### Joint state 
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.07, n_max=0.07))      # [rad]
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.04, n_max=0.04))      # [rad/s]

        def __post_init__(self):
            self.enable_corruption = True   # IDK
            self.concatenate_terms = True   # IDK

    policy: PolicyCfg = PolicyCfg()

### REWARDS ###
@configclass
class RewardsCfg:
    """
    Reward configuration for the AlienGo environment.
    """
    ## Rewards
    # Track linear velocity: 0
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # Track angular velocity: 0
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.9, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    
    ## Penalities
    lin_vel_z_l2    = RewTerm(func=mdp.lin_vel_z_l2,      weight=-0.6)
    ang_vel_xy_l2   = RewTerm(func=mdp.ang_vel_xy_l2,     weight=-0.05)
    dof_torques_l2  = RewTerm(func=mdp.joint_torques_l2,  weight=-1.0e-5)
    dof_acc_l2      = RewTerm(func=mdp.joint_acc_l2,      weight=-2.5e-7)
    action_rate_l2  = RewTerm(func=mdp.action_rate_l2,    weight=-0.01)
    
    dof_pos_dev     = RewTerm(func=mdp.joint_deviation_l1, weight=-0.4) 
    
    ## Penalities but with mdp.obs
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.8,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"]), "target_height": 0.40}, # "target": 0.35         target not a param of base_pos_z
    )
    
    undesired_thigh_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.6,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 1.0},
    )
    undesired_body_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    
### Commands ###
@configclass
class CommandsCfg:
    """Command terms for the MDP."""   # ASKING TO HAVE 0 Velocity

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(0.0, 0.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0, 0)
        ),
    )
    
### EVENTS ###
@configclass
class EventCfg:
    """Configuration for events."""
    
    # Reset the robot with initial velocity
    reset_scene = EventTerm(
        func=mdp.reset_root_state_uniform,
        params={"pose_range": {"x": (-0.1, 0.1), "z": (-0.32, 0.18), # it was z(-0.22, 12)
                               "roll": (-0.15, 0.15), "pitch": (-0.15, 0.15),}, #cancel if want it planar
                "velocity_range": {"x": (-0.4, 0.9), "y": (-0.4, 0.4)},}, 
        mode="reset",
    )
    reset_random_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        params={"position_range": (-0.40, 0.40), "velocity_range": (-0.4, 0.4)},
        mode="reset",
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        params={"velocity_range": {"x": (-0.6, 0.6), "y": (-0.5, 0.5), "z": (-0.15, 0.1)}},
        mode="interval",
        interval_range_s=(0.3,2.2),
    )
    
### TERMINATION ###
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""
    pass


######### ENVIRONMENT #########
@configclass
class AliengoEnvCfg(ManagerBasedRLEnvCfg):   #MBEnv --> _init_, _del_, load_managers(), reset(), step(), seed(), close(), 
    """Configuration for the locomotion velocity-tracking environment."""

    scene : SceneCfg                = SceneCfg(num_envs=1028, env_spacing=2.5)
    actions : ActionsCfg            = ActionsCfg()
    commands : CommandsCfg          = CommandsCfg() 
    observations : ObsCfg           = ObsCfg()
 
    events : EventCfg               = EventCfg()
    rewards : RewardsCfg            = RewardsCfg()
    terminations : TerminationsCfg  = TerminationsCfg()
    curriculum : CurriculumCfg      = CurriculumCfg()


    def __post_init__(self):
        """Initialize additional environment settings."""
        self.decimation = 4  # env decimation -> 50 Hz control
        self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        self.sim.render_interval = self.decimation
        self.episode_length_s = 3.0
        #self.sim.physics_material = self.scene.terrain.physics_material

        # viewer settings
        self.viewer.eye = (5.0, 1.0, 2.0)