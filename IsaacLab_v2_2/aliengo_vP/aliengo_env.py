
import math
import torch

from isaaclab.envs     import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.assets   import ArticulationCfg, AssetBaseCfg
from isaaclab.assets   import Articulation, RigidObject
from isaaclab.sensors  import ContactSensorCfg, RayCasterCfg, patterns

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

from isaaclab.utils.noise  import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils        import configclass
from isaaclab.scene        import InteractiveSceneCfg
from isaaclab.terrains     import TerrainImporterCfg

from isaaclab.terrains.config.rough             import ROUGH_TERRAINS_CFG
from isaaclab_assets.robots.unitree    import AliengoCFG_Color, AliengoCFG_Black #modified in Isaacab_

import isaaclab.sim        as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

###-------------------------------------------------------------------------------------###

"""
    ALIENGO_ENV.PY script STRUCTURE:
    - scene         --> GUI and World appearance 
    - mdp-rl        --> Actions, commmands, observations (policy i defined in ppo.py)
    - environment   --> encapsules ALL objs and cfgs from: SCENE, ACTIONS, COMMANDS, OSERVATIONS

    you will have to create an AliengoEnvCfg object in your main_script.py
    it will contains almost everything, usefull to pass infos and configs to the other functions

"""

global ROUGH_TERRAIN
global HEIGHT_SCAN 

#Print IMU data in report_debug file
DEBUG_IMU = False

ROUGH_TERRAIN = 0
HEIGHT_SCAN = 0

base_command = {}  # for keyboard inputs

######### SCENE #########
terrain_type = "generator" if ROUGH_TERRAIN else "plane"
@configclass
class BaseSceneCfg(InteractiveSceneCfg):

    """
        note::
        The adding of entities to the scene is sensitive to the order of the attributes in the configuration.
        Please make sure to add the entities in the order you want them to be added to the scene.
        The recommended order of specification is terrain, physics-related assets (articulations and rigid bodies),
        sensors and non-physics-related assets (lights). 
    """

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type= terrain_type,
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # ROBOT
    robot: ArticulationCfg = AliengoCFG_Black.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # SENSORS (virtual ones, the real robot does not has thm) 
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # HEIGHT SCAN (robot does not has it, however in sim can lean to a faster training)
    if HEIGHT_SCAN:
        height_scanner= RayCasterCfg(
            prim_path = "{ENV_REGEX_NS}/Robot/base",
            offset = RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 10.0)),
            attach_yaw_only = True,
            pattern_cfg = patterns.GridPatternCfg(resolution=0.1, size=(1.0, 1.0)),
            debug_vis= True,
            mesh_prim_paths = ["/World/ground"],
        )

    # LIGHTS

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))



######### MDP - RL #########

### ACTIONS ###
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)


### COMANDS ###

def constant_commands(env: ManagerBasedEnv) -> torch.Tensor:
    """Generated command"""
    tensor_lst =  torch.tensor([0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    return tensor_lst


@configclass
class CommandsCfg:
    """Command terms for the MDP."""   # ASKING TO HAVE 0 Velocity

    base_velocity = mdp.UniformVelocityCommandCfg( # inherits from CommandTermCfg
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

### OBSERVATIONS ###

# FOR DEBUG ONLY  (SUBSTITUTE: mdp.joint_pos_rel)
def joint_pos_rel_print(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    #print("[ALIENGO-DEBUG] IDS:", asset.data.joint_names)
    print(asset.data.default_joint_vel[:, asset_cfg.joint_ids])
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        ### Command Input (What we requires to do)
        #velocity_commands = ObsTerm(func=constant_commands)     # optional, is 0 
        
        ### Robot State (What we have)
        base_lin_pos = ObsTerm(func=mdp.root_pos_w, noise=Unoise(n_min=-0.01, n_max=0.01))      # [m]
        base_quat_pos = ObsTerm(func=mdp.root_quat_w, noise=Unoise(n_min=-0.02, n_max=0.02))    # [quaternion]
        base_lin_vel = ObsTerm(func=mdp.root_lin_vel_w, noise=Unoise(n_min=-0.1, n_max=0.1))    # [m/s]
        base_ang_vel = ObsTerm(func=mdp.root_ang_vel_w, noise=Unoise(n_min=-0.08, n_max=0.08))    # [rad/s]
            
        ### Joint state 
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.07, n_max=0.07))      # [rad]
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.04, n_max=0.04))      # [rad/s]

        def __post_init__(self):
            self.enable_corruption = True   # IDK
            self.concatenate_terms = True   # IDK

    policy: PolicyCfg = PolicyCfg()

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
    # physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.2, 0.8),
    #         "dynamic_friction_range": (0.2, 0.6),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )
    # random_masses = EventTerm(
    #     func = mdp.randomize_rigid_body_mass,
    #     mode = "startup",
    #     params = {
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "mass_distribution_params": (0.95, 1.05),
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     }
    # )


### REWARDS ###

# Available strings: ['base', 'FL_hip', 'FL_thigh', 'FL_calf', 'FR_hip', 'FR_thigh', 'FR_calf', 
#                             'RL_hip', 'RL_thigh', 'RL_calf', 'RR_hip', 'RR_thigh', 'RR_calf']


def height_goal(
    env: ManagerBasedRLEnv, target_height: float, allowance_radius: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward asset height if close to its target.
       it gives a bit of reward if the robot is close to the target height.
       maybe helping it to reach faster the desired height.

    Note:
        Currently, it assumes a flat terrain, i.e. the target height is in the world frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    height_diff = asset.data.root_pos_w[:, 2] - target_height
    rewards = torch.where(
        torch.abs(height_diff) <= allowance_radius,
        torch.where(height_diff == 0, torch.tensor(0.99), torch.tensor(0.30)),
        torch.tensor(0.0)
    )
    return rewards

def desired_pose_style_l1(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the chosen one."""
    asset: Articulation = env.scene[asset_cfg.name]

                       # 'FL_hi', 'FR_hi', 'RL_hi', 'RR_hi', 'FL_th', 'FR_th', 'RL_th', 'RR_th', 'FL_cl', 'FR_cl', 'RL_cl', 'RR_cl'
    desired_joints_pos = torch.tensor([0.1000, -0.1000,  0.1000, -0.1000,  0.8000,  0.8000,  1.0000,  1.0000, -1.4500, -1.4500, -1.4500, -1.4500], device=env.device)
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - desired_joints_pos[asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
                            ######## Positive weights: TRACKING the BASE Velocity (set to 0) ########
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.9, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # track_height = RewTerm(
    #     func=height_goal,
    #     weight=0.5,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"]), "target_height": 0.40, "allowance_radius": 0.03}, # "target": 0.35         target not a param of base_pos_z
    # )

    #### BODY PENALITIES
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.8,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"]), "target_height": 0.40}, # "target": 0.35         target not a param of base_pos_z
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.7)
    body_lin_acc_l2 = RewTerm(func=mdp.body_lin_acc_l2,  weight=-0.9)
    
    lin_vel_z_l2    = RewTerm(func=mdp.lin_vel_z_l2,     weight=-0.6)
    ang_vel_xy_l2   = RewTerm(func=mdp.ang_vel_xy_l2,    weight=-0.4)

    #### JOINTS PENALITIES
    dof_pos_limits  = RewTerm(func=mdp.joint_pos_limits,  weight=-0.9)
    #dof_pos_dev     = RewTerm(func=desired_pose_style_l1, weight=-0.25) # -0.25 strong
    dof_pos_dev     = RewTerm(func=mdp.joint_deviation_l1, weight=-0.4) 
    dof_vel_l2      = RewTerm(func=mdp.joint_vel_l2,       weight=-0.02)

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

    scene : BaseSceneCfg            = BaseSceneCfg(num_envs=1028, env_spacing=2.5)
    actions : ActionsCfg            = ActionsCfg()
    commands : CommandsCfg          = CommandsCfg() 
    observations : ObservationsCfg  = ObservationsCfg()
 
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
        self.sim.physics_material = self.scene.terrain.physics_material

        # viewer settings
        self.viewer.eye = (5.0, 1.0, 2.0)

        self.sim.physics_material = self.scene.terrain.physics_material
        if HEIGHT_SCAN:
            if self.scene.height_scanner is not None:
                self.scene.height_scanner.update_period = self.decimation * self.sim.dt