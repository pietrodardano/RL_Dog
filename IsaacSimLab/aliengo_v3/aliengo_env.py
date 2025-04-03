
import math
import torch

from omni.isaac.lab.envs     import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.assets   import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.assets   import Articulation
from omni.isaac.lab.sensors  import ContactSensorCfg, RayCasterCfg, patterns

from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm

from omni.isaac.lab.utils.noise  import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.utils        import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.scene        import InteractiveSceneCfg
from omni.isaac.lab.terrains     import TerrainImporterCfg

from omni.isaac.lab.terrains.config.rough   import ROUGH_TERRAINS_CFG
from omni.isaac.lab_assets.unitree          import AliengoCFG_Color, AliengoCFG_Black #modified in IsaacLab_

import omni.isaac.lab.sim        as sim_utils
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp            # contains omni.isaac.lab.envs.mdp

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

ROUGH_TERRAIN = 0
HEIGHT_SCAN = 1


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

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

######### MDP - RL #########

### ACTIONS ###
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.8, use_default_offset=True)


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
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Command Input (What we requires to do)
        velocity_commands = ObsTerm(func=constant_commands)
        
        # Robot State (What we have)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        if not HEIGHT_SCAN and not ROUGH_TERRAIN: 
            floor_dis = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.02, n_max=0.02))

        if HEIGHT_SCAN:
            height_scan = ObsTerm(
                func=mdp.height_scan,
                params={"sensor_cfg": SceneEntityCfg("height_scanner")},
                clip=(-1.0, 1.0),
        )
            
        # Joint state 
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))

        actions   = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True   # IDK
            self.concatenate_terms = True   # IDK

    policy: PolicyCfg = PolicyCfg()

### EVENTS ###
@configclass
class EventCfg:
    """Configuration for events."""

    #reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Reset the robot with initial velocity
    reset_scene = EventTerm(
        func=mdp.reset_root_state_uniform,
        params={"pose_range": {"x": (-0.1, 0.0), "z": (-0.2, 0.08)}, 
                "velocity_range": {"x": (-0.2, 1.0), "y": (-0.05, 0.05)},}, 
        mode="reset",
    )
    reset_random_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        params={"position_range": (-0.15, 0.15), "velocity_range": (-0.05, 0.05)},
        mode="reset",
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.2, 0.2), "z": (-0.05, 0.05)}},
        mode="interval",
        interval_range_s=(0.2,0.8),
    )


### REWARDS ###

# Available strings: ['base', 'FL_hip', 'FL_thigh', 'FL_calf', 'FR_hip', 'FR_thigh', 'FR_calf', 'RL_hip', 'RL_thigh', 'RL_calf', 'RR_hip', 'RR_thigh', 'RR_calf']
# IDK why not "*_foot" (and "trunk") even if is present in URDF

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

                                ######## Positive weights: TRACKING the BASE Velocity (set to 0) ########
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.2)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.8, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=0.9,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"]), "target_height": 0.45}, # "target": 0.35         target not a param of base_pos_z
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.1)
    
    #### PENALITIES
    body_lin_acc_l2 = RewTerm(func=mdp.body_lin_acc_l2,  weight=-0.2)
    lin_vel_z_l2    = RewTerm(func=mdp.lin_vel_z_l2,     weight=-0.2)
    ang_vel_xy_l2   = RewTerm(func=mdp.ang_vel_xy_l2,    weight=-0.3)
    action_rate_l2  = RewTerm(func=mdp.action_rate_l2,   weight=-0.02)

    ## JOINTS
    dof_pos_limits  = RewTerm(func=mdp.joint_pos_limits,  weight=-0.3)
    dof_pos_dev     = RewTerm(func=mdp.joint_deviation_l1, weight=-0.2)
    dof_acc_l2      = RewTerm(func=mdp.joint_acc_l2,       weight=-2.5e-6)
    dof_torques_l2  = RewTerm(func=mdp.joint_torques_l2,   weight=-1.0e-7)
    #dof_vel_l2      = RewTerm(func=mdp.joint_vel_l2,       weight=-0.001)

    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time,
    #     weight=0.04,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),   # *_foot doesen't work even if in URDf is present
    #         "command_name": "base_velocity",
    #         "threshold": 0.5,
    #     },
    # )

    desired_calf_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.06,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"), "threshold": 1.0},    # *_foot doesen't work even if in URDf is present
    )

    undesired_thigh_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.6,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 1.0},
    )
    undesired_body_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.9,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    ### Too strong/angry ###
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 10.0},
    # )

    # upside_down = DoneTerm(
    #     func = mdp.bad_orientation,
    #     params={"limit_angle": 1.48}, # whole robot | radiants: 1.5 ~ 90° Deg
    # )

    
@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""
    pass
    

######### ENVIRONMENT #########
@configclass
class AliengoEnvCfg(ManagerBasedRLEnvCfg):   #MBEnv --> _init_, _del_, load_managers(), reset(), step(), seed(), close(), 
    """Configuration for the locomotion velocity-tracking environment."""

    scene : BaseSceneCfg            = BaseSceneCfg(num_envs=128, env_spacing=2.5)
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
        self.episode_length_s = 2
        self.sim.physics_material = self.scene.terrain.physics_material

        # viewer settings
        self.viewer.eye = (6.0, 0.0, 4.5)

        self.sim.physics_material = self.scene.terrain.physics_material
        if HEIGHT_SCAN:
            if self.scene.height_scanner is not None:
                self.scene.height_scanner.update_period = self.decimation * self.sim.dt