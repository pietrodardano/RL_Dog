
import math
import torch

########### DELETE IT IN FUTURE, NOT NECESSARY
# import os
# import sys
# sys.path.append("~/IsaacLab_/source/extensions/omni.isaac.lab")
# sys.path.append("~/IsaacLab_/source/extensions/omni.isaac.lab_assets")
# sys.path.append("~/IsaacLab_/source/extensions/omni.isaac.lab_tasks")
# os.path.expanduser("~/IsaacLab_")


import omni.isaac.lab.sim        as sim_utils
import omni.isaac.lab.envs.mdp   as mdp

##
#########   import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp    #  !!!!!!!!!!!
##

from omni.isaac.lab.envs     import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.assets   import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.assets   import Articulation

from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm

from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.utils       import configclass

from omni.isaac.lab.scene       import InteractiveSceneCfg
from omni.isaac.lab.terrains    import TerrainImporterCfg
from omni.isaac.lab.terrains.config.rough   import ROUGH_TERRAINS_CFG
from omni.isaac.lab_assets.unitree          import AliengoCFG_Color, AliengoCFG_Black  #modified in IsaacLab_ WORKS 
#from unitree import AliengoCFG_Black, AliengoCFG_Color

"""
    ALIENGO_ENV.PY script STRUCTURE:
    - scene         --> GUI and World appearance 
    - mdp-rl        --> Actions, commmands, observations (policy i defined in ppo.py)
    - environment   --> encapsules ALL objs and cfgs from: SCENE, ACTIONS, COMMANDS, OSERVATIONS

    you will have to create an AliengoEnvCfg object in your main_script.py
    it will contains almost everything, usefull to pass infos and configs to the other functions

"""


######### SCENE #########

@configclass
class BaseSceneCfg(InteractiveSceneCfg):

    #############################################

    def __init__(self, rough_terrain=0, num_envs=16, env_spacing=2.5):

        # do NOT declare variables/parameters exept for Assets, InteractiveScene ..., they will generate error

        self.terrain        = self._terrain(rough_terrain=rough_terrain)
        self.robot          = self._robot()
        self.light          = self._light()
        self.dome_light     = self._dome_light()
        self.num_envs       = num_envs         # Required
        self.env_spacing    = env_spacing      # Required

    #############################################

    ### GROUND - TERRAIN ###
    def _terrain(self, rough_terrain=0):
        if rough_terrain:
            return TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="generator",
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
        else:
            return AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
            )
    
    ### ROBOT ###
    def _robot(self):             # needed 
        robot: ArticulationCfg = AliengoCFG_Black.replace(prim_path="{ENV_REGEX_NS}/Robot") # obj_type: Articulation(Rigid(Asset))
        return robot
        # ``{ENV_REGEX_NS}/Robot`` will be replaced with ``/World/envs/env_.*/Robot``
    # robot: ArticulationCfg = AliengoCFG_Black.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    ### LIGHTS ###
    def _light(self):
        return AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
    def _dome_light(self):
        return AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=400.0),
        )


######### MDP - RL #########

### ACTIONS ###
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


### COMANDS ###
@configclass
class CommandsCfg:
    """Command terms for the MDP. HIGH LEVEL GOALS --> e.g., velocity, task to do"""

    def __init__(self, walk=False, num_envs=16, device="cpu"):          #  walk = [0, 1] -> walk or not !
        dir = [1, 0, 0] if walk else [0, 0, 0]
        self.velocity_cmd = torch.tensor([dir], device=device).repeat(num_envs, 1)


### OBSERVATIONS ###
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        def __init__(self, velocity_cmd):
            
            self.base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
            self.base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
            self.projected_gravity = ObsTerm(
                func=mdp.projected_gravity,
                noise=Unoise(n_min=-0.05, n_max=0.05),
            )

            self.velocity_commands = ObsTerm(func=lambda: velocity_cmd)  # wrap it in a function
            """
                If you didn't use a lambda function and directly passed the tensor, like this:
                    self.velocity_commands = ObsTerm(func=self.cmnd.velocity_cmd)
                - This would set func to the current value of self.cmnd.velocity_cmd at the time of assignment.
                - If self.cmnd.velocity_cmd changes later, ObsTerm would not see the updated value 
                    because it holds the initial tensor value, not a function to fetch the latest value.
            """

            self.joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
            self.joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
            self.floor_dis = ObsTerm(func=mdp.base_pos_z,    noise=Unoise(n_min=-0.02, n_max=0.02))
            self.actions   = ObsTerm(func=mdp.last_action)

            def __post_init__(self):
                self.enable_corruption = True   # IDK
                self.concatenate_terms = True   # IDK

    # observation groups
    def __init__(self, velocity_cmd = [0,0,0]):     #for redundancy/security, it already receives arguments
        self.policy = self.PolicyCfg(velocity_cmd=velocity_cmd)

### EVENTS ###
@configclass
class EventCfg:
    """Configuration for events."""
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

### REWARDS ###
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    # (3) Primary task: keep body raised from the floor
    body_height = RewTerm(
        func=mdp.base_pos_z,
        weight=1.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"]), "target": 0.35},
    )
    
    # (4) Shaping tasks: Keep body almost horizontal
    horiz_orient = RewTerm(
        func=mdp.root_quat_w,
        weight=0.8,
        params={"target": [1, 0, 0, 0]} # Robot's BaseLink x_axis // target ---> be straight
    )
    # another sol is to have the projected_grav as [0, 0, -g] --> [0, 0, -1] --> q = [0.707, 0, 0, -0.707]

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    def __init__(self):
        self.time_out = self.time_out_()
    def time_out_(self):
        return DoneTerm(func=mdp.time_out, time_out=True)
    
@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""
    pass
    

######### ENVIRONMENT #########
@configclass
class AliengoEnvCfg(ManagerBasedRLEnvCfg):   #MBEnv --> _init_, _del_, load_managers(), reset(), step(), seed(), close(), 
    """Configuration for the locomotion velocity-tracking environment."""

    def __init__(self, args, device, rough_terrain=0):
        super().__init__()
        # self.args = args    #num_envs, env_spacing, walk
        # self.walk = args.walk
        # self.num_envs = args.num_envs
        # self.env_spacing = args.env_spacing
        # self.rough_terrain = rough_terrain

        # self.device   = device

        self.scene          = BaseSceneCfg(rough_terrain=rough_terrain, num_envs=args.num_envs, env_spacing=args.env_spacing)
        self.actions        = ActionsCfg()
        self.commands       = CommandsCfg(walk=args.walk, num_envs=args.num_envs, device=device) 
        self.observations   = ObservationsCfg(self.commands.velocity_cmd)
 
        self.events         = EventCfg()
        self.rewards        = RewardsCfg()
        self.terminations   = TerminationsCfg()
        self.curriculum     = CurriculumCfg()

    def __post_init__(self):
        """Initialize additional environment settings."""
        self.decimation = 4  # env decimation -> 50 Hz control
        self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        self.sim.render_interval = self.decimation

        # viewer settings
        self.viewer.eye = (6.0, 0.0, 4.5)

        if False:  # it war: rough_terrain, but generated errors since not passed and cannot be passed
            self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # tick all the sensors based on the smallest update period (physics update period)