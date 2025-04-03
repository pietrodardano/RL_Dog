# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the environment for a quadruped robot with height-scan sensor.

In this example, we use a locomotion policy to control the robot. The robot is commanded to
move forward at a constant velocity. The height-scan sensor is used to detect the height of
the terrain.

    conda activate isaacenv_
    cd
    cd IsaacLab_
    ./isaaclab.sh -p /home/rl_sim/RL_Dog/IsaacSimLab/aliengo_v0/check_trained.py --num_envs 32

Launch Isaac Sim Simulator first.
"""

################################## AFTER TRAINING ONLY ##################################

import argparse

from omni.isaac.lab.app import AppLauncher # type: ignore
parser = argparse.ArgumentParser(description="Tutorial on creating a quadruped base environment.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to spawn.")
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import  omni.isaac.lab.envs.mdp      as mdp # type: ignore
import  omni.isaac.lab.sim           as sim_utils # type: ignore
import  omni.isaac.lab.utils.math    as math_utils
from    omni.isaac.lab.assets        import ArticulationCfg, AssetBaseCfg # type: ignore
from    omni.isaac.lab.envs          import ManagerBasedEnv, ManagerBasedEnvCfg # type: ignore
from    omni.isaac.lab.managers      import EventTermCfg as EventTerm # type: ignore
from    omni.isaac.lab.managers      import ObservationGroupCfg as ObsGroup # type: ignore
from    omni.isaac.lab.managers      import ObservationTermCfg as ObsTerm # type: ignore
from    omni.isaac.lab.managers      import SceneEntityCfg # type: ignore
from    omni.isaac.lab.scene         import InteractiveSceneCfg # type: ignore
from    omni.isaac.lab.sensors       import ContactSensorCfg, RayCasterCfg, patterns
from    omni.isaac.lab.terrains      import TerrainImporterCfg # type: ignore
from    omni.isaac.lab.utils         import configclass # type: ignore
from    omni.isaac.lab.utils.assets  import ISAACLAB_NUCLEUS_DIR, check_file_path, read_file # type: ignore
from    omni.isaac.lab.utils.noise   import AdditiveUniformNoiseCfg as Unoise # type: ignore



##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # type: ignore # isort: skip
from omni.isaac.lab_assets.anymal   import ANYMAL_C_CFG  # type: ignore # isort: skip
from omni.isaac.lab_assets.unitree  import UNITREE_A1_CFG
#from unitree import AliengoCFG_Black, AliengoCFG_Color
from omni.isaac.lab_assets.unitree  import AliengoCFG_Color, AliengoCFG_Black  #modified in IsaacLab_ WORKS 

ROUGH_TERRAIN = 0

##
# Scene definition
##


@configclass
class BaseSceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""

    # add terrain
    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    if ROUGH_TERRAIN:
        terrain= TerrainImporterCfg(
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

    # add robot
    robot: ArticulationCfg = AliengoCFG_Black.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # SENSORS (virtual ones, the real robot does not has thm) 
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=400.0),
    )


##
# MDP settings
##

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

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


### OBSERVATIONS ###

from omni.isaac.lab.assets import Articulation, RigidObject
def my_body_acc_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Acceleration of the asset_body in the WORLD frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    num_envs = asset.data.body_acc_w.shape[0]  # Number of environments (instances)
    body_acc_w = asset.data.body_acc_w[:, asset_cfg.body_ids, :3]
    return body_acc_w.view(num_envs, 3) # Ensure shape (num_envs, obs_term_dim)

def my_body_acc_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Acceleration of the asset_body in the BASE frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    num_envs = asset.data.body_acc_w.shape[0]  # Number of environments (instances)
    body_acc_w = asset.data.body_acc_w[:, asset_cfg.body_ids, :3]
    return math_utils.quat_rotate_inverse(asset.data.root_quat_w, body_acc_w.view(num_envs, 3))

def imu_acc_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Acceleration + Gravity = IMU : of the base in the body frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    num_envs = asset.data.body_acc_w.shape[0]  # Number of environments (instances)
    body_acc_w = asset.data.body_acc_w[:, asset_cfg.body_ids, :3]
    body_acc_b = math_utils.quat_rotate_inverse(asset.data.root_quat_w, body_acc_w.view(num_envs, 3))
    return body_acc_b + asset.data.projected_gravity_b*9.81

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        ### Command Input (What we requires to do)
        velocity_commands = ObsTerm(func=constant_commands)
        
        ### Robot State (What we have)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # projected_gravity = ObsTerm(
        #     func=mdp.projected_gravity, # base frame!
        #     noise=Unoise(n_min=-0.05, n_max=0.05),
        # )
        # accel_base = ObsTerm(
        #     func=my_body_acc_b,   # world frame!
        #     params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"])},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        # )

        imu_like_data = ObsTerm(
            func=imu_acc_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"])},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        ) 

        #TO DO by prof
        # sum (prj_gravity + accel_base) = like IMU -----> DONE !
        # test the trained policy +  controls --> are disturbaances !!! (eg: go2 sim.py)
        # video headless
        # walk x,y
            
        ### Joint state 
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.10, n_max=0.10))

        actions   = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True   # IDK
            self.concatenate_terms = True   # IDK

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


##
# Environment configuration
##


@configclass
class QuadrupedEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    scene : BaseSceneCfg            = BaseSceneCfg(num_envs=128, env_spacing=2.5)
    actions : ActionsCfg            = ActionsCfg()
    commands : CommandsCfg          = CommandsCfg() 
    observations : ObservationsCfg  = ObservationsCfg()
 
    events : EventCfg               = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4  # env decimation -> 50 Hz control
        # simulation settings
        self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        if ROUGH_TERRAIN:
            self.sim.physics_material = self.scene.terrain.physics_material

def main():
    """Main function."""
    # setup base environment
    env_cfg = QuadrupedEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)

    policy_path = '/home/rl_sim/RL_Dog/runs/AlienGo_v3_stoptry_31_07_IMU_81%stable/checkpoints/best_agent.pt'
    policy = torch.jit.load(policy_path).to("cuda").eval()

    count = 0
    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 1000 == 0:
                obs, _ = env.reset()
                count = 0
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # infer action
            action = policy    ################################## PUT HERE THE TRAINED POLICY !!!!!
            # step env
            obs, _ = env.step(action)
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()