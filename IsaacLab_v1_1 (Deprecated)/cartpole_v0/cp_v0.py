"""
    conda activate isaacenv_
    cd
    cd IsaacLab_
    ./isaaclab.sh -p ~/RL_Dog/IsaacSimLab/Tutorials_IsaacLab/cp-v0.py
"""

import argparse
from omni.isaac.lab.app import AppLauncher
parser = argparse.ArgumentParser(description="CartPole RL 1st try")
parser.add_argument("--num_envs", type=int, default=32, help="Num of environments to spawn")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parser_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

"""     ### IsaacLab API ###
https://github.com/isaac-sim/IsaacLab/tree/main/source/extensions/omni.isaac.lab/omni/isaac/lab
"""

import omni.isaac.lab.sim as sim_utils

from omni.isaac.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.utils import configclass


""" 
    i should import pre-defined configs CFGs and MDP but i'll try to do it by myself

"""
from omni.isaac.lab_assets.cartpole import CARTPOLE_CFG  # isort:skip

########### SCENE ###########

@configclass
class CPSceneSFG(InteractiveSceneCfg):
    # ground 
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0,100.0)),
    )

    #cartpole
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    #lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight", 
        spawn=sim_utils.DistantLightCfg(color=(0.9,0.9,0.9), intensity=500.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )

########### MDP - RL ###########

import omni.isaac.lab.envs.mdp.command as mdp_command
import omni.isaac.lab.envs.mdp.actions as mdp_actions

### TO CHECK 


@configclass
class CommandsCfg:                      
    """Command terms for the MDP."""

    # no commands for this MDP
    null = mdp.NullCommandCfg()

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)
