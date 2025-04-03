"""
This script demonstrates the environment for a quadruped robot AlienGo.

    In this example, we use a locomotion policy to control the robot. The robot is commanded to
    move forward at a constant velocity. (?????)

    conda activate isaacenv_
    cd
    cd IsaacLab_
    ./isaaclab.sh -p /home/rl_sim/RL_Dog/IsaacSimLab/aliengo_v3/aliengo_check_main.py --num_envs 256

Launch Isaac Sim Simulator first.
"""

from omni.isaac.lab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser(description='AlienGo_v3 Environment Configuration')
parser.add_argument('--num_envs',       type=int,   default=128,            help='Number of environments')
parser.add_argument('--env_spacing',    type=float, default=2.5,           help='Environment spacing')
parser.add_argument('--walk',           type=int,   default=0,             help='ask to Walk or not (1,0)')
parser.add_argument("--task",           type=str,   default="AlienGo-v3",  help="Name of the task.")

#parser.add_argument("--headless",       action="store_true",    default=False,  help="GUI or not GUI.")
#parser.add_argument("--video",          action="store_true",    default=False,  help="Record videos during training.")
#parser.add_argument("--video_length",   type=int,               default=200,    help="Length of the recorded video (in steps).")
#parser.add_argument("--video_interval", type=int,               default=2000,   help="Interval between video recordings (in steps).")
#parser.add_argument("--device",         type=str,               default="cpu",  help="cpu or cuda.")
args = parser.parse_args()

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from omni.isaac.lab.envs     import ManagerBasedRLEnv
from omni.isaac.lab.envs     import ManagerBasedRLEnvCfg
from aliengo_ppo             import PPO_v1
import aliengo_env_real

from omni.isaac.lab.utils.dict import print_dict

# BASICS
import os
import time
import torch
import datetime

from colorama import Fore, Style

# FOR SIMULATION
import carb
import threading


import tensorboard
"""
cmd -->     tensorboard --logdir = "/home/rl_sim/RL_Dog/runs    (SERVER)
            or
            tensorboard --logdir=/home/rluser/RL_Dog/runs       (DELL)

            http://localhost:6006
"""


MODE = 1

def main():
    device="cuda" if torch.cuda.is_available() else "cpu"

    from aliengo_env import AliengoEnvCfg
    env_cfg = AliengoEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = ManagerBasedRLEnv(cfg=env_cfg)
    print(Fore.GREEN + '[INFO-AlienGo] Env Created' + Style.RESET_ALL)
    
    
    ### RL POLICY ###
    policy_path = "/home/rl_sim/RL_Dog/runs/AlienGo_v3_stoptry_31_07_IMU_81%stable/checkpoints/best_agent.pt"
    policy = torch.jit.load(policy_path).to(env.device).eval()

    print(Fore.GREEN + '[INFO-AlienGo] Policy Loaded' + Style.RESET_ALL)
    count = 0
    cnt_limit = 1000    # Set the sim reset time !!
    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % cnt_limit == 0:               
                obs, _ = env.reset()
                count = 0          #if uncommented it will loop forever
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            
            action = policy     # should be policy(obs) or sort of 
            obs, _ = env.step(action)
            count += 1
            if count == 8*cnt_limit:
                break

    env.close()



def sub_keyboard_event(event, *args, **kwargs) -> bool:

    if len(aliengo_env_real.base_command) > 0:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == 'W':
                aliengo_env_real.base_command[0] = [1, 0, 0]
            if event.input.name == 'S':
                aliengo_env_real.base_command[0] = [-1, 0, 0]
            if event.input.name == 'A':
                aliengo_env_real.base_command[0] = [0, 1, 0]
            if event.input.name == 'D':
                aliengo_env_real.base_command[0] = [0, -1, 0]
            if event.input.name == 'Q':
                aliengo_env_real.base_command[0] = [0, 0, 1]
            if event.input.name == 'E':
                aliengo_env_real.base_command[0] = [0, 0, -1]

            if len(aliengo_env_real.base_command) > 1:
                if event.input.name == 'I':
                    aliengo_env_real.base_command[1] = [1, 0, 0]
                if event.input.name == 'K':
                    aliengo_env_real.base_command[1] = [-1, 0, 0]
                if event.input.name == 'J':
                    aliengo_env_real.base_command[1] = [0, 1, 0]
                if event.input.name == 'L':
                    aliengo_env_real.base_command[1] = [0, -1, 0]
                if event.input.name == 'U':
                    aliengo_env_real.base_command[1] = [0, 0, 1]
                if event.input.name == 'O':
                    aliengo_env_real.base_command[1] = [0, 0, -1]
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            for i in range(len(aliengo_env_real.base_command)):
                aliengo_env_real.base_command[i] = [0, 0, 0]
    return True

def cmd_vel_cb(msg, num_robot):
    x = msg.linear.x
    y = msg.linear.y
    z = msg.angular.z
    aliengo_env_real.base_command[num_robot] = [x, y, z]

def specify_cmd_for_robots(numv_envs):
    base_cmd = []
    for _ in range(numv_envs):
        base_cmd.append([0, 0, 0])
    aliengo_env_real.base_command = base_cmd

if __name__ == "__main__":
    main()
    simulation_app.close()