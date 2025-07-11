"""
    This script demonstrates the environment for a quadruped robot AlienGo.

    --- HEADLESS: ---
    
    conda activate isaacenv
    cd
    cd IsaacLab/
    
    ./isaaclab.sh -p /home/user/Documents/RL_Dog/Isaac_aliengo/aliengo_vReal/aliengo_main.py --num_envs 1028 --headless --enable_cameras

    ./isaaclab.sh -p /home/robotac22/RL_Dog/Isaac_aliengo/aliengo_vReal/aliengo_main.py --num_envs 2056 --headless --enable_cameras

"""

from isaaclab.app import AppLauncher

import argparse
parser = argparse.ArgumentParser(description='AlienGo_vP Env Config')
parser.add_argument('--num_envs',       type=int,       default=2056,              help='Number of environments')
parser.add_argument('--env_spacing',    type=float,     default=2.5,               help='Environment spacing')
parser.add_argument("--task",           type=str,       default="AlienGo_vPaper25",  help="Name of the task.")

parser.add_argument("--my_headless",       action="store_true",    default=True,    help="GUI or not GUI.")
parser.add_argument("--video",          action="store_true",    default=True,    help="Record videos during training.")
parser.add_argument("--video_length",   type=int,               default=500,     help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int,               default=6000,   help="Interval between video recordings (in steps).")

### Launch IsaacSim ###
AppLauncher.add_app_launcher_args(parser)
args_cli        = parser.parse_args()
app_launcher    = AppLauncher(args_cli)
simulation_app  = app_launcher.app

from isaaclab.envs        import ManagerBasedRLEnv
from isaaclab.utils.dict  import print_dict

from aliengo_env import AliengoEnvCfg
from aliengo_ppo import PPO_aliengo

import os
import torch
import datetime
import gymnasium as gym
from colorama import Fore, Style

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    gym.register(
        id=args_cli.task,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={'cfg': AliengoEnvCfg}
    )
    
    env_cfg = AliengoEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.viewer.resolution = (640, 480)
    
    name_task = args_cli.task
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../runs")
    
    try:
        if args_cli.video:
            env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
            timestamp = datetime.datetime.now().strftime("%d_%m_%H:%M")
            log_dir = f"{directory}/{name_task}_{timestamp}/"
            os.makedirs(log_dir, exist_ok=True)
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0 and step > 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print(Fore.GREEN + "[ALIENGO-INFO] Recording videos during training." + Style.RESET_ALL)
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
            env.num_envs = args_cli.num_envs  # Ensure the number of environments is set correctly
        else:
            env = ManagerBasedRLEnv(cfg=env_cfg)
            
    except Exception as e:
        print(Fore.RED + f'[ALIENGO-VIDEO-ERROR] {e}' + Style.RESET_ALL)
        env = ManagerBasedRLEnv(cfg=env_cfg)
        pass
    
    # Here the env will be wrapped with the SKRL wrapper 
    agent = PPO_aliengo(env=env, device=device, name=name_task, directory=log_dir, verbose=1) # SKRL_env_WRAPPER inside
    print(Fore.GREEN + '[ALIENGO-INFO] Start training' + Style.RESET_ALL)

    agent.train_sequential(timesteps=14000, headless=args_cli.my_headless)
    env.close()
    
if __name__ == "__main__":
    main()
    simulation_app.close()