"""
This script demonstrates the environment for a quadruped robot AlienGo.

    In this example, we use a locomotion policy to control the robot. The robot is commanded to
    move forward at a constant velocity. (?????)

    conda activate isaacenv_
    cd
    cd IsaacLab_
    ./isaaclab.sh -p /home/rl_sim/RL_Dog/IsaacSimLab/aliengo_vP/aliengo_simulate.py --num_envs 1

    #IF HEADLESS:
    conda activate isaacenv_
    cd
    cd IsaacLab_
    ./isaaclab.sh -p /home/rl_sim/RL_Dog/IsaacSimLab/aliengo_vP/aliengo_simulate.py --num_envs 1 --headless --enable_cameras

    ./isaaclab.sh -p /home/rl_sim/RL_Dog/IsaacSimLab/aliengo_vP/aliengo_simulate.py --num_envs 4096 --headless --enable_cameras


Launch Isaac Sim Simulator first.
"""

HEADLESS = 1
SIM = 1
# if SIM is FALSE --> Evaluation

from omni.isaac.lab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser(description='AlienGo_v1 Environment Configuration')
parser.add_argument('--num_envs',       type=int,   default=128,            help='Number of environments')
parser.add_argument('--env_spacing',    type=float, default=2.5,           help='Environment spacing')
parser.add_argument('--walk',           type=int,   default=0,             help='ask to Walk or not (1,0)')
parser.add_argument("--task",           type=str,   default="AlienGo-v0",  help="Name of the task.")

#parser.add_argument("--headless",       action="store_true",    default=True,  help="GUI or not GUI.")
parser.add_argument("--video",          action="store_true",    default=HEADLESS,  help="Record videos during training.")
parser.add_argument("--video_length",   type=int,               default=400,    help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int,               default=4000,   help="Interval between video recordings (in steps).")
#parser.add_argument("--device",         type=str,               default="cpu",  help="cpu or cuda.")
#args = parser.parse_args()

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from omni.isaac.lab.envs        import ManagerBasedRLEnv
from omni.isaac.lab.envs        import ManagerBasedRLEnvCfg
from omni.isaac.lab.utils.dict  import print_dict
from omni.isaac.lab_tasks.utils import get_checkpoint_path

from aliengo_env import AliengoEnvCfg
from aliengo_ppo import PPO_v1

import aliengo_env
import carb         #from omni
import omni.appwindow

import os
import sys
import tqdm
import torch
import datetime
import gymnasium as gym
from colorama import Fore, Style

import tensorboard

"""
cmd -->     tensorboard --logdir=/home/rl_sim/RL_Dog/runs         #(SERVER)
            or
            tensorboard --logdir=/home/rluser/RL_Dog/runs         #(DELL)

            http://localhost:6006
"""

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gym.register(
        id=args_cli.task,
        entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
        kwargs={'cfg': AliengoEnvCfg}
    )

    # acquire input interface
    _input = carb.input.acquire_input_interface()
    _appwindow = omni.appwindow.get_default_app_window()
    _keyboard = _appwindow.get_keyboard()
    _sub_keyboard = _input.subscribe_to_keyboard_events(_keyboard, sub_keyboard_event)

    specify_cmd_for_robots(args_cli.num_envs)

    env_cfg = AliengoEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.viewer.resolution = (640, 480)

    #path = "/home/rl_sim/RL_Dog/runs/AlienGo_vP_stoptry_29_08_FULL_STATE_v2/checkpoints/best_agent.pt"
    path = '/home/rl_sim/RL_Dog/runs/AlienGo_vP_stoptry_09_09_FULL_STATE_v3/checkpoints/best_agent.pt'
    base_dir = os.path.dirname(os.path.dirname(path))

    if SIM:
        new_folder_name = os.path.basename(base_dir) + "_SIM"
        path_folder = os.path.join(os.path.dirname(base_dir), new_folder_name)
    else:
        new_folder_name = os.path.basename(base_dir) + "_EVAL"
        path_folder = os.path.join(os.path.dirname(base_dir), new_folder_name)

    try:
        if args_cli.video:
            env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
            log_dir = path_folder
            os.makedirs(log_dir, exist_ok=True)
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print(Fore.GREEN + "[ALIENGO-INFO] Recording videos during training." + Style.RESET_ALL)
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
        else:
            env = ManagerBasedRLEnv(cfg=env_cfg)
    except Exception as e:
        print(Fore.RED + f'[ALIENGO-VIDEO-ERROR] {e}' + Style.RESET_ALL)
        env = ManagerBasedRLEnv(cfg=env_cfg)
        pass

    agent = PPO_v1(env=env, device=device, verbose=1) # SKRL_env_WRAPPER inside

    if SIM:
        agent.agent.init()
        agent.agent.load(path) #["policy"]
        #agent = torch.jit.load(path).to(env.device)

        print(Fore.GREEN + '[ALIENGO-INFO] Policy Loaded' + Style.RESET_ALL)
        count = 0
        cnt_limit = 1000    # Set the sim reset time !!
        obs, _ = env.reset()
        flag_1 = 0
        flag = 0

        while simulation_app.is_running():
            with torch.inference_mode():
                # reset
                if count % cnt_limit == 0:               
                    obs, _ = env.reset()

                    count = 0          #if uncommented it will loop forever
                    print("-" * 80)
                    print("[ALIENGO-INFO]: Resetting environment...")

                ### TO TEST THE POLICY, I'M OVERWRITING
                # obs = torch.tensor([ 0.0000,  0.0000,  0.3927,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                #                      0.0000,  00000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                #                      0.0000,  0.7500,  0.7500,  0.7500,  0.7500, -1.5000, -1.5000, -1.5000,
                #                      -1.5000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                #                      0.0000,  0.0000,  0.0000,  0.0000,  0.0000], device="cuda:0")

                # OBS - J_DEF_POS
                obs = torch.tensor([ 0.0000,  0.0000,  0.3927,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                     0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  -0.1000, 0.1000, -0.1000,
                                     0.1000,  -0.0500,  -0.0500, -0.2500, -0.2500, 0.0000,  0.0000,  0.0000,
                                     0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                                     0.0000,  0.0000,  0.0000,  0.0000,  0.0000], device="cuda:0")
                print(Fore.GREEN + f'[ALIENGO-INFO] OBS {obs}' + Style.RESET_ALL)
                
                action, _, outputs = agent.agent.act(obs, count, cnt_limit) # obs['policy] if not the provided custom one, OBS instead 

                if flag_1==0:
                    print(Fore.GREEN + f'[ALIENGO-INFO] done 1st act' + Style.RESET_ALL)
                    print("ActionShape: ", action.shape)
                    print("[ALIENGO-INFO] ACT ",action)
                    print("[ALIENGO-INFO] OUTPUTS ",outputs)
                    flag_1 = 1

                obs, _, _, _ = env.step(action)
                if flag==0:
                    print(Fore.GREEN + f'[ALIENGO-INFO] done 1st step' + Style.RESET_ALL)
                    flag = 1

                count += 1
                if count == 8*cnt_limit:
                    break

    else:   # EVALUATION
        agent.trainer_seq_eval(path, timesteps=21000)
        #agent.trainer_par_eval(path, timesteps=21000)
    env.close()


def sub_keyboard_event(event, *args, **kwargs) -> bool:

    if len(aliengo_env.base_command) > 0:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == 'W':
                aliengo_env.base_command[0] = [1, 0, 0]
            if event.input.name == 'S':
                aliengo_env.base_command[0] = [-1, 0, 0]
            if event.input.name == 'A':
                aliengo_env.base_command[0] = [0, 1, 0]
            if event.input.name == 'D':
                aliengo_env.base_command[0] = [0, -1, 0]
            if event.input.name == 'Q':
                aliengo_env.base_command[0] = [0, 0, 1]
            if event.input.name == 'E':
                aliengo_env.base_command[0] = [0, 0, -1]

            if len(aliengo_env.base_command) > 1:
                if event.input.name == 'I':
                    aliengo_env.base_command[1] = [1, 0, 0]
                if event.input.name == 'K':
                    aliengo_env.base_command[1] = [-1, 0, 0]
                if event.input.name == 'J':
                    aliengo_env.base_command[1] = [0, 1, 0]
                if event.input.name == 'L':
                    aliengo_env.base_command[1] = [0, -1, 0]
                if event.input.name == 'U':
                    aliengo_env.base_command[1] = [0, 0, 1]
                if event.input.name == 'O':
                    aliengo_env.base_command[1] = [0, 0, -1]
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            for i in range(len(aliengo_env.base_command)):
                aliengo_env.base_command[i] = [0, 0, 0]
    return True

def specify_cmd_for_robots(numv_envs):
    base_cmd = []
    for _ in range(numv_envs):
        base_cmd.append([0, 0, 0])
    aliengo_env.base_command = base_cmd


if __name__ == "__main__":
    main()
    simulation_app.close()