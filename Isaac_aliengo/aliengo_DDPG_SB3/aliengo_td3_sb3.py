import os
import datetime
import inspect
from colorama import Fore, Style

import torch
import torch.nn as nn
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env

from isaaclab.envs import ManagerBasedRLEnv
from aliengo_env import RewardsCfg_ORIGINAL
from aliengo_env import ObsCfg

_RewardsCfg = RewardsCfg_ORIGINAL

# Define custom policy network
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        # Custom actor network
        self.actor = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, self.action_space.shape[0]),
            nn.Tanh()
        )

        # Custom critic network
        self.critic = nn.Sequential(
            nn.Linear(self.observation_space.shape[0] + self.action_space.shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, deterministic=False):
        return self.actor(obs)

    def _predict(self, observation, deterministic=False):
        return self.forward(observation, deterministic)

    def evaluate_actions(self, obs, actions):
        q_value = self.critic(torch.cat([obs, actions], dim=1))
        return q_value

########################## ALIENGO_TD3 ##########################
class Aliengo_TD3:
    def __init__(self,
                 env: ManagerBasedRLEnv,
                 device="cuda",
                 name="Aliengo_XX",
                 directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../runs"),
                 verbose=0):

        self.env = env
        self.device = device
        self.name = name
        self.directory = directory
        self.agent = self._create_agent()

    def _create_agent(self):
        # Define the action noise
        n_actions = self.env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        # Create the TD3 agent
        model = TD3(
            policy=CustomActorCriticPolicy,
            env=self.env,
            action_noise=action_noise,
            learning_rate=5e-4,
            buffer_size=1385,
            batch_size=4096,
            gamma=0.99,
            tau=0.099,
            train_freq=1,
            gradient_steps=1,
            verbose=1
        )
        return model

    ########### TRAINER ###########
    def my_train(self, timesteps=21000, headless=True):
        self.agent.learn(total_timesteps=timesteps)
        self._save_source_code(self.directory)

    ########### WRITERS ###########
    def _save_source_code(self, directory):
        file_paths = {
            "TD3_config.txt": self._get_TD3_config_content(),
            "RewardsCfg_source.txt": inspect.getsource(_RewardsCfg),
            "ObservationsCfg_source.txt": inspect.getsource(ObsCfg.PolicyCfg)
        }

        for file_name, content in file_paths.items():
            file_path = os.path.join(directory, file_name)
            try:
                with open(file_path, 'w') as f:
                    f.write(content)
                print(Fore.BLUE + f'[ALIENGO-TD3] Source code saved in {file_path}' + Style.RESET_ALL)
            except Exception as e:
                print(Fore.RED + f'[ALIENGO-TD3] {e}' + Style.RESET_ALL)

    def _get_TD3_config_content(self):
        return (
            f"####### TD3 TRAINING ####### \n\n"
            f"Num envs           -> {self.env.num_envs:>6} \n"
            "-------------------- TD3 CONFIG ------------------- \n"
            f"Batch_Size         -> {4096:>6} \n"
            f"Learning Rate      -> {5e-4:>6} \n"
            f"Discount Factor    -> {0.99:>6} \n"
            f"Tau                -> {0.099:>6} \n"
            f"Train Freq         -> {1:>6} \n"
            f"Gradient Steps     -> {1:>6} \n"
        )

######################### ORIGINAL CONFIG #########################
def get_default_config(env):
    n_actions = env.action_space.shape[-1]
    TD3_DEFAULT_CONFIG = {
        "gradient_steps": 1,
        "batch_size": 4096,
        "discount_factor": 0.99,
        "tau": 0.099,
        "learning_rate": 5e-4,
        "buffer_size": 1385,
        "train_freq": 1,
        "action_noise": NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)),
    }
    return TD3_DEFAULT_CONFIG
