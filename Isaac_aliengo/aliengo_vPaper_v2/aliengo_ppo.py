import os
import datetime
import inspect
import coloramas

import torch
import torch.nn as nn

#### SKRL ####
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.memories.torch import RandomMemory

from skrl.agents.torch.ppo  import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch      import Agent
from skrl.models.torch      import Model, GaussianMixin, DeterministicMixin

from skrl.trainers.torch import Trainer, SequentialTrainer, ParallelTrainer, StepTrainer 
from skrl.utils import set_seed
from skrl.envs.wrappers.torch import Wrapper, wrap_env

#### OTHER ####
set_seed(42)  # e.g. `set_seed(42)` for fixed seed

from isaaclab.envs  import ManagerBasedRLEnv
from aliengo_env    import RewardsCfg
from aliengo_env    import ObservationsCfg

class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        """
        clip:   Clipping actions means restricting the action values to be within the bounds defined by the action space. 
                This ensures that the actions taken by the agent are valid within the defined environment's action space.
                Clipping the log_of_STD ensures that values for the log STD do not goes below or above specified thresholds.

        reduction: The reduction method specifies how to aggregate the log probability densities when computing the total log probability.
        """

        print(Fore.BLUE + f"[ALIENGO-PPO] Observation Space: {self.num_observations}, Action Space: {self.num_actions}" + Style.RESET_ALL)
        
        # USE THIS TO PRINT LAYER BY LAYER , NEED TO TRAIN 
        self.l1 = nn.Linear(self.num_observations, 256)
        self.l2 = nn.ELU()
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.ELU()
        self.l5 = nn.Linear(256, 128)
        self.l6 = nn.ELU()
        self.net = nn.Sequential(self.l1, self.l2, self.l3, self.l4, self.l5, self.l6)

        self.mean_layer = nn.Linear(128, self.num_actions)       # num_actions: 12
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
        self.value_layer = nn.Linear(128, 1)

    def act(self, inputs, role):
        if role == "policy":            
            return GaussianMixin.act(self, inputs, role)  # ORIGINAL
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            self.o1 = self.l1(inputs["states"])
            self.o2 = self.l2(self.o1)
            self.o3 = self.l3(self.o2)
            self.o4 = self.l4(self.o3)
            self.o5 = self.l5(self.o4)
            self.o6 = self.l6(self.o5)
            self.o7 = self.mean_layer(self.o6)
            self._shared_output = self.net(inputs["states"])   #original  --> shared_output
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}                 #original
        elif role == "value":
            shared_output = self.net(inputs["states"]) if self._shared_output is None else self._shared_output
            self._shared_output = shared_output # it was "None"
            return self.value_layer(shared_output), {}
        
