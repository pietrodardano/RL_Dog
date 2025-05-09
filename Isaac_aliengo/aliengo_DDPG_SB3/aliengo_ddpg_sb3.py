import os
import datetime
import inspect
from colorama import Fore, Style

import torch
import torch.nn as nn

from skrl.memories.torch                import RandomMemory
from skrl.models.torch                  import DeterministicMixin, Model
from skrl.resources.noises.torch        import OrnsteinUhlenbeckNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler

from skrl.trainers.torch import SequentialTrainer, ParallelTrainer

# from skrl.utils import set_seed
# from skrl.envs.loaders.torch import load_isaaclab_env
# from skrl.envs.wrappers.torch import wrap_env

from stable_baselines3 import DDPG
from stable_baselines3.ddpg.policies import MlpPolicy

from isaaclab.envs  import ManagerBasedRLEnv
from aliengo_env    import RewardsCfg_SAFETY, RewardsCfg_ORIGINAL
from aliengo_env    import ObsCfg

_RewardsCfg = RewardsCfg_ORIGINAL
