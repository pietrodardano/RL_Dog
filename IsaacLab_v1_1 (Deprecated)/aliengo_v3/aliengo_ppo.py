### GENERAL ###ù

import os
import datetime
import inspect

import torch
import torch.nn as nn
from colorama import Fore, Style

### SKRL-RL STUFF ###

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import Wrapper, wrap_env
        # equal to --> from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper

from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.memories.torch import RandomMemory

from skrl.agents.torch import Agent
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

from skrl.trainers.torch import Trainer, SequentialTrainer, ParallelTrainer, StepTrainer 
from skrl.utils import set_seed

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed

### ISAACLAB CLASSES ###
from omni.isaac.lab.envs     import ManagerBasedRLEnv


# define shared model (stochastic and deterministic models) using mixins
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

        print(f"[AlienGo - PPO] Observation Space: {self.num_observations}, Action Space: {self.num_actions}")
        self.net = nn.Sequential(nn.Linear(self.num_observations, 256), # activ fcns were ELU
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 128),
                                 nn.ELU())

        self.mean_layer = nn.Linear(128, self.num_actions)       # num_actions: 12
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
        self.value_layer = nn.Linear(128, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            self._shared_output = self.net(inputs["states"])
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            shared_output = self.net(inputs["states"]) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}
        

def get_experiment_name_with_timestamp(base_name):
    timestamp = datetime.datetime.now().strftime("%d_%m_%H:%M")
    experiment_name = f"{base_name}_{timestamp}"
    
    return experiment_name

from aliengo_env import RewardsCfg
from aliengo_env import ObservationsCfg

class PPO_v1:
    def __init__(self, env: ManagerBasedRLEnv, config=PPO_DEFAULT_CONFIG, device = "cuda", verbose=0):
        self.env = wrap_env(env, verbose=verbose, wrapper="isaaclab")    # SKRL: wrapper = "auto", by default,  otherwise--> "isaac-orbit"
        self.config = config
        self.device = "cuda" 
        self.num_envs = env.num_envs   #needed for MEMORY of PPO, num_envs comes from "args" of the env object
        self.agent = self._create_agent()

    def _create_agent(self):

        #Create the model: In --> feedbacks/states , Out --> Actions/Actuators
        model_nn_ = {}
        model_nn_["policy"] = Shared(self.env.observation_space, self.env.action_space, self.device)
        model_nn_["value"] = model_nn_["policy"]

        # instantiate a memory as rollout buffer (any memory can be used for this)
        mem_size = 32
        batch_dim = 6
        memory_rndm_ = RandomMemory(memory_size=mem_size, num_envs=self.num_envs, device=self.device)
        self.config["rollouts"] = mem_size
        self.config["learning_epochs"] = 12
        self.config["mini_batches"] = 4 #min(mem_size * batch_dim / 48, 2 )# 48Gb VRAM of the RTX A6000
        

        self.config["lambda"] = 0.95 # GAE, Generalized Advantage Estimation: bias and variance balance
        self.config["discount_factor"] = 0.98 # ~1 Long Term, ~0 Short Term Rewards | Standard: 0.99
        self.config["entropy_loss_scale"] = 0.01 # Entropy Loss: ~1 --> Exploration vs ~0 --> Exploitation

        # Adjusts Learning Rate
        #self.config["learning_rate"] = 5e-4
        self.config["learning_rate_scheduler"] = KLAdaptiveRL   # Has problems with "verbose" param --> commented
        self.config["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}

        # Standardize the input data by removing the mean and scaling
        self.config["state_preprocessor"] = RunningStandardScaler
        self.config["state_preprocessor_kwargs"] = {"size": self.env.observation_space, "device": self.device}
        self.config["value_preprocessor"] = RunningStandardScaler
        self.config["value_preprocessor_kwargs"] = {"size": 1, "device": self.device}

        self.config["experiment"]["directory"] = "/home/rl_sim/RL_Dog/runs"

        base_name  = "AlienGo_v3_stoptry"
        timestamp = datetime.datetime.now().strftime("%d_%m_%H:%M")
        experiment_name = f"{base_name}_{timestamp}"
        self.config["experiment"]["experiment_name"] = experiment_name

        agent = PPO(
            models=model_nn_,
            memory=memory_rndm_,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            cfg=self.config,
            device=self.device,
        )
        return agent
    
    ########### TRAINING ###########
    def train_sequential(self, timesteps=20000, headless=False):
        cfg_trainer= {"timesteps": timesteps, "headless": headless}
        timestamp = datetime.datetime.now().strftime("%d_%m_%H:%M")
        trainer = SequentialTrainer(cfg=cfg_trainer, env=self.env, agents=self.agent)
               
        try:
            experiment_name = "AlienGo_v3_stoptry"
            directory = f"/home/rl_sim/RL_Dog/runs/{experiment_name}_{timestamp}"
            os.makedirs(directory, exist_ok=True)

            # Paths for source code files
            rewards_cfg_file_path = os.path.join(directory, "RewardsCfg_source.txt")
            observations_cfg_file_path = os.path.join(directory, "ObservationsCfg_source.txt")

            # Save source code for RewardsCfg
            try:
                rewards_cfg_source = inspect.getsource(RewardsCfg)
                with open(rewards_cfg_file_path, 'w') as f:
                    f.write("####### SEQUENTIAL TRAINING ####### \n\n")
                    f.write(rewards_cfg_source)
                print(f"Source code for RewardsCfg saved to {rewards_cfg_file_path}")
            except Exception as e:
                print(f"Failed to save source code for RewardsCfg: {e}")

            # Save source code for ObservationsCfg.PolicyCfg
            try:
                observations_cfg_source = inspect.getsource(ObservationsCfg.PolicyCfg)
                with open(observations_cfg_file_path, 'w') as f:
                    f.write("####### SEQUENTIAL TRAINING ####### \n\n")
                    f.write(observations_cfg_source)
                print(f"Source code for ObservationsCfg.PolicyCfg saved to {observations_cfg_file_path}")
            except Exception as e:
                print(f"Failed to save source code for ObservationsCfg.PolicyCfg: {e}")

        except Exception as e:
            print(f"An error occurred while setting up the experiment directory: {e}")
        trainer.train()

    def train_parallel(self, timesteps=20000, headless=False):
        timestamp = datetime.datetime.now().strftime("%d_%m_%H:%M")
        cfg_trainer = {"timesteps": timesteps, "headless": headless}
        trainer = ParallelTrainer(cfg=cfg_trainer, env=self.env, agents=self.agent)

        try:
            experiment_name = "AlienGo_v3_stoptry"
            directory = f"/home/rl_sim/RL_Dog/runs/{experiment_name}_{timestamp}"
            os.makedirs(directory, exist_ok=True)

            # Paths for source code files
            rewards_cfg_file_path = os.path.join(directory, "RewardsCfg_source.txt")
            observations_cfg_file_path = os.path.join(directory, "ObservationsCfg_source.txt")

            # Save source code for RewardsCfg
            try:
                rewards_cfg_source = inspect.getsource(RewardsCfg)
                with open(rewards_cfg_file_path, 'w') as f:
                    f.write("####### PARALLEL TRAINING ####### \n\n")
                    f.write(rewards_cfg_source)
                print(f"Source code for RewardsCfg saved to {rewards_cfg_file_path}")
            except Exception as e:
                print(f"Failed to save source code for RewardsCfg: {e}")

            # Save source code for ObservationsCfg.PolicyCfg
            try:
                observations_cfg_source = inspect.getsource(ObservationsCfg.PolicyCfg)
                with open(observations_cfg_file_path, 'w') as f:
                    f.write("####### PARALLEL TRAINING ####### \n\n")
                    f.write(observations_cfg_source)
                print(f"Source code for ObservationsCfg.PolicyCfg saved to {observations_cfg_file_path}")
            except Exception as e:
                print(f"Failed to save source code for ObservationsCfg.PolicyCfg: {e}")

        except Exception as e:
            print(f"An error occurred while setting up the experiment directory: {e}")
        trainer.train()

    ########### EVALUAION ###########
    def trainer_seq_eval(self, path: str, timesteps=20000, headless=False):
        self.agent.load(path)
        cfg_trainer= {"timesteps": timesteps, "headless": headless}
        trainer = SequentialTrainer(cfg=cfg_trainer, env=self.env, agents=self.agent)
        trainer.eval()

    def trainer_par_eval(self, path: str, timesteps=20000, headless=False):
        self.agent.load(path)
        cfg_trainer= {"timesteps": timesteps, "headless": headless}
        trainer = ParallelTrainer(cfg=cfg_trainer, env=self.env, agents=self.agent)
        trainer.eval()
    


#########################################################################################


# just to have a look about the values, this is ignored by the code since its after 
PPO_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}

# ADD TRAIN 

# ppo 1
# https://skrl.readthedocs.io/en/latest/_downloads/17f299c7b73f8d5f2c56b336c693da94/torch_velocity_anymal_c_ppo.py

# ppo2  --> EXAMPLE_anymal_PPO.py
# https://skrl.readthedocs.io/en/latest/_downloads/7f665f3e3ea3a391c065747e4d1ef288/torch_anymal_ppo.py

# skrl exampl
# https://skrl.readthedocs.io/en/latest/intro/examples.html#nvidia-isaac-lab