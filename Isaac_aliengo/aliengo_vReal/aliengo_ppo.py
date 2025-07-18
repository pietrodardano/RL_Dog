import os
import datetime
import inspect

from colorama import Fore, Style

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
from skrl.envs.wrappers.torch import wrap_env

#### OTHER ####
set_seed(42)

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
        
class PPO_aliengo:
    def __init__(self, 
                 env: ManagerBasedRLEnv, 
                 config=PPO_DEFAULT_CONFIG, 
                 device="cuda", 
                 name="AlienGo_XX", 
                 directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../runs"), 
                 verbose=0):
        
        self.env        = wrap_env(env, wrapper="isaaclab")
        self.name       = name
        self.directory  = directory
        self.config     = config
        self.device     = device
        self.num_envs   = env.num_envs
        self.agent      = self. _create_agent()
    
    ###### AGENT ######
    def _create_agent(self):
        model_nn_ = {}
        model_nn_["policy"] = Shared(self.env.observation_space, self.env.action_space, self.device)
        model_nn_["value"] = model_nn_["policy"]
        
        mem_size = 24 if self.num_envs > 1028 else 32
        memory_rndm_ = RandomMemory(memory_size=mem_size, num_envs=self.num_envs, device=self.device)
        
        self.config={
            "rollouts": mem_size,           # 24 if many envs, 32 if few ones
            "learning_epochs": 6,           # no more than 12
            "mini_batches": 4,              # min(mem_size * batch_dim / 48, 2)   # 48Gb VRAM of the RTX A6000
            "lambda": 0.95,                 # GAE, Generalized Advantage Estimation: bias and variance balance
            "discount_factor": 0.98,       # ~1 Long Term, ~0 Short Term Rewards | Standard: 0.99
            "entropy_loss_scale": 0.004,    # Entropy Loss: Exploration~1, Eploitation~0 | Standard: [0.0, 0.006]
            "learning_rate": 5e-4,
            "learning_rate_scheduler": KLAdaptiveRL,
            "learning_rate_scheduler_kwargs": {"kl_threshold": 0.008},
            "state_preprocessor": RunningStandardScaler,
            "state_preprocessor_kwargs": {"size": self.env.observation_space, "device": self.device},
            "value_preprocessor": RunningStandardScaler,
            "value_preprocessor_kwargs": {"size": 1, "device": self.device},
            "experiment": {"directory": self.directory, "store_separately": True}
        }

        name_task = self.name
        timestamp = datetime.datetime.now().strftime("%d_%m_%H:%M")
        experiment_name = f"{name_task}_{timestamp}"
        self.config["experiment"]["experiment_name"] = experiment_name

        agent = PPO(
            models            = model_nn_,
            memory            = memory_rndm_,
            observation_space = self.env.observation_space,
            action_space      = self.env.action_space,
            cfg               = self.config,
            device            = self.device,
        )
        return agent
    
    ###### TRAINING ######
    def train_sequential(self, timesteps=20000, headless=False):
        trainer = self.mytrain(timesteps, headless, mode="sequential")
        trainer.train()
        
    ###### EVALUATION ######
    def trainer_seq_eval(self, path: str, timesteps=20000, headless=False):
        self.agent.init()
        self.agent.load(path)
        cfg_trainer = {"timesteps": timesteps, "headless": headless}
        trainer = SequentialTrainer(cfg=cfg_trainer, env=self.env, agents=self.agent)
        trainer.eval()
        
    ############ MY UTILITIES ############
    def mytrain(self, timesteps=20000, headless=False, mode="sequential"):
        cfg_trainer = {"timesteps": timesteps, "headless": headless}
        trainer_cls = SequentialTrainer(cfg=cfg_trainer, env=self.env, agents=self.agent) if mode == "sequential" else ParallelTrainer(cfg=cfg_trainer, env=self.env, agents=self.agent)
        directory   = self.directory
        self._save_source_code(directory, mode)
        return trainer_cls

    def _save_source_code(self, directory, training_type):
        file_paths = {
            "PPO_config.txt": self._get_ppo_config_content(training_type),
            "RewardsCfg_source.txt": inspect.getsource(RewardsCfg),
            "ObservationsCfg_source.txt": inspect.getsource(ObservationsCfg.PolicyCfg)
        }

        for file_name, content in file_paths.items():
            file_path = os.path.join(directory, file_name)
            try:
                with open(file_path, 'w') as f:
                    f.write(content)
                print(Fore.BLUE + f'[ALIENGO-PPO] Source code saved in {file_path}' + Style.RESET_ALL)
            except Exception as e:
                print(Fore.RED + f'[ALIENGO-PPO] {e}' + Style.RESET_ALL)

    def _get_ppo_config_content(self, training_type):
        return (
            f"####### {training_type.upper()} TRAINING ####### \n\n"
            f"Num envs           -> {self.num_envs:>6} \n"
            "-------------------- PPO CONFIG ------------------- \n"
            f"Rollouts           -> {self.config['rollouts']:>6} \n"
            f"Learning Epochs    -> {self.config['learning_epochs']:>6} \n"
            f"Mini Batches       -> {self.config['mini_batches']:>6} \n"
            f"Discount Factor    -> {self.config['discount_factor']:>6} \n"
            f"Lambda             -> {self.config['lambda']:>6} \n"
            f"Learning Rate      -> {self.config['learning_rate']:>6} \n"
            f"Entropy Loss Scale -> {self.config['entropy_loss_scale']:>6} \n"
        )
        


#########################################################################################

# just to have a look about the values, this is ignored by the code
PPO_DEFAULT_CONFIG_insight = {
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
