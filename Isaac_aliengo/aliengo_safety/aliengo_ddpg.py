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

from skrl.utils import set_seed
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env

from my_ddpg import DDPG, DDPG_DEFAULT_CONFIG

from isaaclab.envs  import ManagerBasedRLEnv
from aliengo_env    import RewardsCfg_SAFETY, RewardsCfg_ORIGINAL
from aliengo_env    import ObsCfg

set_seed() 

_RewardsCfg = RewardsCfg_SAFETY

# define models (deterministic models) using mixins
class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        # self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
        #                          nn.ELU(),
        #                          nn.Linear(256, 256),
        #                          nn.ELU(),
        #                          nn.Linear(256, 128),
        #                          nn.ELU(),
        #                          nn.Linear(128, self.num_actions)
        # )
        
        self.l1         = nn.Linear(self.num_observations, 256)
        self.l2         = nn.ELU()
        self.l3         = nn.Linear(256, 256)
        self.l4         = nn.ELU()
        self.l5         = nn.Linear(256, 128)
        self.l6         = nn.ELU()
        self.mean_layer = nn.Linear(128, self.num_actions)  
        self.net = nn.Sequential(self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.mean_layer)
        

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, 0, device)  # 0 for action_space
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + 0, 256), # 0 for action_space
                                 nn.ELU(),
                                 nn.Linear(256, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 1)
        )
        
    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

########################## ALIENGO_DDPG ##########################
class Aliengo_DDPG:
    def __init__(self, 
             env: ManagerBasedRLEnv, 
             device="cuda", 
             config=DDPG_DEFAULT_CONFIG.copy(), 
             name="Aliengo_XX", 
             directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../runs"), 
             verbose=0):
               
        # self.env        = load_isaaclab_env(task_name="AlienGo_safety", num_envs=self.env.num_envs)
        self.env        = wrap_env(env, wrapper="isaaclab")  # or isaaclab-multi-agent
        self.device     = device
        self.name       = name
        self.config     = config
        self.directory  = directory
        self.agent      = self._create_agent()
        
    def _create_agent(self):
        model_nn = {}
        memory = RandomMemory(memory_size=15625, num_envs=self.env.num_envs, device=self.device)
        
        # DDPG reqquires 4 models:   https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#models
        model_nn["policy"] = DeterministicActor(self.env.observation_space, self.env.action_space, self.device)
        pretrained_state_dict = torch.load("/home/user/Documents/GitHub/RL_Dog/Policies/FULL_STATE__NN_v2.pt", 
                                           map_location=self.device)
        
        ## ---- THIS ----
        ## Filter out unexpected keys and missing keys
        # model_state_dict = model_nn["policy"].state_dict()
        # filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}
        # model_state_dict.update(filtered_state_dict)
        # model_nn["policy"].load_state_dict(model_state_dict)
        
        ## ---- OR ----
        model_nn["policy"].migrate(state_dict=pretrained_state_dict)
        
        #model_nn["target_policy"] = DeterministicActor(self.env.observation_space, self.env.action_space, self.device)  # Needed
        model_nn["critic"]        = Critic(self.env.observation_space, self.env.action_space, self.device)
        model_nn["target_critic"] = Critic(self.env.observation_space, self.env.action_space, self.device)
        
        # model_nn["device"]= self.device
        
        self.config = {
            "exploration": {"noise": OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=0.5, device=self.device)},
            "gradient_steps": 1,
            "batch_size": 4096,
            "discount_factor": 0.99,
            "polyak": 0.099,  # 0.005, now higher since we know the policy
            # "actor_learning_rate": 5e-4,
            "critic_learning_rate": 5e-4,
            "random_timesteps": 80,
            "learning_starts": 80,
            "state_preprocessor": RunningStandardScaler,
            "state_preprocessor_kwargs": {"size": self.env.observation_space, "device": self.device},
            "experiment": {
            "write_interval": 1000,
            "checkpoint_interval": 10000,
            "directory": self.directory
            }
        }
        
        agent = DDPG(
            models=model_nn,
            memory=memory,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=self.device,
            cfg=self.config
        )
        return agent
    
    ########### TRAINER ###########
    def my_train(self, timesteps=21000, headless=True, mode="sequential"):
        cfg_trainer = {"timesteps": timesteps, "headless": headless}
        trainer     = SequentialTrainer(cfg=cfg_trainer, env=self.env, agents=self.agent) if mode == "sequential" else ParallelTrainer(cfg=cfg_trainer, env=self.env, agents=self.agent)
        directory   = self.directory
        self._save_source_code(directory, mode)
        return trainer
    
    def train_sequential(self, timesteps=21000, headless=True):
        trainer = self.my_train(timesteps, headless, mode="sequential")
        trainer.train()
        
    ########### WRITERS ###########
    def _save_source_code(self, directory, training_type):
        file_paths = {
            "DDPG_config.txt": self._get_DDPG_config_content(training_type),
            "RewardsCfg_source.txt": inspect.getsource(_RewardsCfg),
            "ObservationsCfg_source.txt": inspect.getsource(ObsCfg.PolicyCfg)
        }

        for file_name, content in file_paths.items():
            file_path = os.path.join(directory, file_name)
            try:
                with open(file_path, 'w') as f:
                    f.write(content)
                print(Fore.BLUE + f'[ALIENGO-DDPG] Source code saved in {file_path}' + Style.RESET_ALL)
            except Exception as e:
                print(Fore.RED + f'[ALIENGO-DDPG] {e}' + Style.RESET_ALL)

    def _get_DDPG_config_content(self, training_type):
        return (
            f"####### {training_type.upper()} TRAINING ####### \n\n"
            f"Num envs           -> {self.env.num_envs:>6} \n"
            "-------------------- DDPG CONFIG ------------------- \n"
            f"Batch_Size         -> {self.config['batch_size']:>6} \n"
            f"Critic_Lrate       -> {self.config['critic_learning_rate']:>6} \n"
            f"Discount Factor    -> {self.config['discount_factor']:>6} \n"
            f"Polyak             -> {self.config['polyak']:>6} \n"
            f"Learning Starts    -> {self.config['learning_starts']:>6} \n"
            f"Gradient Steps     -> {self.config['gradient_steps']:>6} \n"
        )






######################### ORIGINAL CONFIG #########################

DDPG_DEFAULT_CONFIG_insight = {
    "gradient_steps": 1,            # gradient steps
    "batch_size": 64,               # training batch size

    "discount_factor": 0.99,        # discount factor (gamma)
    "polyak": 0.005,                # soft update hyperparameter (tau)

    "actor_learning_rate": 1e-3,    # actor learning rate
    "critic_learning_rate": 1e-3,   # critic learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0,            # clipping coefficient for the norm of the gradients

    "exploration": {
        "noise": None,              # exploration noise
        "initial_scale": 1.0,       # initial scale for the noise
        "final_scale": 1e-3,        # final scale for the noise
        "timesteps": None,          # timesteps for the noise decay
    },

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

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