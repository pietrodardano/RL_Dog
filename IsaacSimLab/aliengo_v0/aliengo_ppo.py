### GENERAL ###

import torch

### SKRL-RL STUFF ###

from skrl.agents.torch.ppo import PPO
#from skrl.envs.loaders.torch import load
from skrl.envs.wrappers.torch import Wrapper, wrap_env
        # equal to --> from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper

from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.memories.torch import RandomMemory
from skrl.agents.torch import Agent
from skrl.trainers.torch import Trainer, SequentialTrainer, ParallelTrainer, StepTrainer 
from skrl.utils import set_seed


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

############# CHECK IF TO-DO ###############
# omni.isaac.lab_tasks.utils.wrappers.skrl.process_skrl_cfg(cfg: dict) → dict:
"""
from omni.isaac.lab_tasks.utils.wrappers.skrl import procss_skrl_cfg
process_skrl_cfg(ALIENGO_ENV_CFG)   # IF ANY (must adapt it)
"""

from omni.isaac.lab.envs     import ManagerBasedRLEnv
class PPO_v1:
    def __init__(self, env: ManagerBasedRLEnv, config=PPO_DEFAULT_CONFIG, device = "cpu"):
        self.env = wrap_env(env, wrapper="isaaclab")    # by SKRL
        self.config = config
        self.device = device
        self.num_envs = env.num_envs   #needed for MEMORY of PPO, num_envs comes from "args" of the env object
        self.agent = self._create_agent()

    def _create_model_nn(self):
        return {
            "policy": torch.nn.Sequential(
                torch.nn.Linear(self.env.observation_space.shape[0], 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, self.env.action_space.shape[0])
            )
        }
    
    def _create_preprocessor(self):
        return {
            "observation_scaler": RunningStandardScaler(shape=self.env.observation_space.shape)
        }

    def _create_agent(self):
        model_nn_ = self._create_model_nn()
        preprocessor_ = self._create_preprocessor()
        
        # instantiate a memory as rollout buffer (any memory can be used for this)
        memory_rndm_ = RandomMemory(memory_size=24, num_envs=self.num_envs, device=self.device)

        agent = PPO(
            models=model_nn_,
            preprocessor = preprocessor_,
            memory=memory_rndm_,
            cfg=self.config,
            env=self.env,
            device=self.device
        )
        return agent
    
    def train_mine_easy(self, num_episodes=1000):
        for ep in range(num_episodes):
            obs = self.env.reset()
            done = False
            tot_reward = 0
            cnt = 0
            while not done:
                action = self.agent.act(obs)  # act if a skrl.agent.torch.ppo --> PPO's method
                """ Process the environment's states to make a decision (actions) using the main policy """
                obs, reward, done, info = self.env.step(actions=action)
                tot_reward += reward 
                cnt += 1
                if cnt % 10 == 0: print(f"Episode: {ep}, Total Reward: {tot_reward}")

    def train_sequential(self, timesteps=20000, headless=False):
        cfg_trainer= {"timesteps": timesteps, "headless": headless}
        trainer = SequentialTrainer(cfg=cfg_trainer, env=self.env, agents=self.agent)
        trainer.train()

    def train_parallel(self, timesteps=20000, headless=False):
        cfg_trainer = {"timesteps": timesteps, "headless": headless}
        trainer = ParallelTrainer(cfg=cfg_trainer, env=self.env, agents=self.agent)
        trainer.train() 



# ADD TRAIN 

# ppo 1
# https://skrl.readthedocs.io/en/latest/_downloads/17f299c7b73f8d5f2c56b336c693da94/torch_velocity_anymal_c_ppo.py

# ppo2  --> EXAMPLE_anymal_PPO.py
# https://skrl.readthedocs.io/en/latest/_downloads/7f665f3e3ea3a391c065747e4d1ef288/torch_anymal_ppo.py

# skrl exampl
# https://skrl.readthedocs.io/en/latest/intro/examples.html#nvidia-isaac-lab