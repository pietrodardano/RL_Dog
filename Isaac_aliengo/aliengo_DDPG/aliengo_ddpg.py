import os
import datetime
import inspect
from colorama import Fore, Style
import numpy as np

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

#from my_ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG

from isaaclab.envs  import ManagerBasedRLEnv
from aliengo_env    import RewardsCfg
from aliengo_env    import ObsCfg

set_seed() 

_RewardsCfg = RewardsCfg

# Custom SequentialTrainer with logging capabilities
class LoggingSequentialTrainer(SequentialTrainer):
    def __init__(self, env, agents, agents_scope=None, cfg=None, debug_dir=None, max_debug_iterations=50):
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=cfg)
        self.debug_dir = debug_dir
        self.max_debug_iterations = max_debug_iterations
        self.debug_counter = 0
        
        # Create debug directory if it doesn't exist
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
            self.states_log_file = os.path.join(self.debug_dir, "states_log.txt")
            self.actions_log_file = os.path.join(self.debug_dir, "actions_log.txt")
            self.rewards_log_file = os.path.join(self.debug_dir, "rewards_log.txt")
            
            # Initialize log files with headers
            with open(self.states_log_file, 'w') as f:
                f.write("# States log for the first 50 iterations\n")
                f.write("# Format: [iteration][env_idx][feature_idx] = value\n\n")
                
            with open(self.actions_log_file, 'w') as f:
                f.write("# Actions log for the first 50 iterations\n")
                f.write("# Format: [iteration][env_idx][action_idx] = value\n\n")
                
            with open(self.rewards_log_file, 'w') as f:
                f.write("# Rewards log for the first 50 iterations\n")
                f.write("# Format: [iteration][env_idx] = value\n\n")
                
            print(Fore.GREEN + f"[ALIENGO-DDPG] Debug logs will be saved to {self.debug_dir}" + Style.RESET_ALL)
    
    def train(self, callback=None):
        """Train with logging for the first 50 iterations"""
        # set running mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("train")
        else:
            self.agents.set_running_mode("train")

        # reset env
        states, infos = self.env.reset()

        import tqdm
        import sys
        
        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):
            # pre-interaction
            if self.num_simultaneous_agents > 1:
                for agent in self.agents:
                    agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)
            else:
                self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                if self.num_simultaneous_agents > 1:
                    actions = torch.vstack(
                        [
                            agent.act(states[scope[0] : scope[1]], timestep=timestep, timesteps=self.timesteps)[0]
                            for agent, scope in zip(self.agents, self.agents_scope)
                        ]
                    )
                else:
                    actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]

                # Log states and actions for the first max_debug_iterations
                if self.debug_dir and self.debug_counter < self.max_debug_iterations:
                    self._log_step(timestep, states, actions)
                    self.debug_counter += 1

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # Log rewards
                if self.debug_dir and self.debug_counter <= self.max_debug_iterations:
                    self._log_rewards(timestep, rewards)

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                if self.num_simultaneous_agents > 1:
                    for agent, scope in zip(self.agents, self.agents_scope):
                        agent.record_transition(
                            states=states[scope[0] : scope[1]],
                            actions=actions[scope[0] : scope[1]],
                            rewards=rewards[scope[0] : scope[1]],
                            next_states=next_states[scope[0] : scope[1]],
                            terminated=terminated[scope[0] : scope[1]],
                            truncated=truncated[scope[0] : scope[1]],
                            infos=infos,
                            timestep=timestep,
                            timesteps=self.timesteps,
                        )
                else:
                    self.agents.record_transition(
                        states=states,
                        actions=actions,
                        rewards=rewards,
                        next_states=next_states,
                        terminated=terminated,
                        truncated=truncated,
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.timesteps,
                    )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            if self.num_simultaneous_agents > 1:
                                for agent in self.agents:
                                    agent.track_data(f"Info / {k}", v.item())
                            else:
                                self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            if self.num_simultaneous_agents > 1:
                for agent in self.agents:
                    agent.post_interaction(timestep=timestep, timesteps=self.timesteps)
            else:
                self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # Execute callback if provided
            if callback is not None:
                callback(timestep, self.timesteps)

            # reset environments
            if terminated.any() or truncated.any():
                with torch.no_grad():
                    states, infos = self.env.reset()
            else:
                states = next_states
    
    def _log_step(self, timestep, states, actions):
        """Log states and actions for debugging"""
        # Convert tensors to numpy for easier formatting
        states_np = states.detach().cpu().numpy()
        actions_np = actions.detach().cpu().numpy()
        
        # Log states
        with open(self.states_log_file, 'a') as f:
            f.write(f"Iteration {timestep}:\n")
            # Limit to first 5 environments to keep file size reasonable
            for env_idx, state in enumerate(states_np[:5]):
                f.write(f"  Env {env_idx}: {state.tolist()}\n")
            if len(states_np) > 5:
                f.write(f"  ... and {len(states_np)-5} more environments\n")
            f.write("\n")
        
        # Log actions
        with open(self.actions_log_file, 'a') as f:
            f.write(f"Iteration {timestep}:\n")
            for env_idx, action in enumerate(actions_np[:5]):
                f.write(f"  Env {env_idx}: {action.tolist()}\n")
            if len(actions_np) > 5:
                f.write(f"  ... and {len(actions_np)-5} more environments\n")
            f.write("\n")
    
    def _log_rewards(self, timestep, rewards):
        """Log rewards for debugging"""
        rewards_np = rewards.detach().cpu().numpy()
        
        with open(self.rewards_log_file, 'a') as f:
            f.write(f"Iteration {timestep}:\n")
            for env_idx, reward in enumerate(rewards_np[:5]):
                f.write(f"  Env {env_idx}: {reward.item()}\n")
            if len(rewards_np) > 5:
                f.write(f"  ... and {len(rewards_np)-5} more environments\n")
            f.write("\n")

# define models (deterministic models) using mixins
class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=True):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        # self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
        #                          nn.ReLU(),
        #                          nn.Linear(256, 256),
        #                          nn.ReLU(),
        #                          nn.Linear(256, self.num_actions),
        #                          nn.Tanh())
        
        #print(Fore.BLUE + f"[ALIENGO-DDPG_Actor] Observation Space: {self.num_observations}, Action Space: {self.num_actions}" + Style.RESET_ALL)
        
        self.l1         = nn.Linear(self.num_observations, 256)
        self.l2         = nn.ReLU()
        self.l3         = nn.Linear(256, 256)
        self.l4         = nn.ReLU()
        self.l5         = nn.Linear(256, 128)
        self.l6         = nn.ReLU()
        self.mean_layer = nn.Linear(128, self.num_actions)  
        self.net = nn.Sequential(self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.mean_layer)
        

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=True):
        Model.__init__(self, observation_space, action_space, device)  # 0 for action_space
        DeterministicMixin.__init__(self, clip_actions)
        
        #print(Fore.BLUE + f"[ALIENGO-DDPG_Critic] Observation Space: {self.num_observations}, Action Space: {self.num_actions}" + Style.RESET_ALL)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))
        
    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}

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
        self.env        = wrap_env(env) #, wrapper="isaaclab")  # or isaaclab-multi-agent
        self.device     = device
        self.name       = name
        self.config     = config
        self.directory  = directory
        self.agent      = self._create_agent()
        
    def _create_agent(self):
        model_nn = {}
        memory = RandomMemory(memory_size=32, num_envs=self.env.num_envs, device=self.device)
        
        # DDPG reqquires 4 models:   https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#models
        model_nn["policy"]        = DeterministicActor(self.env.observation_space, self.env.action_space, self.device)
        model_nn["target_policy"] = DeterministicActor(self.env.observation_space, self.env.action_space, self.device)  # Needed
        model_nn["critic"]        = Critic(self.env.observation_space, self.env.action_space, self.device)
        model_nn["target_critic"] = Critic(self.env.observation_space, self.env.action_space, self.device)
        
        self.config = {
            # "exploration": {"noise": OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=0.5, device=self.device)},
            "gradient_steps": 1,
            "batch_size": 64,             # set to 4096
            "discount_factor": 0.99,
            "polyak": 0.005,
            "actor_learning_rate": 1e-3,
            "critic_learning_rate": 1e-3,
            "random_timesteps": 100,
            "learning_starts": 100,
            "state_preprocessor": RunningStandardScaler,
            "state_preprocessor_kwargs": {"size": self.env.observation_space, "device": self.device},
            "experiment": {
            "write_interval": 1000,     # TensorBoard writing interval (timesteps)
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
        """Create and configure a trainer"""
        cfg_trainer = {"timesteps": timesteps, "headless": headless}
        
        # Create debug directory
        debug_dir = os.path.join(self.directory, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        if mode == "sequential":
            # Use our custom logging trainer instead of the standard one
            trainer = LoggingSequentialTrainer(
                cfg=cfg_trainer, 
                env=self.env, 
                agents=self.agent,
                debug_dir=debug_dir,
                max_debug_iterations=50
            )
        else:
            trainer = ParallelTrainer(cfg=cfg_trainer, env=self.env, agents=self.agent)
            
        self._save_source_code(self.directory, mode)
        return trainer
    
    def train_sequential(self, timesteps=21000, headless=True):
        """Train the agent using sequential trainer with debug logging"""
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
            #f"Actor_Lrate        -> {self.config['actor_learning_rate']:>6} \n"
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