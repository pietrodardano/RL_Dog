import gymnasium as gym
import torch
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.wrappers.torch import GymWrapper
from environment.aliengo_env import AlienGoEnv, myGPU

# Define the PPO model
class Policy(Model):
    def __init__(self, observation_space, action_space, device=myGPU):
        super(Policy, self).__init__(observation_space, action_space, device)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.num_actions)
        )

    def compute(self, states, taken_actions=None, role="policy"):
        return self.model(states), {}

# Create and wrap the AlienGo environment
env = AlienGoEnv()
env = GymWrapper(env)

# Instantiate the memory
memory = Memory(memory_size=10000, num_envs=1, device=myGPU)

# Instantiate the agent's models (policy and value)
models = {
    "policy": Policy(env.observation_space, env.action_space, device=myGPU),
    "value": Policy(env.observation_space, env.action_space, device=myGPU)
}

cfg_agent = PPO_DEFAULT_CONFIG.copy()

# Configure and instantiate the PPO agent
agent = PPO(models=models,
            memory=memory,
            observation_space=env.observation_space,
            action_space=env.action_space,
            cfg=cfg_agent,
            device=myGPU)

# Configure and instantiate the RL trainer
trainer = SequentialTrainer(agents=agent, env=env)

# Train the agent
trainer.train()

# Save the trained models
#agent.save(directory="trained_models")
