import gymnasium as gym
import torch
from skrl.agents.torch import PPO
from skrl.envs.wrappers.torch import GymWrapper
from environment.aliengo_env import AlienGoEnv

# Load the trained models
policy = torch.load("trained_models/policy.pth")

# Create and wrap the AlienGo environment
env = AlienGoEnv()
env = GymWrapper(env)

# Instantiate the memory (not used during evaluation)
memory = None

# Instantiate the agent's models (policy)
models = {
    "policy": policy,
}

# Configure and instantiate the PPO agent
agent = PPO(models=models,
            device="cuda:0")

# Evaluate the agent
obs = env.reset()
for episode in range(10):
    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        env.render()
env.close()
