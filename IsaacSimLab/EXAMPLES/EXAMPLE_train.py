import argparse
import torch
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.utils import configclass

from aliengo_env import AliengoEnvCfg
from ppo import PPOPolicy

def main():
    # Parse command-line arguments
    args = parse_args()

    # Initialize environment configuration
    env_cfg = AliengoEnvCfg(num_envs=args.num_envs, env_spacing=args.env_spacing)

    # Create the environment
    env = ManagerBasedRLEnv(env_cfg)

    # Create the policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PPOPolicy(env.observation_space, env.action_space, device)

    # Training loop
    num_episodes = 1000  # Example value
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            # Select action using the policy
            action = policy.select_action(obs)

            # Step the environment
            next_obs, reward, done, info = env.step(action)

            # Store experience and update policy
            policy.store_experience(obs, action, reward, next_obs, done)
            policy.update()

            obs = next_obs

    # Save the trained policy
    policy.save("ppo_policy.pth")

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for Quadruped Environment')
    parser.add_argument('--num_envs', type=int, default=16, help='Number of environments')
    parser.add_argument('--env_spacing', type=float, default=2.5, help='Environment spacing')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
