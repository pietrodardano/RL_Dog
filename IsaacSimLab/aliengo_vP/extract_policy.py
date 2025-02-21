import torch
import os

path = "/home/rl_sim/RL_Dog/runs/AlienGo_vP_stoptry_09_09_11:32/checkpoints/best_agent.pt"
policy_nn = torch.load(path)['policy']
torch.save(policy_nn, "/home/rl_sim/RL_Dog/models/FULL_STATE__NN_vP.pt")

