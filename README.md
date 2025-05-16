# RL_Dog 
Reinforced-Learning for autonomous walking and suddenly-stopping for Legged Robot (AlienGo by Unitree)

Project by Pietro Dardano, advised by prof. [A. Del Prete](https://andreadelprete.github.io/) - UniTn - Summer 2024 + Spring 2025

Since the amount of opensource code for IsaacLab is still quite limited, this project can be seen as and example (even if still work in progress) on how to create Manager_Based_RL_Envs with different algorithms.

#### **If you find it useful, please consider giving a star to the repository.** :smile: :star:

## Methodology
- **Algorithms and Agents:**
    - Proximal Policy Optimization (PPO)
        - SKRL: Works
    - Deep Deterministic Policy Gradient (DDPG)
        - SKRL: To fix NaN generation
        - SB3: To init and derive from TD3
    - Twind_Delayed_DDPG (TD3)
        - SB3: To init
- **Architecture Inspired by ANYmal (ETH-RSL)**: Our architecture is based on the principles outlined in the [ANYmal paper](https://www.science.org/doi/epdf/10.1126/scirobotics.aau5872).
- **SKRL**: Wrapper for algorithms and agents. [Documentation](https://skrl.readthedocs.io/en/latest/intro/getting_started.html).
- **SB3**: Using it since SKRL is giving problems with DDPG. [Documentation](https://stable-baselines3.readthedocs.io/en/v1.0/guide/algos.html)
- **Python + PyTorch**: Programming languages and framework for development and deep learning.

## Repository
**Branches**:
- **main**: Currently mantained and working code: IsaacSim 4.5, IsaacLab 2.0.2 (v2.1.0 is now available but not yet integrated here)
- **IsaacLab_v1.1**: Code from summer 2024, Legacy versions: IsaacSim 4.1, IsaacLab 1.1.0

## Setup
### Workstation

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![skrl](https://img.shields.io/badge/skrl-1.4.3-green.svg)](https://skrl.readthedocs.io/en/latest/)

- **NVIDIA's [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)**:  provides the high-performance simulation environment necessary for training our models. Refer to the [Orbit](https://isaac-orbit.github.io/) and [Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html) pages for more information. <br>

### Configs: 
#### WorkStation 1
- CPU: AMD® Ryzen 9 7950x
- GPU: 2x NVIDIA RTX A6000 Ada Generation, 48Gb GDDR6, 300W
- RAM: 192Gb
- OS: Ubuntu 22.04.4 LTS

#### WorkStation 2
- CPU: Intel Xeon(R) Gold 6226R
- GPU: NVIDIA RTX A6000, 48Gb GDDR6, 300W
- RAM: 128Gb
- OS: Ubuntu 20.04 LTS

#### WorkStation 3
- CPU: Intel Xeon(R) Gold 5415+
- GPU: NVIDIA RTX A4000, 14Gb GDDR6, 140W
- RAM: 128Gb
- OS: Ubuntu 20.04 LTS

#### Nvidia & CUDA: Driver Version: 570.124.06 | CUDA Version: 12.8 

Please note that IsaacLab contains many OpenAI Gym and Gymnasium features. It is common to find attributes, methods and classes related to them. <br>
It contains [RSL_RL](https://github.com/leggedrobotics/rsl_rl/tree/master) too, helpfull framework by ETH for legged robot training.

### Laptop (lower grade simulations)
**Remark**: for the time being i am mainly working on the Isaac Sim+Lab version for a more complete and realistic simulation. I'll try my best to implement it soon.

- **OpenAI Gymnasium**: Since Isaac Sim is almost not suitable for being installed on **laptops**, I opted for the lightweight Gymnasium as a simulation environment. It supports Python 3.10 and Ubuntu 22.04 and it's well documented. Obviously, it is far from a realistic simulation, as Isaac Sim is, but for quick tests and trainings, I consider it a good trade-off considering my hardware limitations. For more details on Gymnasium, visit the [official documentation](https://gymnasium.farama.org/).

### `Why not Isaac Gym?`: 
It requires Ubuntu 20.04 or earlier and Python 3.8 or 3.9. Having installed Ubuntu 22.04, I excluded this option. <br>
It is deprecated too, everyone now-a-day is transitioning to IsaacLab 


## Understanding the Project

For a comprehensive understanding of the principles and techniques used in this project, refer to the following resources:
- A detailed review of related methodologies can be found in [reference 1](https://journals.sagepub.com/doi/full/10.1177/17298814211007305).
- Insights into recent advancements are discussed in [reference 2](https://arxiv.org/html/2308.12517v2).

## Project Structure
```
RL_Dog
├── Gymn_aliengo          # Envs and tasks for training using OpenAI Gym
├── Isaac_aliengo         # Envs and tasks for training using NVIDIA's Isaac Sim
├── Policies              # Trained Policies
├── assets                # Assets such as models and textures
├── isaaclab              # Just to access to Isaac Lab code, like if forked
│   ├── isaaclab_assets
│   ├── isaaclab_tasks
├── runs                  # Logs and results from training runs
├── LICENSE
├── README.md
├── SETUP_GUIDE.md        # Setup guide for the project
```

## Installation

To set up the project, follow these steps:
1. Setup your OS and Environment
    Instructions in the file: [IsaacSim-Setup_Guide](https://github.com/pietrodardano/RL_Dog/blob/main/SETUP_GUIDE.md) or TODO_Gymnasium-Setup_Guide
1. Clone the repository:
   ```
   git clone https://github.com/pietrodardano/RL_Dog.git
   ```


