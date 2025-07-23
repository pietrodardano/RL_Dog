# RL_Dog 
[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.2.0-orange.svg)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-red.svg)](https://releases.ubuntu.com/22.04/)
[![skrl](https://img.shields.io/badge/skrl-1.4.3-yellow.svg)](https://skrl.readthedocs.io/en/latest/)
[![SB3](https://img.shields.io/badge/SB3-2.6.0-green.svg)](https://github.com/DLR-RM/stable-baselines3)

---------------------------------------------------------------------------------------------------

Reinforced-Learning for autonomous walking and suddenly-stopping for Legged Robot (AlienGo by Unitree)

Project by Pietro Dardano, advised by prof. [A. Del Prete](https://andreadelprete.github.io/) - UniTn - Summer 2024 + Spring 2025

Since the amount of example code for IsaacLab is still quite limited, this project can be seen as and example (even if still work in progress) on how to create Manager_Based_RL_Envs with different algorithms.

#### If you find it useful, please consider giving a star to the repository. :smile: :star:

## Methodology and Latest Updates
- **Algorithms and Agents:**
    - Proximal Policy Optimization (PPO)
        - SKRL: Works
    - Deep Deterministic Policy Gradient (DDPG)
        - SKRL: To fix NaN generation
        - SB3: To init and derive from TD3
    - Twind_Delayed_DDPG (TD3)
        - SB3: To fix and do first train
- **Architecture Inspired by ANYmal (ETH-RSL)**: Our architecture is based on the principles outlined in the [ANYmal paper](https://www.science.org/doi/epdf/10.1126/scirobotics.aau5872).
- **SKRL**: Wrapper for algorithms and agents. [Documentation](https://skrl.readthedocs.io/en/latest/intro/getting_started.html).
- **SB3**: Using it since SKRL is giving problems with DDPG. [Documentation](https://stable-baselines3.readthedocs.io/en/v1.0/guide/algos.html)
- **Python + PyTorch**: Programming languages and framework for development and deep learning.

## Repository
**Branches**:
- **main**: Currently mantained and working code: IsaacSim 4.5, IsaacLab 2.1.0
- **IsaacLab_v1.1**: Code from summer 2024, Legacy versions: IsaacSim 4.1, IsaacLab 1.1.0

## Configs tested: 
#### Nvidia & CUDA: Driver Version: 570.124.06 | CUDA Version: 12.8 
#### If Nvidia Blackwell: Driver 570.133.20 server-open (!!) | CUDA Version 12.8

| WorkStation    | CPU                        | GPU                                         | RAM    | OS                   |
|--------------- |--------------------------- |---------------------------------------------|--------|----------------------|
| WS 1  | AMD® Ryzen 9 7950x         | 2x NVIDIA RTX A6000 Ada Gen, 48Gb GDDR6, 300W | 192Gb  | Ubuntu 22.04.4 LTS   |
| WS 2  | Intel Xeon(R) Gold 6226R   | NVIDIA RTX A6000, 48Gb GDDR6, 300W           | 128Gb  | Ubuntu 20.04 LTS     |
| WS 3  | Intel Xeon(R) Gold 5415+   | NVIDIA RTX A4000, 14Gb GDDR6, 140W           | 128Gb  | Ubuntu 20.04 LTS     |
| WS 4  | AMD® Ryzen Thrd.rip 7970x  | NVIDIA RTX PRO Blackwell A6000, 96GB GDDR7   | 128Gb  | Ubuntu 22.04.4 LTS   |

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
1. Follow the installation [GUIDE](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) of IsaacSim and IsaacLab
2. Clone this repository: (Remark! **main** branch is for IsaacSim > v4.5 and IsaacLab > v2.0.0)
   ```
   git clone https://github.com/pietrodardano/RL_Dog.git
   ```
3. Check that your assets (URDF, config) are installed locally, in your IsaacLab folder in isaaclab_assets directory.
4. I am using Miniconda, be sure to change or use the same environment name.
5. Launch the simulation (headless or not) with the scripts that you can find at the beginnning of each **main.py**

## Extra

**NVIDIA's [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)**: allows for parallel simulation of multiple environments necessary for training our models. Refer to the [Orbit](https://isaac-orbit.github.io/) and [Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html) pages for more information. <br>

Please note that IsaacLab contains many OpenAI Gym and Gymnasium features. It is common to find attributes, methods and classes related to them. <br>

