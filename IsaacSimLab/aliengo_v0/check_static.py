
"""This script demonstrates how to spawn AlienGo and interact with it.

.. code-block:: bash

    conda activate isaacenv_
    cd
    cd IsaacLab_
    ./isaaclab.sh -p ~/RL_Dog/IsaacSimLab/aliengo_v0/check_static.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse
import sys

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.lab.sim as sim_utils

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext

##
# Pre-defined configs
##
#from unitree import AliengoCFG_Black, AliengoCFG_Color
from omni.isaac.lab_assets.unitree          import AliengoCFG_Color, AliengoCFG_Black  #modified in IsaacLab_

def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a robot in it
    origins = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]

    for orig in range(len(origins)):
        prim_utils.create_prim(f"/World/Origin{orig+1}", "Xform", translation=origins[orig])

    ## AlienGo try
    robot_cfg = AliengoCFG_Black.copy()
    #robot_cfg.prim_ppath = "{ENV_REGEX_NS}/Robot"
    robot_cfg.prim_path ="/World/Origin.*/Robot"
    robot = Articulation(cfg=robot_cfg)


    # return the scene information
    scene_entities = {"aliengo":robot}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    robot = entities["aliengo"]

    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_state_to_sim(root_state)

            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1 # --> noise / disturbances / randomness
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            
            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robot state...")
        
        # Apply random action
        # -- generate random joint efforts
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        robot.set_joint_effort_target(efforts)
        robot.write_data_to_sim()
        sim.step()
        count += 1
        robot.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device="cpu", use_gpu_pipeline=False)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)

    sim.reset()

    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()
    simulation_app.close()