
# do -->   conda activate isaaclabpip
#          ctr+shift+p --> select interpreter 

"""
                    COMMAND TO RUN THE SCRIPT 
cd 
cd IsaacLab_
./isaaclab.sh -p ~/RL_Dog/IsaacSimLab/Tutorials_IsaacLab/1empty_scene.py
"""

import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Creating empty stage.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# SCENES IMPORTS

from omni.isaac.lab.sim             import SimulationCfg, SimulationContext

import omni.isaac.lab.sim           as sim_utils
import omni.isaac.lab.utils.math    as math_utils
import omni.isaac.core.utils.prims  as prim_utils

from omni.isaac.lab.assets          import RigidObject, RigidObjectCfg
from omni.isaac.lab.utils.assets    import ISAAC_NUCLEUS_DIR

# EXTRA ISAACSIM

import torch

def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # # spawn distant light
    # cfg_light_distant = sim_utils.DistantLightCfg(
    #     intensity=3000.0,
    #     color=(0.75, 0.75, 0.75),
    # )
    # cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # create a new xform prim for all objects to be spawned under
    prim_utils.create_prim("/World/Objects", "Xform")
    # spawn a red cone
    cfg_cone = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )
    cfg_cone.func("/World/Objects/Cone1", cfg_cone, translation=(-1.0, 1.0, 1.0))
    cfg_cone.func("/World/Objects/Cone2", cfg_cone, translation=(-1.0, -1.0, 1.0))

    # spawn a green cone with colliders and rigid body
    cfg_cone_rigid = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
    cfg_cone_rigid.func(
        "/World/Objects/ConeRigid", cfg_cone_rigid, translation=(0.0, 0.0, 2.0), orientation=(0.5, 0.0, 0.5, 0.0)
    )

    # spawn a usd file of a table into the scene
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func("/World/Objects/Table", cfg, translation=(0.0, 0.0, 1.05))

def main():
    sim_cfg = SimulationCfg(dt=0.01, substeps=1)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    design_scene()

    # Play the simulator
    sim.reset()
    print("[INFO]: Setup completed ")
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()