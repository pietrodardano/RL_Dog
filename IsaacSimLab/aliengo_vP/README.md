
## BODY order:
['base', 'FL_hip', 'FR_hip', 'RL_hip', 'RR_hip', 'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh', 'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']

## JOINT order:
['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']

## To Extract Policy:
```
path = "/home/rluser/RL_Dog/runs/AlienGo_vP_stoptry_**EXPERIMENT_NAME**/checkpoints/best_agent.pt"
policy_nn = torch.load(path)['policy']
torch.save(policy_nn, "/home/rluser/RL_Dog/models/policy_nn_NAME.pt")
```

### Observation Space: 37

NOTE:  JOINT's Position and Velocities are <u>RELATIVE</u> to the default ones
```
# e.g: Jpos_1 = Jpos1_Actual - Jpos1_Default

asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
```

- **`Pose`**: x, y, z 
- **`Quat`**: w, x, y, z
- **`Lvel`**: vx, vy, vz
- **`Avel`**: r, p, y
- **`Jpos`**: 'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint'
 - **`Jvel`**: 'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint'


### DEFAULT JOINT POSITION:
tensor([[ 0.1000, -0.1000,  0.1000, -0.1000,  0.8000,  0.8000,  1.0000,  1.0000, -1.5000, -1.5000, -1.5000, -1.5000]], device='cuda:0')

<font color="red">`Remark`: If setting the deafult joint positions to ALL 0.0, this Error is returned:</font>
```
ValueError: The following joints have default positions out of the limits: 
    - 'FL_calf_joint': 0.000 not in [-2.775, -0.646]
    - 'FR_calf_joint': 0.000 not in [-2.775, -0.646]
    - 'RL_calf_joint': 0.000 not in [-2.775, -0.646]
    - 'RR_calf_joint': 0.000 not in [-2.775, -0.646]

```

or we modify the limits (Dangerous, if for real robot it can break), or we put the calf_default_pose close to the lower limit!

### DEFAULT VELOCITIES: 
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0')

### Action Space: 12
- **`Jpos`**


## HOW I CONFIGURED (by DEFAULT) THE ROBOT:

The following script is for A1 by unitree, but its config is almost the same fo ALIENGO.
```
UNITREE_A1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/A1/a1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=33.5,
            saturation_effort=33.5,
            velocity_limit=21.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
```