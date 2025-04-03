## Description of the directories in this folder

- **`aliengo_v0`** ==> Tried to use OOP style, not working yet! But not entirely wrong
- **`aliengo_v1`** ==> Works, not OOP style, not usefull for training, just the env is correct
- **`aliengo_v2`** ==> Basic WALK training
- **`aliengo_v3`** ==> STOP policy, IDEAL config (sensors)
- **`aliengo_v4`** ==> STOP policy, REAL config for Sim2Real
- **`aliengo_vP`** ==> STOP policy, Full robot STATE config for Paper
- **`aliengo_vW`** ==> WALK policy, IDEAL config

## Assets: URDF and USD
URDF are supported and IsaacLab can work directly with them, however is suggested to use USDs

**2 Methods** for URDF --> USD conversion: [GUIDE](https://isaac-sim.github.io/IsaacLab/source/how-to/import_new_asset.html)
- `SUGGESTED` 

    As explained in the .txt file in "assets" folder:
    ```
    conda activate isaacenv_
    cd 
    cd IsaacLab_

    ./isaaclab.sh -p source/standalone/tools/convert_urdf.py \
    ~/RL_Dog/assets/URDF/aliengo_color.urdf \
    source/extensions/omni.isaac.lab_assets/data/Robots/Unitree/aliengo_color.usd \
    --merge-joints \
    --make-instanceable
    ```
- `USD can be NOT correct` 

    Launch Omniverse-Launcher (app), run IsaacSim: IsaacUtils -> Workflows -> URDF Importer

**NOTE:** you can import directly the USD file in the `unitree.py` located in each subfolder `/aliengo_v*`
however it is <u>**STRONGLY SUGGESTED**</u> to import the RobotCFG that you can find and modify in you local `IsaacLab/../omni` directories:

- **DATA** : Where the USD file must be located
    ```
    ~/IsaacLab_/source/extensions/omni.isaac.lab_assets/data/Robots/Unitree
    ```

- **Unitree.py** (or RobotBrand.py): Where you crreate the Robot-Config
    ```
    ~/IsaacLab_/source/extensions/omni.isaac.lab_assets/omni/isaac/lab_assets/unitree.py
    ```

    `Open the .py file and add your Robot-Config` then you import THIS file, and its RobotsCFG

## Code structure

### In each `aliengo_v*` you will find:
- **`aliengo_main.py`** : here you import and call the methods bothfor training, evaluation, env_creation ...

- **`aliengo_env.py`** : here you create your custom environment **ManagerBasedRLEnv** style is suggested.

- **`aliengo_ppo.py`** : here you code your algorithm and the **agent** methods / trainers.

- **`aliengo_check.py`** : here you find a similar "main.py" just for the check of the trained robot or the environment.

You will launch **only** the `aliengo_main.py` file: usually in the top lines there are commented the CMD commands to launch your scripts and simulations

### Example to launch: 
Open terminal doing `Ctrl+Alt+T` then paste the following:
```
conda activate isaacenv_
cd
cd IsaacLab_
./isaaclab.sh -p /home/rl_sim/RL_Dog/IsaacSimLab/aliengo_v0/check_trained.py --num_envs 16
```
Or if everything is already set, do directly:
``` 
./isaaclab.sh -p /home/rl_sim/RL_Dog/IsaacSimLab/aliengo_v0/check_trained.py --num_envs 16 
```