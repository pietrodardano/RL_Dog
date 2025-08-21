# RL_Dog Training Launcher


### Using the Python Launcher (Recommended - Cross-platform)

```bash
python run_training.py --help
```

```bash
# From the project root directory
python run_training.py walk-real                    # Run walk-real scenario with defaults
python run_training.py walk-ideal --gui             # Run with GUI instead of headless
python run_training.py ddpg --num-envs 1024         # Use 1024 environments
python run_training.py safety --video               # Enable video recording
```

Most used:
```bash
python run_training.py walk-ideal --num-envs 2048 --video

python run_training.py walk-real --num-envs 2048 --video
```

### Using the Bash Launcher (Linux/Mac)
```bash
./run_training.sh --help
```

```bash
# From the project root directory
./run_training.sh walk-real                         # Run walk-real scenario
./run_training.sh walk-ideal --gui --num-envs 1024  # Custom configuration
./run_training.sh ddpg --video                      # With video recording
```

## Available Scenarios

The launchers automatically detect available scenarios from the `Isaac_aliengo` directory:

- `walk-real`    — Aliengo Walk Real Environment
- `walk-ideal`   — Aliengo Walk Ideal Environment
- `stop-real`    — Aliengo Stop Real Environment
- `stop-full`    — Aliengo Stop Full State Environment
- `ddpg`         — Aliengo DDPG Training
- `ddpg-sb3`     — Aliengo DDPG with Stable Baselines3
- `safety`       — Aliengo Safety Training
- `vwalk-real`   — Aliengo vWalk Real Environment

## Command Options

- `--num-envs N`   — Number of environments (default: 2056)
- `--gui`          — Run with GUI instead of headless mode
- `--no-cameras`   — Disable camera rendering
- `--video`        — Enable video recording
- `--help`         — Show help message

## Legacy Commands

If you prefer the old manual approach, you can still use:

```bash
cd ~/IsaacLab
./isaaclab.sh -p /path_to/RL_Dog/Isaac_aliengo/aliengo_Walk_Real/aliengo_main.py --num_envs 2056 --headless --enable_cameras
```
