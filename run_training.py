#!/usr/bin/env python3
"""
RL_Dog Training Launcher - Python version for cross-platform compatibility
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Import our config
sys.path.append(str(Path(__file__).parent))
from config import PROJECT_ROOT, get_isaac_aliengo_path

class Colors:
    """ANSI color codes for terminal output"""
    RED     = '\033[0;31m'
    GREEN   = '\033[0;32m'
    YELLOW  = '\033[1;33m'
    BLUE    = '\033[0;34m'
    NC      = '\033[0m'  # No Color

def print_colored(text, color):
    """Print colored text to terminal"""
    print(f"{color}{text}{Colors.NC}")

def get_available_scenarios():
    """Get list of available training scenarios"""
    scenarios = {}
    isaac_dir = PROJECT_ROOT / "Isaac_aliengo"
    
    if isaac_dir.exists():
        for subdir in isaac_dir.iterdir():
            if subdir.is_dir() and (subdir / "aliengo_main.py").exists():
                scenario_name = subdir.name.replace("aliengo_", "").replace("_", "-").lower()
                scenarios[scenario_name] = str(subdir / "aliengo_main.py")
    
    return scenarios

def find_isaaclab_path():
    """Try to find IsaacLab installation"""
    possible_paths = [
        Path.home() / "IsaacLab",
        Path.home() / "isaac-lab", 
        Path.home() / "isaaclab",
        Path("/opt/isaaclab"),
        Path("/usr/local/isaaclab")
    ]
    
    for path in possible_paths:
        if path.exists() and (path / "isaaclab.sh").exists():
            return path
    
    return None

def run_training(scenario, args):
    """Run the training scenario"""
    scenarios = get_available_scenarios()
    
    if scenario not in scenarios:
        print_colored(f"Error: Unknown scenario '{scenario}'", Colors.RED)
        print_colored("Available scenarios:", Colors.YELLOW)
        for s in scenarios.keys():
            print(f"  {s}")
        return 1
    
    script_path = scenarios[scenario]
    
    # Find IsaacLab
    isaaclab_path = find_isaaclab_path()
    if not isaaclab_path:
        print_colored("Error: Could not find IsaacLab installation", Colors.RED)
        print_colored("Please make sure IsaacLab is installed in your home directory", Colors.YELLOW)
        return 1
    
    # Build the command
    cmd = [
        str(isaaclab_path / "isaaclab.sh"),
        "-p", script_path,
        "--num_envs", str(args.num_envs)
    ]
    
    if args.headless:
        cmd.append("--headless")
    
    if args.enable_cameras:
        cmd.append("--enable_cameras")
    
    if args.video:
        cmd.append("--video")
    
    print_colored(f"Running scenario: {scenario}", Colors.GREEN)
    print_colored(f"Script: {script_path}", Colors.BLUE)
    print_colored(f"Command: {' '.join(cmd)}", Colors.BLUE)
    print_colored(f"IsaacLab path: {isaaclab_path}", Colors.BLUE)
    print()
    
    # Change to IsaacLab directory and run
    try:
        os.chdir(isaaclab_path)
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print_colored("\nTraining interrupted by user", Colors.YELLOW)
        return 1
    except Exception as e:
        print_colored(f"Error running training: {e}", Colors.RED)
        return 1

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="RL_Dog Training Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available scenarios are automatically detected from Isaac_aliengo subdirectories.

Examples:
  python run_training.py walk-real
  python run_training.py walk-ideal --num-envs 1024 --gui
  python run_training.py ddpg --video
        """
    )
    
    scenarios = get_available_scenarios()
    
    parser.add_argument(
        "scenario",
        choices=list(scenarios.keys()),
        help="Training scenario to run"
    )
    
    parser.add_argument(
        "--num-envs",
        type=int,
        default=2056,
        help="Number of environments (default: 2056)"
    )
    
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run with GUI (default: headless)"
    )
    
    parser.add_argument(
        "--no-cameras",
        action="store_true",
        help="Disable cameras"
    )
    
    parser.add_argument(
        "--video",
        action="store_true",
        help="Enable video recording"
    )
    
    args = parser.parse_args()
    
    # Process arguments
    args.headless = not args.gui
    args.enable_cameras = not args.no_cameras
    
    print_colored("RL_Dog Training Launcher", Colors.GREEN)
    print_colored(f"Project Root: {PROJECT_ROOT}", Colors.BLUE)
    print()
    
    return run_training(args.scenario, args)

if __name__ == "__main__":
    sys.exit(main())
