#!/bin/bash

# RL_Dog Training Launcher
# This script provides convenient commands to run different training scenarios

# Colors for output
RED     ='\033[0;31m'
GREEN   ='\033[0;32m'
YELLOW  ='\033[1;33m'
BLUE    ='\033[0;34m'
NC      ='\033[0m' # No Color

# Get the directory where this script is located (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo -e "${GREEN}RL_Dog Training Launcher${NC}"
echo -e "${BLUE}Project Root: $PROJECT_ROOT${NC}"
echo ""

# Default values
DEFAULT_NUM_ENVS=2056
DEFAULT_HEADLESS=true
DEFAULT_ENABLE_CAMERAS=true

# Function to show usage
show_usage() {
    echo -e "${YELLOW}Usage: $0 [SCENARIO] [OPTIONS]${NC}"
    echo ""
    echo -e "${YELLOW}Available Scenarios:${NC}"
    echo "  walk-real     - Aliengo Walk Real Environment"
    echo "  walk-ideal    - Aliengo Walk Ideal Environment"
    echo "  stop-real     - Aliengo Stop Real Environment" 
    echo "  stop-full     - Aliengo Stop Full State Environment"
    echo "  ddpg          - Aliengo DDPG Training"
    echo "  ddpg-sb3      - Aliengo DDPG with Stable Baselines3"
    echo "  safety        - Aliengo Safety Training"
    echo ""
    echo -e "${YELLOW}Options:${NC}"
    echo "  --num-envs N        Number of environments (default: $DEFAULT_NUM_ENVS)"
    echo "  --gui               Run with GUI (default: headless)"
    echo "  --no-cameras        Disable cameras"
    echo "  --video             Enable video recording"
    echo "  --help              Show this help message"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0 walk-real                           # Run walk-real with defaults"
    echo "  $0 walk-ideal --num-envs 1024 --gui    # Run walk-ideal with GUI and 1024 envs"
    echo "  $0 ddpg --video                        # Run DDPG with video recording"
}

# Function to run training
run_training() {
    local scenario=$1
    local script_path=""
    local num_envs=$DEFAULT_NUM_ENVS
    local headless_flag="--headless"
    local cameras_flag="--enable_cameras"
    local video_flag=""
    
    # Parse additional arguments
    shift
    while [[ $# -gt 0 ]]; do
        case $1 in
            --num-envs)
                num_envs="$2"
                shift 2
                ;;
            --gui)
                headless_flag=""
                shift
                ;;
            --no-cameras)
                cameras_flag=""
                shift
                ;;
            --video)
                video_flag="--video"
                shift
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                return 1
                ;;
        esac
    done
    
    # Determine script path based on scenario
    case $scenario in
        walk-real)
            script_path="$PROJECT_ROOT/Isaac_aliengo/aliengo_vWalk_Real/aliengo_main.py"
            ;;
        walk-ideal)
            script_path="$PROJECT_ROOT/Isaac_aliengo/aliengo_Walk_Ideal/aliengo_main.py"
            ;;
        stop-real)
            script_path="$PROJECT_ROOT/Isaac_aliengo/aliengo_Stop_Real/aliengo_main.py"
            ;;
        stop-full)
            script_path="$PROJECT_ROOT/Isaac_aliengo/aliengo_Stop_FullState/aliengo_main.py"
            ;;
        ddpg)
            script_path="$PROJECT_ROOT/Isaac_aliengo/aliengo_DDPG/aliengo_main.py"
            ;;
        ddpg-sb3)
            script_path="$PROJECT_ROOT/Isaac_aliengo/aliengo_DDPG_SB3/aliengo_main.py"
            ;;
        safety)
            script_path="$PROJECT_ROOT/Isaac_aliengo/aliengo_safety/aliengo_main.py"
            ;;
        *)
            echo -e "${RED}Unknown scenario: $scenario${NC}"
            show_usage
            return 1
            ;;
    esac
    
    # Check if script exists
    if [[ ! -f "$script_path" ]]; then
        echo -e "${RED}Error: Script not found at $script_path${NC}"
        return 1
    fi
    
    # Activate conda environment if needed
    if command -v conda &> /dev/null; then
        echo -e "${BLUE}Activating isaacenv conda environment...${NC}"
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate isaacenv
    fi
    
    # Change to IsaacLab directory (assuming it's in the home directory)
    ISAACLAB_DIR="$HOME/IsaacLab"
    if [[ -d "$ISAACLAB_DIR" ]]; then
        echo -e "${BLUE}Changing to IsaacLab directory: $ISAACLAB_DIR${NC}"
        cd "$ISAACLAB_DIR"
    else
        echo -e "${YELLOW}Warning: IsaacLab directory not found at $ISAACLAB_DIR${NC}"
        echo -e "${YELLOW}Make sure IsaacLab is installed and the path is correct${NC}"
    fi
    
    # Build the command
    local cmd="./isaaclab.sh -p $script_path --num_envs $num_envs $headless_flag $cameras_flag $video_flag"
    
    echo -e "${GREEN}Running scenario: $scenario${NC}"
    echo -e "${BLUE}Command: $cmd${NC}"
    echo -e "${BLUE}Script: $script_path${NC}"
    echo -e "${BLUE}Environments: $num_envs${NC}"
    echo ""
    
    # Execute the command
    eval "$cmd"
}

# Main script logic
if [[ $# -eq 0 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_usage
    exit 0
fi

# Run the training
run_training "$@"
