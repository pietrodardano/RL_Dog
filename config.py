"""
Base configuration for RL_Dog project.
This file automatically detects the project root directory and provides 
paths that work for any user regardless of their system setup.
"""

import os
import sys
from pathlib import Path

def find_project_root():
    """
    Find the project root by looking for characteristic files/directories.
    This works regardless of where the script is called from.
    """
    # Start from the current file's directory
    current_path = Path(__file__).parent.absolute()
    
    # Look for characteristic files that indicate this is the project root
    root_indicators = [
        "Isaac_aliengo",
        "isaaclab", 
        "assets",
        "README.md",
        "LICENSE"
    ]
    
    # Walk up the directory tree
    for path in [current_path] + list(current_path.parents):
        # Check if this directory contains our indicators
        if any((path / indicator).exists() for indicator in root_indicators):
            return path
    
    # Fallback to the directory containing this config file
    return current_path

# Automatically detect the project root directory
PROJECT_ROOT = find_project_root()

# Common directories
ISAAC_ALIENGO_DIR = PROJECT_ROOT / "Isaac_aliengo"
ISAACLAB_DIR = PROJECT_ROOT / "isaaclab"
ISAACLAB_ASSETS_DIR = PROJECT_ROOT / "isaaclab_assets"
ISAACLAB_TASKS_DIR = PROJECT_ROOT / "isaaclab_tasks"
RUNS_DIR = PROJECT_ROOT / "runs"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Debug/logging directories
REPORT_DEBUG_DIR = PROJECT_ROOT / "report_debug"

# Ensure required directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [RUNS_DIR, REPORT_DEBUG_DIR]
    for directory in directories:
        directory.mkdir(exist_ok=True)
        
def get_project_root():
    """Get the absolute path to the project root."""
    return str(PROJECT_ROOT)

def get_isaac_aliengo_path(subproject_name):
    """Get the path to a specific Isaac_aliengo subproject."""
    return str(ISAAC_ALIENGO_DIR / subproject_name)

def get_runs_path():
    """Get the path to the runs directory."""
    return str(RUNS_DIR)

def get_debug_path():
    """Get the path to the debug/logging directory."""
    return str(REPORT_DEBUG_DIR)

def add_project_to_path():
    """Add the project root to Python path if not already there."""
    project_root_str = str(PROJECT_ROOT)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

# Initialize directories on import
ensure_directories()

# Add project to path automatically
add_project_to_path()

# Print project info when run directly
if __name__ == "__main__":
    print(f"RL_Dog Project Root: {PROJECT_ROOT}")
    print(f"Isaac Aliengo Dir: {ISAAC_ALIENGO_DIR}")
    print(f"Runs Dir: {RUNS_DIR}")
    print(f"Debug Dir: {REPORT_DEBUG_DIR}")
    print(f"Project added to Python path: {PROJECT_ROOT in sys.path}")
