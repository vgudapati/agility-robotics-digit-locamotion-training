"""
Digit Locomotion Training Extension for Isaac Lab.

This extension provides environments and configurations for training
locomotion policies for the Agility Robotics Digit humanoid robot.
"""

import os
import toml

# Read extension version from config
EXTENSION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTENSION_TOML = os.path.join(EXTENSION_DIR, "config", "extension.toml")

if os.path.exists(EXTENSION_TOML):
    ext_config = toml.load(EXTENSION_TOML)
    __version__ = ext_config.get("package", {}).get("version", "0.1.0")
else:
    __version__ = "0.1.0"

# Import submodules to register environments
from . import assets
from . import tasks
from . import agents
