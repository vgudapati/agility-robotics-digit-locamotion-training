# Digit Locomotion Training

Train locomotion policies for the Agility Robotics Digit humanoid robot using NVIDIA Isaac Lab.

## Overview

This project provides a complete framework for training reinforcement learning policies that enable the Digit robot to walk in the NVIDIA Isaac Sim simulation environment. The trained policies can be deployed on the real Digit robot through sim-to-real transfer.

### Features

- **Isaac Lab Integration**: Built on NVIDIA's Isaac Lab framework for high-performance robot learning
- **Velocity Tracking**: Train policies that follow commanded forward, lateral, and angular velocities
- **Domain Randomization**: Comprehensive randomization for robust sim-to-real transfer
- **Terrain Curriculum**: Progressive training from flat to rough terrain
- **Modular Design**: Easy to customize rewards, observations, and training parameters
- **Cross-Platform**: Runs on both Linux and Windows

## Prerequisites

### System Requirements

- Ubuntu 20.04/22.04 or Windows 10/11
- NVIDIA GPU with RTX 30xx or better (recommended: RTX 4090)
- CUDA 11.8 or later
- At least 32GB RAM

### Software Requirements

- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) (2023.1.1 or later)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) (latest)
- Python 3.10 or 3.11

## Installation

### 1. Install Isaac Sim

Download and install Isaac Sim through the [NVIDIA Omniverse Launcher](https://www.nvidia.com/en-us/omniverse/).

### 2. Install Isaac Lab

#### Linux

```bash
# Clone Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Install Isaac Lab (creates conda environment)
./isaaclab.sh --install

# Activate the environment
conda activate isaaclab
```

#### Windows

```powershell
# Clone Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Install Isaac Lab (creates conda environment)
.\isaaclab.bat --install

# Activate the environment
conda activate isaaclab

# Set environment variable (add to your profile for persistence)
$env:ISAACLAB_PATH = "C:\path\to\IsaacLab"
```

### 3. Install This Project

#### Linux

```bash
# Clone this repository
git clone https://github.com/your-org/agility-robotics-digit-locamotion-training.git
cd agility-robotics-digit-locamotion-training

# Install the extension
pip install -e source/digit_locomotion

# Verify installation
python -c "import digit_locomotion; print('Installation successful!')"
```

#### Windows

```powershell
# Clone this repository
git clone https://github.com/your-org/agility-robotics-digit-locamotion-training.git
cd agility-robotics-digit-locamotion-training

# Install the extension
pip install -e source\digit_locomotion

# Verify installation
python -c "import digit_locomotion; print('Installation successful!')"
```

## Quick Start

### Train on Flat Terrain

Start with flat terrain training to establish basic walking.

#### Linux

```bash
# From the IsaacLab directory
./isaaclab.sh -p /path/to/this/repo/scripts/train.py \
    --task Digit-Velocity-Flat-v0 \
    --num_envs 4096 \
    --headless
```

#### Windows (PowerShell)

> **Important:** In PowerShell, you must use `.\` prefix to run scripts from the current directory.
> Also, deactivate conda first to use Isaac Sim's Python.

```powershell
# Navigate to IsaacLab directory
cd C:\IsaacLab

# Deactivate conda (required to use Isaac Sim's Python)
$env:CONDA_PREFIX = ""

# Run training (note the .\ prefix for PowerShell)
.\isaaclab.bat -p c:\path\to\this\repo\scripts\train.py --task Digit-Velocity-Flat-v0 --num_envs 4096 --headless
```

#### Windows (Command Prompt)

```cmd
cd C:\IsaacLab
set CONDA_PREFIX=
isaaclab.bat -p c:\path\to\this\repo\scripts\train.py --task Digit-Velocity-Flat-v0 --num_envs 4096 --headless
```

### Train on Rough Terrain

Once flat terrain training converges, train on rough terrain for robustness.

#### Linux

```bash
./isaaclab.sh -p /path/to/this/repo/scripts/train.py \
    --task Digit-Velocity-Rough-v0 \
    --num_envs 4096 \
    --headless
```

#### Windows (PowerShell)

```powershell
cd C:\IsaacLab
$env:CONDA_PREFIX = ""
.\isaaclab.bat -p c:\path\to\this\repo\scripts\train.py --task Digit-Velocity-Rough-v0 --num_envs 4096 --headless
```

#### Windows (Command Prompt)

```cmd
cd C:\IsaacLab
set CONDA_PREFIX=
isaaclab.bat -p c:\path\to\this\repo\scripts\train.py --task Digit-Velocity-Rough-v0 --num_envs 4096 --headless
```

### Evaluate a Trained Policy

#### Linux

```bash
./isaaclab.sh -p /path/to/this/repo/scripts/play.py \
    --task Digit-Velocity-Flat-v0 \
    --checkpoint logs/digit_flat/model_15000.pt \
    --num_envs 16
```

#### Windows (PowerShell)

```powershell
cd C:\IsaacLab
$env:CONDA_PREFIX = ""

# With visualization (4 robots for better viewing)
.\isaaclab.bat -p c:\path\to\this\repo\scripts\play.py `
    --task Digit-Velocity-Flat-v0 `
    --checkpoint C:\IsaacLab\logs\digit_flat\final_model.pt `
    --num_envs 4

# Headless evaluation (faster, more robots)
.\isaaclab.bat -p c:\path\to\this\repo\scripts\play.py `
    --headless --num_envs 64
```

#### Windows (Command Prompt)

```cmd
cd C:\IsaacLab
set CONDA_PREFIX=
isaaclab.bat -p c:\path\to\this\repo\scripts\play.py --task Digit-Velocity-Flat-v0 --checkpoint C:\IsaacLab\logs\digit_flat\final_model.pt --num_envs 4
```

### Export Policy for Deployment

```bash
# Works on both Linux and Windows
python scripts/export_policy.py \
    --checkpoint logs/digit_flat/model_15000.pt \
    --output policies/digit_walking.onnx \
    --verify
```

## Docker (Recommended for Reproducibility)

Docker provides a consistent environment across platforms. See [docker/README.md](docker/README.md) for detailed instructions.

### Quick Docker Start

#### Linux

```bash
# Build and train
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run train-flat
```

#### Windows (with Docker Desktop + WSL2)

```powershell
# Ensure Docker Desktop is running with WSL2 backend and GPU support enabled
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run train-flat
```

## Project Structure

```
agility-robotics-digit-locamotion-training/
├── scripts/
│   ├── train.py              # Training entry point
│   ├── train.bat             # Windows batch launcher
│   ├── train.ps1             # Windows PowerShell launcher
│   ├── play.py               # Policy evaluation
│   ├── play.bat              # Windows batch launcher
│   └── export_policy.py      # ONNX export for deployment
├── docker/
│   ├── Dockerfile            # Container definition
│   ├── docker-compose.yml    # Service definitions
│   └── README.md             # Docker documentation
├── source/digit_locomotion/
│   └── digit_locomotion/
│       ├── assets/
│       │   └── digit.py      # Digit robot configuration
│       ├── tasks/locomotion/
│       │   ├── velocity_env_cfg.py   # Environment configuration
│       │   └── mdp/
│       │       ├── rewards.py        # Reward functions
│       │       ├── observations.py   # Observation functions
│       │       ├── terminations.py   # Termination conditions
│       │       └── events.py         # Domain randomization
│       └── agents/
│           └── rsl_rl_cfg.py # PPO hyperparameters
├── logs/                     # Training logs (gitignored)
├── checkpoints/              # Model checkpoints (gitignored)
└── policies/                 # Exported policies
```

## Configuration

### Environment Configuration

The environment is configured in `velocity_env_cfg.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_envs` | 4096 | Number of parallel environments |
| `episode_length_s` | 20.0 | Episode duration in seconds |
| `decimation` | 4 | Policy runs at 50 Hz (200 Hz physics / 4) |

### Reward Weights

The reward function balances multiple objectives:

| Reward Term | Weight | Purpose |
|-------------|--------|---------|
| `track_lin_vel_xy_exp` | +1.5 | Track commanded XY velocity |
| `track_ang_vel_z_exp` | +0.75 | Track commanded yaw rate |
| `lin_vel_z_l2` | -2.0 | Penalize vertical bouncing |
| `ang_vel_xy_l2` | -0.05 | Smooth body motion |
| `flat_orientation_l2` | -1.0 | Keep body upright |
| `feet_air_time` | +0.125 | Encourage stepping gait |
| `action_rate_l2` | -0.01 | Smooth actions |

### PPO Hyperparameters

Training uses RSL-RL's PPO implementation:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 1e-3 | Adam learning rate |
| `num_epochs` | 5 | PPO epochs per update |
| `mini_batches` | 4 | Mini-batches per epoch |
| `gamma` | 0.99 | Discount factor |
| `lam` | 0.95 | GAE lambda |
| `clip_param` | 0.2 | PPO clip parameter |

## Training Tips

### Recommended Training Progression

1. **Phase 1: Flat Terrain** (~10-15k iterations)
   - Start with flat terrain
   - Focus on basic walking and standing
   - Expected reward: 15-20

2. **Phase 2: Rough Terrain** (~20-30k iterations)
   - Switch to rough terrain environment
   - Enable terrain curriculum
   - Expected reward: 12-18

3. **Phase 3: Full Curriculum** (~40-50k iterations)
   - Full domain randomization
   - All terrain types
   - Expected reward: 10-15

### Common Issues

**Robot falls immediately**
- Check initial joint positions in `digit.py`
- Reduce initial velocity randomization
- Increase standing reward weight

**Robot shuffles instead of stepping**
- Increase `feet_air_time` reward weight
- Add foot clearance reward
- Check foot contact sensor configuration

**Unstable training**
- Reduce learning rate
- Increase `num_mini_batches`
- Check for NaN in observations

## Customization

### Adding New Rewards

Create custom rewards in `mdp/rewards.py`:

```python
def my_custom_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """My custom reward description."""
    # Access robot data
    robot = env.scene["robot"]
    base_height = robot.data.root_pos_w[:, 2]

    # Compute reward
    return torch.exp(-torch.square(base_height - 1.0))
```

Then add to `RewardsCfg` in `velocity_env_cfg.py`:

```python
my_reward = RewardTermCfg(
    func=mdp.my_custom_reward,
    weight=1.0,
)
```

### Modifying the Robot

Edit `assets/digit.py` to change:
- Joint limits and default positions
- Actuator parameters (stiffness, damping, torque limits)
- Collision properties

## Sim-to-Real Transfer

For deploying on real hardware:

1. **Enable full domain randomization** during training
2. **Export to ONNX** using `export_policy.py`
3. **Match observation pipeline** on real robot
4. **Start with low gains** and gradually increase

Key randomizations for sim-to-real:
- Mass: ±10%
- Friction: 0.6-1.4
- Motor strength: ±20%
- Observation noise: Based on real sensor specs

## Acknowledgments

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) by NVIDIA
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) by ETH Zurich RSL
- [Agility Robotics](https://agilityrobotics.com/) for Digit robot

## License

MIT License - see [LICENSE](LICENSE) for details.
