# Docker Environment for Digit Locomotion Training

This directory contains Docker configuration for running Digit locomotion training in a containerized environment.

## Prerequisites

### Linux

1. **NVIDIA GPU** with compute capability 7.0+ (RTX 20xx or newer)
2. **NVIDIA Driver** version 525.60.11 or later
3. **Docker** with NVIDIA Container Toolkit:
   ```bash
   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
       sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

### Windows

1. **NVIDIA GPU** with compute capability 7.0+ (RTX 20xx or newer)
2. **NVIDIA Driver** version 525.60.11 or later (Windows Game Ready or Studio Driver)
3. **Windows 10/11** with WSL2 enabled
4. **Docker Desktop for Windows** with WSL2 backend:
   - Download from [Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - Enable WSL2 backend in Docker Desktop settings
   - Enable GPU support in Docker Desktop: Settings → Resources → WSL Integration

#### Windows GPU Setup

```powershell
# 1. Ensure WSL2 is installed
wsl --install

# 2. Update WSL
wsl --update

# 3. Verify GPU is accessible in WSL2
wsl nvidia-smi

# 4. Enable Docker Desktop WSL2 backend
# Docker Desktop → Settings → General → "Use WSL 2 based engine"

# 5. Enable GPU support
# Docker Desktop → Settings → Resources → WSL Integration → Enable for your distro
```

## Quick Start

### Build the Docker Image

**Linux:**
```bash
# From project root directory
docker compose -f docker/docker-compose.yml build
```

**Windows (PowerShell):**
```powershell
# From project root directory
docker compose -f docker/docker-compose.yml build
```

**Windows (Command Prompt):**
```cmd
docker compose -f docker\docker-compose.yml build
```

### Train on Flat Terrain

**Linux:**
```bash
docker compose -f docker/docker-compose.yml run train-flat
```

**Windows:**
```powershell
docker compose -f docker/docker-compose.yml run train-flat
```

### Train on Rough Terrain

```bash
# Linux/Windows (same command)
docker compose -f docker/docker-compose.yml run train-rough
```

### Monitor Training with TensorBoard

```bash
# Start TensorBoard (in separate terminal)
docker compose -f docker/docker-compose.yml up tensorboard

# Open browser to http://localhost:6006
```

### Interactive Development

**Linux:**
```bash
docker compose -f docker/docker-compose.yml run dev

# Inside container:
$ISAAC_SIM_PATH/python.sh scripts/train.py --task Digit-Velocity-Flat-v0 --headless
```

**Windows:**
```powershell
docker compose -f docker/docker-compose.yml run dev

# Inside container:
$ISAAC_SIM_PATH/python.sh scripts/train.py --task Digit-Velocity-Flat-v0 --headless
```

## Services

| Service | Description |
|---------|-------------|
| `train-flat` | Train on flat terrain (15k iterations) |
| `train-rough` | Train on rough terrain (30k iterations) |
| `train-resume` | Resume training from checkpoint |
| `evaluate` | Evaluate trained policy with visualization |
| `dev` | Interactive development environment |
| `tensorboard` | TensorBoard for monitoring |
| `export` | Export policy to ONNX format |

## Manual Docker Commands

### Build

**Linux:**
```bash
# Build base image
docker build -t digit-locomotion:latest -f docker/Dockerfile ..

# Build development image
docker build -t digit-locomotion:dev --target dev -f docker/Dockerfile ..
```

**Windows (PowerShell):**
```powershell
# Build base image
docker build -t digit-locomotion:latest -f docker/Dockerfile .

# Build development image
docker build -t digit-locomotion:dev --target dev -f docker/Dockerfile .
```

### Run Training

**Linux:**
```bash
docker run --gpus all --rm \
    -v $(pwd)/logs:/workspace/logs \
    -v $(pwd)/checkpoints:/workspace/checkpoints \
    digit-locomotion:latest \
    bash -c "$ISAAC_SIM_PATH/python.sh scripts/train.py \
        --task Digit-Velocity-Flat-v0 \
        --num_envs 4096 \
        --headless"
```

**Windows (PowerShell):**
```powershell
docker run --gpus all --rm `
    -v ${PWD}/logs:/workspace/logs `
    -v ${PWD}/checkpoints:/workspace/checkpoints `
    digit-locomotion:latest `
    bash -c "$ISAAC_SIM_PATH/python.sh scripts/train.py --task Digit-Velocity-Flat-v0 --num_envs 4096 --headless"
```

**Windows (Command Prompt):**
```cmd
docker run --gpus all --rm ^
    -v %cd%/logs:/workspace/logs ^
    -v %cd%/checkpoints:/workspace/checkpoints ^
    digit-locomotion:latest ^
    bash -c "$ISAAC_SIM_PATH/python.sh scripts/train.py --task Digit-Velocity-Flat-v0 --num_envs 4096 --headless"
```

### Run with Display (Linux only)

```bash
# Allow X11 forwarding
xhost +local:docker

# Run with display
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    digit-locomotion:latest \
    bash -c "$ISAAC_SIM_PATH/python.sh scripts/play.py \
        --task Digit-Velocity-Flat-v0 \
        --checkpoint logs/digit_flat/model_15000.pt"
```

> **Note:** X11 forwarding for GUI visualization is more complex on Windows. For Windows visualization, it's recommended to run Isaac Sim natively instead of in Docker.

## Volume Mounts

The following directories are mounted as volumes:

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./logs` | `/workspace/logs` | Training logs and TensorBoard |
| `./checkpoints` | `/workspace/checkpoints` | Model checkpoints |
| `./policies` | `/workspace/policies` | Exported ONNX models |
| `./videos` | `/workspace/videos` | Recorded evaluation videos |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ISAAC_SIM_PATH` | `/isaac-sim` | Path to Isaac Sim installation |
| `ISAACLAB_PATH` | `/opt/isaaclab` | Path to Isaac Lab |
| `HEADLESS` | `1` | Run without display (for training) |

## Troubleshooting

### GPU Not Detected

**Linux:**
```bash
# Verify NVIDIA runtime is working
docker run --gpus all nvidia/cuda:11.8-base nvidia-smi
```

**Windows:**
```powershell
# First check GPU in WSL2
wsl nvidia-smi

# Then check in Docker
docker run --gpus all nvidia/cuda:11.8-base nvidia-smi
```

If GPU is not detected on Windows:
1. Ensure you have the latest NVIDIA driver (Game Ready or Studio)
2. Update WSL2: `wsl --update`
3. Restart Docker Desktop
4. Restart your computer

### Out of Memory

Reduce the number of environments:
```bash
docker compose -f docker/docker-compose.yml run train-flat --num_envs 2048
```

### Display Issues (Linux)

```bash
# Allow Docker to access X11
xhost +local:docker

# Verify DISPLAY is set
echo $DISPLAY
```

### Display Issues (Windows)

For visualization on Windows, it's recommended to:
1. Run Isaac Sim natively (not in Docker)
2. Use the native Windows scripts (`scripts\train.bat`, `scripts\play.bat`)
3. Or use an X server like VcXsrv (advanced)

### Slow Training

Ensure you're using all available GPUs:
```bash
# Check GPU utilization during training
nvidia-smi -l 1
```

### Windows-Specific Issues

**"docker: Error response from daemon: could not select device driver"**
- Install/update NVIDIA Container Toolkit in WSL2
- Restart Docker Desktop

**"OCI runtime create failed"**
- Update WSL2: `wsl --update`
- Update NVIDIA driver
- Restart Docker Desktop

**Volume mount permission errors**
- Ensure the folder exists on the host
- Check Docker Desktop file sharing settings
