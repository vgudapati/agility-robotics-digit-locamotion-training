# Dockerfile for Digit Locomotion Training with Isaac Lab
#
# This Dockerfile creates an environment for training locomotion policies
# for the Agility Robotics Digit robot using NVIDIA Isaac Lab.
#
# Prerequisites:
#   - NVIDIA Docker runtime (nvidia-container-toolkit)
#   - NVIDIA GPU with driver 525.60.11 or later
#
# Build:
#   docker build -t digit-locomotion:latest -f docker/Dockerfile .
#
# Run (interactive):
#   docker run --gpus all -it --rm \
#       -v $(pwd):/workspace \
#       -v /tmp/.X11-unix:/tmp/.X11-unix \
#       -e DISPLAY=$DISPLAY \
#       digit-locomotion:latest
#
# Run training (headless):
#   docker run --gpus all --rm \
#       -v $(pwd)/logs:/workspace/logs \
#       digit-locomotion:latest \
#       python scripts/train.py --task Digit-Velocity-Flat-v0 --headless

# Base image: Isaac Sim with Isaac Lab
# Use the official NVIDIA Isaac Sim container as base
ARG ISAAC_SIM_VERSION=4.2.0
FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION} AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ACCEPT_EULA=Y

# Isaac Sim paths
ENV ISAAC_SIM_PATH=/isaac-sim
ENV ISAACSIM_PATH=/isaac-sim
ENV OMNI_KIT_ACCEPT_EULA=YES

# Python environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    htop \
    tmux \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Clone and install Isaac Lab
ARG ISAACLAB_VERSION=main
RUN git clone --branch ${ISAACLAB_VERSION} --depth 1 \
    https://github.com/isaac-sim/IsaacLab.git /opt/isaaclab

# Install Isaac Lab
WORKDIR /opt/isaaclab
RUN ${ISAAC_SIM_PATH}/python.sh -m pip install --upgrade pip && \
    ${ISAAC_SIM_PATH}/python.sh -m pip install -e source/extensions/omni.isaac.lab && \
    ${ISAAC_SIM_PATH}/python.sh -m pip install -e source/extensions/omni.isaac.lab_tasks && \
    ${ISAAC_SIM_PATH}/python.sh -m pip install -e source/extensions/omni.isaac.lab_assets

# Install RSL-RL for PPO training
RUN ${ISAAC_SIM_PATH}/python.sh -m pip install rsl-rl>=2.0.0

# Install additional dependencies
RUN ${ISAAC_SIM_PATH}/python.sh -m pip install \
    tensorboard \
    matplotlib \
    onnx \
    onnxruntime \
    toml

# Copy project files
WORKDIR /workspace
COPY . /workspace/

# Install the digit_locomotion extension
RUN ${ISAAC_SIM_PATH}/python.sh -m pip install -e source/digit_locomotion

# Create directories for outputs
RUN mkdir -p logs checkpoints policies videos

# Set Isaac Lab environment variables
ENV ISAACLAB_PATH=/opt/isaaclab
ENV PATH="${ISAAC_SIM_PATH}:${ISAACLAB_PATH}:${PATH}"

# Default command: show help
CMD ["bash", "-c", "${ISAAC_SIM_PATH}/python.sh scripts/train.py --help"]


# ============================================================================
# Development stage with additional tools
# ============================================================================
FROM base AS dev

# Install development dependencies
RUN ${ISAAC_SIM_PATH}/python.sh -m pip install \
    pytest \
    black \
    isort \
    mypy \
    ipython \
    jupyterlab

# Expose JupyterLab port
EXPOSE 8888

# Development command: start bash
CMD ["/bin/bash"]


# ============================================================================
# Training stage optimized for headless training
# ============================================================================
FROM base AS train

# Set headless mode by default
ENV HEADLESS=1
ENV DISPLAY=""

# Training entrypoint
ENTRYPOINT ["bash", "-c", "${ISAAC_SIM_PATH}/python.sh scripts/train.py \"$@\"", "--"]
CMD ["--task", "Digit-Velocity-Flat-v0", "--num_envs", "4096", "--headless"]
