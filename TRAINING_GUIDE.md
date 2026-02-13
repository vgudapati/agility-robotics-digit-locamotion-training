# Digit V4 Locomotion Teacher Training - Reproducibility Guide

This document describes the complete training pipeline to reproduce the Digit V4
locomotion teacher policy from scratch. The teacher is an MLP policy [512, 512, 256, 128]
trained with PPO (RSL-RL) using a velocity-command curriculum from walking (0-1 m/s)
to fast running (0-8 m/s / ~18 mph).

## Prerequisites

- **IsaacLab** installed (e.g., `~/IsaacLab` on Linux or `C:\IsaacLab` on Windows)
- **NVIDIA GPU** with >= 12 GB VRAM (tested on NVIDIA RTX A5000)
- This repository installed as a pip package in the IsaacLab Python environment

### Install this package

```bash
cd source/digit_locomotion
pip install -e .
```

## Architecture Overview

- **Policy**: MLP ActorCritic [512, 512, 256, 128], ELU activation
- **Action space**: 36 joints (legs/body only, arms locked via IdealPD actuator)
- **Observation space**: 148 dims (commands + base vel/ang_vel/gravity + joint pos/vel + prev actions)
- **PPO config**: 48 steps/env, 5 epochs, gamma=0.99, lambda=0.95, clip=0.2
- **Simulation**: 200 Hz physics, 50 Hz policy (decimation=4)

### GPU Profiles

| GPU | VRAM | `num_envs` | `num_mini_batches` | Mini-batch size | Notes |
|-----|------|------------|-------------------|-----------------|-------|
| RTX 4090 | 24 GB | 16,384 | 8 | ~98K | Baseline (original dev GPU) |
| RTX 5090 | 32 GB | 32,768 | 16 | ~98K | 2x envs, 2x mini-batches |

**Scaling rule**: When doubling `num_envs`, double `num_mini_batches` to keep the
mini-batch size constant (~98K transitions). This preserves the gradient noise/signal
ratio that the hyperparameters were tuned for. Failing to scale `num_mini_batches`
will slow convergence significantly (the larger mini-batches make gradient updates
too stable for effective early exploration).

## Curriculum Stages

| Stage | Task ID | Vel X (m/s) | Vel Y | Yaw | Episode |
|-------|---------|-------------|-------|-----|---------|
| 1. Walking | `Digit-Baseline-v0` | 0 - 1.0 | -0.5 to 0.5 | -1.0 to 1.0 | 20s |
| 2. Fast Walking | `Digit-BaselineFastWalking-v0` | 0 - 2.0 | -0.4 to 0.4 | -0.6 to 0.6 | 100s |
| 3. Slow Jogging | `Digit-BaselineSlowJogging-v0` | 0 - 2.5 | -0.35 to 0.35 | -0.55 to 0.55 | 100s |
| 4. Jogging | `Digit-BaselineJogging-v0` | 0 - 3.0 | -0.3 to 0.3 | -0.5 to 0.5 | 100s |
| 5. Moderate Running | `Digit-BaselineModerateRunning-v0` | 0 - 4.0 | -0.25 to 0.25 | -0.4 to 0.4 | 100s |
| 6. Running | `Digit-BaselineRunning-v0` | 0 - 5.0 | -0.2 to 0.2 | -0.3 to 0.3 | 100s |
| 7. Fast Running | `Digit-BaselineFastRunning-v0` | 0 - 8.0 | -0.1 to 0.1 | -0.2 to 0.2 | 100s |

## Key Hyperparameter Rules

| Parameter | Walking (0-3 m/s) | Running (4-5 m/s) | Fast Running (0-8 m/s) |
|-----------|-------------------|-------------------|------------------------|
| `--learning_rate` | 0.001 | 1e-5 | 5e-5 |
| `--lr_cap` | 0.001 | 1e-4 | 5e-5 |
| `--entropy_coef` | 0.005 | 0.005 | 0.001 then 0.0005 |
| `--init_noise_std` | 0.5 (from scratch only) | inherited | inherited |
| `--warmup_iters` | default (50) | default (50) | 50 |

**Critical rules:**
- `lr=0.001` works for stages 1-5 (up to 3 m/s), but causes collapse at higher speeds
- For stages 6+, reduce LR to `1e-5 → 5e-5` range
- `entropy_coef=0.005` for curriculum/exploration, then switch to `0.001` once the full
  speed range is learned, and `0.0005` for final fine-tuning
- Never use `--seed 42` with `cudnn.deterministic` — it hurts training. Use `--seed null`

---

## Training Commands — Linux

### Environment Setup (Linux)

```bash
export ISAACLAB_DIR=~/IsaacLab
export TRAIN_SCRIPT=/path/to/agility-robotics-digit-locamotion-training/scripts/train_baseline.py
export LOG_DIR=${ISAACLAB_DIR}/logs/digit_baseline_teacher
```

### Stage 1: Walking (0-1 m/s) — From Scratch

**RTX 4090 (16K envs):**
```bash
${ISAACLAB_DIR}/isaaclab.sh -p ${TRAIN_SCRIPT} \
  --task Digit-Baseline-v0 \
  --num_envs 16384 \
  --max_iterations 3000 \
  --headless \
  --episode_length_s 20.0 \
  --learning_rate 0.001 \
  --lr_cap 0.001 \
  --entropy_coef 0.005 \
  --init_noise_std 0.5
```

**RTX 5090 (32K envs):**
```bash
${ISAACLAB_DIR}/isaaclab.sh -p ${TRAIN_SCRIPT} \
  --task Digit-Baseline-v0 \
  --num_envs 32768 \
  --num_mini_batches 16 \
  --max_iterations 3000 \
  --headless \
  --episode_length_s 20.0 \
  --learning_rate 0.001 \
  --lr_cap 0.001 \
  --entropy_coef 0.005 \
  --init_noise_std 0.5
```

**Expected**: Converges in ~60-100 iterations (~15-25 min). Episode length should max
out at 1000 (=20s). Look for `track_lin_vel_xy_exp > 1.0`.

**Best checkpoint**: `model_400.pt` (or whenever ep_length stabilizes at ~1000).

### Stage 2: Fast Walking (0-2 m/s)

**RTX 4090 (16K envs):**
```bash
${ISAACLAB_DIR}/isaaclab.sh -p ${TRAIN_SCRIPT} \
  --task Digit-BaselineFastWalking-v0 \
  --num_envs 16384 \
  --max_iterations 3000 \
  --headless \
  --episode_length_s 100.0 \
  --resume --checkpoint ${LOG_DIR}/<stage1_run>/model_400.pt \
  --learning_rate 0.001 \
  --lr_cap 0.001 \
  --entropy_coef 0.005
```

**RTX 5090 (32K envs):**
```bash
${ISAACLAB_DIR}/isaaclab.sh -p ${TRAIN_SCRIPT} \
  --task Digit-BaselineFastWalking-v0 \
  --num_envs 32768 \
  --num_mini_batches 16 \
  --max_iterations 3000 \
  --headless \
  --episode_length_s 100.0 \
  --resume --checkpoint ${LOG_DIR}/<stage1_run>/model_400.pt \
  --learning_rate 0.001 \
  --lr_cap 0.001 \
  --entropy_coef 0.005
```

**Expected**: Converges in ~200-400 iterations. Episode length should approach 5000
(=100s). Use the checkpoint with highest `time_out` ratio.

**Best checkpoint**: Around `model_600.pt`.

### Stage 3: Slow Jogging (0-2.5 m/s)

**RTX 4090 (16K envs):**
```bash
${ISAACLAB_DIR}/isaaclab.sh -p ${TRAIN_SCRIPT} \
  --task Digit-BaselineSlowJogging-v0 \
  --num_envs 16384 \
  --max_iterations 3000 \
  --headless \
  --episode_length_s 100.0 \
  --resume --checkpoint ${LOG_DIR}/<stage2_run>/model_600.pt \
  --learning_rate 0.001 \
  --lr_cap 0.001 \
  --entropy_coef 0.005
```

**RTX 5090 (32K envs):**
```bash
${ISAACLAB_DIR}/isaaclab.sh -p ${TRAIN_SCRIPT} \
  --task Digit-BaselineSlowJogging-v0 \
  --num_envs 32768 \
  --num_mini_batches 16 \
  --max_iterations 3000 \
  --headless \
  --episode_length_s 100.0 \
  --resume --checkpoint ${LOG_DIR}/<stage2_run>/model_600.pt \
  --learning_rate 0.001 \
  --lr_cap 0.001 \
  --entropy_coef 0.005
```

**Expected**: Quick convergence (~100-200 iters) since it's only a 0.5 m/s increment.

**Best checkpoint**: Around `model_750.pt`.

### Stage 4: Jogging (0-3 m/s)

**RTX 4090 (16K envs):**
```bash
${ISAACLAB_DIR}/isaaclab.sh -p ${TRAIN_SCRIPT} \
  --task Digit-BaselineJogging-v0 \
  --num_envs 16384 \
  --max_iterations 3000 \
  --headless \
  --episode_length_s 100.0 \
  --resume --checkpoint ${LOG_DIR}/<stage3_run>/model_750.pt \
  --learning_rate 0.001 \
  --lr_cap 0.001 \
  --entropy_coef 0.005
```

**RTX 5090 (32K envs):**
```bash
${ISAACLAB_DIR}/isaaclab.sh -p ${TRAIN_SCRIPT} \
  --task Digit-BaselineJogging-v0 \
  --num_envs 32768 \
  --num_mini_batches 16 \
  --max_iterations 3000 \
  --headless \
  --episode_length_s 100.0 \
  --resume --checkpoint ${LOG_DIR}/<stage3_run>/model_750.pt \
  --learning_rate 0.001 \
  --lr_cap 0.001 \
  --entropy_coef 0.005
```

**Expected**: Converges in ~200-500 iters. time_out should reach 85%+.

**Best checkpoint**: Around `model_1000.pt`.

### Stage 5: Moderate Running (0-4 m/s)

**RTX 4090 (16K envs):**
```bash
${ISAACLAB_DIR}/isaaclab.sh -p ${TRAIN_SCRIPT} \
  --task Digit-BaselineModerateRunning-v0 \
  --num_envs 16384 \
  --max_iterations 3000 \
  --headless \
  --episode_length_s 100.0 \
  --resume --checkpoint ${LOG_DIR}/<stage4_run>/model_1000.pt \
  --learning_rate 0.001 \
  --lr_cap 0.001 \
  --entropy_coef 0.005
```

**RTX 5090 (32K envs):**
```bash
${ISAACLAB_DIR}/isaaclab.sh -p ${TRAIN_SCRIPT} \
  --task Digit-BaselineModerateRunning-v0 \
  --num_envs 32768 \
  --num_mini_batches 16 \
  --max_iterations 3000 \
  --headless \
  --episode_length_s 100.0 \
  --resume --checkpoint ${LOG_DIR}/<stage4_run>/model_1000.pt \
  --learning_rate 0.001 \
  --lr_cap 0.001 \
  --entropy_coef 0.005
```

**WARNING**: `entropy_coef=0.002` causes collapse at this stage (noise goes negative).
Keep at 0.005.

**Expected**: Peaks around 88% time_out, may degrade to ~70% before stabilizing.

**Best checkpoint**: Around `model_1250.pt` (take the peak before degradation).

### Stage 6: Running (0-5 m/s) — Lower LR Required

**IMPORTANT**: `lr=0.001` is too high for this speed range. It peaked at 30% timeout
then degraded. Use `lr=1e-5, lr_cap=1e-4` instead.

**RTX 4090 (16K envs):**
```bash
${ISAACLAB_DIR}/isaaclab.sh -p ${TRAIN_SCRIPT} \
  --task Digit-BaselineRunning-v0 \
  --num_envs 16384 \
  --max_iterations 3000 \
  --headless \
  --episode_length_s 100.0 \
  --resume --checkpoint ${LOG_DIR}/<stage5_run>/model_1250.pt \
  --learning_rate 0.00001 \
  --lr_cap 0.0001 \
  --entropy_coef 0.005
```

**RTX 5090 (32K envs):**
```bash
${ISAACLAB_DIR}/isaaclab.sh -p ${TRAIN_SCRIPT} \
  --task Digit-BaselineRunning-v0 \
  --num_envs 32768 \
  --num_mini_batches 16 \
  --max_iterations 3000 \
  --headless \
  --episode_length_s 100.0 \
  --resume --checkpoint ${LOG_DIR}/<stage5_run>/model_1250.pt \
  --learning_rate 0.00001 \
  --lr_cap 0.0001 \
  --entropy_coef 0.005
```

**Expected**: Slower convergence. time_out should reach 80%+ by ~300 iterations.
May need to run for full 3000 iterations.

**Best checkpoint**: Around `model_1740.pt` (81%+ time_out).

### Stage 7a: Fast Running (0-8 m/s) — Initial Training

**RTX 4090 (16K envs):**
```bash
${ISAACLAB_DIR}/isaaclab.sh -p ${TRAIN_SCRIPT} \
  --task Digit-BaselineFastRunning-v0 \
  --num_envs 16384 \
  --max_iterations 3000 \
  --headless \
  --episode_length_s 100.0 \
  --resume --checkpoint ${LOG_DIR}/<stage6_run>/model_1740.pt \
  --learning_rate 0.0000001 \
  --lr_cap 0.00001 \
  --warmup_iters 400 \
  --entropy_coef 0.005
```

**RTX 5090 (32K envs):**
```bash
${ISAACLAB_DIR}/isaaclab.sh -p ${TRAIN_SCRIPT} \
  --task Digit-BaselineFastRunning-v0 \
  --num_envs 32768 \
  --num_mini_batches 16 \
  --max_iterations 3000 \
  --headless \
  --episode_length_s 100.0 \
  --resume --checkpoint ${LOG_DIR}/<stage6_run>/model_1740.pt \
  --learning_rate 0.0000001 \
  --lr_cap 0.00001 \
  --warmup_iters 400 \
  --entropy_coef 0.005
```

**Note**: Very conservative initial LR (1e-7 → 1e-5 over 400 iters) prevents collapse
when jumping to 0-8 m/s. This run is slow but stable.

**Expected**: ~4500 iterations to reach 80%+ time_out. Run may need to be continued.

**Best checkpoint**: Around `model_4600.pt`.

### Stage 7b: Fast Running — LR Increase

Once the policy is stable at 0-8 m/s, increase LR to 5e-5 for faster progress:

**RTX 4090 (16K envs):**
```bash
${ISAACLAB_DIR}/isaaclab.sh -p ${TRAIN_SCRIPT} \
  --task Digit-BaselineFastRunning-v0 \
  --num_envs 16384 \
  --max_iterations 3000 \
  --headless \
  --episode_length_s 100.0 \
  --resume --checkpoint ${LOG_DIR}/<stage7a_run>/model_4600.pt \
  --learning_rate 0.00001 \
  --lr_cap 0.00005 \
  --warmup_iters 50 \
  --entropy_coef 0.005
```

**RTX 5090 (32K envs):**
```bash
${ISAACLAB_DIR}/isaaclab.sh -p ${TRAIN_SCRIPT} \
  --task Digit-BaselineFastRunning-v0 \
  --num_envs 32768 \
  --num_mini_batches 16 \
  --max_iterations 3000 \
  --headless \
  --episode_length_s 100.0 \
  --resume --checkpoint ${LOG_DIR}/<stage7a_run>/model_4600.pt \
  --learning_rate 0.00001 \
  --lr_cap 0.00005 \
  --warmup_iters 50 \
  --entropy_coef 0.005
```

**Expected**: time_out ~81-85%, noise_std plateaus around 2.16.

**Best checkpoint**: Around `model_4700.pt`.

### Stage 7c: Fast Running — Entropy Fine-Tuning (entropy=0.001)

This is the breakthrough step. Reducing entropy from 0.005 to 0.001 unlocks noise_std
descent, which dramatically improves policy quality:

**RTX 4090 (16K envs):**
```bash
${ISAACLAB_DIR}/isaaclab.sh -p ${TRAIN_SCRIPT} \
  --task Digit-BaselineFastRunning-v0 \
  --num_envs 16384 \
  --max_iterations 3000 \
  --headless \
  --episode_length_s 100.0 \
  --resume --checkpoint ${LOG_DIR}/<stage7b_run>/model_4700.pt \
  --learning_rate 0.00005 \
  --lr_cap 0.00005 \
  --warmup_iters 50 \
  --entropy_coef 0.001
```

**RTX 5090 (32K envs):**
```bash
${ISAACLAB_DIR}/isaaclab.sh -p ${TRAIN_SCRIPT} \
  --task Digit-BaselineFastRunning-v0 \
  --num_envs 32768 \
  --num_mini_batches 16 \
  --max_iterations 3000 \
  --headless \
  --episode_length_s 100.0 \
  --resume --checkpoint ${LOG_DIR}/<stage7b_run>/model_4700.pt \
  --learning_rate 0.00005 \
  --lr_cap 0.00005 \
  --warmup_iters 50 \
  --entropy_coef 0.001
```

**Expected over 3000 iterations (~12 hours)**:
- noise_std: 2.11 → 1.05
- reward: -1 → 328
- bad_orientation: 25% → 3.35%
- time_out: 75% → 96.65%
- action_rate: -1.1 → -0.38

**Best checkpoint**: `model_7699.pt` (final).

### Stage 7d: Fast Running — Final Polish (entropy=0.0005)

Further entropy reduction for the last stretch of noise_std improvement:

**RTX 4090 (16K envs):**
```bash
${ISAACLAB_DIR}/isaaclab.sh -p ${TRAIN_SCRIPT} \
  --task Digit-BaselineFastRunning-v0 \
  --num_envs 16384 \
  --max_iterations 3000 \
  --headless \
  --episode_length_s 100.0 \
  --resume --checkpoint ${LOG_DIR}/<stage7c_run>/model_7699.pt \
  --learning_rate 0.00005 \
  --lr_cap 0.00005 \
  --warmup_iters 50 \
  --entropy_coef 0.0005
```

**RTX 5090 (32K envs):**
```bash
${ISAACLAB_DIR}/isaaclab.sh -p ${TRAIN_SCRIPT} \
  --task Digit-BaselineFastRunning-v0 \
  --num_envs 32768 \
  --num_mini_batches 16 \
  --max_iterations 3000 \
  --headless \
  --episode_length_s 100.0 \
  --resume --checkpoint ${LOG_DIR}/<stage7c_run>/model_7699.pt \
  --learning_rate 0.00005 \
  --lr_cap 0.00005 \
  --warmup_iters 50 \
  --entropy_coef 0.0005
```

**Expected (still in progress)**:
- noise_std: 1.05 → 0.91 (after ~700 iters), projected to reach ~0.80-0.85
- All other metrics maintained or improved

---

## Training Commands — Windows

> **RTX 5090 users**: Add `--num_envs 32768 --num_mini_batches 16` to each command
> below (replacing `--num_envs 16384`). See the GPU Profiles table above.

### Environment Setup (Windows)

```cmd
set ISAACLAB_DIR=C:\IsaacLab
set TRAIN_SCRIPT=c:\Users\vguda\projects\custom\agility-robotics-digit-locamotion-training\scripts\train_baseline.py
set LOG_DIR=%ISAACLAB_DIR%\logs\digit_baseline_teacher
```

### Stage 1: Walking (0-1 m/s) — From Scratch

```cmd
%ISAACLAB_DIR%\isaaclab.bat -p %TRAIN_SCRIPT% ^
  --task Digit-Baseline-v0 ^
  --num_envs 16384 ^
  --max_iterations 3000 ^
  --headless ^
  --episode_length_s 20.0 ^
  --learning_rate 0.001 ^
  --lr_cap 0.001 ^
  --entropy_coef 0.005 ^
  --init_noise_std 0.5
```

### Stage 2: Fast Walking (0-2 m/s)

```cmd
%ISAACLAB_DIR%\isaaclab.bat -p %TRAIN_SCRIPT% ^
  --task Digit-BaselineFastWalking-v0 ^
  --num_envs 16384 ^
  --max_iterations 3000 ^
  --headless ^
  --episode_length_s 100.0 ^
  --resume --checkpoint %LOG_DIR%\<stage1_run>\model_400.pt ^
  --learning_rate 0.001 ^
  --lr_cap 0.001 ^
  --entropy_coef 0.005
```

### Stage 3: Slow Jogging (0-2.5 m/s)

```cmd
%ISAACLAB_DIR%\isaaclab.bat -p %TRAIN_SCRIPT% ^
  --task Digit-BaselineSlowJogging-v0 ^
  --num_envs 16384 ^
  --max_iterations 3000 ^
  --headless ^
  --episode_length_s 100.0 ^
  --resume --checkpoint %LOG_DIR%\<stage2_run>\model_600.pt ^
  --learning_rate 0.001 ^
  --lr_cap 0.001 ^
  --entropy_coef 0.005
```

### Stage 4: Jogging (0-3 m/s)

```cmd
%ISAACLAB_DIR%\isaaclab.bat -p %TRAIN_SCRIPT% ^
  --task Digit-BaselineJogging-v0 ^
  --num_envs 16384 ^
  --max_iterations 3000 ^
  --headless ^
  --episode_length_s 100.0 ^
  --resume --checkpoint %LOG_DIR%\<stage3_run>\model_750.pt ^
  --learning_rate 0.001 ^
  --lr_cap 0.001 ^
  --entropy_coef 0.005
```

### Stage 5: Moderate Running (0-4 m/s)

```cmd
%ISAACLAB_DIR%\isaaclab.bat -p %TRAIN_SCRIPT% ^
  --task Digit-BaselineModerateRunning-v0 ^
  --num_envs 16384 ^
  --max_iterations 3000 ^
  --headless ^
  --episode_length_s 100.0 ^
  --resume --checkpoint %LOG_DIR%\<stage4_run>\model_1000.pt ^
  --learning_rate 0.001 ^
  --lr_cap 0.001 ^
  --entropy_coef 0.005
```

### Stage 6: Running (0-5 m/s) — Lower LR Required

```cmd
%ISAACLAB_DIR%\isaaclab.bat -p %TRAIN_SCRIPT% ^
  --task Digit-BaselineRunning-v0 ^
  --num_envs 16384 ^
  --max_iterations 3000 ^
  --headless ^
  --episode_length_s 100.0 ^
  --resume --checkpoint %LOG_DIR%\<stage5_run>\model_1250.pt ^
  --learning_rate 0.00001 ^
  --lr_cap 0.0001 ^
  --entropy_coef 0.005
```

### Stage 7a: Fast Running (0-8 m/s) — Initial Training

```cmd
%ISAACLAB_DIR%\isaaclab.bat -p %TRAIN_SCRIPT% ^
  --task Digit-BaselineFastRunning-v0 ^
  --num_envs 16384 ^
  --max_iterations 3000 ^
  --headless ^
  --episode_length_s 100.0 ^
  --resume --checkpoint %LOG_DIR%\<stage6_run>\model_1740.pt ^
  --learning_rate 0.0000001 ^
  --lr_cap 0.00001 ^
  --warmup_iters 400 ^
  --entropy_coef 0.005
```

### Stage 7b: Fast Running — LR Increase

```cmd
%ISAACLAB_DIR%\isaaclab.bat -p %TRAIN_SCRIPT% ^
  --task Digit-BaselineFastRunning-v0 ^
  --num_envs 16384 ^
  --max_iterations 3000 ^
  --headless ^
  --episode_length_s 100.0 ^
  --resume --checkpoint %LOG_DIR%\<stage7a_run>\model_4600.pt ^
  --learning_rate 0.00001 ^
  --lr_cap 0.00005 ^
  --warmup_iters 50 ^
  --entropy_coef 0.005
```

### Stage 7c: Fast Running — Entropy Fine-Tuning (entropy=0.001)

```cmd
%ISAACLAB_DIR%\isaaclab.bat -p %TRAIN_SCRIPT% ^
  --task Digit-BaselineFastRunning-v0 ^
  --num_envs 16384 ^
  --max_iterations 3000 ^
  --headless ^
  --episode_length_s 100.0 ^
  --resume --checkpoint %LOG_DIR%\<stage7b_run>\model_4700.pt ^
  --learning_rate 0.00005 ^
  --lr_cap 0.00005 ^
  --warmup_iters 50 ^
  --entropy_coef 0.001
```

### Stage 7d: Fast Running — Final Polish (entropy=0.0005)

```cmd
%ISAACLAB_DIR%\isaaclab.bat -p %TRAIN_SCRIPT% ^
  --task Digit-BaselineFastRunning-v0 ^
  --num_envs 16384 ^
  --max_iterations 3000 ^
  --headless ^
  --episode_length_s 100.0 ^
  --resume --checkpoint %LOG_DIR%\<stage7c_run>\model_7699.pt ^
  --learning_rate 0.00005 ^
  --lr_cap 0.00005 ^
  --warmup_iters 50 ^
  --entropy_coef 0.0005
```

---

## Platform Differences

| Aspect | Linux | Windows |
|--------|-------|---------|
| IsaacLab launcher | `isaaclab.sh` | `isaaclab.bat` |
| Line continuation | `\` | `^` |
| Environment variables | `export VAR=val` / `${VAR}` | `set VAR=val` / `%VAR%` |
| Path separator | `/` | `\` |
| Default IsaacLab path | `~/IsaacLab` | `C:\IsaacLab` |
| Log directory | `~/IsaacLab/logs/...` | `C:\IsaacLab\logs\...` |
| `--headless` flag | Works as-is | Works as-is |
| `--hide_ui` flag | Do NOT use | Do NOT use (causes errors) |

**Note**: The training script, environment configs, and reward functions are
platform-independent Python code. Only the launcher and path conventions differ.

---

## Monitoring Training

Key metrics to watch (printed every iteration):

| Metric | Good Sign | Bad Sign |
|--------|-----------|----------|
| `Mean episode length` | Increasing or at max | Decreasing |
| `Episode_Termination/time_out` | > 90% | < 70% or dropping |
| `Episode_Termination/bad_orientation` | < 5% | > 10% or rising |
| `Mean action noise std` | Decreasing | Flat at high value or increasing |
| `Mean reward` | Increasing | Dropping consistently |
| `Metrics/base_velocity/error_vel_xy` | < 1.5 | > 2.0 |
| `Episode_Reward/action_rate_l2` | Closer to 0 | Large negative |

### When to advance to next stage

- `time_out > 85%` AND `bad_orientation < 10%`
- Episode length near the maximum (episode_length_s / dt / decimation)
- Reward has plateaued for 100+ iterations

### Signs of trouble

- **Reward dropping** for 50+ iterations → LR too high, try reducing lr_cap by 2-5x
- **noise_std increasing** → entropy_coef too high, or policy is losing confidence
- **bad_orientation spiking** → check if it's variance (1-2%) or a trend (3%+)
- **Episode length near 0** → catastrophic collapse, roll back to previous checkpoint

## Reward Configuration

All 16 active reward terms with their weights:

| Term | Weight | Purpose |
|------|--------|---------|
| `is_alive` | 5.0 | Survival bonus |
| `track_lin_vel_xy_exp` | 5.0 | XY velocity tracking (primary objective) |
| `track_ang_vel_z_exp` | 2.5 | Yaw rate tracking |
| `feet_air_time` | 1.0 | Encourage alternating foot contacts |
| `base_height` | -5.0 | Prevent crouching (target: 1.05m) |
| `forward_lean_penalty` | -10.0 | Excessive lean (max: 0.05 rad) |
| `joint_pos_limits` | -1.0 | Joint limit violations |
| `undesired_contacts` | -1.0 | Torso/rod contact |
| `flat_orientation_l2` | -0.5 | Stay upright |
| `lin_vel_z_l2` | -0.5 | Penalize vertical bouncing |
| `foot_contact_symmetry` | -0.5 | Penalize single-leg hopping |
| `ang_vel_xy_l2` | -0.05 | Penalize roll/pitch rotation |
| `action_rate_l2` | -0.005 | Action smoothness |
| `joint_torques_l2` | -1e-6 | Energy minimization (torques) |
| `mechanical_power` | -1e-5 | Energy minimization (power) |
| `joint_acc_l2` | -5e-8 | Joint acceleration smoothness |
| `termination_penalty` | -25.0 | Early termination (falling) |

## Important Implementation Details

### RSL-RL Bugs and Workarounds

The training script (`scripts/train_baseline.py`) applies several critical fixes:

1. **Adaptive KL scheduler disabled**: RSL-RL's adaptive scheduler runs 40x per
   iteration (8 mini-batches x 5 epochs) and overrides external LR caps. Fix:
   `desired_kl=None, schedule="fixed"` (lines 673-676).

2. **Noise std clamping**: RSL-RL stores action noise as a raw `nn.Parameter` (not
   log_std), so the optimizer can push it negative causing a crash. Fix: monkey-patch
   `optimizer.step()` to clamp `std >= 0.01` after every step (lines 636-652).

3. **`learn()` iteration counter**: RSL-RL's `learn(N)` treats N as RELATIVE iterations.
   The script calls `learn(1)` in a loop for per-iteration control, with manual
   increment of `runner.current_learning_iteration`.

### Arms Locked

Arm joints (14 joints matching `.*_arm_.*`) are excluded from the action space via
regex: `joint_names=["(?!.*_arm_).*"]`. Arms are held at default pose by an
`IdealPDActuatorCfg` with stiffness=200, damping=10. This was necessary because all
attempts to control arms via reward shaping failed (the policy consistently chose
T-pose for balance).

### Checkpoint Compatibility

All curriculum stages use the same network architecture [512, 512, 256, 128] and
observation space (148 dims), so checkpoints can be loaded across stages without
any modification.

## Total Training Time

| Stage | Approximate Time | Iterations |
|-------|-----------------|------------|
| Walking | 15-25 min | ~400 |
| Fast Walking | 1-2 hours | ~600 |
| Slow Jogging | 30-60 min | ~350 |
| Jogging | 1-3 hours | ~1000 |
| Moderate Running | 2-4 hours | ~1250 |
| Running | 4-8 hours | ~1740 |
| Fast Running (initial) | 16-20 hours | ~4600 |
| Fast Running (LR tune) | 1-2 hours | ~100-200 |
| Fast Running (entropy=0.001) | 12 hours | 3000 |
| Fast Running (entropy=0.0005) | 12 hours | 3000 |
| **Total** | **~50-70 hours** | **~16,000+** |

## Original Run References

The successful checkpoint chain used in development (Windows):

```
run_072 (Walking, from scratch)
  └─ model_400 → run_073 (Fast Walking)
       └─ model_600 → run_074 (Slow Jogging)
            └─ model_750 → run_075 (Jogging)
                 └─ model_1000 → run_081 (Moderate Running)
                      └─ model_1250 → run_082 (Running, lr=0.001 — degraded)
                           └─ model_1440 → run_083/084 (Running, lr=1e-5 — fixed)
                                └─ model_1740 → run_088 (Fast Running, lr=1e-7→1e-5)
                                     └─ model_4600 → run_090 (Fast Running, lr=5e-5)
                                          └─ model_4700 → run_092 (entropy=0.001)
                                               └─ model_7699 → run_093 (entropy=0.0005)
```

## Final Policy Quality (run_093, iter 8407, in progress)

- **noise_std**: 0.91
- **reward**: 336
- **time_out**: 96.5%
- **bad_orientation**: 3.5%
- **error_vel_xy**: 1.31 m/s
- **action_rate**: -0.30
