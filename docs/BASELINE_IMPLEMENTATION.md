# Baseline Implementation: Radosavovic et al. (2024)

This document describes our baseline implementation following the approach from:
> "Real-world humanoid locomotion with reinforcement learning"
> Radosavovic, Xiao, Zhang, Darrell, Malik, Sreenath
> Science Robotics 2024

## Overview

The baseline implementation provides a foundation that aligns with the state-of-the-art approach for humanoid locomotion training. This serves as a reference point before extending to high-speed (5-13 m/s) running.

## Key Components

### 1. Transformer Policy Architecture

**File:** `source/digit_locomotion/digit_locomotion/networks/transformer_policy.py`

```
Architecture (matching paper):
- 4 transformer blocks (causal attention)
- 192 embedding dimension
- 4 attention heads
- MLP ratio: 2.0
- Context window: 16 timesteps
- ~1.4M parameters

Input projection: MLP [512, 512] → embed_dim
Action head: MLP [256, 128] → num_actions
```

The transformer processes (observation, action) pairs as tokens and uses causal masking to only attend to past timesteps. This enables in-context adaptation without weight updates.

### 2. Observation History

The transformer automatically maintains a rolling history of 16 timesteps:
- Each timestep contains: observation + previous_action
- History is reset when episodes terminate
- Enables implicit system identification from observation patterns

### 3. Teacher-Student Training Pipeline

**Paper approach:**
1. Train teacher policy with privileged state (no noise)
2. Distill teacher to student using: `L = L_RL + λ * D_KL(student || teacher)`
3. Anneal λ to zero at training midpoint

**Our implementation:**
- `Digit-BaselineTeacher-v0`: Teacher environment (privileged state)
- `Digit-Baseline-v0`: Student environment (noisy observations)
- Separate training scripts for each phase

### 4. Reward Design (Emergent Arm Swing)

**Key principle: NO explicit arm swing rewards!**

The paper demonstrates that natural arm swing emerges from energy minimization alone:

```python
# Energy minimization (critical for emergent arm swing)
joint_torques_l2:   weight = -1e-5   # Strong energy penalty
joint_acc_l2:       weight = -2.5e-7
action_rate_l2:     weight = -0.01
joint_vel_l2:       weight = -1e-4

# Tracking (primary objectives)
track_lin_vel_xy:   weight = 1.5
track_ang_vel_z:    weight = 0.75

# Stability
lin_vel_z_l2:       weight = -2.0
ang_vel_xy_l2:      weight = -0.05
flat_orientation:   weight = -0.5

# Gait quality
feet_air_time:      weight = 0.125
```

## File Structure

```
source/digit_locomotion/digit_locomotion/
├── networks/
│   ├── __init__.py
│   └── transformer_policy.py      # CausalTransformer, ActorCriticTransformer
├── agents/
│   ├── __init__.py
│   ├── rsl_rl_cfg.py              # Original MLP configs
│   └── baseline_cfg.py            # Baseline configs (teacher/student)
└── tasks/locomotion/
    ├── __init__.py                 # Environment registration
    ├── velocity_env_cfg.py         # Original environments
    └── baseline_env_cfg.py         # Baseline environments

scripts/
└── train_baseline.py               # Training script
```

## Registered Environments

| Task ID | Description | Policy Type |
|---------|-------------|-------------|
| `Digit-Baseline-v0` | Baseline with MLP (comparison) | MLP [1024, 512, 256] |
| `Digit-BaselineTeacher-v0` | Teacher with privileged state | MLP [512, 512, 256, 128] |
| `Digit-BaselineLSTM-v0` | LSTM baseline (comparison) | LSTM + MLP |

## Training Commands

### Stage 1: Train Teacher (Fast)
```bash
# Teacher with privileged state information
./isaaclab.bat -p scripts/train_baseline.py --task Digit-BaselineTeacher-v0 --headless
```
Expected: ~5-10k iterations to convergence

### Stage 2a: Train MLP Baseline (Comparison)
```bash
# Standard MLP without transformer
./isaaclab.bat -p scripts/train_baseline.py --task Digit-Baseline-v0 --headless
```

### Stage 2b: Train LSTM Baseline (Comparison)
```bash
# LSTM for comparison (paper shows transformer >> LSTM)
./isaaclab.bat -p scripts/train_baseline.py --task Digit-BaselineLSTM-v0 --headless
```

### Resume Training
```bash
./isaaclab.bat -p scripts/train_baseline.py --task Digit-Baseline-v0 --resume \
    --checkpoint logs/digit_baseline_mlp/run_001/model_5000.pt --headless
```

## Comparison: Paper vs Our Implementation

| Aspect | Paper | Our Implementation | Status |
|--------|-------|-------------------|--------|
| Architecture | Causal Transformer | ✅ Implemented | Ready |
| Context Window | 16 timesteps | ✅ 16 timesteps | Ready |
| Teacher Policy | MLP [512, 512, 256, 128] | ✅ Implemented | Ready |
| Student Policy | Transformer | ✅ Implemented | Ready |
| IL + RL Loss | KL divergence + PPO | ⚠️ Manual two-stage | Partial |
| Arm Rewards | None (emergent) | ✅ None | Ready |
| Energy Terms | Joint torque penalty | ✅ Implemented | Ready |
| Obs Noise | Gaussian | ✅ Implemented | Ready |
| Obs Delay | Randomized | 📝 Noted for future | TODO |
| Policy Freq | 50 Hz | ✅ 50 Hz | Ready |
| Training Scale | 4x A100, thousands envs | ✅ 8k-16k envs | Ready |

## Expected Results

Based on the paper:
- Walking reliability: No falls in week of testing
- Velocity tracking: ~1 m/s forward walking
- Emergent arm swing: Contralateral with legs
- In-context adaptation: Gait changes on terrain changes
- Robustness: External pushes, varied surfaces

## Architecture Ablation (Paper Results)

Success rate on challenging terrain:
- **Transformer: ~95%** ← Target
- LSTM: ~60%
- TCN: ~45%
- MLP: ~25%

This motivates our transformer implementation as the primary approach.

## Next Steps

After establishing the baseline:
1. **Validate emergent arm swing** - Train baseline and confirm arm swing emerges
2. **Compare architectures** - Run MLP, LSTM, Transformer ablation
3. **Extend to high-speed** - Build on baseline for 5-13 m/s running
4. **Add explicit arm control** - May be needed at high speeds where biomechanics differ

## References

1. Radosavovic et al. "Real-world humanoid locomotion with reinforcement learning" Science Robotics 2024
2. RSL-RL: https://github.com/leggedrobotics/rsl_rl
3. Isaac Lab: https://isaac-sim.github.io/IsaacLab/
