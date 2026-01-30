# Digit Locomotion Training - Architecture & Improvements

## Current Architecture

### Robot Configuration
**File:** `source/digit_locomotion/digit_locomotion/assets/digit.py`

- Uses NVIDIA's built-in Digit V4 USD asset
- 50 DOF total (legs + arms)
- IdealPD actuators with stiffness=80, damping=4 for legs
- Initial height: 1.05m with slight knee bend

### Environment Configuration
**File:** `source/digit_locomotion/digit_locomotion/tasks/locomotion/velocity_env_cfg.py`

| Setting | Value |
|---------|-------|
| Physics rate | 200 Hz |
| Policy rate | 50 Hz (decimation=4) |
| Episode length | 20 seconds |
| Default num_envs | 8192 (optimized for RTX 4090) |

### Observation Space (162 dimensions)

| Observation | Dimensions |
|-------------|------------|
| Velocity commands | 3 |
| Base linear velocity | 3 |
| Base angular velocity | 3 |
| Projected gravity | 3 |
| Joint positions | 50 |
| Joint velocities | 50 |
| Previous actions | 50 |

### Action Space

- 50 joint position targets (full body control)

### Reward Function

| Term | Weight | Purpose |
|------|--------|---------|
| track_lin_vel_xy_exp | +1.5 | Follow XY velocity commands |
| track_ang_vel_z_exp | +0.75 | Follow yaw rate commands |
| lin_vel_z_l2 | -2.0 | Penalize vertical bouncing |
| ang_vel_xy_l2 | -0.05 | Smooth body rotation |
| flat_orientation_l2 | -1.0 | Keep torso upright |
| feet_air_time | +0.125 | Encourage stepping gait |
| action_rate_l2 | -0.01 | Smooth actions |
| joint_acc_l2 | -2.5e-7 | Smooth joint motion |
| joint_torques_l2 | -1e-5 | Energy efficiency |
| undesired_contacts | -1.0 | Penalize body collisions |
| joint_pos_limits | -1.0 | Stay within joint limits |

### PPO Configuration
**File:** `source/digit_locomotion/digit_locomotion/agents/rsl_rl_cfg.py`

| Parameter | Value (Optimized) |
|-----------|-------------------|
| Network | MLP [1024, 512, 256] |
| num_steps_per_env | 48 |
| Learning rate | 1e-3 |
| Epochs per update | 5 |
| Mini-batches | 8 |
| Gamma | 0.99 |
| GAE Lambda | 0.95 |
| Clip param | 0.2 |
| Max iterations | 15,000 |

---

## GPU Optimization

### Problem
With default settings, GPU utilization was only 30-40% and VRAM usage was 4-5 GB on RTX 4090 (24 GB).

### Solution - Optimized Settings

| Parameter | Old Value | New Value | Impact |
|-----------|-----------|-----------|--------|
| num_envs | 4096 | 8192 | 2x more parallel simulations |
| num_steps_per_env | 24 | 48 | 2x more data per update |
| Network size | [512,256,128] | [1024,512,256] | Larger model capacity |
| num_mini_batches | 4 | 8 | Better batch utilization |

### Expected Performance Improvement

| Metric | Before | After (Estimated) |
|--------|--------|-------------------|
| GPU Utilization | 30-40% | 60-80% |
| VRAM Usage | 4-5 GB | 10-15 GB |
| Steps/second | ~700 | ~1200-1500 |
| Training time (15k iter) | ~9 hours | ~4-5 hours |

### Command for Maximum GPU Utilization (RTX 4090)

```powershell
# Use 16384 environments for full GPU utilization
.\isaaclab.bat -p scripts\train.py --task Digit-Velocity-Flat-v0 --num_envs 16384 --headless
```

### Scaling Guide

| GPU | VRAM | Recommended num_envs |
|-----|------|---------------------|
| RTX 3080 | 10 GB | 4096 |
| RTX 3090 | 24 GB | 8192-12288 |
| RTX 4080 | 16 GB | 6144-8192 |
| RTX 4090 | 24 GB | 8192-16384 |
| A100 | 40/80 GB | 16384-32768 |

---

## Training Results (15,000 iterations, old settings)

| Metric | Start | End |
|--------|-------|-----|
| Episode length | 7.6 steps | 36.2 steps (4.8x improvement) |
| Mean reward | -0.35 | +0.07 (positive!) |
| Track lin vel reward | 0.0027 | 0.0379 (14x improvement) |
| Training time | - | ~9 hours on RTX 4090 |
| Primary failure mode | - | `bad_orientation` (robot falls over) |

---

## Recommended Improvements

### 1. Increase Training Duration

```python
# In rsl_rl_cfg.py
max_iterations = 50000  # Currently 15000
```

### 2. Strengthen Balance Rewards

```python
# In velocity_env_cfg.py - RewardsCfg
flat_orientation_l2 = RewardTermCfg(
    func=mdp.flat_orientation_l2,
    weight=-2.0,  # Increase from -1.0
)
```

### 3. Add Base Height Reward

```python
# Encourage maintaining proper standing height
base_height = RewardTermCfg(
    func=mdp.base_height_l2,
    weight=-1.0,
    params={"target_height": 1.0, "asset_cfg": SceneEntityCfg("robot")},
)
```

### 4. Curriculum Learning for Velocity Commands

```python
# Start with small velocities, gradually increase
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.5),   # Start smaller (was -1.0, 1.0)
            lin_vel_y=(-0.2, 0.2),   # Start smaller (was -1.0, 1.0)
            ang_vel_z=(-0.5, 0.5),   # Start smaller (was -1.0, 1.0)
        ),
    )
```

### 5. Reduce Action Rate Penalty

```python
# Allow faster corrections for balance
action_rate_l2 = RewardTermCfg(
    func=mdp.action_rate_l2,
    weight=-0.005,  # Reduce from -0.01
)
```

### 6. Add Feet Contact Symmetry

```python
# Encourage alternating foot contacts (proper gait)
feet_contact_symmetry = RewardTermCfg(
    func=mdp.feet_contact_symmetry,
    weight=0.5,
)
```

### 7. Domain Randomization (for sim-to-real)

```python
# In EventsCfg
randomize_mass = EventTermCfg(
    func=mdp.randomize_rigid_body_mass,
    mode="startup",
    params={"mass_range": (0.9, 1.1)},  # ±10%
)
randomize_friction = EventTermCfg(
    func=mdp.randomize_physics_material_friction,
    mode="startup",
    params={"friction_range": (0.6, 1.4)},
)
```

### 8. Network Architecture

```python
# Consider larger network for complex humanoid
actor_hidden_dims = [1024, 512, 256]  # Currently [512, 256, 128]
critic_hidden_dims = [1024, 512, 256]
```

---

## Suggested Training Progression

| Phase | Task | Iterations | Focus |
|-------|------|------------|-------|
| 1 | Flat terrain, low velocity | 20k | Standing + basic balance |
| 2 | Flat terrain, full velocity | 30k | Walking stability |
| 3 | Rough terrain | 30k | Robustness |
| 4 | Full randomization | 20k | Sim-to-real transfer |

---

## File Structure

```
source/digit_locomotion/digit_locomotion/
├── assets/
│   └── digit.py              # Robot configuration (actuators, init pose)
├── tasks/locomotion/
│   └── velocity_env_cfg.py   # Environment config (rewards, observations, terminations)
└── agents/
    └── rsl_rl_cfg.py         # PPO hyperparameters
```

---

## Key Findings

1. **Balance is the primary challenge** - Robot falls due to `bad_orientation` termination
2. **Episode length is a good metric** - Increased 4.8x during training
3. **Velocity tracking improved significantly** - 14x improvement in tracking reward
4. **More training needed** - 15k iterations shows learning but not convergence
5. **Arm control adds complexity** - 50 DOF is high; consider fixing arms initially

---

## Commands Reference

### Training (Windows PowerShell)
```powershell
cd C:\IsaacLab
$env:CONDA_PREFIX = ""
.\isaaclab.bat -p c:\path\to\scripts\train.py --task Digit-Velocity-Flat-v0 --num_envs 4096 --headless
```

### Evaluation (Windows PowerShell)
```powershell
cd C:\IsaacLab
$env:CONDA_PREFIX = ""
.\isaaclab.bat -p c:\path\to\scripts\play.py --checkpoint C:\IsaacLab\logs\digit_flat\final_model.pt --num_envs 4
```

### Training (Linux)
```bash
./isaaclab.sh -p scripts/train.py --task Digit-Velocity-Flat-v0 --num_envs 4096 --headless
```

---

## Checkpoints Location

Trained models are saved to: `C:\IsaacLab\logs\digit_flat\`

- `final_model.pt` - Final trained model
- `model_*.pt` - Checkpoints every 500 iterations
- `events.out.tfevents.*` - TensorBoard logs

View training metrics:
```bash
tensorboard --logdir C:\IsaacLab\logs\digit_flat
```
