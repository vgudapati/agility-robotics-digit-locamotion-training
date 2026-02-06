# Training Tuning Options for Digit Locomotion

This document outlines options for improving training performance when the robot has learned basic balance but needs further optimization.

## Current Best Performance (Phase 1: Walking)
- **Mean episode length**: ~730 steps (max 750 at 50Hz for 15s episodes)
- **Success rate (time_out)**: 94%
- **Fall rate (bad_orientation)**: 5.6%
- **Arm swing coordination**: 0.45 (positive, working well)
- **Status**: Walking policy trained successfully with shoulder-based arm swing

---

## Curriculum Training Phases

| Phase | Task ID | Velocity Range | Status |
|-------|---------|----------------|--------|
| 1 | `Digit-Walking-v0` | 0-2 m/s | ✅ Complete |
| 2 | `Digit-Jogging-v0` | 0-5 m/s | Next |
| 3 | `Digit-Running-v0` | 0-8 m/s | Pending |
| 4 | `Digit-FastRunning-v0` | 0-10 m/s | Pending |
| 5 | `Digit-Sprint-v0` | 0-13.5 m/s | Pending |

**Progression command:**
```bash
# Continue to next phase with checkpoint
.\isaaclab.bat -p scripts\train.py --task Digit-Jogging-v0 --num_envs 16384 --checkpoint <path_to_walking_model.pt> --headless
```

---

## Option 1: Reward Weight Tuning

Adjust reward weights to better balance training objectives.

### File: `velocity_env_cfg.py` → `RewardsRunningCfg`

| Reward | Current Weight | Suggested Change | Rationale |
|--------|----------------|------------------|-----------|
| `track_lin_vel_xy_exp` | 1.5 | 2.0 - 3.0 | Prioritize velocity tracking |
| `track_ang_vel_z_exp` | 0.75 | 1.0 - 1.5 | Better yaw tracking |
| `arm_swing` | 0.3 | 0.1 - 0.2 | Reduce if interfering with balance |
| `flat_orientation_l2` | -0.5 | -1.0 to -2.0 | Stricter posture control |
| `lin_vel_z_l2` | -2.0 | -1.0 | Less strict vertical velocity |

**When to use**: When robot walks but doesn't track velocity commands well, or has poor posture.

---

## Option 2: Training Duration

Increase training iterations - the policy may still be improving.

### File: `rsl_rl_cfg.py` → `DigitRunningPPORunnerCfg`

```python
max_iterations = 3000  # Was 1000, try 3000-5000
```

**When to use**: When reward is still increasing at end of training, or learning curve hasn't plateaued.

---

## Option 3: Learning Rate Schedule

Adjust learning dynamics for more stable or faster learning.

### File: `rsl_rl_cfg.py` → `DigitRunningPPORunnerCfg.algorithm`

| Parameter | Current | Conservative | Aggressive |
|-----------|---------|--------------|------------|
| `learning_rate` | 1.0e-3 | 5.0e-4 | 2.0e-3 |
| `desired_kl` | 0.01 | 0.008 | 0.015 |
| `entropy_coef` | 0.01 | 0.005 | 0.02 |

**Conservative**: More stable but slower learning
**Aggressive**: Faster but potentially unstable

**When to use**:
- Conservative: If training is unstable (reward oscillates)
- Aggressive: If learning is too slow

---

## Option 4: Network Architecture

Adjust neural network size for the task complexity.

### File: `rsl_rl_cfg.py` → `DigitRunningPPORunnerCfg.policy`

| Config | Hidden Dims | Use Case |
|--------|-------------|----------|
| Large (current) | [1024, 512, 256] | Complex dynamics, sprint |
| Medium | [512, 256, 128] | Walking/jogging |
| Small | [256, 128, 64] | Fast experiments |

```python
policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
    actor_hidden_dims=[512, 256, 128],   # Smaller for walking
    critic_hidden_dims=[512, 256, 128],
    activation="elu",
)
```

**When to use**: Smaller networks converge faster for simpler tasks (walking). Use larger for complex dynamics (sprint).

---

## Option 5: Episode Length

Give the robot more time to learn long-horizon behavior.

### File: `velocity_env_cfg.py` → Environment `__post_init__`

```python
def __post_init__(self):
    super().__post_init__()
    self.episode_length_s = 20.0  # Was 15.0, try 20-30
```

**When to use**: When robot needs to learn behaviors that span longer time periods.

---

## Option 6: Termination Tolerance

Allow robot to recover from larger errors during training.

### File: `velocity_env_cfg.py` → `TerminationsRunningCfg`

```python
bad_orientation = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": 0.7},  # Was 0.5 (~30°), now ~40°
)
```

| Angle (rad) | Degrees | Description |
|-------------|---------|-------------|
| 0.5 | ~29° | Conservative (current) |
| 0.6 | ~34° | Moderate |
| 0.7 | ~40° | Permissive |
| 0.8 | ~46° | Very permissive |

**When to use**: If robot is being terminated before it can learn recovery behaviors.

---

## Option 7: Domain Randomization

Adjust randomization for cleaner learning signal or robustness.

### File: `velocity_env_cfg.py` → `EventsCfg`

**Disable for cleaner learning** (initial training):
```python
# Comment out or set to None in environment config
events: EventsCfg = None  # No randomization
```

**Enable for robustness** (after basic gait learned):
```python
events: EventsRoughCfg = EventsRoughCfg()  # Full randomization
```

**When to use**:
- Disable: When robot can't learn basic gait
- Enable: When basic gait is stable and you want robustness

---

## Option 8: Standing Environment Ratio

Increase practice on standing still for better balance.

### File: `velocity_env_cfg.py` → Command configs

```python
base_velocity = mdp.UniformVelocityCommandCfg(
    rel_standing_envs=0.1,  # Was 0.02 (2%), try 10%
    ...
)
```

| Ratio | Description |
|-------|-------------|
| 0.02 | Minimal standing (current) |
| 0.05 | Some standing practice |
| 0.10 | Significant standing practice |
| 0.20 | Heavy balance focus |

**When to use**: When robot has trouble maintaining balance at low speeds or when standing.

---

## Option 9: Stability Improvements

If the robot walks but appears wobbly or unstable, try these adjustments:

### 9a: Stricter Orientation Penalty

**File:** `velocity_env_cfg.py` → `RewardsRunningCfg`

```python
flat_orientation_l2 = RewardTermCfg(
    func=mdp.flat_orientation_l2,
    weight=-1.0,  # Was -0.5, stricter posture control
)
```

### 9b: More Standing Practice

**File:** `velocity_env_cfg.py` → Command configs

```python
base_velocity = mdp.UniformVelocityCommandCfg(
    rel_standing_envs=0.05,  # Was 0.02, more balance practice
    ...
)
```

### 9c: Continue Training

Simply train for more iterations:
```bash
.\isaaclab.bat -p scripts\train.py --task Digit-Walking-v0 --num_envs 16384 --checkpoint <model.pt> --max_iterations 500 --headless
```

**When to use**: When robot completes episodes but looks unstable or wobbly during movement.

---

## Option 10: Elbow Bend for Jogging/Running (Future)

For more natural arm movement at higher speeds, add elbow coordination.

**File:** `rewards.py` - Add new reward function:

```python
def elbow_bend_while_moving(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_bend: float = 0.5,  # ~30 degrees
) -> torch.Tensor:
    """Reward bent elbows during locomotion (like human running).

    Humans bend their elbows at ~90° when running for efficient arm swing.

    Args:
        env: The environment instance.
        command_name: Name of the velocity command.
        target_bend: Target elbow bend angle in radians.

    Returns:
        Negative distance from target bend (to be used with positive weight).
    """
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos

    # Elbow joints for Digit V4
    # left_arm_elbow (index 16), right_arm_elbow (index 21)
    left_elbow = joint_pos[:, 16]
    right_elbow = joint_pos[:, 21]

    # Reward being close to target bend angle
    bend_error = torch.abs(left_elbow - target_bend) + torch.abs(right_elbow - target_bend)

    # Only when moving
    cmd_vel = env.command_manager.get_command(command_name)
    moving = torch.norm(cmd_vel[:, :2], dim=1) > 0.1

    return -bend_error * moving.float()
```

**File:** `velocity_env_cfg.py` → Add to `RewardsRunningCfg`:

```python
elbow_bend = RewardTermCfg(
    func=custom_mdp.elbow_bend_while_moving,
    weight=0.1,  # Small weight initially
    params={"command_name": "base_velocity", "target_bend": 0.5},
)
```

**When to use**: Phase 2 (Jogging) and beyond for more natural arm movement at higher speeds.

---

## Recommended Tuning Order

1. **First**: Try longer training (Option 2) - easiest change
2. **Second**: Adjust termination tolerance (Option 6) - lets robot explore more
3. **Third**: Tune reward weights (Option 1) - if specific behaviors need improvement
4. **Fourth**: Adjust learning rate (Option 3) - if training is unstable
5. **Fifth**: Try smaller network (Option 4) - if convergence is slow
6. **Last**: Domain randomization (Option 7) - after basic gait works

---

## Quick Reference Commands

```powershell
# === TRAINING (Windows PowerShell) ===

# Phase 1: Walking (0-2 m/s)
.\isaaclab.bat -p scripts\train.py --task Digit-Walking-v0 --num_envs 16384 --headless

# Phase 2: Jogging (0-5 m/s) - from walking checkpoint
.\isaaclab.bat -p scripts\train.py --task Digit-Jogging-v0 --num_envs 16384 --checkpoint <walking_model.pt> --headless

# Phase 3: Running (0-8 m/s) - from jogging checkpoint
.\isaaclab.bat -p scripts\train.py --task Digit-Running-v0 --num_envs 16384 --checkpoint <jogging_model.pt> --headless

# Phase 4: Fast Running (0-10 m/s) - from running checkpoint
.\isaaclab.bat -p scripts\train.py --task Digit-FastRunning-v0 --num_envs 16384 --checkpoint <running_model.pt> --headless

# Phase 5: Sprint (0-13.5 m/s / 30 mph) - from fast running checkpoint
.\isaaclab.bat -p scripts\train.py --task Digit-Sprint-v0 --num_envs 16384 --checkpoint <fast_running_model.pt> --headless

# Continue training same phase (--resume keeps iteration count)
.\isaaclab.bat -p scripts\train.py --task Digit-Walking-v0 --resume --checkpoint <model.pt> --headless

# === EVALUATION ===

# Play with random velocity commands
.\isaaclab.bat -p scripts\play.py --task Digit-Walking-v0 --checkpoint <model.pt> --num_envs 4

# Play with fixed velocity
.\isaaclab.bat -p scripts\play.py --task Digit-Walking-v0 --checkpoint <model.pt> --num_envs 1 --vel_x 1.0

# Play headless (faster, for metrics only)
.\isaaclab.bat -p scripts\play.py --task Digit-Walking-v0 --checkpoint <model.pt> --num_envs 64 --headless
```

---

## Monitoring Progress

Watch these metrics during training:

| Metric | Good Sign | Problem Sign |
|--------|-----------|--------------|
| Mean reward | Increasing | Decreasing or flat |
| Mean episode length | Approaching max (750) | Stuck below 100 |
| track_lin_vel_xy_exp | > 0.5 | < 0.1 |
| bad_orientation termination | < 50% | > 90% |
| time_out termination | > 50% | 0% |

---

## Option 11: Fixing Wobbly/Jittery Motion

If the robot has learned to walk but appears wobbly, shaky, or has jerky movements, try these adjustments.

### 11a: Increase Action Smoothness Penalty (Recommended First)

Penalizes rapid changes in actions, forcing smoother motion.

**File:** `baseline_env_cfg.py` → `BaselineRewardsCfg`

| Parameter | Current | Suggested | Effect |
|-----------|---------|-----------|--------|
| `action_rate_l2` | -0.01 | -0.05 to -0.1 | Stronger smoothing |

```python
action_rate_l2 = RewardTermCfg(
    func=mdp.action_rate_l2,
    weight=-0.05,  # Was -0.01, stronger smoothing
)
```

**When to use**: Robot makes jerky, rapid movements between timesteps.

---

### 11b: Increase Angular Velocity Penalty

Penalizes roll/pitch oscillation (side-to-side wobble).

**File:** `baseline_env_cfg.py` → `BaselineRewardsCfg`

| Parameter | Current | Suggested | Effect |
|-----------|---------|-----------|--------|
| `ang_vel_xy_l2` | -0.05 | -0.1 to -0.2 | Reduces wobble |

```python
ang_vel_xy_l2 = RewardTermCfg(
    func=mdp.ang_vel_xy_l2,
    weight=-0.15,  # Was -0.05, stronger anti-wobble
)
```

**When to use**: Robot rocks side-to-side or has unstable upper body.

---

### 11c: Reduce Action Scale

Smaller actions = finer control = less overshoot.

**File:** `baseline_env_cfg.py` → `BaselineActionsCfg`

| Parameter | Current | Suggested | Effect |
|-----------|---------|-----------|--------|
| `scale` | 0.25 | 0.15 to 0.1 | Finer control |

```python
joint_pos = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=[".*"],
    scale=0.15,  # Was 0.25, smaller actions for finer control
    use_default_offset=True,
)
```

**When to use**: Robot overshoots target positions, causing oscillation.

---

### 11d: Add Joint Velocity Penalty (Careful!)

Penalizes high joint speeds. Can cause collapse if too strong.

**File:** `baseline_env_cfg.py` → `BaselineRewardsCfg`

```python
# Add this reward term (be careful with weight!)
joint_vel_l2 = RewardTermCfg(
    func=mdp.joint_vel_l2,
    weight=-1e-5,  # Start very small, increase if stable
)
```

| Weight | Effect |
|--------|--------|
| -1e-6 | Very gentle, minimal effect |
| -1e-5 | Noticeable smoothing |
| -1e-4 | Strong smoothing (may cause slowdown) |
| -1e-3 | **Too aggressive - will cause collapse** |

**When to use**: Robot moves joints too fast. Start with -1e-6 and increase gradually.

---

### 11e: Increase Energy Penalty

Higher energy penalty encourages smoother, more efficient movements.

**File:** `baseline_env_cfg.py` → `BaselineRewardsCfg`

| Parameter | Current | Suggested | Effect |
|-----------|---------|-----------|--------|
| `joint_torques_l2` | -1e-6 | -5e-6 to -1e-5 | More energy-efficient motion |

```python
joint_torques_l2 = RewardTermCfg(
    func=mdp.joint_torques_l2,
    weight=-5e-6,  # Was -1e-6, encourages smoother motion
)
```

**When to use**: Robot uses excessive torque, leading to jerky movements.

---

### 11f: Reduce Observation Noise (Testing Only)

Lower noise can help during development but reduces sim-to-real robustness.

**File:** `baseline_env_cfg.py` → `BaselineObservationsCfg.PolicyCfg`

| Observation | Current std | Testing std | Effect |
|-------------|-------------|-------------|--------|
| `base_lin_vel` | 0.05 | 0.02 | Less velocity noise |
| `base_ang_vel` | 0.1 | 0.05 | Less angular noise |
| `joint_vel` | 0.5 | 0.2 | Less joint velocity noise |

```python
base_lin_vel = ObservationTermCfg(
    func=mdp.base_lin_vel,
    noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.02),  # Was 0.05
)
```

**When to use**: Testing to see if noise is causing jitter. Restore for sim-to-real.

---

### 11g: Increase Vertical Bounce Penalty

Penalizes up/down bouncing motion.

**File:** `baseline_env_cfg.py` → `BaselineRewardsCfg`

| Parameter | Current | Suggested | Effect |
|-----------|---------|-----------|--------|
| `lin_vel_z_l2` | -2.0 | -3.0 to -4.0 | Less bouncing |

```python
lin_vel_z_l2 = RewardTermCfg(
    func=mdp.lin_vel_z_l2,
    weight=-3.0,  # Was -2.0, stronger anti-bounce
)
```

**When to use**: Robot bounces up and down while walking.

---

### Recommended Wobble Fix Order

1. **First**: Increase `action_rate_l2` to -0.05 (most direct fix)
2. **Second**: Increase `ang_vel_xy_l2` to -0.15 (reduces body wobble)
3. **Third**: Reduce `scale` to 0.15 (if still overshooting)
4. **Fourth**: Add `joint_vel_l2` at -1e-5 (if joints move too fast)
5. **Fifth**: Increase `joint_torques_l2` to -5e-6 (for energy efficiency)

### Example: Anti-Wobble Baseline Config

```python
@configclass
class BaselineRewardsSmoothCfg(BaselineRewardsCfg):
    """Smoother version of baseline rewards for less wobbly motion."""

    # Stronger action smoothing
    action_rate_l2 = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.05,  # Was -0.01
    )

    # Stronger anti-wobble
    ang_vel_xy_l2 = RewardTermCfg(
        func=mdp.ang_vel_xy_l2,
        weight=-0.15,  # Was -0.05
    )

    # Gentle joint velocity penalty
    joint_vel_l2 = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-1e-5,
    )
```
