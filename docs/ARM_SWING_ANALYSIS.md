# Arm Swing Behavior Analysis: Berkeley Paper vs DIGIT Implementation

## Overview

This document analyzes why emergent arm swing behavior observed in the Radosavovic et al. (2024) Science Robotics paper does not naturally occur when training on the Agility Robotics DIGIT robot, and documents our solution.

## The Berkeley Paper Approach

**Paper**: "Real-world humanoid locomotion with reinforcement learning" (Radosavovic et al., Science Robotics 2024)

### Key Findings from the Paper

1. **No Explicit Arm Constraints**: The researchers did NOT impose explicit constraints on arm swing motion in the reward function or use reference trajectories for arms.

2. **Emergent Behavior**: Arm swing emerged naturally through energy minimization terms in the reward function.

3. **Quote from Paper**:
   > "When training our neural network controller, we do not impose explicit constraints on the arm swing motion in the reward function or use any reference trajectories to guide the arm motions."

4. **Energy Hypothesis**:
   > "Our reward function includes energy minimization terms which might suggest that the emergent arm swing motion might lead to energy savings in humanoid locomotion."

5. **Contralateral Coordination**: The learned policy exhibited human-like coordination - when left leg lifted, right arm swung forward.

### The Robot Used

- **Custom Berkeley Humanoid** - NOT the DIGIT robot
- The robot's physical design likely had arms that naturally hang down when unpowered
- Default arm position was probably near the body (relaxed pose)

## The DIGIT Robot Reality

### Key Differences

| Aspect | Berkeley Humanoid | DIGIT v4 |
|--------|-------------------|----------|
| Default arm pose | Likely hanging/relaxed | T-pose (arms horizontal) |
| Arm behavior when unpowered | Falls to sides | Holds position (damping) |
| USD/URDF default | Unknown | T-pose configuration |
| Action reference | Unknown | Relative to T-pose |

### Why Emergent Arm Swing Fails on DIGIT

1. **T-Pose Default**: The DIGIT USD file's default pose has arms extended horizontally (T-pose).

2. **Action Configuration**: Our configuration uses:
   ```python
   joint_pos = mdp.JointPositionActionCfg(
       ...
       use_default_offset=True,  # Actions relative to default pose
   )
   ```
   This means action=0 keeps arms in T-pose, and the policy must actively output non-zero values to move arms down.

3. **Joint Damping**: DIGIT's arm joints may have high damping, allowing them to hold positions with minimal torque. This reduces the energy penalty for keeping arms extended.

4. **Mass Distribution**: DIGIT's arms may be well-balanced at the shoulder, requiring less torque to hold horizontal than a poorly-balanced arm would.

## Agility Robotics' Own Approach

According to Agility Robotics (source: The Robot Report, 2024):

1. **Free-Space Control**: Agility uses free-space positions and orientations for arms, NOT joint-space parameters.

2. **Explicit Hand Tracking**: They use "a reward term that considers the translational and rotational error between the current hand pose and the target hand pose."

3. **Different Paradigm**: This is fundamentally different from the Berkeley paper's emergent approach - Agility explicitly controls arm positions.

4. **Training Infrastructure**: They train in NVIDIA Isaac Sim with an LSTM network (~1M parameters) for "decades of simulated time over three or four days."

## Our Solution

Since emergent arm swing does not naturally occur on DIGIT, we implemented explicit arm control:

### Arm-Specific Torque Penalty

```python
# In BaselineRewardsCfg

# General energy penalty (all joints)
joint_torques_l2 = RewardTermCfg(
    func=mdp.joint_torques_l2,
    weight=-1e-5,
)

# Arm-specific penalty (100x stronger for arms only)
arm_torques_l2 = RewardTermCfg(
    func=mdp.joint_torques_l2,
    weight=-1e-3,  # 100x stronger than general penalty
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_arm_.*"])},
)
```

### Rationale

- Arms extended horizontally require constant torque against gravity
- By heavily penalizing arm torques specifically, we encourage the policy to find low-energy arm positions
- This targets arms without affecting leg stability (which needs torque for locomotion)

### Alternative Approaches Considered

1. **Increase general energy penalty**: Failed - destabilized walking before affecting arms
2. **Change default arm position**: Would require modifying USD file
3. **Arm position deviation penalty**: More direct but less physically motivated

## Lessons Learned

1. **Paper results don't always transfer**: The Berkeley paper's emergent arm swing relied on their specific robot's physical properties.

2. **Default pose matters**: When using `use_default_offset=True`, the USD default pose significantly affects what the policy learns.

3. **Energy penalties aren't uniform**: Leg torques dominate total energy, so general energy penalties barely affect arms.

4. **Robot-specific solutions needed**: Each robot may need custom reward terms based on its mechanical properties.

## References

1. Radosavovic, I., et al. (2024). "Real-world humanoid locomotion with reinforcement learning." Science Robotics.
   - Paper: https://www.science.org/doi/10.1126/scirobotics.adi9579
   - arXiv: https://arxiv.org/abs/2303.03381

2. Agility Robotics Training Approach
   - https://www.therobotreport.com/agility-robotics-explains-train-whole-body-control-foundation-model/

3. Berkeley Hybrid Robotics Lab
   - https://hybrid-robotics.berkeley.edu/publications/ScienceRobotics2024_Learning_Humanoid_Locomotion.pdf

## Configuration Files

- Reward configuration: `source/digit_locomotion/digit_locomotion/tasks/locomotion/baseline_env_cfg.py`
- DIGIT robot definition: `C:/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/agility.py`
- Arm joint pattern: `".*_arm_.*"`
