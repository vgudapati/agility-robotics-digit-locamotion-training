# Arm Swing Fix: Two Approaches

## Background

The Digit V4 robot consistently develops a "spear-arm" / T-pose during running training,
where both arms extend fully horizontal. This provides balance but looks unnatural.

### Root Cause Analysis

Multiple soft approaches were tried and failed:
- **Reward penalties** (shoulder_roll=-0.5, shoulder_pitch=-2.0): Only -0.12/episode, negligible
- **Per-joint action scaling** (0.1-0.2 for arms): Policy still finds T-pose optimal
- **ImplicitActuatorCfg stiffness=100**: PhysX drive maxForce caps limit actual stiffness
- **IdealPDActuatorCfg stiffness=200**: Arms track commands but policy CHOOSES T-pose
- **mechanical_power_penalty**: -0.05/episode, negligible vs tracking rewards

### Joint Name Verification

The USD joint names DO contain `_arm_` (e.g., `left_arm_shoulder_pitch`, `right_arm_elbow`).
The patterns `.*_arm_.*` and `(?!.*_arm_).*` are correct.

Total: 50 joints (36 leg/body + 14 arm including 6 wrist joints).
Arm indices: [1, 3, 5, 7, 9, 11, 16, 21, 23, 25, 29, 33, 41, 49]

Note: The custom digit.py had incorrect joint names (`left_shoulder_roll` instead of
`left_arm_shoulder_roll`). The USD is the source of truth.

### Gap Analysis vs Radosavovic et al. (2024)

The paper achieved natural arm swing through energy minimization alone. Key gaps:

| Gap | Paper | Our Setup |
|-----|-------|-----------|
| Energy terms | Primary driver of arm swing | -0.05/ep (negligible, 280x weaker than joint_acc_l2) |
| joint_acc_l2 | Balanced with other terms | Dominates at -14/ep (80% of all penalty) |
| Arm constraints | None — let swing emerge | shoulder penalties actively fight emergence |
| Observation delay | Randomized | Not implemented |
| Terrain variety | Rough planes, slopes | Flat plane only |
| Training scale | ~10 billion samples | ~1.2 billion (12%) |

---

## Approach 1: Lock Arms (Safe — Working Teacher)

**Goal**: Get a stable 5 m/s running teacher with arms held at default pose.

### Changes Required

1. **Lock arms in action space**:
   - `joint_names=["(?!.*_arm_).*"]` — exclude 14 arm joints from policy control (36 remaining)
   - IdealPDActuatorCfg (`.*_arm_.*`) with high stiffness holds arms at default position

3. **Remove arm reward terms**:
   - Delete `shoulder_roll_penalty` and `shoulder_pitch_penalty`
   - Keep `mechanical_power` (still useful for leg energy efficiency)

4. **Checkpoint transplant**:
   - Load best checkpoint (run_042 model_3350.pt or run_055 best)
   - Remove arm output neurons from actor output layer
   - Keep all hidden layers (leg gait knowledge preserved)
   - Save new checkpoint compatible with reduced action space

### Expected Outcome
- Arms stay at default pose (hanging down) via PD controller
- Policy focuses entirely on leg control
- Smaller action space → potentially faster convergence
- Usable teacher for student distillation

### Limitations
- No arm swing (arms are passive)
- Student distillation will also have no arm swing
- Need Approach 2 or unlocking later for natural arm behavior

---

## Approach 2: Natural Arm Swing Emergence (Experimental)

**Goal**: Reproduce the Radosavovic et al. result where arm swing emerges naturally
from energy minimization, with no explicit arm constraints.

### Changes Required

1. **Fix joint name patterns** (P0 bug — same as Approach 1)

2. **Keep arms in action space**:
   - `joint_names=[".*"]` — all joints including arms
   - `scale=0.25` uniform for all joints (no per-joint scaling)

3. **Rebalance rewards** (the critical changes):
   - **Reduce `joint_acc_l2`** weight from `-2.5e-7` to `-2.5e-8` (10x reduction)
     - Currently dominates at -14/ep; after reduction should be ~-1.4/ep
     - Frees up reward budget for energy and tracking terms
   - **Increase `mechanical_power`** weight from `-1e-5` to `-2e-4` (20x increase)
     - Should become ~-1.0/ep, significant enough to incentivize efficiency
     - This is THE driver of arm swing per the paper
   - **Delete `shoulder_roll_penalty`** — actively fights emergence
   - **Delete `shoulder_pitch_penalty`** — actively fights emergence
   - Consider increasing `joint_torques_l2` from `-1e-6` to `-1e-5` (additional energy signal)

4. **Proper arm actuator setup**:
   - IdealPDActuatorCfg with moderate stiffness (40-80, not 200)
   - Lower damping (2-4) to allow natural swing dynamics
   - Keep effort/velocity limits reasonable

5. **Train fresh from scratch**:
   - Cannot resume from checkpoint — old policy learned T-pose
   - Start at walking speed (0-1 m/s) and progress through curriculum
   - Arm swing should emerge early at walking speeds if energy terms are right

### Optional Enhancements (P3)
- Add observation delay randomization (1-3 timesteps)
- Add terrain variety (rough planes, gentle slopes)
- Increase training budget (more iterations per curriculum stage)

### Expected Outcome
- Arms develop contralateral swing (right arm forward when left leg lifts)
- Biomechanically efficient gait similar to human running
- Natural-looking motion suitable for student distillation

### Risks
- Arm swing may not emerge if energy terms aren't balanced correctly
- May require multiple tuning iterations to find the right reward weights
- Longer training time (fresh start through full curriculum)
- T-pose could still re-emerge at higher speeds if stability dominates

### Success Criteria
- `shoulder_pitch_penalty` (if monitored) shows periodic oscillation, not constant
- `mechanical_power` shows meaningful penalty (> -0.5/ep)
- Visual inspection shows arm swing coordinated with leg stride

---

## Recommended Execution

1. **Run Approach 1 first** — produces a working teacher in hours
2. **Run Approach 2 in parallel** — experimental, may need iteration
3. If Approach 2 succeeds, use that teacher instead
4. If Approach 2 fails, use Approach 1 teacher and consider unlocking arms
   in a later fine-tuning stage with tight constraints

---

## Training Commands

### Approach 1 (Locked Arms, from checkpoint)
```bash
./isaaclab.bat -p scripts/train_baseline.py \
  --task Digit-BaselineRunning-v0 \
  --num_envs 16384 --headless \
  --max_iterations 1500 --episode_length_s 40 \
  --resume --checkpoint <transplanted_checkpoint.pt> \
  --learning_rate 1e-7 --lr_cap 1e-6
```

### Approach 2 (Natural Swing, fresh start)
```bash
./isaaclab.bat -p scripts/train_baseline.py \
  --task Digit-Baseline-v0 \
  --num_envs 16384 --headless \
  --max_iterations 2000 --episode_length_s 20 \
  --learning_rate 1e-3 --lr_cap 1e-3
```
