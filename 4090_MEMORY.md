# Project Memory - Digit Locomotion Training

## Key Files
- `scripts/train_baseline.py` - Main training script (LR warmup, fixed scheduler, iteration fixes)
- `source/digit_locomotion/digit_locomotion/agents/baseline_cfg.py` - PPO configs
- `source/digit_locomotion/digit_locomotion/tasks/locomotion/baseline_env_cfg.py` - Env configs
- `source/digit_locomotion/digit_locomotion/tasks/locomotion/__init__.py` - Task registration
- `source/digit_locomotion/digit_locomotion/tasks/locomotion/mdp/rewards.py` - Custom rewards

## Critical Bugs Found & Fixed
- **RSL-RL adaptive scheduler**: Runs 40x/iter (8 mini-batches × 5 epochs), overrides external LR cap. Fix: `desired_kl=None`, `schedule="fixed"`.
- **RSL-RL `learn()` iteration counter**: Sets `current_learning_iteration = it` not `it+1`. Fix: increment manually after each `learn(1)`.
- **RSL-RL `num_learning_iterations` is RELATIVE**: `learn(N)` runs N iters from current position. Was passing cumulative values.
- **LR cap only on optimizer was insufficient**: Must also sync `runner.alg.learning_rate`.
- **weight=0.0 reward terms are still ACTIVE** — IsaacLab RewardManager computes them. Always DELETE unused terms.

## RSL-RL Integration Pattern
- Call `learn(1)` in a loop for per-iteration control
- Disable adaptive scheduler: `runner.alg.desired_kl = None; runner.alg.schedule = "fixed"`
- Manual LR warmup from `--learning_rate` to `--lr_cap` over 50 iters
- Increment `runner.current_learning_iteration += 1` after each `learn(1)`
- Adam optimizer state is calibrated to previous LR — large jumps cause collapse.

## Learning Rate Rules
- **LR=1e-3 causes peak-then-degrade** at high-speed stages (4+ m/s)
- **lr_cap=1e-4**: Too noisy/volatile for sustained training (run_089)
- **lr_cap=5e-5**: Sweet spot — steady progress, low variance (run_090)
- **lr_cap=1e-5**: Too conservative — 11 hours with minimal progress (run_088)
- **Rule: Use `--learning_rate 0.00001 --lr_cap 0.00005` for fine-tuning at high speed**
- LR=1e-3 worked fine for Walking through Jogging (0-3 m/s)

## Lessons Learned
- `mdp.is_alive` doesn't exist in isaaclab_tasks; use custom `is_alive()` in rewards.py
- Checkpoints save to `C:\IsaacLab\logs\<experiment_name>\run_XXX\`
- **Original teacher config works** (48 steps, 5 epochs, 8 mini-batches, [512,512,256,128])
- **Seed=42 + cudnn.deterministic hurt training** — use seed=None
- Default Digit pose = arms DOWN (shoulder_pitch=0.3, shoulder_roll=0.0, elbow=-0.3)
- Digit V4 spawn height = 1.05m
- **joint_acc_l2 dominates reward** at -14/episode (weight=-2.5e-7). Consider reducing.

## Arm T-pose / Spear-Arm Problem — UNSOLVED by soft methods
- **Reward penalties FAILED** — shoulder_pitch=-2.0 only -0.12/episode, negligible vs tracking reward
- **Action scaling (0.1-0.2) FAILED** — policy still finds T-pose optimal
- **ImplicitActuatorCfg stiffness=100 FAILED** — PhysX drive maxForce caps
- **IdealPDActuatorCfg FAILED** — arms track commands but policy CHOOSES T-pose for balance
- **mechanical_power_penalty FAILED** — -0.05/episode, negligible
- **SOLUTION: Lock arms** — exclude arm joints from action space (`joint_names=["(?!.*_arm_).*"]`). IdealPDActuatorCfg holds arms at default pose. Removed shoulder penalties (no longer needed).
- Current: 16 reward terms, arms locked, legs-only action space

## Digit V4 Joint Names (from USD via print_joint_names.py)
- **50 total joints**: 36 leg/body + 14 arm (including 6 wrist)
- Joint names use `left_arm_*` / `right_arm_*` prefix (NOT `left_shoulder_*`)
- Pattern `.*_arm_.*` correctly matches all 14 arm joints
- Pattern `(?!.*_arm_).*` correctly matches 36 leg/body joints
- Arm indices: [1, 3, 5, 7, 9, 11, 16, 21, 23, 25, 29, 33, 41, 49]
- **Custom digit.py has WRONG names** — use USD as source of truth
- Checkpoint action space = 50 (all joints). Locked arms = 36 actions.

## Checkpoint Transplant (scripts/transplant_checkpoint.py)
- Removes 14 arm outputs from actor final layer (50→36 actions)
- Trims action columns from actor/critic input layers (162→148 obs)
- Obs layout: `[commands(3), lin_vel(3), ang_vel(3), gravity(3), joint_pos(50), joint_vel(50), actions(50→36)]`
- `--digit-v4` flag uses hardcoded verified joint names (no IsaacSim needed)
- `--auto-detect` hangs for hours — avoid; use `--digit-v4` instead
- Optimizer state RESET (Adam state invalid for changed architecture)

## Curriculum Training Progress
- Walking→FastWalking→SlowJogging→Jogging→ModerateRunning→Running→FastRunning
- **Run_055** (0-5 m/s, lr_cap=1e-6): ep_length peaked 1955, best ever. Arms still T-pose.
- **Run_059** (transplant, lr_cap=1e-6): Collapsed 1966→840 in 87 iters. LR too aggressive.
- **Run_060** (transplant, lr_cap=5e-7): Peaked 1972, collapsed to 20 by iter 210. Same root cause.
- **TRANSPLANT APPROACH FAILED** — hidden layers co-adapted with arm balance. Any LR unravels it.
- Branch: `stage2_locomation_training_for_running_approach1`

## Transplant Failure Analysis
- Hidden layers learned leg gait patterns that DEPEND on arm T-pose for balance
- Removing arm outputs doesn't remove the arm-dependent balance strategy in hidden weights
- Any optimization pressure (even 5e-7) eventually degrades arm-dependent balance without replacement
- Solution: train from scratch with locked arms — no co-adaptation to corrupt

## From-Scratch Training (Locked Arms)
- **Reward rebalancing CRITICAL**: is_alive=10, termination=-50, joint_acc=-5e-8, action_rate=-0.005, lin_vel_z=-0.5
- Old weights caused "suicide" local minimum (policy learned to die to avoid penalties)
- -200 termination caused value function divergence (3.7B value loss)
- **Run_063** (0-1 m/s, from scratch): ep_length maxed 1000, track=1.30. Walking converged in ~60 iters.
- **Run_067** (0-2 m/s, resumed from run_063): ep_length ~3740/4000, track=1.16. Stable overnight.

## Training CLI Notes
- **--hide_ui** flag causes errors — do NOT use
- **--headless** alone is sufficient for headless training
- Added CLI args: `--entropy_coef`, `--init_noise_std`

## Entropy/Noise Tuning
- **entropy_coef=0.01 + init_noise_std=1.0**: noise exploded 1.0→2.5, collapsed by iter 140
- **entropy_coef=0.005 + init_noise_std=0.5**: noise stable, converged walking in 170 iters
- **entropy_coef=0.005**: Good for curriculum learning (exploration phase). Noise climbs 0.5→2.16 and stays flat.
- **entropy_coef=0.002**: DANGEROUS early — caused value function collapse (noise went negative → crash)
- **entropy_coef=0.001**: BREAKTHROUGH for fine-tuning. After policy learned full speed range (0-8 m/s):
  - noise_std dropped 2.16→1.47 in ~800 iters (was stuck flat at 2.16 with 0.005)
  - reward: 140→278, bad_orient: 19%→7%, time_out: 81%→93%, action_rate: -1.1→-0.68
  - No brittleness observed — bad_orient kept IMPROVING as noise dropped
- **RULE: entropy_coef=0.005 for curriculum/exploration, then 0.001 for fine-tuning once skill is learned**

## RSL-RL Noise Std Crash & Fix
- RSL-RL stores noise as raw `nn.Parameter` (not log_std) — optimizer can push it negative
- `Normal(mean, std)` crashes with `RuntimeError: normal expects all elements of std >= 0.0`
- **Fix**: Monkey-patch `optimizer.step()` to clamp `std >= 0.01` after every step (runs inside `learn()`)
- Old fix (clamp after `learn()` returns) was too late — crash happens between mini-batches inside `update()`
- Fix is in `train_baseline.py` lines 636-652

## Curriculum Progress (Locked Arms, From Scratch)
- run_072: Walking 0-1 m/s (entropy=0.005, noise=0.5) — converged iter 170, 93% timeout
- run_073: FastWalking 0-2 m/s (from run_072 ckpt 400) — converged ~iter 515, 95% timeout
- run_074: SlowJogging 0-2.5 m/s (from run_073) — converged ~iter 750, 97% timeout
- run_075: Jogging 0-3 m/s (from run_074 ckpt 750) — plateaued iter 935+, 93% timeout, 7% bad_orient
- run_076: ModerateRunning 0-4 m/s (entropy=0.002) — COLLAPSED (noise went negative, reward -10.6B)
- run_080: ModerateRunning 0-4 m/s (entropy=0.002, ckpt run_076/1070) — COLLAPSED same pattern
- **run_081**: ModerateRunning 0-4 m/s (entropy=0.005, ckpt run_075/1000) — peaked 88%/12% at iter ~1230, degraded to 70%/30%. Best ckpt: model_1250.pt
- **run_082**: Running 0-5 m/s (LR=1e-3, ckpt run_081/1250) — peaked 30% timeout at iter ~1440, degraded to 18%. **LR too high for this speed range.**
- **run_083**: Running 0-5 m/s (LR=1e-5→1e-4, ckpt run_082/1440) — 81.4% timeout at iter 1672, still climbing. **Lower LR dramatically better.**
- Next: Continue run_083 until plateau, then FastRunning (0-8 m/s) with same LR settings

## Research Resources
See [ppo_convergence_research.md](ppo_convergence_research.md) for full reference list.
