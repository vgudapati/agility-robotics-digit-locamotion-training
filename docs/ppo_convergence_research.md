# Digit Locomotion Training - Research & Lessons Learned

## Project Key Files
- `scripts/train_baseline.py` - Main training script (seed fixing, LR capping, adaptive LR)
- `source/digit_locomotion/digit_locomotion/agents/baseline_cfg.py` - PPO configs
- `source/digit_locomotion/digit_locomotion/tasks/locomotion/baseline_env_cfg.py` - Env configs
- `source/digit_locomotion/digit_locomotion/tasks/locomotion/__init__.py` - Task registration
- `source/digit_locomotion/digit_locomotion/tasks/locomotion/mdp/rewards.py` - Custom rewards

## Lessons Learned

### LR Spike / Training Collapse
- The adaptive LR scheduler (`schedule="adaptive"`) increases LR when KL divergence is low
- In early training KL is naturally low (random policy), causing LR to spike to 0.4-0.5
- **Fix**: Lower initial LR from 1e-3 to 3e-4, add hard LR cap at 5e-3 (`--lr_cap`)

### Missing `is_alive` Reward
- `mdp.is_alive` doesn't exist in `isaaclab_tasks` — it silently fails
- **Fix**: Custom `is_alive()` function in `rewards.py` returning `torch.ones(env.num_envs)`
- Updated config to use `custom_mdp.is_alive` instead of `mdp.is_alive`

### Checkpoints
- Checkpoints save to `C:\IsaacLab\logs\<experiment_name>\run_XXX\`
- `model_best.pt` is saved by adaptive LR training when episode length improves
- `model_final.pt` is always saved at end of training

### Training Diagnostics
- 99.96% of terminations from `bad_orientation` = robot falling over
- `action_rate_l2` penalty was the largest negative reward contributor (~-0.08)
- Episode length is the most reliable early indicator of learning progress
- Mean action noise std drifting above 1.3 indicates too much exploration

### Seed Fixing
- Set seeds for: `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`
- Also set `torch.backends.cudnn.deterministic = True` and `benchmark = False`
- Default seed: 42

---

# PPO Convergence Research for Humanoid Locomotion

## Community-Standard PPO Hyperparameters

| Parameter | legged_gym | AnymalB (IsaacLab) | Unitree G1 | Our Original | Community Config |
|---|---|---|---|---|---|
| num_steps_per_env | 24 | 24 | 24 | 48 | 24 |
| num_learning_epochs | 5 | 5 | 5 | 5 | 8 |
| num_mini_batches | 4 | 4 | 4 | 8 | 4 |
| init_noise_std | 1.0 | 1.0 | 0.8 | 1.0 | 0.8 |
| gamma | 0.99 | 0.99 | 0.99 | 0.99 | 0.97 |
| Network | varies | [512,256,128] | [32]+LSTM | [512,512,256,128] | [512,256,128] |

## Key Insights
- `num_steps_per_env=24` is universal across all successful legged_gym/RSL-RL configs
- For humanoids with 29+ joints, `init_noise_std=0.8` prevents chaotic initial actions
- Reward design matters more than PPO hyperparameters for convergence
- Episode length is the most reliable early indicator of learning progress
- Use 128-256 envs for debugging, 4096-16384 for production
- Radosavovic et al. 2024 used ~10 billion samples total, trained in ~1 day

## Reference Sources

### 1. legged_gym default config
- URL: https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py
- Value: Gold standard defaults for RSL-RL locomotion (24 steps, 5 epochs, 4 mini-batches)

### 2. IsaacLab AnymalB RSL-RL PPO config
- URL: https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_b/agents/rsl_rl_ppo_cfg.py
- Value: Official NVIDIA reference confirming legged_gym defaults carry over to IsaacLab

### 3. Unitree G1 humanoid config
- URL: https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_config.py
- Value: Most comparable to Digit - high-DOF humanoid (29 joints), uses init_noise_std=0.8

### 4. RSL-RL NaN std issue #33
- URL: https://github.com/leggedrobotics/rsl_rl/issues/33
- Value: Documents failure mode where high init_noise_std causes NaN action std

### 5. Bipedal Walking Deep Dive (Hussein Lezzaik)
- URL: https://husseinlezzaik.github.io/2025/05/01/bipedal/
- Value: Practical guide showing reward design matters more than PPO params

### 6. Radosavovic et al. 2024 - Real-world humanoid locomotion with RL
- URL: https://arxiv.org/abs/2303.03381 (arXiv), https://www.science.org/doi/10.1126/scirobotics.adi9579 (paper)
- Value: The paper this training pipeline is based on (architecture, training approach)

### 7. Humanoid-Gym (roboterax)
- URL: https://github.com/roboterax/humanoid-gym
- Value: Alternative humanoid RL framework for PPO parameter reference

### 8. NVIDIA Spot Quadruped Blog
- URL: https://developer.nvidia.com/blog/closing-the-sim-to-real-gap-training-spot-quadruped-locomotion-with-nvidia-isaac-lab/
- Value: NVIDIA's sim-to-real recommendations, confirms curriculum + domain randomization importance

### 9. robot_lab (fan-ziqi) - RL extension for IsaacLab
- URL: https://github.com/fan-ziqi/robot_lab
- Value: Extended IsaacLab framework with additional locomotion examples

### 10. Booster Gym paper
- URL: https://arxiv.org/html/2506.15132v1
- Value: Recent research on accelerating humanoid locomotion training
