"""Print all joint names from the Digit V4 robot to identify arm vs leg indices."""
import re
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([])
args_cli.headless = True
args_cli.num_envs = 1
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab_assets.robots.agility import DIGIT_V4_CFG

# Minimal scene with just the robot
scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
scene_cfg.robot = DIGIT_V4_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
scene_cfg.ground = AssetBaseCfg(
    prim_path="/World/defaultGroundPlane",
    spawn=sim_utils.GroundPlaneCfg(),
)

sim_cfg = sim_utils.SimulationCfg(dt=0.005)
sim = sim_utils.SimulationContext(sim_cfg)
sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

scene = InteractiveScene(scene_cfg)
sim.reset()
scene.reset()

robot = scene["robot"]
joint_names = robot.joint_names
print(f"\n{'='*60}")
print(f"Total joints: {len(joint_names)}")
print(f"{'='*60}")

arm_pattern = re.compile(r".*(shoulder|elbow|arm).*")
for i, name in enumerate(joint_names):
    is_arm = bool(arm_pattern.match(name))
    marker = " <-- ARM" if is_arm else ""
    print(f"  [{i:2d}] {name}{marker}")

arm_indices = [i for i, name in enumerate(joint_names) if arm_pattern.match(name)]
leg_indices = [i for i, name in enumerate(joint_names) if not arm_pattern.match(name)]
print(f"\nArm indices ({len(arm_indices)}): {arm_indices}")
print(f"Leg/body indices ({len(leg_indices)}): {leg_indices}")

# Also test the regex patterns we're using
arm_regex = re.compile(r".*_arm_.*")
arm_matches = [name for name in joint_names if arm_regex.fullmatch(name)]
print(f"\nPattern '.*_arm_.*' matches ({len(arm_matches)}): {arm_matches}")

noarm_regex = re.compile(r"(?!.*_arm_).*")
noarm_matches = [name for name in joint_names if noarm_regex.fullmatch(name)]
print(f"Pattern '(?!.*_arm_).*' matches ({len(noarm_matches)}): {noarm_matches}")

simulation_app.close()
