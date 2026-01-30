"""Check NVIDIA Nucleus connection and Digit asset availability."""
import sys

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, check_file_path

digit_path = f"{ISAAC_NUCLEUS_DIR}/Robots/Agility/Digit/digit_v4.usd"
print(f"\nNucleus directory: {ISAAC_NUCLEUS_DIR}")
print(f"Digit USD path: {digit_path}")

status = check_file_path(digit_path)
if status == 0:
    print("\nStatus: FILE NOT FOUND")
    print("The Digit robot USD is not available locally or on the Nucleus server.")
    print("\nTo fix this, you need to:")
    print("1. Sign in to NVIDIA Omniverse (omniverse://launcher)")
    print("2. Connect to the Nucleus server through the Omniverse Launcher")
    print("3. Or download the Digit asset manually from NVIDIA's asset library")
elif status == 1:
    print("\nStatus: FOUND LOCALLY")
    print("The Digit robot USD is available locally.")
elif status == 2:
    print("\nStatus: FOUND ON NUCLEUS SERVER")
    print("The Digit robot USD is available on the Nucleus server.")
    print("It will be downloaded when you run training.")

simulation_app.close()
