"""Verify digit_locomotion installation.

This script verifies that the digit_locomotion package is installed
and can be imported after Isaac Sim is properly initialized.
"""

import sys
import traceback

# First, check if the package is installed (without full import)
try:
    from importlib.util import find_spec
    spec = find_spec("digit_locomotion")
    if spec is None:
        print("FAILED: digit_locomotion package not found")
        sys.exit(1)
    print(f"Package location: {spec.origin}")
except Exception as e:
    print(f"FAILED: Could not locate package: {e}")
    sys.exit(1)

# Initialize Isaac Sim (required before importing modules that use pxr)
print("Initializing Isaac Sim...")
print("(This may take a moment on first run...)")
sys.stdout.flush()

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

print("Isaac Sim initialized successfully!")
sys.stdout.flush()

# Now we can import digit_locomotion
try:
    print("Importing digit_locomotion...")
    sys.stdout.flush()

    import digit_locomotion
    print("OK - digit_locomotion imported successfully!")
    sys.stdout.flush()

    # Verify key components
    print("Loading DIGIT_CFG...")
    sys.stdout.flush()

    from digit_locomotion.assets import DIGIT_CFG
    print(f"  - DIGIT_CFG loaded: {DIGIT_CFG.prim_path}")

    print("\n" + "=" * 50)
    print("Installation verified successfully!")
    print("=" * 50)
    sys.stdout.flush()

except Exception as e:
    print(f"\nFAILED: {e}")
    traceback.print_exc()
    sys.exit(1)
finally:
    print("\nClosing simulation...")
    sys.stdout.flush()
    simulation_app.close()
    print("Done.")
