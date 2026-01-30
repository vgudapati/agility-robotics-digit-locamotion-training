"""Test pxr import."""
try:
    from pxr import Usd, UsdGeom
    print("pxr works!")
except ImportError as e:
    print(f"pxr import failed: {e}")

    # Check what paths are available
    import sys
    print("\nPython path:")
    for p in sys.path:
        print(f"  {p}")
