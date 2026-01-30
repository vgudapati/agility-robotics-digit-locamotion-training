"""Test isaacsim import."""
try:
    import isaacsim
    print(f"isaacsim found at: {isaacsim.__file__}")
    print(f"isaacsim version: {getattr(isaacsim, '__version__', 'unknown')}")

    # Check what's available
    print(f"\nisaacsim attributes: {dir(isaacsim)}")
except ImportError as e:
    print(f"isaacsim import failed: {e}")
