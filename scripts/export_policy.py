#!/usr/bin/env python3
"""Export trained policy to ONNX format for deployment.

This script converts a trained PyTorch policy to ONNX format,
which can be used for deployment on real hardware or other
inference frameworks.

Usage:
    python scripts/export_policy.py \
        --checkpoint logs/digit_flat/model_15000.pt \
        --output policies/digit_walking.onnx \
        --obs_dim 102

Note: This script does not require Isaac Sim to run.
"""

from __future__ import annotations

import argparse
import os

import torch
import onnx


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export trained policy to ONNX format"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained policy checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="policy.onnx",
        help="Output path for the ONNX model",
    )
    parser.add_argument(
        "--obs_dim",
        type=int,
        default=102,
        help="Observation dimension (must match training)",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=11,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the exported model",
    )

    return parser.parse_args()


def export_jit_to_onnx(
    checkpoint_path: str,
    output_path: str,
    obs_dim: int,
    opset_version: int = 11,
) -> None:
    """Export a JIT-compiled policy to ONNX.

    Args:
        checkpoint_path: Path to the JIT model checkpoint.
        output_path: Path to save the ONNX model.
        obs_dim: Dimension of the observation space.
        opset_version: ONNX opset version to use.
    """
    print(f"Loading JIT model from: {checkpoint_path}")

    # Load the JIT model
    policy = torch.jit.load(checkpoint_path, map_location="cpu")
    policy.eval()

    # Create dummy input
    dummy_input = torch.randn(1, obs_dim)

    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        policy,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
    )

    print("Export complete!")


def verify_onnx_model(onnx_path: str, obs_dim: int) -> None:
    """Verify the exported ONNX model.

    Args:
        onnx_path: Path to the ONNX model.
        obs_dim: Dimension of the observation space.
    """
    print(f"\nVerifying ONNX model: {onnx_path}")

    # Load and check the model
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("ONNX model is valid!")

    # Print model info
    print(f"\nModel inputs:")
    for input_tensor in model.graph.input:
        print(f"  - {input_tensor.name}: {[d.dim_value for d in input_tensor.type.tensor_type.shape.dim]}")

    print(f"\nModel outputs:")
    for output_tensor in model.graph.output:
        print(f"  - {output_tensor.name}: {[d.dim_value for d in output_tensor.type.tensor_type.shape.dim]}")

    # Test inference with ONNX Runtime
    try:
        import onnxruntime as ort

        print("\nTesting inference with ONNX Runtime...")
        session = ort.InferenceSession(onnx_path)

        # Create test input
        import numpy as np
        test_input = np.random.randn(1, obs_dim).astype(np.float32)

        # Run inference
        outputs = session.run(None, {"observation": test_input})
        print(f"Inference successful!")
        print(f"Output shape: {outputs[0].shape}")

    except ImportError:
        print("\nNote: Install onnxruntime to test inference")
        print("  pip install onnxruntime")


def main():
    """Main export function."""
    args = parse_args()

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Create output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Export the model
    export_jit_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        obs_dim=args.obs_dim,
        opset_version=args.opset_version,
    )

    # Verify if requested
    if args.verify:
        verify_onnx_model(args.output, args.obs_dim)

    print(f"\nExported policy saved to: {args.output}")


if __name__ == "__main__":
    main()
