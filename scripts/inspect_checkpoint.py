"""Inspect a checkpoint to determine model architecture and action space size."""
import sys
import torch

if len(sys.argv) < 2:
    print("Usage: python inspect_checkpoint.py <checkpoint_path>")
    sys.exit(1)

checkpoint_path = sys.argv[1]
print(f"Loading checkpoint: {checkpoint_path}")
loaded = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

print(f"\nCheckpoint keys: {list(loaded.keys())}")
print(f"Saved iteration: {loaded.get('iter', 'N/A')}")

model_state = loaded["model_state_dict"]
print(f"\n{'='*60}")
print("Model Architecture (all layers):")
print(f"{'='*60}")
for key, tensor in sorted(model_state.items()):
    print(f"  {key:50s} {str(tensor.shape):>20s}")

# Find actor output layer (last layer with 'actor' in name)
actor_layers = {k: v for k, v in model_state.items() if 'actor' in k.lower()}
print(f"\n{'='*60}")
print("Actor layers:")
print(f"{'='*60}")
for key, tensor in sorted(actor_layers.items()):
    print(f"  {key:50s} {str(tensor.shape):>20s}")

# The output layer should be the last one — its first dimension is num_actions
# For weight: shape is [out_features, in_features]
# For bias: shape is [out_features]
actor_biases = {k: v for k, v in actor_layers.items() if 'bias' in k}
if actor_biases:
    last_bias_key = sorted(actor_biases.keys())[-1]
    num_actions = actor_biases[last_bias_key].shape[0]
    print(f"\nDetected num_actions (from {last_bias_key}): {num_actions}")

# Also check if there's info about observations
critic_layers = {k: v for k, v in model_state.items() if 'critic' in k.lower()}
if critic_layers:
    first_weight = None
    for key in sorted(critic_layers.keys()):
        if 'weight' in key:
            first_weight = critic_layers[key]
            break
    if first_weight is not None:
        print(f"Critic input size (num_obs): {first_weight.shape[1]}")

# Check optimizer state for additional info
if "optimizer_state_dict" in loaded:
    opt_state = loaded["optimizer_state_dict"]
    print(f"\nOptimizer param groups: {len(opt_state.get('param_groups', []))}")
    for i, pg in enumerate(opt_state.get('param_groups', [])):
        print(f"  Group {i}: lr={pg.get('lr', 'N/A')}, weight_decay={pg.get('weight_decay', 'N/A')}")

print(f"\n{'='*60}")
print(f"Checkpoint info dict: {loaded.get('infos', {})}")
