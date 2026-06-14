"""
Run this once in your .venv:
    python export_weights.py

It reads model.pth and writes weights.json next to this file.
weights.json is what index.html loads to run the real trained agent.
"""

import torch
import json
import os

MODEL_PATH = "model.pth"
OUTPUT_PATH = "weights.json"

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: {MODEL_PATH} not found. Train the model first.")
    exit(1)

# Load checkpoint — weights_only=False because we saved epsilon and learn_step too
checkpoint = torch.load(MODEL_PATH, weights_only=False)

state_dict = checkpoint["policy_network"]

# Each key is like "fc1.weight", "fc1.bias", "fc2.weight" etc.
# .tolist() converts the tensor to a plain Python list so json.dump can handle it
weights = {}
for key, tensor in state_dict.items():
    weights[key] = tensor.tolist()

# Also export epsilon so the visualizer can show what it was at the end of training
weights["_epsilon"] = float(checkpoint["epsilon"])
weights["_learn_step"] = int(checkpoint["learn_step"])

with open(OUTPUT_PATH, "w") as f:
    json.dump(weights, f)

print(f"Done. Exported {len(state_dict)} tensors to {OUTPUT_PATH}")
print(f"  Epsilon at save time : {checkpoint['epsilon']:.4f}")
print(f"  Learn steps          : {checkpoint['learn_step']}")
print(f"\nNow put index.html and weights.json in the same folder and open index.html")