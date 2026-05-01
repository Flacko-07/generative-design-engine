#!/usr/bin/env python3
"""
Phase 3 – Inverse Design with the trained cGAN
Loads generator and scalers, then generates designs for a user‑provided Cd.
"""

import numpy as np
from tensorflow.keras.models import load_model
import joblib

# ─────────────────────────────────────────────────────────────────
# 1. Load the trained assets
# ─────────────────────────────────────────────────────────────────
generator = load_model("cgan_generator.h5")
scaler_X = joblib.load("scaler_X.gz")
scaler_Y = joblib.load("scaler_Y.gz")

LATENT_DIM = 10   # must match the training script

print("✅ Model and scalers loaded.\n")

# ─────────────────────────────────────────────────────────────────
# 2. Generation function (identical to the one used in training)
# ─────────────────────────────────────────────────────────────────
def design_for_target(Cd_target, num_samples=5):
    """Generate cylinder designs for a desired Cd."""
    Cd_norm = scaler_X.transform(np.array([[Cd_target]]))
    Cd_norm = np.tile(Cd_norm, (num_samples, 1))
    noise = np.random.normal(0, 1, (num_samples, LATENT_DIM))
    designs_norm = generator.predict([Cd_norm, noise], verbose=0)
    designs_real = scaler_Y.inverse_transform(designs_norm)
    return designs_real

# ─────────────────────────────────────────────────────────────────
# 3. Interactive loop (or just hard‑coded examples)
# ─────────────────────────────────────────────────────────────────
try:
    while True:
        target = float(input("🎯 Enter target Cd (or press Ctrl+C to quit): "))
        designs = design_for_target(target, num_samples=5)
        print(f"\nGenerated designs for Cd ≈ {target}:")
        for i, (d, x) in enumerate(designs, 1):
            print(f"  {i}: diameter = {d:.4f}   x_center = {x:.4f}")
        print()
except KeyboardInterrupt:
    print("\n👋 Done.")