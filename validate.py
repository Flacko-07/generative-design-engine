#!/usr/bin/env python3
"""
Validate the cGAN on multiple target Cds and plot errors.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
import sys, os

# Import the simulation function from your pipeline
from parametric_cylinder import build_and_run_cylinder

# ──────────────────────────────────────────────────────
# 1. Load the trained cGAN generator and scalers
# ──────────────────────────────────────────────────────
generator = load_model("cgan_generator.keras")   # or .h5 if that's what you saved
scaler_X = joblib.load("scaler_X.gz")
scaler_Y = joblib.load("scaler_Y.gz")

LATENT_DIM = 10   # must match training script

# ──────────────────────────────────────────────────────
# 2. Generation function
# ──────────────────────────────────────────────────────
def design_for_target(Cd_target, num_samples=3):
    Cd_norm = scaler_X.transform(np.array([[Cd_target]]))
    Cd_norm = np.tile(Cd_norm, (num_samples, 1))
    noise = np.random.normal(0, 1, (num_samples, LATENT_DIM))
    gen_input = np.concatenate([noise, Cd_norm], axis=1)
    designs_norm = generator.predict(gen_input, verbose=0)
    return scaler_Y.inverse_transform(designs_norm)

# ──────────────────────────────────────────────────────
# 3. Targets and storage
# ──────────────────────────────────────────────────────
targets = [1.50, 1.55, 1.60, 1.65, 1.70]
results = []

for target in targets:
    designs = design_for_target(target, num_samples=3)
    for i, (dia, xc) in enumerate(designs):
        # Ensure parameters are within reasonable bounds
        dia = max(0.05, min(0.6, dia))
        xc = max(0.5, min(3.5, xc))
        print(f"Target Cd={target:.2f} | Design {i+1}: D={dia:.4f}, x={xc:.4f}")
        Re, actual_cd = build_and_run_cylinder(
            diameter=dia,
            x_center=xc,
            U_inlet=0.001,
            nu=1e-5,
            target_cell_size=0.05,
            base_dir="validation_cases"
        )
        error = actual_cd - target
        rel_error = abs(error) / target * 100
        results.append({
            "target_Cd": target,
            "diameter": dia,
            "x_center": xc,
            "actual_Cd": actual_cd,
            "absolute_error": error,
            "relative_error_%": rel_error,
            "Re": Re
        })
        print(f"     Actual Cd={actual_cd:.4f}  (error {rel_error:.1f}%)")

# ──────────────────────────────────────────────────────
# 4. Save results and plot
# ──────────────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv("validation_results.csv", index=False)
print("\n📊 Validation results saved to validation_results.csv")

plt.figure(figsize=(8, 6))
for target in targets:
    subset = df[df["target_Cd"] == target]
    plt.scatter(subset["target_Cd"], subset["actual_Cd"],
                label=f"Target {target:.2f}", s=80)
# Ideal line
lims = [min(df["target_Cd"]), max(df["target_Cd"])]
plt.plot(lims, lims, 'k--', label="Perfect match")
plt.xlabel("Target Cd")
plt.ylabel("Simulated Cd")
plt.title("cGAN Inverse Design Validation")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("validation_error.png", dpi=150)
print("📈 Error plot saved as validation_error.png")