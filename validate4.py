#!/usr/bin/env python3
"""Validate the final diffusion model on laminar & turbulent designs."""

import numpy as np
import tensorflow as tf
import joblib
import csv
import matplotlib.pyplot as plt
from pathlib import Path

# Import the CFD case builders (make sure these are in the same directory)
from parametric_cylinder import build_and_run_cylinder      # laminar
from turb4 import build_turbulent_case_4param as build_turbulent_case_ch       # turbulent 4-param

# ----------------------------------------------------------------------
# 1. Load trained model & scalers
# ----------------------------------------------------------------------
denoiser = tf.keras.models.load_model("diffusion_denoiser.keras")
scaler_c = joblib.load("scaler_c.pkl")
scaler_d = joblib.load("scaler_d.pkl")

# Diffusion constants (must match training)
T = 1000
betas = np.linspace(1e-4, 0.02, T, dtype=np.float32)
alphas = 1.0 - betas
alpha_bars = np.cumprod(alphas)

@tf.function
def p_sample(x_t, t, cond_batch):
    noise_pred = denoiser([x_t, t, cond_batch])
    beta_t = tf.reshape(tf.gather(betas, t), [-1, 1])
    alpha_t = tf.reshape(tf.gather(alphas, t), [-1, 1])
    alpha_bar_t = tf.reshape(tf.gather(alpha_bars, t), [-1, 1])
    coef1 = 1.0 / tf.sqrt(alpha_t)
    coef2 = beta_t / tf.sqrt(1.0 - alpha_bar_t)
    mean = coef1 * (x_t - coef2 * noise_pred)
    noise = tf.random.normal(shape=tf.shape(x_t))
    sigma = tf.sqrt(beta_t)
    return tf.where(tf.reshape(t, [-1, 1]) > 0, mean + sigma * noise, mean)

def generate_design(target_Cd, regime, num_samples=2):
    cond_raw = np.array([[target_Cd, regime]] * num_samples, dtype=np.float32)
    cond_tensor = tf.constant(scaler_c.transform(cond_raw))
    x = tf.random.normal((num_samples, 3))
    for t in reversed(range(T)):
        t_batch = tf.fill([num_samples], t)
        x = p_sample(x, t_batch, cond_tensor)
    designs = scaler_d.inverse_transform(x.numpy())
    # Clip outputs to physically meaningful ranges
    designs[:, 0] = np.clip(designs[:, 0], 0.02, 0.5)    # diameter
    designs[:, 1] = np.clip(designs[:, 1], 0.5, 4.0)     # x_center
    designs[:, 2] = np.clip(designs[:, 2], 1.5, 4.0)     # channel_height
    return designs

# ----------------------------------------------------------------------
# 2. Validation targets
# ----------------------------------------------------------------------
targets = [
    # (target_Cd, regime)  0 = laminar, 1 = turbulent
    (1.50, 0), (1.55, 0), (1.60, 0), (1.65, 0),
    (0.90, 1), (1.00, 1), (1.10, 1), (1.20, 1),
]

results = []
print("🔍 Running validation simulations...\n")

for tgt_Cd, regime in targets:
    designs = generate_design(tgt_Cd, regime, num_samples=2)
    for i, (dia, xc, ch) in enumerate(designs):
        regime_str = "lam" if regime == 0 else "turb"
        print(f"Target Cd={tgt_Cd:.2f} ({regime_str}) D={dia:.4f} x={xc:.4f} H={ch:.4f}", end="  ")

        try:
            if regime == 0:   # Laminar
                Re, actual_cd = build_and_run_cylinder(
                    diameter=dia, x_center=xc,
                       # make sure laminar builder accepts this
                    U_inlet=0.001, nu=1e-5, target_cell_size=0.05,
                    base_dir="validation_final"
                )
            else:             # Turbulent
                Re, actual_cd, _ = build_turbulent_case_ch(
                    dia, xc, U_inlet=10.0, channel_height=ch,
                    nu=1.5e-5, target_cell_size=0.05,
                    base_dir="validation_final"
                )

            error = actual_cd - tgt_Cd
            rel_err = abs(error) / tgt_Cd * 100
            results.append((tgt_Cd, regime, dia, xc, ch, Re, actual_cd, error, rel_err))
            print(f"→ Cd={actual_cd:.4f} (err {rel_err:.1f}%)")

        except Exception as e:
            print(f"❌ Failed: {e}")

# ----------------------------------------------------------------------
# 3. Save & plot
# ----------------------------------------------------------------------
with open("validation_final.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["target_Cd", "regime", "diameter", "x_center", "channel_height",
                     "Re", "actual_Cd", "absolute_error", "relative_error_%"])
    writer.writerows(results)

# Compute per-regime stats
lam_errors = [r[8] for r in results if r[1] == 0]
turb_errors = [r[8] for r in results if r[1] == 1]

print("\n📊 Summary:")
print(f"{'Target':>8} {'Regime':>8} {'Actual Cd':>10} {'Abs Err':>8} {'Rel Err':>8}")
for row in results:
    print(f"{row[0]:8.2f} {'lam' if row[1]==0 else 'turb':>8} {row[6]:10.4f} {row[7]:8.4f} {row[8]:7.1f}%")

print(f"\nLaminar mean rel. error:   {np.mean(lam_errors):.1f}%")
print(f"Turbulent mean rel. error: {np.mean(turb_errors):.1f}%")

# Scatter plot
plt.figure(figsize=(8, 6))
for regime, color, marker in [(0, "blue", "o"), (1, "red", "s")]:
    subset = [(r[0], r[6]) for r in results if r[1] == regime]
    if subset:
        targets, actuals = zip(*subset)
        plt.scatter(targets, actuals, c=color, marker=marker,
                    label="Laminar" if regime==0 else "Turbulent", edgecolors="k")

lims = [min(r[0] for r in results), max(r[0] for r in results)]
plt.plot(lims, lims, 'k--', label="Perfect match")
plt.xlabel("Target Cd")
plt.ylabel("Simulated Cd")
plt.title("Diffusion Model Validation (1,292 training points)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("validation_final.png", dpi=150)
print("📈 Plot saved as validation_final.png")