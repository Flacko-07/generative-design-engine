#!/usr/bin/env python3
"""Validate diffusion model on both laminar and turbulent designs."""

import numpy as np
import tensorflow as tf
import joblib
import csv
import time
from turbulent_test import build_turbulent_case
from parametric_cylinder import build_and_run_cylinder  # laminar case builder

# ---------- load diffusion model ----------
denoiser = tf.keras.models.load_model("diffusion_denoiser.keras")
scaler_c = joblib.load("diffusion_scaler_c.gz")
scaler_d = joblib.load("diffusion_scaler_d.gz")

T = 1000
betas = np.linspace(1e-4, 0.02, T, dtype=np.float32)
alphas = 1.0 - betas
alpha_bars = np.cumprod(alphas)

@tf.function
def p_sample(x_t, t, cond_batch):
    noise_pred = denoiser([x_t, t, cond_batch])

    # Gather scheduling parameters and reshape to [batch, 1]
    beta_t      = tf.reshape(tf.gather(betas, t),      [-1, 1])
    alpha_t     = tf.reshape(tf.gather(alphas, t),     [-1, 1])
    alpha_bar_t = tf.reshape(tf.gather(alpha_bars, t), [-1, 1])

    coef1 = 1.0 / tf.sqrt(alpha_t)
    coef2 = beta_t / tf.sqrt(1.0 - alpha_bar_t)

    mean = coef1 * (x_t - coef2 * noise_pred)

    # Add noise for t > 0
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
    return scaler_d.inverse_transform(x.numpy())

# ---------- validation targets ----------
targets = [
    # (target_Cd, regime)  regime: 0=laminar, 1=turbulent
    (1.50, 0), (1.55, 0), (1.60, 0), (1.65, 0),
    (0.85, 1), (0.95, 1), (1.05, 1), (1.15, 1),
]

results = []
print("🔍 Running validation simulations...\n")

for tgt_Cd, regime in targets:
    designs = generate_design(tgt_Cd, regime, num_samples=2)
    for i, (dia, xc, br) in enumerate(designs):
        dia = max(0.02, min(0.5, dia))
        xc  = max(0.5, min(4.0, xc))
        print(f"Target Cd={tgt_Cd:.2f} ({'lam' if regime==0 else 'turb'}) "
              f"D={dia:.4f} x={xc:.4f} br={br:.4f}", end="  ")
        try:
            if regime == 0:   # laminar
                Re, actual_cd = build_and_run_cylinder(
                    dia, xc, U_inlet=0.001, nu=1e-5, target_cell_size=0.05,
                    base_dir="validation_laminar"
                )
            else:             # turbulent
                Re, actual_cd = build_turbulent_case(
                    dia, xc, U_inlet=10.0, nu=1.5e-5, target_cell_size=0.05,
                    base_dir="validation_turbulent"
                )
            error = actual_cd - tgt_Cd
            rel_err = abs(error) / tgt_Cd * 100
            results.append((tgt_Cd, regime, dia, xc, br, Re, actual_cd, error, rel_err))
            print(f"→ Cd={actual_cd:.4f} (err {rel_err:.1f}%)")
        except Exception as e:
            print(f"❌ Failed: {e}")

# ---------- save & print summary ----------
with open("diffusion_validation.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["target_Cd", "regime", "diameter", "x_center", "blockage_ratio",
                     "Re", "actual_Cd", "absolute_error", "relative_error_%"])
    writer.writerows(results)

print("\n📊 Summary:")
print(f"{'Target':>8} {'Regime':>8} {'Actual Cd':>10} {'Abs Err':>8} {'Rel Err':>8}")
for row in results:
    print(f"{row[0]:8.2f} {'lam' if row[1]==0 else 'turb':>8} {row[6]:10.4f} {row[7]:8.4f} {row[8]:7.1f}%")

# Per‑regime statistics
lam_errors = [r[8] for r in results if r[1]==0]
turb_errors = [r[8] for r in results if r[1]==1]
print(f"\nLaminar mean rel. error:   {np.mean(lam_errors):.1f}%")
print(f"Turbulent mean rel. error: {np.mean(turb_errors):.1f}%")