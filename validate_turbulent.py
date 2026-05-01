#!/usr/bin/env python3
"""Validate diffusion‑generated turbulent cylinder designs."""

import numpy as np
import tensorflow as tf
import joblib
from turbulent_test import build_turbulent_case   # reuse the working case builder

# ----------------------------------------------------------------------
# 1. Load the trained diffusion model (inference only)
# ----------------------------------------------------------------------
denoiser = tf.keras.models.load_model("diffusion_denoiser.keras")
scaler_c = joblib.load("diffusion_scaler_c.gz")
scaler_d = joblib.load("diffusion_scaler_d.gz")

# Diffusion constants (must match training)
T = 1000
beta_start, beta_end = 1e-4, 0.02
betas = np.linspace(beta_start, beta_end, T, dtype=np.float32)
alphas = 1.0 - betas
alpha_bars = np.cumprod(alphas)

@tf.function
def p_sample(x_t, t, cond_batch):
    noise_pred = denoiser([x_t, t, cond_batch])
    beta_t = tf.gather(betas, t)
    alpha_t = tf.gather(alphas, t)
    alpha_bar_t = tf.gather(alpha_bars, t)
    coef1 = 1.0 / tf.sqrt(alpha_t)
    coef2 = beta_t / tf.sqrt(1.0 - alpha_bar_t)
    mean = coef1 * (x_t - coef2 * noise_pred)
    noise = tf.random.normal(shape=tf.shape(x_t))
    sigma = tf.sqrt(beta_t)
    return tf.where(tf.reshape(t, [-1, 1]) > 0, mean + sigma * noise, mean)

def generate_design(target_Cd, regime, num_samples=1):
    cond_raw = np.array([[target_Cd, regime]] * num_samples, dtype=np.float32)
    cond_tensor = tf.constant(scaler_c.transform(cond_raw))
    x = tf.random.normal((num_samples, 3))
    for t in reversed(range(T)):
        t_batch = tf.fill([num_samples], t)
        x = p_sample(x, t_batch, cond_tensor)
    return scaler_d.inverse_transform(x.numpy())

# ----------------------------------------------------------------------
# 2. Validation targets (turbulent regime)
# ----------------------------------------------------------------------
targets = [
    (0.90, 1),
    (1.00, 1),
    (1.10, 1),
]

print("🔍 Validating diffusion‑generated designs against OpenFOAM\n")
results = []

for target_cd, regime in targets:
    design = generate_design(target_cd, regime, num_samples=1)[0]
    dia, xc, br = design
    
    # Ensure parameters are physically sensible
    dia = max(0.02, min(0.5, dia))
    xc  = max(0.5, min(4.0, xc))
    
    print(f"Target Cd = {target_cd:.2f} → generated D={dia:.4f}  x={xc:.4f}  br={br:.4f}")
    
    # Run turbulent OpenFOAM simulation
    Re, actual_cd = build_turbulent_case(
        diameter=dia,
        x_center=xc,
        U_inlet=10.0,      # keep consistent with training data
        nu=1.5e-5,
        base_dir="diffusion_validation"
    )
    
    error = abs(actual_cd - target_cd)
    rel_error_pct = error / target_cd * 100
    results.append((target_cd, dia, xc, br, Re, actual_cd, error, rel_error_pct))
    
    print(f"  Actual Cd = {actual_cd:.4f}  (error = {error:.4f}, {rel_error_pct:.1f}%)")
    print()

# ----------------------------------------------------------------------
# 3. Summary
# ----------------------------------------------------------------------
print("📊 Validation Summary:")
print(f"{'Target Cd':>10} {'Diam':>8} {'x_center':>10} {'Block':>8} {'Re':>10} {'Actual Cd':>10} {'Error':>8} {'Rel. err':>10}")
for (tcd, dia, xc, br, Re, actual, err, rel) in results:
    print(f"{tcd:10.2f} {dia:8.4f} {xc:10.4f} {br:8.4f} {Re:10.0f} {actual:10.4f} {err:8.4f} {rel:9.1f}%")