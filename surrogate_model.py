#!/usr/bin/env python3
"""
Phase 3 – Surrogate Model for Parametric Cylinder
Fits a Gaussian Process to (diameter, x_center) → Cd,
plots the design landscape, and performs inverse design.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.optimize import minimize

# ─────────────────────────────────────────────────────────────────
# 1. Load dataset
# ─────────────────────────────────────────────────────────────────
def load_dataset(csv_path="cylinder_dataset.csv"):
    diameters, x_centers, Res, Cds = [], [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            diameters.append(float(row[0]))
            x_centers.append(float(row[1]))
            Res.append(float(row[2]))
            Cds.append(float(row[3]))
    return (np.array(diameters), np.array(x_centers),
            np.array(Res), np.array(Cds))

diam, xc, Re, Cd = load_dataset()

# ─────────────────────────────────────────────────────────────────
# 2. Quick scatter / contour plot of raw data
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter: Cd vs diameter, coloured by x_center
sc = axes[0].scatter(diam, Cd, c=xc, cmap='viridis', edgecolors='k')
axes[0].set_xlabel("Diameter")
axes[0].set_ylabel("Cd")
axes[0].set_title("Raw Data: Cd vs D (colour = x)")
plt.colorbar(sc, ax=axes[0], label="x_center")

# Contour of the existing points (tricontourf)
axes[1].tricontourf(diam, xc, Cd, levels=14, cmap='plasma')
axes[1].scatter(diam, xc, c='white', edgecolors='k')
axes[1].set_xlabel("Diameter")
axes[1].set_ylabel("x_center")
axes[1].set_title("Measured Cd landscape")
plt.tight_layout()
plt.savefig("raw_data_landscape.png", dpi=150)
print("📊 Raw data plots saved as raw_data_landscape.png")

# ─────────────────────────────────────────────────────────────────
# 3. Gaussian Process surrogate
# ─────────────────────────────────────────────────────────────────
# Inputs: (diameter, x_center), output: Cd
X = np.column_stack([diam, xc])
y = Cd

# Normalise inputs to mean=0, std=1 for better GP performance
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

# Kernel: RBF + white noise (to capture data noise)
kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(X_norm, y)

# ─────────────────────────────────────────────────────────────────
# 4. Predict on a fine grid for visualisation
# ─────────────────────────────────────────────────────────────────
diam_grid = np.linspace(diam.min()*0.9, diam.max()*1.1, 50)
xc_grid = np.linspace(xc.min()*0.9, xc.max()*1.1, 50)
D, Xcg = np.meshgrid(diam_grid, xc_grid)
grid_X = np.column_stack([D.ravel(), Xcg.ravel()])
grid_X_norm = (grid_X - X_mean) / X_std
Cd_pred, Cd_std = gp.predict(grid_X_norm, return_std=True)
Cd_pred = Cd_pred.reshape(D.shape)
Cd_std = Cd_std.reshape(D.shape)

# Plot predicted surface and uncertainty
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
cs1 = axes2[0].contourf(D, Xcg, Cd_pred, levels=20, cmap='plasma')
axes2[0].scatter(diam, xc, c='white', edgecolors='k')
axes2[0].set_xlabel("Diameter")
axes2[0].set_ylabel("x_center")
axes2[0].set_title("GP Predicted Cd")
plt.colorbar(cs1, ax=axes2[0])

cs2 = axes2[1].contourf(D, Xcg, Cd_std, levels=20, cmap='inferno')
axes2[1].scatter(diam, xc, c='white', edgecolors='k')
axes2[1].set_xlabel("Diameter")
axes2[1].set_ylabel("x_center")
axes2[1].set_title("Prediction uncertainty (std)")
plt.colorbar(cs2, ax=axes2[1])
plt.tight_layout()
plt.savefig("gp_surrogate.png", dpi=150)
print("📈 GP surrogate plots saved as gp_surrogate.png")

# ─────────────────────────────────────────────────────────────────
# 5. Inverse design: find (D, x) for a target Cd
# ─────────────────────────────────────────────────────────────────
target_Cd = float(input("\n🎯 Enter target Cd (e.g. 1.5): "))

def objective(params):
    """Distance between GP prediction and target Cd"""
    pt = np.array(params).reshape(1, -1)
    pt_norm = (pt - X_mean) / X_std
    pred = gp.predict(pt_norm)[0]
    return (pred - target_Cd)**2

# Bounds: stay within observed range (with 10% margin)
bounds = [(diam.min()*0.9, diam.max()*1.1),
          (xc.min()*0.9, xc.max()*1.1)]

# Multiple random starts to find global minimum
best_result = None
best_x = None
for _ in range(20):
    x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
    res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    if best_result is None or res.fun < best_result.fun:
        best_result = res
        best_x = res.x

print(f"\n✅ Optimal design for Cd ≈ {target_Cd}:")
print(f"   Diameter  = {best_x[0]:.4f}")
print(f"   x_center  = {best_x[1]:.4f}")
print(f"   Predicted Cd = {gp.predict(np.array([best_x]).reshape(1,-1) - X_mean / X_std)[0]:.4f}")