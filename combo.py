#!/usr/bin/env python3
"""Merge laminar and turbulent datasets."""
import pandas as pd

# Load laminar data
lam = pd.read_csv("cylinder_dataset_large.csv")   # columns: diameter, x_center, Re, Cd
lam["blockage_ratio"] = lam["diameter"] / 2.0      # channel height = 2.0
lam["flow_regime"] = 0                              # 0 = laminar

# Load turbulent data (from recovery)
turb = pd.read_csv("turbulent_dataset.csv")        # columns: diameter, x_center, U_inlet, Re, Cd
turb["blockage_ratio"] = turb["diameter"] / 2.0
turb["flow_regime"] = 1                             # 1 = turbulent

# Combine and save
df = pd.concat([lam, turb], ignore_index=True)
df.to_csv("combined_dataset.csv", index=False)
print(f"Combined dataset: {len(df)} designs (laminar: {len(lam)}, turbulent: {len(turb)})")