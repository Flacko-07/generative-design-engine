#!/usr/bin/env python3
"""Generate additional turbulent cases for fine‑tuning."""
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import time
from turbulent_scaled import build_turbulent_case  # reuse the same function

def worker(args):
    dia, xc, U = args
    Re, Cd = build_turbulent_case(dia, xc, U, nu=1.5e-5, target_cell_size=0.05, base_dir="finetune_cases")
    return dia, xc, U, Re, Cd

if __name__ == "__main__":
    np.random.seed(42)
    n_extra = 200

    # Extended ranges: slightly outside original to improve extrapolation
    dias = np.random.uniform(0.03, 0.45, n_extra)
    xcs  = np.random.uniform(0.5, 4.0, n_extra)
    Us   = np.random.uniform(4.0, 22.0, n_extra)

    total = len(dias)
    print(f"🚀 Launching {total} extra turbulent cases...")

    results = []
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(worker, (d, x, u)): (d, x, u) for d, x, u in zip(dias, xcs, Us)}
        for fut in futures:
            try:
                d, x, u, Re, Cd = fut.result()
                results.append((d, x, u, Re, Cd))
                print(f"  D={d:.3f} x={x:.3f} U={u:.1f} Re={Re:.0f} Cd={Cd:.4f}")
            except Exception as e:
                print(f"  ❌ Failed: {e}")

    # Save to CSV
    import csv
    with open("finetune_dataset.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["diameter", "x_center", "U_inlet", "Re", "Cd"])
        writer.writerows(results)
    print(f"✅ Saved {len(results)} cases to finetune_dataset.csv")