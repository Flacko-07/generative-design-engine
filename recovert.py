#!/usr/bin/env python3
"""Recover turbulent dataset from completed cases."""
import csv, re
from pathlib import Path

def parse_forces_file(path):
    with open(path) as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.startswith('#')]
    last = lines[-1].replace('(', ' ').replace(')', ' ').split()
    return float(last[1]), float(last[2]), float(last[3])

base = Path("turbulent_cases")
rows = []
for case_dir in sorted(base.glob("turb_D*_X*_U*")):
    # Extract parameters from directory name
    parts = case_dir.name.split('_')
    dia = float(parts[1].lstrip('D'))
    xc  = float(parts[2].lstrip('X'))
    U   = float(parts[3].lstrip('U'))

    # Find latest time directory
    time_dirs = [d for d in case_dir.iterdir()
                 if d.is_dir() and re.match(r'^\d+(\.\d+)?$', d.name)]
    if not time_dirs:
        print(f"  No time dirs in {case_dir.name}, skipping")
        continue
    latest = sorted(time_dirs, key=lambda d: float(d.name))[-1]

    # Find forces.dat (either name)
    forces_path = None
    for name in ("forces.dat", "force.dat"):
        cand = list((case_dir / "postProcessing" / "forces").rglob(name))
        if cand:
            forces_path = cand[-1]
            break
    if not forces_path:
        print(f"  No forces file in {case_dir.name}, skipping")
        continue

    try:
        fx, fy, fz = parse_forces_file(forces_path)
        Re = U * dia / 1.5e-5
        Cd = 2.0 * fx / (1.0 * U**2 * (dia * 0.5))
        rows.append((dia, xc, U, Re, Cd))
        print(f"  OK: D={dia:.3f} x={xc:.3f} U={U:.1f} Re={Re:.0f} Cd={Cd:.4f}")
    except Exception as e:
        print(f"  Failed parsing {case_dir.name}: {e}")

with open("turbulent_dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["diameter", "x_center", "U_inlet", "Re", "Cd"])
    writer.writerows(rows)
print(f"\n✅ Recovered {len(rows)} cases into turbulent_dataset.csv")