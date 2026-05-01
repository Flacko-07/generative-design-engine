#!/usr/bin/env python3
"""Fast CSV-backed inference for the Generative Design Engine.

This script generates candidate cylinder-flow designs for any target Cd and
regime without launching OpenFOAM or loading TensorFlow model artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path


DATASET = Path(__file__).with_name("combined_final.csv")


def load_rows(path: Path = DATASET) -> list[dict[str, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: float(value) for key, value in row.items()} for row in reader]


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def candidate_from_neighbors(neighbors: list[dict[str, float]], target_cd: float, index: int) -> dict[str, float]:
    phase = index + 1
    mixed = {
        "diameter": 0.0,
        "x_center": 0.0,
        "channel_height": 0.0,
        "Re": 0.0,
        "Cd": 0.0,
        "U_inlet": 0.0,
        "blockage_ratio": 0.0,
    }
    weight_total = 0.0

    for neighbor_index, row in enumerate(neighbors):
        distance = abs(row["Cd"] - target_cd)
        weight = 1 / (distance + 0.015 + neighbor_index * 0.002)
        weight_total += weight
        for key in mixed:
            mixed[key] += row[key] * weight

    for key in mixed:
        mixed[key] /= weight_total

    closest = neighbors[0]
    spread = max(0.015, abs(closest["Cd"] - target_cd) * 0.25)
    diameter_jitter = math.sin(target_cd * 13.7 + phase) * spread
    x_jitter = math.cos(target_cd * 8.3 + phase * 0.7) * spread * 4
    h_jitter = math.sin(target_cd * 5.1 + phase * 1.3) * spread * 2

    return {
        "rank": index + 1,
        "target_Cd": round(target_cd, 4),
        "predicted_Cd": round(mixed["Cd"], 4),
        "diameter": round(clamp(mixed["diameter"] + diameter_jitter, 0.02, 0.5), 5),
        "x_center": round(clamp(mixed["x_center"] + x_jitter, 0.5, 4.0), 5),
        "channel_height": round(clamp(mixed["channel_height"] + h_jitter, 1.5, 4.0), 5),
        "Re": round(max(0.0, mixed["Re"]), 2),
        "U_inlet": round(mixed["U_inlet"], 4),
        "blockage_ratio": round(clamp(mixed["blockage_ratio"], 0.0, 1.0), 5),
        "nearest_training_Cd": round(closest["Cd"], 4),
    }


def generate_designs(target_cd: float, regime: int, count: int = 5) -> dict[str, object]:
    if target_cd <= 0:
        raise ValueError("target Cd must be positive")
    if regime not in {0, 1}:
        raise ValueError("regime must be 0 for laminar or 1 for turbulent")

    started = time.perf_counter()
    pool = sorted(
        (row for row in load_rows() if int(row["flow_regime"]) == regime),
        key=lambda row: abs(row["Cd"] - target_cd),
    )
    count = int(clamp(count, 1, 12))
    designs = [candidate_from_neighbors(pool[index : index + 8], target_cd, index) for index in range(count)]

    return {
        "input": {
            "cd": target_cd,
            "regime": regime,
            "regime_label": "laminar" if regime == 0 else "turbulent",
            "count": count,
        },
        "method": "regime-aware nearest-neighbor inference over combined_final.csv",
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "designs": designs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CFD design candidates for target Cd and regime.")
    parser.add_argument("--cd", type=float, required=True, help="Target drag coefficient")
    parser.add_argument("--regime", type=int, choices=[0, 1], required=True, help="0=laminar, 1=turbulent")
    parser.add_argument("--count", type=int, default=5, help="Number of candidates to generate")
    args = parser.parse_args()

    print(json.dumps(generate_designs(args.cd, args.regime, args.count), indent=2))


if __name__ == "__main__":
    main()

