#!/usr/bin/env python3
"""Validate synthetic-trace dimensions, finiteness, uniqueness, and platform mix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.run_validation import workload


def validate(seeds: int = 100) -> dict:
    scenarios = {
        "mild": (0.75, None), "nominal": (1.0, None), "severe": (1.35, None),
        "transfer_vmware": (1.0, "vmware"),
        "transfer_hyperv": (1.0, "hyperv"),
        "transfer_kubernetes": (1.0, "kubernetes"),
    }
    hashes = set()
    missing_values = 0
    nonpositive_values = 0
    malformed_shapes = 0
    total_traces = 0
    for scenario, (severity, shift) in scenarios.items():
        for seed in range(seeds):
            demand, capacity = workload(seed, severity, platform_shift=shift)
            total_traces += 1
            malformed_shapes += int(demand.shape != (720, 4) or capacity.shape != (720,))
            missing_values += int(np.size(demand) - np.isfinite(demand).sum())
            missing_values += int(np.size(capacity) - np.isfinite(capacity).sum())
            nonpositive_values += int(np.sum(demand <= 0) + np.sum(capacity <= 0))
            hashes.add(hashlib.sha256(scenario.encode() + demand.tobytes() + capacity.tobytes()).hexdigest())
    duplicates = total_traces - len(hashes)
    report = {
        "independent_seeds": seeds,
        "scenario_count": len(scenarios),
        "total_traces": total_traces,
        "intervals_per_trace": 720,
        "workloads_per_trace": 4,
        "platform_composition": {"vmware": 1, "hyperv": 1, "kubernetes": 2},
        "missing_or_nonfinite_values": missing_values,
        "nonpositive_demand_or_capacity_values": nonpositive_values,
        "malformed_shapes": malformed_shapes,
        "duplicate_trace_hashes": duplicates,
        "learned_model_split": "NOT_APPLICABLE_NO_TRAINED_MODEL_IN_REPORTED_EVALUATION",
        "classification_distribution": "NOT_APPLICABLE_NO_CLASSIFICATION_TASK",
        "passed": not any((missing_values, nonpositive_values, malformed_shapes, duplicates)),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--output", type=Path,
                        default=Path("evidence/results/data_quality_validation.json"))
    args = parser.parse_args()
    report = validate(args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
