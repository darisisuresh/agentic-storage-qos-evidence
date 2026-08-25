#!/usr/bin/env python3
"""Deterministic PAQO synthetic validation and wall-clock timing benchmark.

The model is intentionally transparent and is not a production-array emulator.
Independent random seeds are the experimental units; interval samples are not.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy import stats


BASE = np.array([5000.0, 4500.0, 3500.0, 4000.0])
SLO = np.array([12.0, 18.0, 30.0, 15.0])
BASE_LAT = np.array([3.2, 4.4, 6.5, 3.8])
STATIC_LIMIT = np.full(4, 7000.0)
MODEL_SPEC = {
    "backlog_request_weight": 0.10,
    "backlog_calibration_divisor": 110.0,
    "backlog_min": 0.0,
    "backlog_max": 240.0,
    "backlog_latency_coefficient": 0.020,
    "shared_utilization_coefficient": 2.0,
    "shared_utilization_threshold": 0.74,
    "workload_utilization_coefficient": 0.70,
    "p95_tail_factor": 1.32,
    "p99_tail_factor": 1.58,
    "backlog_semantics": "DIMENSIONLESS_SYNTHETIC_INDEX_NOT_OUTSTANDING_IOS",
}


@dataclass
class RunResult:
    seed: int
    scenario: str
    controller: str
    p95_ms: float
    p99_ms: float
    violation_pct: float
    served_iops: float
    queue_depth: float
    actions: int
    fairness: float
    decision_us: float


def workload(seed: int, severity: float, intervals: int = 720,
             platform_shift: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(intervals)
    demand = BASE * (1 + 0.08 * np.sin(2 * np.pi * t[:, None] / np.array([180, 150, 210, 170])))
    demand += rng.normal(0, BASE * 0.035, size=(intervals, 4))
    shift_map = {"vmware": [0], "hyperv": [1], "kubernetes": [2, 3]}
    if platform_shift:
        demand[:, shift_map[platform_shift]] *= 1.15
    widths = [10, 18, 24, 12]
    counts = [4, 3, 4, 4]
    amps = np.array([0.80, 0.58, 0.72, 0.68]) * severity
    for w in range(4):
        starts = rng.choice(np.arange(40, intervals - widths[w] - 40), counts[w], replace=False)
        for start in starts:
            x = np.arange(widths[w])
            demand[start:start + widths[w], w] += BASE[w] * amps[w] * np.sin(np.pi * (x + 1) / (widths[w] + 1))
    capacity = np.full(intervals, 28000.0)
    backup_start = int(rng.integers(250, 390))
    capacity[backup_start:backup_start + 45] *= 1 - 0.10 * severity
    demand[backup_start:backup_start + 45] += 600 * severity
    for start in rng.choice(np.arange(30, intervals - 15), 5, replace=False):
        capacity[start:start + int(rng.integers(4, 11))] *= 1 - rng.uniform(0.06, 0.14) * severity
    return np.maximum(demand, 100.0), capacity


def decide(name: str, history: np.ndarray, limits: np.ndarray, p99: np.ndarray,
           cooldown: np.ndarray, t: int) -> tuple[np.ndarray, int]:
    if name == "static":
        return limits, 0
    eligible = cooldown <= 0
    pressure = p99 / SLO
    if name == "reactive":
        allocation_signal = history[t]
        trigger = bool(np.any(pressure > 1.0))
        hold = 5
    elif name == "predictive":
        recent = history[max(0, t - 12):t + 1]
        smooth = np.average(recent, axis=0, weights=np.linspace(0.5, 1.5, len(recent)))
        slope = (recent[-1] - recent[max(0, len(recent) - 5)]) / min(4, len(recent) - 1) if len(recent) > 1 else 0
        envelope = smooth + 6 * np.maximum(slope, 0) + 1.25 * np.std(recent, axis=0)
        allocation_signal = envelope
        trigger = bool(np.any(envelope / np.maximum(limits, 1) > 0.91))
        hold = 4
    elif name == "predictive_no_forecast":
        allocation_signal = history[t]
        trigger = bool(np.any(history[t] / np.maximum(limits, 1) > 0.98))
        hold = 4
    elif name == "mpc":
        recent = history[max(0, t - 16):t + 1]
        slope = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
        horizons = np.arange(1, 9)[:, None]
        forecast = np.maximum(recent[-1] + horizons * slope, 100.0)
        allocation_signal = np.quantile(forecast, 0.90, axis=0)
        trigger = bool(np.any(allocation_signal / np.maximum(limits, 1) > 0.93))
        hold = 6
    else:
        raise ValueError(name)
    if not trigger or not np.any(eligible):
        return limits, 0
    desired = 28000.0 * allocation_signal / np.sum(allocation_signal)
    desired = np.clip(desired, 4000.0, 10000.0)
    desired *= 28000.0 / np.sum(desired)
    delta = np.clip(desired - limits, -900.0, 900.0)
    delta -= np.mean(delta)
    if np.max(np.abs(delta)) < 100:
        return limits, 0
    limits = np.clip(limits + delta, 3500.0, 10500.0)
    limits *= 28000.0 / np.sum(limits)
    cooldown[:] = hold
    return limits, 1


def simulate(seed: int, controller: str, scenario: str, severity: float,
             platform_shift: str | None = None) -> RunResult:
    demand, capacity = workload(seed, severity, platform_shift=platform_shift)
    # This calibrated state is a dimensionless synthetic backlog index.  It is
    # retained as queue_depth in the public CSV for backward compatibility.
    backlog = np.zeros(4)
    limits = STATIC_LIMIT.copy()
    cooldown = np.zeros(4)
    p99 = BASE_LAT * 1.55
    records = []
    actions = 0
    decision_ns = 0
    for t in range(len(demand)):
        start = time.perf_counter_ns()
        limits, changed = decide(controller, demand, limits, p99, cooldown, t)
        decision_ns += time.perf_counter_ns() - start
        actions += changed
        requested = np.minimum(demand[t] + MODEL_SPEC["backlog_request_weight"] * backlog, limits)
        scale = min(1.0, capacity[t] / max(np.sum(requested), 1.0))
        served = requested * scale
        backlog = np.clip(
            backlog + (demand[t] - served) / MODEL_SPEC["backlog_calibration_divisor"],
            MODEL_SPEC["backlog_min"], MODEL_SPEC["backlog_max"])
        utilization = np.sum(served) / capacity[t]
        own = served / np.maximum(limits, 1)
        mean = BASE_LAT * (
            1 + MODEL_SPEC["backlog_latency_coefficient"] * backlog
            + MODEL_SPEC["shared_utilization_coefficient"]
            * max(0, utilization - MODEL_SPEC["shared_utilization_threshold"]) ** 3
            + MODEL_SPEC["workload_utilization_coefficient"] * own ** 3)
        p95 = mean * MODEL_SPEC["p95_tail_factor"]
        p99 = mean * MODEL_SPEC["p99_tail_factor"]
        satisfaction = np.minimum(1.0, SLO / np.maximum(p99, 1e-9))
        fairness = np.sum(satisfaction) ** 2 / (4 * np.sum(satisfaction ** 2))
        records.append((p95.mean(), p99.mean(), (p99 > SLO).mean(), served.sum(), backlog.mean(), fairness))
        cooldown -= 1
    values = np.asarray(records)
    means = values.mean(axis=0)
    return RunResult(seed, scenario, controller, means[0], means[1], 100 * means[2], means[3], means[4], actions, means[5], decision_ns / len(demand) / 1000)


def ci(values: np.ndarray) -> tuple[float, float]:
    half = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
    return float(values.mean()), float(half)


def summarize(rows: list[RunResult]) -> dict:
    out: dict[str, dict] = {}
    for scenario in sorted({r.scenario for r in rows}):
        out[scenario] = {}
        for controller in sorted({r.controller for r in rows if r.scenario == scenario}):
            group = [r for r in rows if r.scenario == scenario and r.controller == controller]
            metrics = {}
            for key in ("p95_ms", "p99_ms", "violation_pct", "served_iops", "queue_depth", "actions", "fairness", "decision_us"):
                mean, half = ci(np.array([getattr(r, key) for r in group], dtype=float))
                metrics[key] = {"mean": mean, "ci95_half_width": half}
            out[scenario][controller] = metrics
    base = {(r.seed, r.scenario): r for r in rows if r.controller == "static"}
    paired = {}
    for controller in ("reactive", "predictive", "predictive_no_forecast", "mpc"):
        group = [r for r in rows if r.scenario == "nominal" and r.controller == controller]
        if not group:
            continue
        paired[controller] = {}
        for key in ("p99_ms", "violation_pct", "queue_depth"):
            x = np.array([getattr(base[(r.seed, r.scenario)], key) for r in group])
            y = np.array([getattr(r, key) for r in group])
            diff = x - y
            test = stats.wilcoxon(diff, alternative="greater")
            paired[controller][key] = {
                "mean_improvement_pct": float(100 * diff.mean() / x.mean()),
                "paired_difference_ci95": list(ci(diff)),
                "wilcoxon_p": float(test.pvalue),
                "cohens_dz": float(diff.mean() / diff.std(ddof=1)),
            }
    return {"design": {"independent_seeds": len({r.seed for r in rows}), "intervals_per_seed": 720,
                       "interval_seconds": 10, "confidence_level": 0.95,
                       "model_specification": MODEL_SPEC}, "summary": out, "paired_tests": paired}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("evidence/results"))
    args = parser.parse_args()
    rows = []
    scenarios = {
        "nominal": (1.0, None),
        "mild": (0.75, None),
        "severe": (1.35, None),
        "transfer_vmware": (1.0, "vmware"),
        "transfer_hyperv": (1.0, "hyperv"),
        "transfer_kubernetes": (1.0, "kubernetes"),
    }
    controllers = ("static", "reactive", "predictive", "predictive_no_forecast", "mpc")
    wall_start = time.perf_counter()
    for scenario, (severity, platform_shift) in scenarios.items():
        for seed in range(args.seeds):
            for controller in controllers:
                rows.append(simulate(seed, controller, scenario, severity, platform_shift))
    elapsed = time.perf_counter() - wall_start
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per_seed_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])), lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)
    report = summarize(rows)
    report["execution"] = {"wall_clock_seconds": elapsed, "python": "3", "timing_clock": "perf_counter_ns"}
    (args.output / "validation_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
