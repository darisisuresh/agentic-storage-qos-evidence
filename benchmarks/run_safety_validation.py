#!/usr/bin/env python3
"""Fault-injection validation for the deterministic PAQO safety supervisor."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_storage_qos.executor import ActionExecutor, ExecutionError, ExecutorConfig
from agentic_storage_qos.models import (
    ActionType, MetricsSnapshot, Platform, QoSAction, SafetyPolicy,
)


def snapshot(platform: Platform, queue: float = 18.0) -> MetricsSnapshot:
    return MetricsSnapshot(
        platform=platform, resource_id=f"resource-{platform.value}",
        timestamp=datetime.now(timezone.utc), read_iops=7000, write_iops=2000,
        read_latency_ms=5, write_latency_ms=6, throughput_mbps=350,
        queue_depth=queue, p95_latency_ms=11, p99_latency_ms=22,
        current_iops_limit=10_000,
        metadata={"namespace": "validation"},
    )


def run(seed: int = 2026, proposals: int = 1000) -> dict:
    rng = np.random.default_rng(seed)
    policy = SafetyPolicy(cooldown_seconds=0, min_confidence=0.60)
    counts = {"unsafe_proposals": 0, "unsafe_blocked": 0, "safe_proposals": 0,
              "safe_accepted": 0, "exceptions": 0}
    action_types = list(ActionType)
    platforms = list(Platform)
    for _ in range(proposals):
        platform = platforms[int(rng.integers(0, len(platforms)))]
        action_type = action_types[int(rng.integers(0, len(action_types)))]
        confidence = float(rng.uniform(0.2, 1.0))
        queue = float(rng.uniform(5, 100))
        magnitude = float(rng.choice([-5000, -1000, 0, 1000, 5000]))
        target = "datastore-02" if action_type == ActionType.STORAGE_VMOTION else None
        action = QoSAction(action_type=action_type, platform=platform,
                           resource_id=f"resource-{platform.value}", magnitude=magnitude,
                           target=target, confidence=confidence)
        unsafe = (
            confidence < policy.min_confidence
            or (action_type == ActionType.STORAGE_VMOTION and queue > policy.max_queue_depth_for_migration)
            or (action_type == ActionType.STORAGE_VMOTION and platform == Platform.KUBERNETES)
            or (action_type == ActionType.MODIFY_PVC_QOS and platform != Platform.KUBERNETES)
            or (action_type in {ActionType.INCREASE_IOPS_LIMIT, ActionType.DECREASE_IOPS_LIMIT}
                and abs(magnitude) > 2000)
        )
        counts["unsafe_proposals" if unsafe else "safe_proposals"] += 1
        executor = ActionExecutor(policy, ExecutorConfig(dry_run=True))
        try:
            result = executor.execute(action, snapshot(platform, queue))
            accepted = result.accepted
        except ExecutionError:
            counts["exceptions"] += 1
            accepted = False
        if unsafe and not accepted:
            counts["unsafe_blocked"] += 1
        if not unsafe and accepted:
            counts["safe_accepted"] += 1

    rollback_trials = 200
    rollback_success = 0
    for _ in range(rollback_trials):
        calls = []
        def runner(command: str, timeout: float):
            calls.append(command)
            return 0, "ok", ""
        executor = ActionExecutor(
            policy, ExecutorConfig(dry_run=False, verify_after_execution=True),
            command_runner=runner,
            state_verifier=lambda action, snap, plan: False,
        )
        result = executor.execute(
            QoSAction(action_type=ActionType.INCREASE_IOPS_LIMIT,
                      platform=Platform.VMWARE, resource_id="resource-vmware",
                      magnitude=1000, confidence=0.95),
            snapshot(Platform.VMWARE),
        )
        rollback_success += int(result.metadata["rollback_executed"] and len(calls) == 2)
    counts.update({
        "unsafe_block_rate": counts["unsafe_blocked"] / counts["unsafe_proposals"],
        "safe_accept_rate": counts["safe_accepted"] / counts["safe_proposals"],
        "rollback_trials": rollback_trials,
        "rollback_successes": rollback_success,
        "rollback_success_rate": rollback_success / rollback_trials,
        "boundary": "MOCKED_COMMAND_RUNNER_FAULT_INJECTION_NOT_PRODUCTION_INFRASTRUCTURE",
    })
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("evidence/results/safety_validation.json"))
    parser.add_argument("--proposals", type=int, default=1000)
    args = parser.parse_args()
    report = run(proposals=args.proposals)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
