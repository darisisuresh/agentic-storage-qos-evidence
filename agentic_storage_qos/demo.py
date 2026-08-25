"""Runnable dry-run demonstration of the agentic storage QoS framework."""

from __future__ import annotations

import asyncio
import logging

from .agent import AgentConfig, DQNAgent
from .collector import (
    CollectorConfig,
    CollectorManager,
    HyperVMetricCollector,
    KubernetesMetricCollector,
    VMwareMetricCollector,
)
from .engine import EngineConfig, OptimizationEngine
from .executor import ActionExecutor, ExecutorConfig
from .models import SafetyPolicy


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    collector_config = CollectorConfig(random_seed=42)
    manager = CollectorManager(
        [
            VMwareMetricCollector(["vm-app-01", "vm-db-01"], collector_config),
            HyperVMetricCollector(["hv-kafka-01"], collector_config),
            KubernetesMetricCollector(
                ["spark-data-pvc", "postgres-pvc"],
                collector_config,
            ),
        ]
    )

    safety = SafetyPolicy(
        min_iops_limit=500,
        max_iops_limit=50_000,
        max_iops_step_fraction=0.20,
        min_confidence=0.55,
        cooldown_seconds=0,
    )
    agent = DQNAgent(
        config=AgentConfig(
            batch_size=8,
            replay_capacity=2_000,
            target_sync_interval=25,
            random_seed=42,
        ),
        safety_policy=safety,
    )
    executor = ActionExecutor(
        safety_policy=safety,
        config=ExecutorConfig(dry_run=True),
    )
    engine = OptimizationEngine(
        collector_manager=manager,
        agent=agent,
        executor=executor,
        config=EngineConfig(
            poll_interval_seconds=1.0,
            training_enabled=True,
        ),
    )

    for cycle in range(5):
        print(f"\n=== Optimization cycle {cycle + 1} ===")
        results = await engine.run_once()
        for result in results:
            print(
                f"{result.platform.value:10s} {result.resource_id:18s} "
                f"accepted={result.accepted!s:5s} executed={result.executed!s:5s} "
                f"command={result.command}"
            )
        await asyncio.sleep(0.2)


if __name__ == "__main__":
    asyncio.run(main())
