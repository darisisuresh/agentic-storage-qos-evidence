"""Closed-loop orchestration for collection, decision, execution, and learning."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
import logging
from typing import Deque

from .agent import DQNAgent
from .collector import CollectorManager
from .executor import ActionExecutor
from .models import ActionResult, MetricsSnapshot

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class EngineConfig:
    """Optimization loop controls."""

    poll_interval_seconds: float = 10.0
    training_enabled: bool = True
    history_per_resource: int = 4

    def __post_init__(self) -> None:
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if self.history_per_resource < 2:
            raise ValueError("history_per_resource must be >= 2")


class OptimizationEngine:
    """Coordinate telemetry, DQN inference, safety checks, and online replay."""

    def __init__(
        self,
        collector_manager: CollectorManager,
        agent: DQNAgent,
        executor: ActionExecutor,
        config: EngineConfig | None = None,
    ) -> None:
        self.collector_manager = collector_manager
        self.agent = agent
        self.executor = executor
        self.config = config or EngineConfig()
        self._history: dict[
            tuple[str, str], Deque[MetricsSnapshot]
        ] = defaultdict(
            lambda: deque(maxlen=self.config.history_per_resource)
        )

    async def run_once(self) -> list[ActionResult]:
        """Execute one complete optimization cycle."""

        snapshots = await self.collector_manager.collect_all()
        results: list[ActionResult] = []

        for snapshot in snapshots:
            key = (snapshot.platform.value, snapshot.resource_id)
            history = self._history[key]
            previous = history[-1] if history else None

            action = self.agent.select_action(
                snapshot,
                previous=previous,
                training=self.config.training_enabled,
            )
            result = self.executor.execute(action, snapshot)
            results.append(result)

            if previous is not None and self.config.training_enabled:
                state = self.agent.encode_state(previous)
                next_state = self.agent.encode_state(snapshot, previous)
                reward = self.agent.compute_reward(previous, snapshot, action)
                self.agent.remember(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=False,
                    next_action_mask=self.agent.action_mask(snapshot),
                )
                loss = self.agent.train_step()
                if loss is not None:
                    LOGGER.debug(
                        "DQN update",
                        extra={
                            "loss": loss,
                            "epsilon": self.agent.epsilon,
                            "resource_id": snapshot.resource_id,
                        },
                    )

            history.append(snapshot)

        return results

    async def run_forever(self, stop_event: asyncio.Event) -> None:
        """Run until ``stop_event`` is set."""

        while not stop_event.is_set():
            try:
                await self.run_once()
            except Exception:
                LOGGER.exception("Optimization cycle failed")
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self.config.poll_interval_seconds,
                )
            except asyncio.TimeoutError:
                continue
