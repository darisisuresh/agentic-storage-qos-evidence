"""Shared domain models for the storage QoS control plane."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping
import math
import uuid


class Platform(str, Enum):
    """Supported infrastructure platforms."""

    VMWARE = "vmware"
    HYPERV = "hyperv"
    KUBERNETES = "kubernetes"


class ActionType(str, Enum):
    """Discrete actions available to the reinforcement-learning policy."""

    NOOP = "noop"
    INCREASE_IOPS_LIMIT = "increase_iops_limit"
    DECREASE_IOPS_LIMIT = "decrease_iops_limit"
    STORAGE_VMOTION = "storage_vmotion"
    RESCHEDULE_BACKUP = "reschedule_backup"
    MODIFY_PVC_QOS = "modify_pvc_qos"


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    """Normalized storage telemetry for one managed workload.

    All rates and latency measurements must describe the same observation window.
    Queue depth is represented as an average outstanding-I/O count.
    """

    platform: Platform
    resource_id: str
    timestamp: datetime
    read_iops: float
    write_iops: float
    read_latency_ms: float
    write_latency_ms: float
    throughput_mbps: float
    queue_depth: float
    p95_latency_ms: float
    p99_latency_ms: float
    current_iops_limit: int
    criticality: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.resource_id.strip():
            raise ValueError("resource_id must not be empty")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        numeric_fields = (
            self.read_iops,
            self.write_iops,
            self.read_latency_ms,
            self.write_latency_ms,
            self.throughput_mbps,
            self.queue_depth,
            self.p95_latency_ms,
            self.p99_latency_ms,
            self.criticality,
        )
        if any(not math.isfinite(value) for value in numeric_fields):
            raise ValueError("metric values must be finite")
        if any(value < 0 for value in numeric_fields):
            raise ValueError("metric values must be non-negative")
        if self.p99_latency_ms < self.p95_latency_ms:
            raise ValueError("p99_latency_ms must be >= p95_latency_ms")
        if self.current_iops_limit <= 0:
            raise ValueError("current_iops_limit must be positive")
        if not 0.1 <= self.criticality <= 10.0:
            raise ValueError("criticality must be in [0.1, 10.0]")

    @property
    def total_iops(self) -> float:
        """Return total read plus write IOPS."""

        return self.read_iops + self.write_iops

    @property
    def mean_latency_ms(self) -> float:
        """Return a weighted mean latency, falling back to a simple mean."""

        total = self.total_iops
        if total <= 0:
            return (self.read_latency_ms + self.write_latency_ms) / 2.0
        return (
            self.read_iops * self.read_latency_ms
            + self.write_iops * self.write_latency_ms
        ) / total


@dataclass(frozen=True, slots=True)
class QoSAction:
    """A policy action proposed by the agent."""

    action_type: ActionType
    platform: Platform
    resource_id: str
    magnitude: float = 0.0
    target: str | None = None
    reason: str = ""
    confidence: float = 1.0
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    requested_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        if not self.resource_id.strip():
            raise ValueError("resource_id must not be empty")
        if not math.isfinite(self.magnitude):
            raise ValueError("magnitude must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.requested_at.tzinfo is None:
            raise ValueError("requested_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class ActionResult:
    """Execution result returned by the platform adapter."""

    action_id: str
    accepted: bool
    executed: bool
    platform: Platform
    resource_id: str
    command: str
    message: str
    completed_at: datetime
    rollback_command: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SafetyPolicy:
    """Deterministic limits applied before any learned action is executed."""

    min_iops_limit: int = 500
    max_iops_limit: int = 100_000
    max_iops_step_fraction: float = 0.20
    max_queue_depth_for_migration: float = 64.0
    min_confidence: float = 0.60
    cooldown_seconds: int = 120
    allow_storage_vmotion: bool = True
    allow_backup_reschedule: bool = True
    allow_pvc_mutation: bool = True

    def __post_init__(self) -> None:
        if self.min_iops_limit <= 0:
            raise ValueError("min_iops_limit must be positive")
        if self.max_iops_limit <= self.min_iops_limit:
            raise ValueError("max_iops_limit must exceed min_iops_limit")
        if not 0.0 < self.max_iops_step_fraction <= 1.0:
            raise ValueError("max_iops_step_fraction must be in (0, 1]")
        if self.max_queue_depth_for_migration <= 0:
            raise ValueError("max_queue_depth_for_migration must be positive")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")
