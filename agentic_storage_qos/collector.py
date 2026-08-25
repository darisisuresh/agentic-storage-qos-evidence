"""Asynchronous metric collectors for VMware, Hyper-V, and Kubernetes.

The default implementations simulate the transport layer while preserving the
shape and failure modes of production collectors. Replace only the `_fetch_raw`
methods with real SDK or API calls; normalization, validation, retry behavior,
and collection orchestration remain unchanged.
"""

from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import random
from typing import Any, Iterable, Mapping, Sequence

from .models import MetricsSnapshot, Platform

LOGGER = logging.getLogger(__name__)


class CollectionError(RuntimeError):
    """Raised when one or more platform collectors cannot obtain metrics."""


@dataclass(slots=True)
class CollectorConfig:
    """Runtime settings shared by collectors."""

    request_timeout_seconds: float = 5.0
    max_attempts: int = 3
    base_backoff_seconds: float = 0.25
    jitter_fraction: float = 0.20
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_backoff_seconds < 0:
            raise ValueError("base_backoff_seconds must be non-negative")
        if not 0.0 <= self.jitter_fraction <= 1.0:
            raise ValueError("jitter_fraction must be in [0, 1]")


class BaseMetricCollector(abc.ABC):
    """Base class implementing timeout, retry, and normalization handling."""

    platform: Platform

    def __init__(
        self,
        resource_ids: Sequence[str],
        config: CollectorConfig | None = None,
    ) -> None:
        if not resource_ids:
            raise ValueError("at least one resource_id is required")
        if any(not value.strip() for value in resource_ids):
            raise ValueError("resource_ids must not contain empty values")
        self._resource_ids = tuple(resource_ids)
        self._config = config or CollectorConfig()
        self._rng = random.Random(self._config.random_seed)

    async def collect(self) -> list[MetricsSnapshot]:
        """Collect and normalize metrics for every configured resource."""

        snapshots: list[MetricsSnapshot] = []
        failures: list[str] = []

        for resource_id in self._resource_ids:
            try:
                raw = await self._collect_with_retry(resource_id)
                snapshots.append(self._normalize(resource_id, raw))
            except Exception as exc:  # defensive boundary around external APIs
                LOGGER.exception(
                    "Metric collection failed",
                    extra={"platform": self.platform.value, "resource_id": resource_id},
                )
                failures.append(f"{resource_id}: {exc}")

        if failures and not snapshots:
            raise CollectionError(
                f"{self.platform.value} collection failed: {'; '.join(failures)}"
            )

        if failures:
            LOGGER.warning(
                "Partial collection failure for %s: %s",
                self.platform.value,
                "; ".join(failures),
            )
        return snapshots

    async def _collect_with_retry(self, resource_id: str) -> Mapping[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                return await asyncio.wait_for(
                    self._fetch_raw(resource_id),
                    timeout=self._config.request_timeout_seconds,
                )
            except Exception as exc:
                last_error = exc
                if attempt == self._config.max_attempts:
                    break
                base = self._config.base_backoff_seconds * (2 ** (attempt - 1))
                jitter = base * self._config.jitter_fraction * self._rng.random()
                await asyncio.sleep(base + jitter)

        raise CollectionError(
            f"exhausted {self._config.max_attempts} attempts for {resource_id}: "
            f"{last_error}"
        )

    @abc.abstractmethod
    async def _fetch_raw(self, resource_id: str) -> Mapping[str, Any]:
        """Return platform-specific raw metrics."""

    @abc.abstractmethod
    def _normalize(
        self,
        resource_id: str,
        raw: Mapping[str, Any],
    ) -> MetricsSnapshot:
        """Convert raw metrics to the common storage telemetry schema."""

    def _bounded_gaussian(
        self,
        mean: float,
        stddev: float,
        minimum: float = 0.0,
    ) -> float:
        return max(minimum, self._rng.gauss(mean, stddev))

    def _simulate_common(
        self,
        *,
        iops_mean: float,
        latency_mean: float,
        throughput_mean: float,
        queue_mean: float,
        iops_limit: int,
    ) -> dict[str, float | int]:
        """Generate correlated metrics with occasional contention bursts."""

        burst = 1.0 + (self._rng.uniform(0.8, 2.0) if self._rng.random() < 0.12 else 0.0)
        read_iops = self._bounded_gaussian(iops_mean * 0.65 * burst, iops_mean * 0.08)
        write_iops = self._bounded_gaussian(iops_mean * 0.35 * burst, iops_mean * 0.06)
        queue_depth = self._bounded_gaussian(queue_mean * burst, queue_mean * 0.15)
        contention_multiplier = 1.0 + max(0.0, queue_depth - queue_mean) / max(queue_mean, 1.0)
        read_latency = self._bounded_gaussian(
            latency_mean * contention_multiplier,
            latency_mean * 0.12,
            0.05,
        )
        write_latency = self._bounded_gaussian(
            latency_mean * 1.20 * contention_multiplier,
            latency_mean * 0.18,
            0.05,
        )
        p95 = max(read_latency, write_latency) * self._rng.uniform(1.35, 1.75)
        p99 = p95 * self._rng.uniform(1.15, 1.55)
        throughput = self._bounded_gaussian(
            throughput_mean * min(burst, 1.7),
            throughput_mean * 0.08,
        )

        return {
            "read_iops": read_iops,
            "write_iops": write_iops,
            "read_latency_ms": read_latency,
            "write_latency_ms": write_latency,
            "throughput_mbps": throughput,
            "queue_depth": queue_depth,
            "p95_latency_ms": p95,
            "p99_latency_ms": p99,
            "current_iops_limit": iops_limit,
        }


class VMwareMetricCollector(BaseMetricCollector):
    """Mock vSphere SDK collector.

    Real integration should obtain counters from pyVmomi's PerformanceManager
    and resolve VM -> virtual disk -> datastore relationships.
    """

    platform = Platform.VMWARE

    async def _fetch_raw(self, resource_id: str) -> Mapping[str, Any]:
        await asyncio.sleep(self._rng.uniform(0.01, 0.05))
        return {
            "entity": resource_id,
            "counters": self._simulate_common(
                iops_mean=9_000,
                latency_mean=3.5,
                throughput_mean=420,
                queue_mean=14,
                iops_limit=12_000,
            ),
            "datastore": f"ds-{resource_id[-4:]}",
            "sdk": "pyVmomi.PerformanceManager",
        }

    def _normalize(
        self,
        resource_id: str,
        raw: Mapping[str, Any],
    ) -> MetricsSnapshot:
        counters = raw.get("counters")
        if not isinstance(counters, Mapping):
            raise CollectionError("VMware response missing counters")
        return _snapshot_from_mapping(
            platform=self.platform,
            resource_id=resource_id,
            values=counters,
            metadata={
                "datastore": raw.get("datastore"),
                "source": raw.get("sdk"),
            },
        )


class HyperVMetricCollector(BaseMetricCollector):
    """Mock Hyper-V WMI/PowerShell collector.

    Real integration can invoke CIM/WMI counters or signed PowerShell remoting
    commands against Hyper-V hosts and Cluster Shared Volumes.
    """

    platform = Platform.HYPERV

    async def _fetch_raw(self, resource_id: str) -> Mapping[str, Any]:
        await asyncio.sleep(self._rng.uniform(0.01, 0.05))
        sample = self._simulate_common(
            iops_mean=7_500,
            latency_mean=4.2,
            throughput_mean=360,
            queue_mean=16,
            iops_limit=10_000,
        )
        return {
            "vm_name": resource_id,
            "wmi_counters": {
                "DiskReadOperationsPerSec": sample["read_iops"],
                "DiskWriteOperationsPerSec": sample["write_iops"],
                "ReadLatencyMs": sample["read_latency_ms"],
                "WriteLatencyMs": sample["write_latency_ms"],
                "ThroughputMBps": sample["throughput_mbps"],
                "CurrentDiskQueueLength": sample["queue_depth"],
                "P95LatencyMs": sample["p95_latency_ms"],
                "P99LatencyMs": sample["p99_latency_ms"],
                "MaximumIOPS": sample["current_iops_limit"],
            },
            "csv": f"CSV-{resource_id[-3:]}",
            "source": "Get-CimInstance/Measure-VM",
        }

    def _normalize(
        self,
        resource_id: str,
        raw: Mapping[str, Any],
    ) -> MetricsSnapshot:
        counters = raw.get("wmi_counters")
        if not isinstance(counters, Mapping):
            raise CollectionError("Hyper-V response missing wmi_counters")
        remapped = {
            "read_iops": counters["DiskReadOperationsPerSec"],
            "write_iops": counters["DiskWriteOperationsPerSec"],
            "read_latency_ms": counters["ReadLatencyMs"],
            "write_latency_ms": counters["WriteLatencyMs"],
            "throughput_mbps": counters["ThroughputMBps"],
            "queue_depth": counters["CurrentDiskQueueLength"],
            "p95_latency_ms": counters["P95LatencyMs"],
            "p99_latency_ms": counters["P99LatencyMs"],
            "current_iops_limit": counters["MaximumIOPS"],
        }
        return _snapshot_from_mapping(
            platform=self.platform,
            resource_id=resource_id,
            values=remapped,
            metadata={"csv": raw.get("csv"), "source": raw.get("source")},
        )


class KubernetesMetricCollector(BaseMetricCollector):
    """Mock Prometheus/CSI collector for stateful Kubernetes workloads."""

    platform = Platform.KUBERNETES

    async def _fetch_raw(self, resource_id: str) -> Mapping[str, Any]:
        await asyncio.sleep(self._rng.uniform(0.01, 0.05))
        sample = self._simulate_common(
            iops_mean=6_000,
            latency_mean=5.0,
            throughput_mean=300,
            queue_mean=18,
            iops_limit=8_000,
        )
        namespace = "data-platform"
        return {
            "prometheus": {
                "storage_read_iops": sample["read_iops"],
                "storage_write_iops": sample["write_iops"],
                "storage_read_latency_ms": sample["read_latency_ms"],
                "storage_write_latency_ms": sample["write_latency_ms"],
                "storage_throughput_mbps": sample["throughput_mbps"],
                "storage_queue_depth": sample["queue_depth"],
                "storage_latency_p95_ms": sample["p95_latency_ms"],
                "storage_latency_p99_ms": sample["p99_latency_ms"],
                "storage_iops_limit": sample["current_iops_limit"],
            },
            "namespace": namespace,
            "pvc": resource_id,
            "storage_class": "gold-csi",
            "source": "Prometheus/CSI",
        }

    def _normalize(
        self,
        resource_id: str,
        raw: Mapping[str, Any],
    ) -> MetricsSnapshot:
        counters = raw.get("prometheus")
        if not isinstance(counters, Mapping):
            raise CollectionError("Kubernetes response missing prometheus metrics")
        remapped = {
            "read_iops": counters["storage_read_iops"],
            "write_iops": counters["storage_write_iops"],
            "read_latency_ms": counters["storage_read_latency_ms"],
            "write_latency_ms": counters["storage_write_latency_ms"],
            "throughput_mbps": counters["storage_throughput_mbps"],
            "queue_depth": counters["storage_queue_depth"],
            "p95_latency_ms": counters["storage_latency_p95_ms"],
            "p99_latency_ms": counters["storage_latency_p99_ms"],
            "current_iops_limit": counters["storage_iops_limit"],
        }
        return _snapshot_from_mapping(
            platform=self.platform,
            resource_id=resource_id,
            values=remapped,
            metadata={
                "namespace": raw.get("namespace"),
                "pvc": raw.get("pvc"),
                "storage_class": raw.get("storage_class"),
                "source": raw.get("source"),
            },
        )


def _snapshot_from_mapping(
    *,
    platform: Platform,
    resource_id: str,
    values: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> MetricsSnapshot:
    required = (
        "read_iops",
        "write_iops",
        "read_latency_ms",
        "write_latency_ms",
        "throughput_mbps",
        "queue_depth",
        "p95_latency_ms",
        "p99_latency_ms",
        "current_iops_limit",
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise CollectionError(f"missing normalized metrics: {', '.join(missing)}")

    try:
        return MetricsSnapshot(
            platform=platform,
            resource_id=resource_id,
            timestamp=datetime.now(timezone.utc),
            read_iops=float(values["read_iops"]),
            write_iops=float(values["write_iops"]),
            read_latency_ms=float(values["read_latency_ms"]),
            write_latency_ms=float(values["write_latency_ms"]),
            throughput_mbps=float(values["throughput_mbps"]),
            queue_depth=float(values["queue_depth"]),
            p95_latency_ms=float(values["p95_latency_ms"]),
            p99_latency_ms=float(values["p99_latency_ms"]),
            current_iops_limit=int(values["current_iops_limit"]),
            metadata=dict(metadata),
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise CollectionError(f"invalid metric payload: {exc}") from exc


class CollectorManager:
    """Collect telemetry concurrently from multiple platform adapters."""

    def __init__(self, collectors: Iterable[BaseMetricCollector]) -> None:
        self._collectors = tuple(collectors)
        if not self._collectors:
            raise ValueError("at least one collector is required")

    async def collect_all(self) -> list[MetricsSnapshot]:
        """Return all successful snapshots or raise when every collector fails."""

        results = await asyncio.gather(
            *(collector.collect() for collector in self._collectors),
            return_exceptions=True,
        )

        snapshots: list[MetricsSnapshot] = []
        errors: list[str] = []
        for collector, result in zip(self._collectors, results, strict=True):
            if isinstance(result, Exception):
                errors.append(f"{collector.platform.value}: {result}")
            else:
                snapshots.extend(result)

        if not snapshots:
            raise CollectionError(
                "all metric collectors failed: " + "; ".join(errors)
            )
        if errors:
            LOGGER.warning("Some collectors failed: %s", "; ".join(errors))
        return snapshots
