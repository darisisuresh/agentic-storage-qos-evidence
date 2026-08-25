from datetime import datetime, timezone

from agentic_storage_qos.executor import ActionExecutor, ExecutorConfig
from agentic_storage_qos.models import (
    ActionType,
    MetricsSnapshot,
    Platform,
    QoSAction,
    SafetyPolicy,
)


def vmware_snapshot() -> MetricsSnapshot:
    return MetricsSnapshot(
        platform=Platform.VMWARE,
        resource_id="vm-db-01",
        timestamp=datetime.now(timezone.utc),
        read_iops=7000,
        write_iops=2000,
        read_latency_ms=5,
        write_latency_ms=6,
        throughput_mbps=350,
        queue_depth=18,
        p95_latency_ms=11,
        p99_latency_ms=22,
        current_iops_limit=10_000,
    )


def test_executor_generates_vmware_qos_command() -> None:
    policy = SafetyPolicy(cooldown_seconds=0)
    executor = ActionExecutor(policy, ExecutorConfig(dry_run=True))
    action = QoSAction(
        action_type=ActionType.INCREASE_IOPS_LIMIT,
        platform=Platform.VMWARE,
        resource_id="vm-db-01",
        magnitude=2000,
        confidence=0.9,
    )
    result = executor.execute(action, vmware_snapshot())
    assert result.accepted
    assert "Set-SpbmStoragePolicy" in result.command
    assert result.metadata["new_iops_limit"] == 12_000


def test_executor_rejects_low_confidence() -> None:
    policy = SafetyPolicy(min_confidence=0.8, cooldown_seconds=0)
    executor = ActionExecutor(policy, ExecutorConfig(dry_run=True))
    action = QoSAction(
        action_type=ActionType.DECREASE_IOPS_LIMIT,
        platform=Platform.VMWARE,
        resource_id="vm-db-01",
        magnitude=-1000,
        confidence=0.5,
    )
    result = executor.execute(action, vmware_snapshot())
    assert not result.accepted
    assert "confidence" in result.message.lower()
