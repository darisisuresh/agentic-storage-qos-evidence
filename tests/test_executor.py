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


def test_post_action_failure_executes_rollback() -> None:
    commands = []

    def runner(command: str, timeout: float):
        commands.append(command)
        return 0, "ok", ""

    executor = ActionExecutor(
        SafetyPolicy(cooldown_seconds=0),
        ExecutorConfig(dry_run=False, verify_after_execution=True),
        command_runner=runner,
        state_verifier=lambda action, snapshot, plan: False,
    )
    result = executor.execute(
        QoSAction(
            action_type=ActionType.INCREASE_IOPS_LIMIT,
            platform=Platform.VMWARE,
            resource_id="vm-db-01",
            magnitude=1000,
            confidence=0.95,
        ),
        vmware_snapshot(),
    )
    assert result.executed
    assert result.metadata["rollback_executed"] is True
    assert len(commands) == 2
    assert "IopsLimit 10000" in commands[1]


def test_failed_rollback_raises() -> None:
    calls = 0

    def runner(command: str, timeout: float):
        nonlocal calls
        calls += 1
        return (0, "ok", "") if calls == 1 else (7, "", "rollback error")

    executor = ActionExecutor(
        SafetyPolicy(cooldown_seconds=0),
        ExecutorConfig(dry_run=False, verify_after_execution=True),
        command_runner=runner,
        state_verifier=lambda action, snapshot, plan: False,
    )
    import pytest
    with pytest.raises(Exception, match="rollback failed"):
        executor.execute(
            QoSAction(
                action_type=ActionType.INCREASE_IOPS_LIMIT,
                platform=Platform.VMWARE,
                resource_id="vm-db-01",
                magnitude=1000,
                confidence=0.95,
            ),
            vmware_snapshot(),
        )
