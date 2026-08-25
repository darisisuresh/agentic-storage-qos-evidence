"""Safety-gated action execution engine.

The engine emits realistic mock commands for vSphere, Hyper-V, and Kubernetes.
Production adapters can execute the same validated execution plans through
pyVmomi, PowerShell remoting, or the Kubernetes API.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import shlex
import threading
from typing import Callable

from .models import (
    ActionResult,
    ActionType,
    MetricsSnapshot,
    Platform,
    QoSAction,
    SafetyPolicy,
)

LOGGER = logging.getLogger(__name__)


class ExecutionError(RuntimeError):
    """Raised when an action cannot be validated or rendered."""


@dataclass(slots=True)
class ExecutorConfig:
    """Execution behavior for mock and production modes."""

    dry_run: bool = True
    verify_after_execution: bool = True
    command_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.command_timeout_seconds <= 0:
            raise ValueError("command_timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Validated platform command plus rollback metadata."""

    command: str
    rollback_command: str | None
    new_iops_limit: int | None = None


class ActionExecutor:
    """Validate, render, and execute storage-policy actions safely."""

    def __init__(
        self,
        safety_policy: SafetyPolicy | None = None,
        config: ExecutorConfig | None = None,
        command_runner: Callable[[str, float], tuple[int, str, str]] | None = None,
    ) -> None:
        self.safety_policy = safety_policy or SafetyPolicy()
        self.config = config or ExecutorConfig()
        self._command_runner = command_runner
        self._last_execution: dict[tuple[Platform, str], datetime] = {}
        self._lock = threading.Lock()

    def execute(
        self,
        action: QoSAction,
        snapshot: MetricsSnapshot,
    ) -> ActionResult:
        """Execute or dry-run a policy action after deterministic validation."""

        if action.platform != snapshot.platform:
            raise ExecutionError("action platform does not match snapshot platform")
        if action.resource_id != snapshot.resource_id:
            raise ExecutionError("action resource does not match snapshot resource")

        rejection = self._validate(action, snapshot)
        if rejection:
            return ActionResult(
                action_id=action.action_id,
                accepted=False,
                executed=False,
                platform=action.platform,
                resource_id=action.resource_id,
                command="",
                message=rejection,
                completed_at=datetime.now(timezone.utc),
            )

        plan = self._build_plan(action, snapshot)

        if self.config.dry_run or action.action_type == ActionType.NOOP:
            executed = action.action_type == ActionType.NOOP
            self._mark_execution(action)
            return ActionResult(
                action_id=action.action_id,
                accepted=True,
                executed=executed,
                platform=action.platform,
                resource_id=action.resource_id,
                command=plan.command,
                rollback_command=plan.rollback_command,
                message=(
                    "No operation required"
                    if action.action_type == ActionType.NOOP
                    else "Dry-run command generated; no external change executed"
                ),
                completed_at=datetime.now(timezone.utc),
                metadata={"new_iops_limit": plan.new_iops_limit},
            )

        if self._command_runner is None:
            raise ExecutionError(
                "dry_run=False requires a command_runner implementation"
            )

        code, stdout, stderr = self._command_runner(
            plan.command,
            self.config.command_timeout_seconds,
        )
        if code != 0:
            raise ExecutionError(
                f"platform command failed with exit code {code}: {stderr.strip()}"
            )

        self._mark_execution(action)
        return ActionResult(
            action_id=action.action_id,
            accepted=True,
            executed=True,
            platform=action.platform,
            resource_id=action.resource_id,
            command=plan.command,
            rollback_command=plan.rollback_command,
            message="Command executed successfully",
            completed_at=datetime.now(timezone.utc),
            metadata={
                "stdout": stdout,
                "stderr": stderr,
                "new_iops_limit": plan.new_iops_limit,
            },
        )

    def _validate(
        self,
        action: QoSAction,
        snapshot: MetricsSnapshot,
    ) -> str | None:
        if action.confidence < self.safety_policy.min_confidence:
            return (
                f"Rejected: confidence {action.confidence:.2f} is below "
                f"{self.safety_policy.min_confidence:.2f}"
            )

        key = (action.platform, action.resource_id)
        with self._lock:
            last = self._last_execution.get(key)
        if last is not None:
            remaining = timedelta(seconds=self.safety_policy.cooldown_seconds) - (
                datetime.now(timezone.utc) - last
            )
            if remaining.total_seconds() > 0:
                return f"Rejected: cooldown active for {remaining.total_seconds():.0f}s"

        if (
            action.action_type == ActionType.STORAGE_VMOTION
            and not self.safety_policy.allow_storage_vmotion
        ):
            return "Rejected: storage vMotion is disabled by policy"

        if (
            action.action_type == ActionType.STORAGE_VMOTION
            and snapshot.queue_depth
            > self.safety_policy.max_queue_depth_for_migration
        ):
            return "Rejected: queue depth is too high for migration"

        if (
            action.action_type == ActionType.RESCHEDULE_BACKUP
            and not self.safety_policy.allow_backup_reschedule
        ):
            return "Rejected: backup rescheduling is disabled by policy"

        if (
            action.action_type == ActionType.MODIFY_PVC_QOS
            and (
                action.platform != Platform.KUBERNETES
                or not self.safety_policy.allow_pvc_mutation
            )
        ):
            return "Rejected: PVC QoS mutation is not allowed"

        if (
            action.action_type == ActionType.STORAGE_VMOTION
            and action.platform == Platform.KUBERNETES
        ):
            return "Rejected: storage vMotion is not valid for Kubernetes"

        return None

    def _build_plan(
        self,
        action: QoSAction,
        snapshot: MetricsSnapshot,
    ) -> ExecutionPlan:
        if action.action_type == ActionType.NOOP:
            return ExecutionPlan(command="# noop", rollback_command=None)

        if action.action_type in {
            ActionType.INCREASE_IOPS_LIMIT,
            ActionType.DECREASE_IOPS_LIMIT,
        }:
            proposed = int(round(snapshot.current_iops_limit + action.magnitude))
            bounded = min(
                self.safety_policy.max_iops_limit,
                max(self.safety_policy.min_iops_limit, proposed),
            )
            allowed_delta = int(
                snapshot.current_iops_limit
                * self.safety_policy.max_iops_step_fraction
            )
            actual_delta = bounded - snapshot.current_iops_limit
            if abs(actual_delta) > max(allowed_delta, 1):
                raise ExecutionError(
                    "requested IOPS change exceeds max_iops_step_fraction"
                )
            return self._qos_limit_plan(action, snapshot, bounded)

        if action.action_type == ActionType.STORAGE_VMOTION:
            if not action.target:
                raise ExecutionError("storage vMotion requires a target datastore")
            vm = shlex.quote(action.resource_id)
            datastore = shlex.quote(action.target)
            command = (
                f"Move-VM -VM {vm} -Datastore {datastore} "
                "-Confirm:$false -RunAsync"
            )
            rollback = (
                f"# Rollback requires the original datastore recorded before "
                f"moving {vm}"
            )
            return ExecutionPlan(command=command, rollback_command=rollback)

        if action.action_type == ActionType.RESCHEDULE_BACKUP:
            delay = max(1, int(round(abs(action.magnitude))))
            command = (
                f"Set-BackupSchedule -Resource "
                f"{shlex.quote(action.resource_id)} -DelayMinutes {delay}"
            )
            rollback = (
                f"Set-BackupSchedule -Resource "
                f"{shlex.quote(action.resource_id)} -RestorePrevious"
            )
            return ExecutionPlan(command=command, rollback_command=rollback)

        if action.action_type == ActionType.MODIFY_PVC_QOS:
            if action.platform != Platform.KUBERNETES:
                raise ExecutionError("PVC QoS action requires Kubernetes")
            proposed = int(round(snapshot.current_iops_limit + action.magnitude))
            bounded = min(
                self.safety_policy.max_iops_limit,
                max(self.safety_policy.min_iops_limit, proposed),
            )
            namespace = str(snapshot.metadata.get("namespace", "default"))
            pvc = shlex.quote(action.resource_id)
            command = (
                "kubectl patch pvc "
                f"{pvc} -n {shlex.quote(namespace)} --type merge "
                f"""-p '{{"metadata":{{"annotations":{{"storage.example.com/iops-limit":"{bounded}"}}}}}}'"""
            )
            rollback = (
                "kubectl patch pvc "
                f"{pvc} -n {shlex.quote(namespace)} --type merge "
                f"""-p '{{"metadata":{{"annotations":{{"storage.example.com/iops-limit":"{snapshot.current_iops_limit}"}}}}}}'"""
            )
            return ExecutionPlan(
                command=command,
                rollback_command=rollback,
                new_iops_limit=bounded,
            )

        raise ExecutionError(f"unsupported action: {action.action_type.value}")

    def _qos_limit_plan(
        self,
        action: QoSAction,
        snapshot: MetricsSnapshot,
        new_limit: int,
    ) -> ExecutionPlan:
        resource = shlex.quote(action.resource_id)

        if action.platform == Platform.VMWARE:
            command = (
                "Set-SpbmStoragePolicy "
                f"-Entity {resource} -IopsLimit {new_limit} -Confirm:$false"
            )
            rollback = (
                "Set-SpbmStoragePolicy "
                f"-Entity {resource} -IopsLimit {snapshot.current_iops_limit} "
                "-Confirm:$false"
            )
        elif action.platform == Platform.HYPERV:
            command = (
                f"Set-VMHardDiskDrive -VMName {resource} "
                f"-MaximumIOPS {new_limit}"
            )
            rollback = (
                f"Set-VMHardDiskDrive -VMName {resource} "
                f"-MaximumIOPS {snapshot.current_iops_limit}"
            )
        elif action.platform == Platform.KUBERNETES:
            namespace = str(snapshot.metadata.get("namespace", "default"))
            command = (
                "kubectl patch pvc "
                f"{resource} -n {shlex.quote(namespace)} --type merge "
                f"""-p '{{"metadata":{{"annotations":{{"storage.example.com/iops-limit":"{new_limit}"}}}}}}'"""
            )
            rollback = (
                "kubectl patch pvc "
                f"{resource} -n {shlex.quote(namespace)} --type merge "
                f"""-p '{{"metadata":{{"annotations":{{"storage.example.com/iops-limit":"{snapshot.current_iops_limit}"}}}}}}'"""
            )
        else:  # pragma: no cover - exhaustive Enum handling
            raise ExecutionError(f"unsupported platform: {action.platform}")

        return ExecutionPlan(
            command=command,
            rollback_command=rollback,
            new_iops_limit=new_limit,
        )

    def _mark_execution(self, action: QoSAction) -> None:
        if action.action_type == ActionType.NOOP:
            return
        with self._lock:
            self._last_execution[(action.platform, action.resource_id)] = (
                datetime.now(timezone.utc)
            )
