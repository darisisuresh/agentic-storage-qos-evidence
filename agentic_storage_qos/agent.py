"""Predictive DQN agent for storage QoS optimization.

The implementation uses a small NumPy neural network to keep the framework
portable. It includes replay memory, target-network synchronization, Huber
loss, gradient clipping, epsilon-greedy exploration, action masking, model
serialization, and a tail-latency-aware reward function.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import random
from typing import Deque, Iterable, Sequence

import numpy as np

from .models import ActionType, MetricsSnapshot, Platform, QoSAction, SafetyPolicy

LOGGER = logging.getLogger(__name__)


class AgentError(RuntimeError):
    """Raised for invalid agent state, model data, or training input."""


@dataclass(slots=True)
class AgentConfig:
    """Hyperparameters and normalization limits for the DQN agent."""

    state_size: int = 12
    hidden_size: int = 64
    learning_rate: float = 1e-3
    discount_factor: float = 0.97
    epsilon_start: float = 0.20
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 32
    replay_capacity: int = 20_000
    target_sync_interval: int = 100
    gradient_clip_norm: float = 5.0
    p95_slo_ms: float = 10.0
    p99_slo_ms: float = 20.0
    queue_safe_depth: float = 24.0
    throughput_reference_mbps: float = 500.0
    random_seed: int = 17

    def __post_init__(self) -> None:
        positive = (
            self.state_size,
            self.hidden_size,
            self.learning_rate,
            self.discount_factor,
            self.batch_size,
            self.replay_capacity,
            self.target_sync_interval,
            self.gradient_clip_norm,
            self.p95_slo_ms,
            self.p99_slo_ms,
            self.queue_safe_depth,
            self.throughput_reference_mbps,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("agent configuration values must be positive")
        if not 0 < self.discount_factor <= 1:
            raise ValueError("discount_factor must be in (0, 1]")
        if not 0 <= self.epsilon_min <= self.epsilon_start <= 1:
            raise ValueError(
                "epsilon values must satisfy 0 <= min <= start <= 1"
            )
        if not 0 < self.epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class Experience:
    """One replay-buffer transition."""

    state: np.ndarray
    action_index: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_action_mask: np.ndarray


class ReplayBuffer:
    """Bounded random replay memory."""

    def __init__(self, capacity: int, rng: random.Random) -> None:
        self._items: Deque[Experience] = deque(maxlen=capacity)
        self._rng = rng

    def append(self, experience: Experience) -> None:
        self._items.append(experience)

    def sample(self, size: int) -> list[Experience]:
        if size > len(self._items):
            raise AgentError(
                f"cannot sample {size} experiences from buffer of {len(self._items)}"
            )
        return self._rng.sample(list(self._items), size)

    def __len__(self) -> int:
        return len(self._items)


class DenseQNetwork:
    """Two-layer fully connected Q-network with manual backpropagation."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        rng: np.random.Generator,
    ) -> None:
        scale1 = np.sqrt(2.0 / input_size)
        scale2 = np.sqrt(2.0 / hidden_size)
        self.w1 = rng.normal(0.0, scale1, (input_size, hidden_size))
        self.b1 = np.zeros(hidden_size, dtype=np.float64)
        self.w2 = rng.normal(0.0, scale2, (hidden_size, output_size))
        self.b2 = np.zeros(output_size, dtype=np.float64)

    def predict(self, states: np.ndarray) -> np.ndarray:
        states = _ensure_2d(states)
        hidden = np.maximum(states @ self.w1 + self.b1, 0.0)
        return hidden @ self.w2 + self.b2

    def train_batch(
        self,
        states: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
        gradient_clip_norm: float,
    ) -> float:
        states = _ensure_2d(states)
        targets = _ensure_2d(targets)
        if states.shape[0] != targets.shape[0]:
            raise AgentError("states and targets must contain the same batch size")

        z1 = states @ self.w1 + self.b1
        hidden = np.maximum(z1, 0.0)
        predictions = hidden @ self.w2 + self.b2
        error = predictions - targets

        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, 1.0)
        linear = abs_error - quadratic
        loss = np.mean(0.5 * quadratic**2 + linear)

        grad_output = np.where(abs_error <= 1.0, error, np.sign(error))
        grad_output /= states.shape[0]

        grad_w2 = hidden.T @ grad_output
        grad_b2 = np.sum(grad_output, axis=0)
        grad_hidden = grad_output @ self.w2.T
        grad_z1 = grad_hidden * (z1 > 0)
        grad_w1 = states.T @ grad_z1
        grad_b1 = np.sum(grad_z1, axis=0)

        gradients = [grad_w1, grad_b1, grad_w2, grad_b2]
        total_norm = float(
            np.sqrt(sum(float(np.sum(grad * grad)) for grad in gradients))
        )
        if total_norm > gradient_clip_norm:
            scale = gradient_clip_norm / (total_norm + 1e-12)
            gradients = [grad * scale for grad in gradients]
            grad_w1, grad_b1, grad_w2, grad_b2 = gradients

        self.w1 -= learning_rate * grad_w1
        self.b1 -= learning_rate * grad_b1
        self.w2 -= learning_rate * grad_w2
        self.b2 -= learning_rate * grad_b2
        return float(loss)

    def copy_from(self, other: "DenseQNetwork") -> None:
        self.w1 = other.w1.copy()
        self.b1 = other.b1.copy()
        self.w2 = other.w2.copy()
        self.b2 = other.b2.copy()

    def to_dict(self) -> dict[str, list]:
        return {
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
        }

    def load_dict(self, payload: dict[str, list]) -> None:
        try:
            self.w1 = np.asarray(payload["w1"], dtype=np.float64)
            self.b1 = np.asarray(payload["b1"], dtype=np.float64)
            self.w2 = np.asarray(payload["w2"], dtype=np.float64)
            self.b2 = np.asarray(payload["b2"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise AgentError(f"invalid model payload: {exc}") from exc


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim != 2:
        raise AgentError("network inputs must be one- or two-dimensional")
    return array


class DQNAgent:
    """Tail-latency-aware DQN policy with deterministic safety masking."""

    ACTIONS: tuple[ActionType, ...] = (
        ActionType.NOOP,
        ActionType.INCREASE_IOPS_LIMIT,
        ActionType.DECREASE_IOPS_LIMIT,
        ActionType.STORAGE_VMOTION,
        ActionType.RESCHEDULE_BACKUP,
        ActionType.MODIFY_PVC_QOS,
    )

    def __init__(
        self,
        config: AgentConfig | None = None,
        safety_policy: SafetyPolicy | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.safety_policy = safety_policy or SafetyPolicy()
        if self.config.state_size != 12:
            raise ValueError("the current feature encoder requires state_size=12")

        self._py_rng = random.Random(self.config.random_seed)
        self._np_rng = np.random.default_rng(self.config.random_seed)
        self._online = DenseQNetwork(
            self.config.state_size,
            self.config.hidden_size,
            len(self.ACTIONS),
            self._np_rng,
        )
        self._target = DenseQNetwork(
            self.config.state_size,
            self.config.hidden_size,
            len(self.ACTIONS),
            self._np_rng,
        )
        self._target.copy_from(self._online)
        self._replay = ReplayBuffer(self.config.replay_capacity, self._py_rng)
        self._epsilon = self.config.epsilon_start
        self._train_steps = 0

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def encode_state(
        self,
        current: MetricsSnapshot,
        previous: MetricsSnapshot | None = None,
    ) -> np.ndarray:
        """Convert telemetry to a bounded, platform-neutral state vector."""

        if previous and (
            previous.resource_id != current.resource_id
            or previous.platform != current.platform
        ):
            raise AgentError("previous snapshot must describe the same resource")

        total_iops_ratio = current.total_iops / max(current.current_iops_limit, 1)
        read_ratio = current.read_iops / max(current.total_iops, 1.0)
        latency_delta = (
            current.p99_latency_ms - previous.p99_latency_ms
            if previous
            else 0.0
        )
        queue_delta = (
            current.queue_depth - previous.queue_depth
            if previous
            else 0.0
        )
        throughput_delta = (
            current.throughput_mbps - previous.throughput_mbps
            if previous
            else 0.0
        )

        platform_one_hot = {
            Platform.VMWARE: (1.0, 0.0, 0.0),
            Platform.HYPERV: (0.0, 1.0, 0.0),
            Platform.KUBERNETES: (0.0, 0.0, 1.0),
        }[current.platform]

        vector = np.asarray(
            [
                np.clip(total_iops_ratio, 0.0, 4.0) / 4.0,
                np.clip(read_ratio, 0.0, 1.0),
                np.clip(
                    current.p95_latency_ms / self.config.p95_slo_ms,
                    0.0,
                    5.0,
                )
                / 5.0,
                np.clip(
                    current.p99_latency_ms / self.config.p99_slo_ms,
                    0.0,
                    5.0,
                )
                / 5.0,
                np.clip(
                    current.queue_depth / self.config.queue_safe_depth,
                    0.0,
                    5.0,
                )
                / 5.0,
                np.clip(
                    current.throughput_mbps
                    / self.config.throughput_reference_mbps,
                    0.0,
                    4.0,
                )
                / 4.0,
                np.tanh(latency_delta / max(self.config.p99_slo_ms, 1.0)),
                np.tanh(queue_delta / max(self.config.queue_safe_depth, 1.0)),
                np.tanh(
                    throughput_delta
                    / max(self.config.throughput_reference_mbps, 1.0)
                ),
                *platform_one_hot,
            ],
            dtype=np.float64,
        )
        if vector.shape != (self.config.state_size,):
            raise AgentError(f"unexpected state shape: {vector.shape}")
        return vector

    def action_mask(self, snapshot: MetricsSnapshot) -> np.ndarray:
        """Return a boolean mask of actions allowed for the current resource."""

        mask = np.ones(len(self.ACTIONS), dtype=bool)

        if snapshot.current_iops_limit >= self.safety_policy.max_iops_limit:
            mask[self.ACTIONS.index(ActionType.INCREASE_IOPS_LIMIT)] = False
        if snapshot.current_iops_limit <= self.safety_policy.min_iops_limit:
            mask[self.ACTIONS.index(ActionType.DECREASE_IOPS_LIMIT)] = False
        if (
            not self.safety_policy.allow_storage_vmotion
            or snapshot.platform == Platform.KUBERNETES
            or snapshot.queue_depth
            > self.safety_policy.max_queue_depth_for_migration
        ):
            mask[self.ACTIONS.index(ActionType.STORAGE_VMOTION)] = False
        if not self.safety_policy.allow_backup_reschedule:
            mask[self.ACTIONS.index(ActionType.RESCHEDULE_BACKUP)] = False
        if (
            not self.safety_policy.allow_pvc_mutation
            or snapshot.platform != Platform.KUBERNETES
        ):
            mask[self.ACTIONS.index(ActionType.MODIFY_PVC_QOS)] = False

        mask[self.ACTIONS.index(ActionType.NOOP)] = True
        return mask

    def select_action(
        self,
        snapshot: MetricsSnapshot,
        previous: MetricsSnapshot | None = None,
        training: bool = False,
    ) -> QoSAction:
        """Select a QoS action using masked epsilon-greedy inference."""

        state = self.encode_state(snapshot, previous)
        mask = self.action_mask(snapshot)
        allowed = np.flatnonzero(mask)
        if allowed.size == 0:
            raise AgentError("action mask excluded every action")

        if training and self._py_rng.random() < self._epsilon:
            action_index = int(self._py_rng.choice(allowed.tolist()))
            confidence = 0.5
        else:
            q_values = self._online.predict(state)[0]
            masked = np.where(mask, q_values, -np.inf)
            action_index = int(np.argmax(masked))
            finite = q_values[mask]
            spread = float(np.std(finite)) if finite.size > 1 else 0.0
            confidence = float(
                np.clip(0.5 + abs(masked[action_index]) / (1.0 + spread) * 0.1, 0.5, 0.99)
            )

        action_type = self.ACTIONS[action_index]
        magnitude, target = self._derive_parameters(action_type, snapshot)
        reason = self._build_reason(action_type, snapshot)
        return QoSAction(
            action_type=action_type,
            platform=snapshot.platform,
            resource_id=snapshot.resource_id,
            magnitude=magnitude,
            target=target,
            reason=reason,
            confidence=confidence,
        )

    def _derive_parameters(
        self,
        action_type: ActionType,
        snapshot: MetricsSnapshot,
    ) -> tuple[float, str | None]:
        step = max(
            100.0,
            snapshot.current_iops_limit
            * self.safety_policy.max_iops_step_fraction,
        )
        if action_type == ActionType.INCREASE_IOPS_LIMIT:
            return step, None
        if action_type == ActionType.DECREASE_IOPS_LIMIT:
            return -step, None
        if action_type == ActionType.STORAGE_VMOTION:
            return 0.0, "datastore-low-contention"
        if action_type == ActionType.RESCHEDULE_BACKUP:
            return 30.0, "delay-minutes"
        if action_type == ActionType.MODIFY_PVC_QOS:
            return step, "gold-csi-expanded"
        return 0.0, None

    def _build_reason(
        self,
        action_type: ActionType,
        snapshot: MetricsSnapshot,
    ) -> str:
        return (
            f"action={action_type.value}; p95={snapshot.p95_latency_ms:.2f}ms; "
            f"p99={snapshot.p99_latency_ms:.2f}ms; "
            f"queue={snapshot.queue_depth:.2f}; "
            f"throughput={snapshot.throughput_mbps:.2f}MB/s"
        )

    def compute_reward(
        self,
        previous: MetricsSnapshot,
        current: MetricsSnapshot,
        action: QoSAction,
    ) -> float:
        """Calculate the paper's normalized tail-latency reward.

        R_t penalizes P95/P99 SLO exceedance, queue saturation, and disruptive
        actions while rewarding positive throughput change.
        """

        if (
            previous.resource_id != current.resource_id
            or previous.platform != current.platform
        ):
            raise AgentError("reward snapshots must describe the same resource")

        p95_violation = max(
            0.0,
            (current.p95_latency_ms - self.config.p95_slo_ms)
            / self.config.p95_slo_ms,
        )
        p99_violation = max(
            0.0,
            (current.p99_latency_ms - self.config.p99_slo_ms)
            / self.config.p99_slo_ms,
        )
        queue_violation = max(
            0.0,
            (current.queue_depth - self.config.queue_safe_depth)
            / self.config.queue_safe_depth,
        )
        throughput_delta = (
            current.throughput_mbps - previous.throughput_mbps
        ) / self.config.throughput_reference_mbps

        action_costs = {
            ActionType.NOOP: 0.0,
            ActionType.INCREASE_IOPS_LIMIT: 0.02,
            ActionType.DECREASE_IOPS_LIMIT: 0.02,
            ActionType.STORAGE_VMOTION: 0.25,
            ActionType.RESCHEDULE_BACKUP: 0.05,
            ActionType.MODIFY_PVC_QOS: 0.04,
        }

        reward = (
            -1.0 * p95_violation
            -2.5 * p99_violation
            -1.2 * queue_violation
            +0.8 * throughput_delta
            -action_costs[action.action_type]
        )
        return float(current.criticality * reward)

    def remember(
        self,
        state: np.ndarray,
        action: QoSAction,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray,
    ) -> None:
        """Store a validated transition in replay memory."""

        try:
            action_index = self.ACTIONS.index(action.action_type)
        except ValueError as exc:
            raise AgentError(f"unsupported action: {action.action_type}") from exc

        state = np.asarray(state, dtype=np.float64)
        next_state = np.asarray(next_state, dtype=np.float64)
        next_action_mask = np.asarray(next_action_mask, dtype=bool)
        if state.shape != (self.config.state_size,):
            raise AgentError("state has an invalid shape")
        if next_state.shape != (self.config.state_size,):
            raise AgentError("next_state has an invalid shape")
        if next_action_mask.shape != (len(self.ACTIONS),):
            raise AgentError("next_action_mask has an invalid shape")

        self._replay.append(
            Experience(
                state=state.copy(),
                action_index=action_index,
                reward=float(reward),
                next_state=next_state.copy(),
                done=bool(done),
                next_action_mask=next_action_mask.copy(),
            )
        )

    def train_step(self) -> float | None:
        """Run one DQN update and return the Huber loss.

        Returns ``None`` until the replay buffer contains a complete batch.
        """

        if len(self._replay) < self.config.batch_size:
            return None

        batch = self._replay.sample(self.config.batch_size)
        states = np.stack([item.state for item in batch])
        next_states = np.stack([item.next_state for item in batch])

        current_q = self._online.predict(states)
        target_q = current_q.copy()
        next_online_q = self._online.predict(next_states)
        next_target_q = self._target.predict(next_states)

        for row, item in enumerate(batch):
            if item.done:
                bootstrap = 0.0
            else:
                masked_online = np.where(
                    item.next_action_mask,
                    next_online_q[row],
                    -np.inf,
                )
                next_index = int(np.argmax(masked_online))
                bootstrap = next_target_q[row, next_index]

            target_q[row, item.action_index] = (
                item.reward + self.config.discount_factor * bootstrap
            )

        loss = self._online.train_batch(
            states,
            target_q,
            self.config.learning_rate,
            self.config.gradient_clip_norm,
        )
        self._train_steps += 1
        self._epsilon = max(
            self.config.epsilon_min,
            self._epsilon * self.config.epsilon_decay,
        )

        if self._train_steps % self.config.target_sync_interval == 0:
            self._target.copy_from(self._online)

        return loss

    def train_offline(
        self,
        transitions: Iterable[
            tuple[
                MetricsSnapshot,
                QoSAction,
                MetricsSnapshot,
                bool,
            ]
        ],
        epochs: int = 1,
    ) -> list[float]:
        """Populate replay memory from historical transitions and train."""

        if epochs < 1:
            raise ValueError("epochs must be >= 1")

        losses: list[float] = []
        cached = list(transitions)
        for _ in range(epochs):
            for previous, action, current, done in cached:
                state = self.encode_state(previous)
                next_state = self.encode_state(current, previous)
                reward = self.compute_reward(previous, current, action)
                self.remember(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    self.action_mask(current),
                )
                loss = self.train_step()
                if loss is not None:
                    losses.append(loss)
        return losses

    def save(self, path: str | Path) -> None:
        """Serialize model weights and agent state to JSON."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "epsilon": self._epsilon,
            "train_steps": self._train_steps,
            "online": self._online.to_dict(),
            "target": self._target.to_dict(),
        }
        destination.write_text(json.dumps(payload), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        """Load model weights from a file created by :meth:`save`."""

        source = Path(path)
        if not source.is_file():
            raise AgentError(f"model file does not exist: {source}")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            self._online.load_dict(payload["online"])
            self._target.load_dict(payload["target"])
            self._epsilon = float(payload["epsilon"])
            self._train_steps = int(payload["train_steps"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AgentError(f"invalid model file: {exc}") from exc
