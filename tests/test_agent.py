from datetime import datetime, timezone

import numpy as np

from agentic_storage_qos.agent import AgentConfig, DQNAgent
from agentic_storage_qos.models import (
    ActionType,
    MetricsSnapshot,
    Platform,
    QoSAction,
    SafetyPolicy,
)


def snapshot(
    *,
    p95: float = 9.0,
    p99: float = 15.0,
    queue: float = 12.0,
    throughput: float = 400.0,
    limit: int = 10_000,
) -> MetricsSnapshot:
    return MetricsSnapshot(
        platform=Platform.VMWARE,
        resource_id="vm-01",
        timestamp=datetime.now(timezone.utc),
        read_iops=5000,
        write_iops=2500,
        read_latency_ms=3.0,
        write_latency_ms=4.0,
        throughput_mbps=throughput,
        queue_depth=queue,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        current_iops_limit=limit,
    )


def test_state_encoding_shape_and_bounds() -> None:
    agent = DQNAgent(AgentConfig(random_seed=1))
    state = agent.encode_state(snapshot())
    assert state.shape == (12,)
    assert np.all(np.isfinite(state))
    assert np.all(state <= 1.0)
    assert np.all(state >= -1.0)


def test_reward_penalizes_tail_latency() -> None:
    agent = DQNAgent(AgentConfig(random_seed=1))
    previous = snapshot(p95=8, p99=12, queue=10, throughput=350)
    good = snapshot(p95=8, p99=12, queue=10, throughput=380)
    bad = snapshot(p95=30, p99=60, queue=50, throughput=300)
    noop = QoSAction(
        action_type=ActionType.NOOP,
        platform=Platform.VMWARE,
        resource_id="vm-01",
    )
    assert agent.compute_reward(previous, good, noop) > agent.compute_reward(
        previous, bad, noop
    )


def test_action_mask_blocks_kubernetes_only_action_on_vmware() -> None:
    agent = DQNAgent(
        AgentConfig(random_seed=1),
        SafetyPolicy(cooldown_seconds=0),
    )
    mask = agent.action_mask(snapshot())
    index = agent.ACTIONS.index(ActionType.MODIFY_PVC_QOS)
    assert not mask[index]



def test_model_save_and_load(tmp_path) -> None:
    agent = DQNAgent(AgentConfig(random_seed=3))
    model_path = tmp_path / "agent.json"
    before = agent._online.predict(agent.encode_state(snapshot())).copy()
    agent.save(model_path)

    restored = DQNAgent(AgentConfig(random_seed=99))
    restored.load(model_path)
    after = restored._online.predict(restored.encode_state(snapshot()))
    assert np.allclose(before, after)
