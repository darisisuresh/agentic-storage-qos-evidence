import numpy as np

from benchmarks.run_validation import MODEL_SPEC, simulate, workload


def test_workload_is_seed_reproducible():
    a, ac = workload(7, 1.0)
    b, bc = workload(7, 1.0)
    assert np.array_equal(a, b)
    assert np.array_equal(ac, bc)


def test_predictive_improves_tail_in_nominal_case():
    static = [simulate(seed, "static", "nominal", 1.0) for seed in range(5)]
    predictive = [simulate(seed, "predictive", "nominal", 1.0) for seed in range(5)]
    assert np.mean([r.p99_ms for r in predictive]) < np.mean([r.p99_ms for r in static])
    assert np.mean([r.violation_pct for r in predictive]) < np.mean([r.violation_pct for r in static])


def test_platform_shift_changes_only_requested_platform_demand():
    nominal, capacity = workload(7, 1.0)
    shifted, shifted_capacity = workload(7, 1.0, platform_shift="vmware")
    assert shifted[:, 0].mean() > nominal[:, 0].mean() * 1.12
    assert np.array_equal(shifted[:, 1:], nominal[:, 1:])
    assert np.array_equal(shifted_capacity, capacity)


def test_published_synthetic_model_constants_are_pinned():
    assert MODEL_SPEC == {
        "backlog_request_weight": 0.10,
        "backlog_calibration_divisor": 110.0,
        "backlog_min": 0.0,
        "backlog_max": 240.0,
        "backlog_latency_coefficient": 0.020,
        "shared_utilization_coefficient": 2.0,
        "shared_utilization_threshold": 0.74,
        "workload_utilization_coefficient": 0.70,
        "p95_tail_factor": 1.32,
        "p99_tail_factor": 1.58,
        "backlog_semantics": "DIMENSIONLESS_SYNTHETIC_INDEX_NOT_OUTSTANDING_IOS",
    }
