import numpy as np

from benchmarks.run_validation import simulate, workload


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
