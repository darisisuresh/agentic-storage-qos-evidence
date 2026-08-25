from benchmarks.run_safety_validation import run


def test_fault_injection_blocks_unsafe_and_rolls_back():
    result = run(seed=17, proposals=100)
    assert result["unsafe_block_rate"] == 1.0
    assert result["rollback_success_rate"] == 1.0
    assert result["safe_accept_rate"] > 0.90
