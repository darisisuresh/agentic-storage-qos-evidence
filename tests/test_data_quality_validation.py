from benchmarks.run_data_quality_validation import validate


def test_synthetic_trace_quality_and_uniqueness():
    report = validate(seeds=10)
    assert report["passed"] is True
    assert report["total_traces"] == 60
    assert report["duplicate_trace_hashes"] == 0
    assert report["missing_or_nonfinite_values"] == 0


def test_report_declares_no_train_split_or_classification_task():
    report = validate(seeds=2)
    assert report["learned_model_split"].startswith("NOT_APPLICABLE")
    assert report["classification_distribution"].startswith("NOT_APPLICABLE")
