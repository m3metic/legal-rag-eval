"""Tests for the evaluation gate's pass/fail logic."""

from legalrag.gate import compare_to_baseline, report

BASE = {"recall@10": 0.75, "mrr": 0.67}


def test_equal_scores_pass():
    checks = compare_to_baseline("exp", {"recall@10": 0.75, "mrr": 0.67}, BASE, tolerance=0.005)
    assert all(c.passed for c in checks)


def test_improvement_passes():
    checks = compare_to_baseline("exp", {"recall@10": 0.80, "mrr": 0.70}, BASE, tolerance=0.005)
    assert all(c.passed for c in checks)


def test_drop_within_tolerance_passes():
    checks = compare_to_baseline("exp", {"recall@10": 0.746, "mrr": 0.67}, BASE, tolerance=0.005)
    assert all(c.passed for c in checks)


def test_drop_beyond_tolerance_fails_only_that_metric():
    checks = compare_to_baseline("exp", {"recall@10": 0.74, "mrr": 0.67}, BASE, tolerance=0.005)
    assert [c.metric for c in checks if not c.passed] == ["recall@10"]


def test_missing_metric_fails():
    checks = compare_to_baseline("exp", {"recall@10": 0.75}, BASE, tolerance=0.005)
    assert [c.metric for c in checks if not c.passed] == ["mrr"]


def test_report_states_verdict():
    good = compare_to_baseline("exp", {"recall@10": 0.75, "mrr": 0.67}, BASE, 0.005)
    bad = compare_to_baseline("exp", {"recall@10": 0.70, "mrr": 0.67}, BASE, 0.005)
    assert "Gate passed" in report(good)
    assert "Gate failed" in report(bad) and "-0.0500" in report(bad)
