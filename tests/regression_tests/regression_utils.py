"""
Shared utilities for performance regression tests.
"""

import json
import os
from pathlib import Path

import pytest


BASELINE_FILE = Path(__file__).parent / "performance_baselines.json"


def get_regression_threshold():
    """Get regression threshold based on hardware.
    
    Configurable via environment variables:
    - REGRESSION_THRESHOLD_8_CORES: threshold for 8+ cores (default: 0.20)
    - REGRESSION_THRESHOLD_4_CORES: threshold for 4-7 cores (default: 0.30)
    - REGRESSION_THRESHOLD_LOW_CORES: threshold for <4 cores (default: 0.60)
    - REGRESSION_THRESHOLD_CI_ADJUSTMENT: additional threshold for CI (default: 0.15)
    """    
    # Get CPU count
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=True) or 4
    except (ImportError, AttributeError):
        cpu_count = os.cpu_count() or 4
    
    # Check if CI environment
    is_ci = any([os.getenv("CI"), os.getenv("GITHUB_ACTIONS"), os.getenv("GITLAB_CI")])
    
    # Get threshold values from environment or use defaults
    # Base: 20% for 8+ cores, 30% for 4-7 cores, 60% for <4 cores
    # CI adds +15% to any threshold
    threshold_8_cores = float(os.getenv("REGRESSION_THRESHOLD_8_CORES", "0.20"))
    threshold_4_cores = float(os.getenv("REGRESSION_THRESHOLD_4_CORES", "0.30"))
    threshold_low_cores = float(os.getenv("REGRESSION_THRESHOLD_LOW_CORES", "0.60"))
    ci_adjustment = float(os.getenv("REGRESSION_THRESHOLD_CI_ADJUSTMENT", "0.15"))
    
    if cpu_count >= 8:
        threshold = threshold_8_cores
    elif cpu_count >= 4:
        threshold = threshold_4_cores
    else:
        threshold = threshold_low_cores
    
    if is_ci:
        threshold += ci_adjustment
    
    print(f"Regression threshold: {threshold:.0%} (CPU: {cpu_count} cores, CI: {is_ci})")
    return threshold


def load_baselines():
    """Load performance baselines from JSON."""
    if not BASELINE_FILE.exists():
        return {}
    with open(BASELINE_FILE, "r") as f:
        return json.load(f)


def save_baseline(test_name, median_ms, p95_ms):
    """Save baseline for a specific test."""
    baselines = load_baselines()
    baselines[test_name] = {"median_ms": median_ms, "p95_ms": p95_ms}
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_FILE, "w") as f:
        json.dump(baselines, f, indent=2)


def check_regression(test_name, median_ms, p95_ms, baselines, regression_threshold):
    """Check if current performance regressed compared to baseline."""
    if test_name not in baselines:
        pytest.skip(f"No baseline for {test_name}. Run with --update-baseline")
    
    baseline = baselines[test_name]
    baseline_median = baseline["median_ms"]
    baseline_p95 = baseline["p95_ms"]
    
    median_regression = (median_ms - baseline_median) / baseline_median
    p95_regression = (p95_ms - baseline_p95) / baseline_p95
    
    # Determine environment info for display
    is_ci = any([os.getenv("CI"), os.getenv("GITHUB_ACTIONS"), os.getenv("GITLAB_CI")])
    env_label = "CI" if is_ci else "Local"
    
    print(f"\n{'='*60}")
    print(f"  {test_name} [{env_label}]")
    print(f"{'='*60}")
    print(f"  Median:  {median_ms:.2f}ms (baseline: {baseline_median:.2f}ms, {median_regression:+.1%} / +{regression_threshold:.0%} threshold)")
    print(f"  P95:     {p95_ms:.2f}ms (baseline: {baseline_p95:.2f}ms, {p95_regression:+.1%} / +{regression_threshold:.0%} threshold)")
    
    if median_regression > regression_threshold:
        print(f"  FAIL: Median regression: +{median_regression*100:.1f}%")
        pytest.fail(
            f"Median regression detected: {median_ms:.2f}ms vs baseline {baseline_median:.2f}ms "
            f"(+{median_regression*100:.1f}%)"
        )
    
    if p95_regression > regression_threshold:
        print(f"  FAIL: P95 regression: +{p95_regression*100:.1f}%")
        pytest.fail(
            f"P95 regression detected: {p95_ms:.2f}ms vs baseline {baseline_p95:.2f}ms "
            f"(+{p95_regression*100:.1f}%)"
        )
    
    print(f"  PASS: No regression detected")
    print(f"{'='*60}\n")

