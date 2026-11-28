"""LATENCY_BUCKETS covers long-running LLM calls (histograms are in seconds)."""

import math

from litellm.types.integrations.prometheus import LATENCY_BUCKETS


def test_latency_buckets_include_seven_and_ten_minutes():
    """Buckets beyond 5 min so histograms resolve requests up to default LLM timeouts."""
    assert 300.0 in LATENCY_BUCKETS
    assert 420.0 in LATENCY_BUCKETS  # 7 min
    assert 600.0 in LATENCY_BUCKETS  # 10 min
    assert math.isinf(LATENCY_BUCKETS[-1])
    idx_300 = LATENCY_BUCKETS.index(300.0)
    idx_420 = LATENCY_BUCKETS.index(420.0)
    idx_600 = LATENCY_BUCKETS.index(600.0)
    assert idx_300 < idx_420 < idx_600


def test_latency_buckets_resolve_overhead_slo_band():
    """histogram_quantile interpolates linearly inside a bucket, so a bucket wider
    than the SLO it serves lets a p95 read far above every observed sample (#54682)."""
    band = [b for b in LATENCY_BUCKETS if 0.1 <= b <= 0.25]
    for lower, upper in zip(band, band[1:]):
        assert round(upper - lower, 6) <= 0.05, (lower, upper)
