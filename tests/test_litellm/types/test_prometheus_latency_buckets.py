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
