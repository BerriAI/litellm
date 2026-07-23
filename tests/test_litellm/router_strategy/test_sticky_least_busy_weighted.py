"""
Tests for the weighted sticky-least-busy routing strategy
(StickyLeastBusyWeightedLoggingHandler).

Covers the weighting-specific behavior: weighted virtual-node distribution,
weight-aware ring cache invalidation, and capacity-normalized load selection.
The unweighted ring/selection behavior is covered by test_sticky_least_busy.py.
"""

import pytest

from litellm.caching.caching import DualCache
from litellm.router_strategy.sticky_least_busy_weighted import (
    StickyLeastBusyWeightedLoggingHandler,
)


def _make_deployment(dep_id: str) -> dict:
    return {
        "model_name": "test-model",
        "model_info": {"id": dep_id},
        "litellm_params": {
            "model": "openai/gpt-4",
            "api_base": f"http://node-{dep_id}:8000",
        },
    }


def _make_weighted_deployment(dep_id: str, weight) -> dict:
    """Deployment with a sticky_weight set in model_info."""
    d = _make_deployment(dep_id)
    d["model_info"]["sticky_weight"] = weight
    return d


MG = "test-model"  # default model group for tests


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the class-level singleton between tests for isolation."""
    StickyLeastBusyWeightedLoggingHandler._instance = None
    yield
    StickyLeastBusyWeightedLoggingHandler._instance = None


class TestWeightedHashRing:
    def test_vnodes_scale_with_weight(self):
        handler = StickyLeastBusyWeightedLoggingHandler(
            router_cache=DualCache(), virtual_nodes=150
        )
        assert handler._vnodes_for_weight(1.0) == 150
        assert handler._vnodes_for_weight(2.0) == 300
        assert handler._vnodes_for_weight(1.7) == 255  # integer-space, deterministic
        assert handler._vnodes_for_weight(0.5) == 75
        # any positive weight keeps at least one vnode (never dropped from ring)
        assert handler._vnodes_for_weight(0.001) == 1

    def test_weighted_distribution(self):
        """A deployment with 3x the weight owns ~3x the key space."""
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=DualCache())
        handler._build_hash_ring(MG, {"light": 1.0, "heavy": 3.0})
        counts = {"light": 0, "heavy": 0}
        for i in range(4000):
            counts[handler._get_deployment_for_key(MG, f"key-{i}")] += 1
        heavy_share = counts["heavy"] / 4000
        # heavy should get ~3/4 of keys
        assert 0.68 < heavy_share < 0.82, f"heavy got {heavy_share:.2%}"

    def test_decimal_weight_distribution(self):
        """Decimal weights (e.g. 1.7) are honored proportionally."""
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=DualCache())
        handler._build_hash_ring(MG, {"a": 1.0, "b": 1.7})
        counts = {"a": 0, "b": 0}
        for i in range(4000):
            counts[handler._get_deployment_for_key(MG, f"key-{i}")] += 1
        b_share = counts["b"] / 4000
        expected = 1.7 / 2.7  # ~0.63
        assert abs(b_share - expected) < 0.06, (
            f"b got {b_share:.2%}, expected ~{expected:.2%}"
        )

    def test_weight_change_rebuilds_ring(self):
        """Changing a weight (same id set) must invalidate the cached ring."""
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=DualCache())
        handler._build_hash_ring(MG, {"dep-1": 1.0, "dep-2": 1.0})
        ring_before = handler._rings[MG][1]
        handler._build_hash_ring(MG, {"dep-1": 1.0, "dep-2": 3.0})
        assert handler._rings[MG][1] is not ring_before  # rebuilt

    def test_same_weights_not_rebuilt(self):
        """Identical (id, weight) set reuses the cached ring."""
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=DualCache())
        handler._build_hash_ring(MG, {"dep-1": 2.0, "dep-2": 1.0})
        ring_before = handler._rings[MG][1]
        handler._build_hash_ring(MG, {"dep-1": 2.0, "dep-2": 1.0})
        assert handler._rings[MG][1] is ring_before  # cache hit

    def test_weight_increase_is_cache_cheap(self):
        """Raising a node's weight only adds vnodes — existing keys mostly stable,
        and any key that moves moves TO the node that gained capacity."""
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=DualCache())
        handler._build_hash_ring(MG, {"dep-1": 1.0, "dep-2": 1.0, "dep-3": 1.0})
        keys = [f"key-{i}" for i in range(1000)]
        before = {k: handler._get_deployment_for_key(MG, k) for k in keys}

        handler._build_hash_ring(MG, {"dep-1": 2.0, "dep-2": 1.0, "dep-3": 1.0})
        after = {k: handler._get_deployment_for_key(MG, k) for k in keys}

        changed = sum(1 for k in keys if before[k] != after[k])
        assert changed < 400, f"Too many keys remapped on weight bump: {changed}/1000"
        moved_to = [after[k] for k in keys if before[k] != after[k]]
        assert all(dst == "dep-1" for dst in moved_to), (
            "weight increase should only pull keys toward the heavier node"
        )

    def test_list_input_is_equal_weight(self):
        """Passing a list of ids is shorthand for equal weight 1.0."""
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=DualCache())
        handler._build_hash_ring(MG, ["dep-1", "dep-2"])
        list_ring = handler._rings[MG][1]
        # equivalent explicit weights produce the identical cached ring signature
        handler._build_hash_ring(MG, {"dep-1": 1.0, "dep-2": 1.0})
        assert handler._rings[MG][1] is list_ring


class TestWeightForResolution:
    def test_absent_weight_defaults_to_one(self):
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=DualCache())
        assert handler._weight_for(_make_deployment("x")) == 1.0

    def test_non_positive_weight_guarded_to_one(self):
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=DualCache())
        assert handler._weight_for(_make_weighted_deployment("y", 0)) == 1.0
        assert handler._weight_for(_make_weighted_deployment("z", -3)) == 1.0

    def test_invalid_weight_guarded_to_one(self):
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=DualCache())
        assert handler._weight_for(_make_weighted_deployment("w", "not-a-number")) == 1.0

    def test_decimal_weight_parsed(self):
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=DualCache())
        assert handler._weight_for(_make_weighted_deployment("h", 1.7)) == 1.7
        assert handler._weight_for(_make_weighted_deployment("s", "2.5")) == 2.5


class TestCapacityNormalizedSelection:
    def test_heavy_node_tolerates_higher_raw_load(self):
        """A high-weight node stays sticky at a raw load that would overload an
        equal-weight node, because the load check is capacity-normalized."""
        handler = StickyLeastBusyWeightedLoggingHandler(
            router_cache=DualCache(), imbalance_threshold=1.5
        )
        weights = {"dep-0": 1.0, "dep-1": 1.0, "dep-heavy": 4.0}
        deployments = [
            _make_weighted_deployment("dep-0", 1.0),
            _make_weighted_deployment("dep-1", 1.0),
            _make_weighted_deployment("dep-heavy", 4.0),
        ]
        # Find a sticky key that maps to the heavy node under these weights
        handler._build_hash_ring(MG, weights)
        sticky_key = next(
            k
            for k in (f"k-{i}" for i in range(2000))
            if handler._get_deployment_for_key(MG, k) == "dep-heavy"
        )

        # raw loads: heavy=6, others=2. normalized: heavy=6/4=1.5, others=2.
        # avg=(2+2+1.5)/3=1.83, min=1.5, reference=1.67, threshold=1.5*1.67=2.5
        # preferred normalized 1.5 < 2.5 → stays sticky (an unweighted node at
        # raw 6 would have normalized load 6 >> threshold and rebalance).
        request_counts = {"dep-0": 2, "dep-1": 2, "dep-heavy": 6}
        result = handler._select_deployment(MG, deployments, request_counts, sticky_key)
        assert result["model_info"]["id"] == "dep-heavy"

    def test_least_busy_prefers_higher_weight_at_equal_raw_load(self):
        """With no sticky key and equal raw load, the higher-capacity node wins
        (lower normalized load)."""
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=DualCache())
        deployments = [
            _make_weighted_deployment("small", 1.0),
            _make_weighted_deployment("big", 4.0),
        ]
        # same raw count, but big has 4x capacity → normalized 4 vs 1 → big chosen
        request_counts = {"small": 4, "big": 4}
        result = handler._select_deployment(MG, deployments, request_counts, None)
        assert result["model_info"]["id"] == "big"

    def test_equal_weights_match_legacy_behavior(self):
        """With all weights 1.0, normalized load == raw load: least-busy picks the
        lowest raw count, exactly as before weighting existed."""
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=DualCache())
        deployments = [
            _make_weighted_deployment("dep-0", 1.0),
            _make_weighted_deployment("dep-1", 1.0),
            _make_weighted_deployment("dep-2", 1.0),
        ]
        request_counts = {"dep-0": 10, "dep-1": 2, "dep-2": 10}
        result = handler._select_deployment(MG, deployments, request_counts, None)
        assert result["model_info"]["id"] == "dep-1"
