"""
Tests for the weighted sticky-least-busy routing strategy
(StickyLeastBusyWeightedLoggingHandler).

Covers the weighting-specific behavior: weighted virtual-node distribution,
weight-aware ring cache invalidation, and capacity-normalized load selection.
The unweighted ring/selection behavior is covered by test_sticky_least_busy.py.
"""

import asyncio

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


def _cache_key_for_deployment(
    handler: StickyLeastBusyWeightedLoggingHandler, model_group: str, deployment: dict
) -> str:
    dep_id = str(deployment["model_info"]["id"])
    load_key, _ = handler._get_load_key_for_deployment(deployment)
    return handler._get_request_count_cache_key(model_group, dep_id, load_key)


def _metric_sample_value(metric, expected_labels: dict) -> float:
    values = [
        sample.value
        for family in metric.collect()
        for sample in family.samples
        if not sample.name.endswith("_created")
        and all(
            sample.labels.get(key) == value for key, value in expected_labels.items()
        )
    ]
    assert len(values) == 1
    return values[0]


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the class-level singleton between tests for isolation."""
    StickyLeastBusyWeightedLoggingHandler._instance = None
    yield
    StickyLeastBusyWeightedLoggingHandler._instance = None


class RecordingDualCache(DualCache):
    def __init__(self):
        super().__init__()
        self.increment_calls = []
        self.async_increment_calls = []

    def increment_cache(self, key, value: int, local_only: bool = False, **kwargs) -> int:
        self.increment_calls.append((key, value))
        return super().increment_cache(
            key=key,
            value=value,
            local_only=local_only,
            **kwargs,
        )

    async def async_increment_cache(self, key, value, **kwargs):
        self.async_increment_calls.append((key, value))
        return await super().async_increment_cache(key=key, value=value, **kwargs)


class FakeRedisClient:
    def __init__(self, value):
        self.value = value
        self.eval_calls = []

    def eval(self, script, numkeys, key, owner):
        self.eval_calls.append((script, numkeys, key, owner))
        if self.value == owner:
            self.value = None
            return 1
        return 0


class FakeRedisCache:
    def __init__(self, redis_client):
        self.redis_client = redis_client

    def check_and_fix_namespace(self, key):
        return f"test:{key}"


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
    def test_dynamic_imbalance_threshold_for_deployment_count(self):
        handler = StickyLeastBusyWeightedLoggingHandler(
            router_cache=DualCache(),
            imbalance_threshold=1.5,
            dynamic_imbalance_thresholds={2: 1.35, "4": "1.45"},
        )

        assert handler._imbalance_threshold_for_deployment_count(2) == 1.35
        assert handler._imbalance_threshold_for_deployment_count(4) == 1.45
        assert handler._imbalance_threshold_for_deployment_count(5) == 1.5

    def test_dynamic_threshold_overrides_sticky_for_small_group(self):
        handler = StickyLeastBusyWeightedLoggingHandler(
            router_cache=DualCache(),
            imbalance_threshold=1.5,
            dynamic_imbalance_thresholds={2: 1.35},
        )
        deployments = [
            _make_weighted_deployment("sticky", 1.0),
            _make_weighted_deployment("least", 1.0),
        ]
        handler._build_hash_ring(MG, {"sticky": 1.0, "least": 1.0})
        sticky_key = next(
            k
            for k in (f"k-{i}" for i in range(2000))
            if handler._get_deployment_for_key(MG, k) == "sticky"
        )

        result = handler._select_deployment(
            MG,
            deployments,
            {"sticky": 4, "least": 2},
            sticky_key,
        )

        assert result["model_info"]["id"] == "least"

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


class TestRequestCounters:
    def test_top_level_metadata_is_counted(self):
        cache = RecordingDualCache()
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=cache)
        dep_id = "top-level-deployment"
        cache_key = handler._get_request_count_cache_key(MG, dep_id)
        kwargs = {
            "litellm_call_id": "top-level-call",
            "metadata": {"model_group": MG},
            "model_info": {"id": dep_id},
        }

        handler.log_pre_api_call(None, None, kwargs)
        handler._decrement_request_count(kwargs, callback_type="SYNC-SUCCESS")

        assert cache.increment_calls == [(cache_key, 1), (cache_key, -1)]
        assert cache.get_cache(key=cache_key, redis_only=True) == 0

    def test_route_map_recovers_missing_callback_metadata(self):
        cache = RecordingDualCache()
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=cache)
        dep_id = "route-map-deployment"
        deployment = _make_deployment(dep_id)
        cache_key = _cache_key_for_deployment(handler, MG, deployment)
        call_id = "route-map-call"

        handler._remember_selected_deployment_for_kwargs(
            {"litellm_call_id": call_id},
            MG,
            deployment,
        )
        kwargs = {"litellm_call_id": call_id}

        handler.log_pre_api_call(None, None, kwargs)
        handler._decrement_request_count(kwargs, callback_type="SYNC-SUCCESS")

        assert cache.increment_calls == [(cache_key, 1), (cache_key, -1)]
        assert cache.get_cache(key=cache_key, redis_only=True) == 0

    def test_aliases_share_backend_request_count(self):
        cache = RecordingDualCache()
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=cache)
        glm_deployment = _make_deployment("glm-deployment")
        open_deployment = _make_deployment("open-deployment")
        open_deployment["model_name"] = "open-large"
        open_deployment["litellm_params"] = glm_deployment["litellm_params"].copy()
        shared_cache_key = _cache_key_for_deployment(handler, "glm-latest", glm_deployment)

        handler.log_pre_api_call(
            None,
            None,
            {
                "litellm_call_id": "open-large-call",
                "metadata": {
                    "model_group": "open-large",
                    "deployment": open_deployment["litellm_params"]["model"],
                    "api_base": open_deployment["litellm_params"]["api_base"],
                },
                "model_info": open_deployment["model_info"],
            },
        )

        request_counts = handler._get_request_counts("glm-latest", [glm_deployment])

        assert cache.increment_calls == [(shared_cache_key, 1)]
        assert request_counts["glm-deployment"] == 1

    def test_route_map_keeps_decrement_on_backend_key_when_callback_lacks_api_base(self):
        cache = RecordingDualCache()
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=cache)
        deployment = _make_deployment("backend-deployment")
        cache_key = _cache_key_for_deployment(handler, MG, deployment)
        call_id = "metadata-partial-call"

        handler._remember_selected_deployment_for_kwargs(
            {"litellm_call_id": call_id},
            MG,
            deployment,
        )
        handler.log_pre_api_call(
            None,
            None,
            {
                "litellm_call_id": call_id,
                "metadata": {
                    "model_group": MG,
                    "deployment": deployment["litellm_params"]["model"],
                    "api_base": deployment["litellm_params"]["api_base"],
                },
                "model_info": deployment["model_info"],
            },
        )
        handler._decrement_request_count(
            {
                "litellm_call_id": call_id,
                "metadata": {"model_group": MG},
                "model_info": deployment["model_info"],
            },
            callback_type="SYNC-SUCCESS",
        )

        assert cache.increment_calls == [(cache_key, 1), (cache_key, -1)]
        assert cache.get_cache(key=cache_key, redis_only=True) == 0

    def test_duplicate_success_callback_does_not_double_decrement(self):
        cache = RecordingDualCache()
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=cache)
        dep_id = "duplicate-success-deployment"
        cache_key = handler._get_request_count_cache_key(MG, dep_id)
        kwargs = {
            "litellm_call_id": "duplicate-success-call",
            "litellm_params": {
                "metadata": {"model_group": MG},
                "model_info": {"id": dep_id},
            },
        }

        handler.log_pre_api_call(None, None, kwargs)
        handler._decrement_request_count(kwargs, callback_type="SYNC-SUCCESS")
        handler._decrement_request_count(kwargs, callback_type="ASYNC-SUCCESS")

        assert cache.increment_calls == [(cache_key, 1), (cache_key, -1)]
        assert cache.get_cache(key=cache_key, redis_only=True) == 0

    def test_failure_callback_allows_retry_increment_for_same_call_id(self):
        cache = RecordingDualCache()
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=cache)
        dep_id = "retry-deployment"
        cache_key = handler._get_request_count_cache_key(MG, dep_id)
        kwargs = {
            "litellm_call_id": "retry-call",
            "litellm_params": {
                "metadata": {"model_group": MG},
                "model_info": {"id": dep_id},
            },
        }

        handler.log_pre_api_call(None, None, kwargs)
        handler._decrement_request_count(kwargs, callback_type="SYNC-FAILURE")
        handler._decrement_request_count(kwargs, callback_type="ASYNC-FAILURE")
        handler.log_pre_api_call(None, None, kwargs)

        assert cache.increment_calls == [(cache_key, 1), (cache_key, -1), (cache_key, 1)]
        assert cache.get_cache(key=cache_key, redis_only=True) == 1

    def test_decrement_without_matching_increment_is_skipped(self):
        cache = RecordingDualCache()
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=cache)
        kwargs = {
            "litellm_call_id": "no-increment-call",
            "litellm_params": {
                "metadata": {"model_group": MG},
                "model_info": {"id": "no-increment-deployment"},
            },
        }

        handler._decrement_request_count(kwargs, callback_type="SYNC-FAILURE")

        assert cache.increment_calls == []


class TestObservedLoad:
    def test_parse_vllm_observed_load(self):
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=DualCache())
        metrics_text = """
vllm:num_requests_running{model_name="glm"} 2
vllm:num_requests_running{model_name="open-large"} 3
vllm:num_requests_waiting{model_name="glm"} 1
"""

        assert handler._parse_observed_load(metrics_text) == (6, 5, 1, "vllm")

    def test_parse_sglang_observed_load_uses_rank_max(self):
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=DualCache())
        metrics_text = """
sglang:num_running_reqs{rank="0"} 4
sglang:num_running_reqs{rank="1"} 4
sglang:num_queue_reqs{rank="0"} 2
sglang:num_queue_reqs{rank="1"} 2
"""

        assert handler._parse_observed_load(metrics_text) == (6, 4, 2, "sglang")

    def test_observed_load_preferred_over_request_count(self):
        cache = DualCache()
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=cache)
        handler.observed_load_enabled = True
        deployment = _make_deployment("observed-deployment")
        dep_id = str(deployment["model_info"]["id"])
        load_key, _ = handler._get_load_key_for_deployment(deployment)
        request_key = handler._get_request_count_cache_key(MG, dep_id, load_key)
        observed_key = handler._get_observed_load_cache_key(load_key)
        cache.set_cache(key=request_key, value=99, ttl=600)
        cache.set_cache(key=observed_key, value=5, ttl=15)

        request_counts = handler._get_request_counts(MG, [deployment])

        assert request_counts[dep_id] == 5

    def test_observed_load_falls_back_to_request_count_when_missing(self):
        cache = DualCache()
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=cache)
        handler.observed_load_enabled = True
        deployment = _make_deployment("fallback-deployment")
        dep_id = str(deployment["model_info"]["id"])
        load_key, _ = handler._get_load_key_for_deployment(deployment)
        request_key = handler._get_request_count_cache_key(MG, dep_id, load_key)
        cache.set_cache(key=request_key, value=9, ttl=600)

        request_counts = handler._get_request_counts(MG, [deployment])

        assert request_counts[dep_id] == 9

    def test_model_list_sync_prunes_removed_backends(self):
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=DualCache())
        handler.observed_load_enabled = True
        deployment = _make_deployment("active-deployment")
        active_load_key, _ = handler._get_load_key_for_deployment(deployment)
        handler._observed_load_backends = {"backend:removed": "http://removed-node:8000/v1"}
        handler._observed_load_backend_last_seen = {"backend:removed": 1}
        handler._observed_load_model_list_load_keys = {"backend:removed"}
        handler._observed_load_last_error_log = {"backend:removed": 1}
        handler._observed_load_success_logged = {"backend:removed": True}

        handler._sync_observed_load_backends_from_model_list([deployment])

        assert handler._observed_load_backends == {
            active_load_key: deployment["litellm_params"]["api_base"]
        }
        assert "backend:removed" not in handler._observed_load_backend_last_seen
        assert handler._observed_load_model_list_load_keys == {active_load_key}
        assert "backend:removed" not in handler._observed_load_last_error_log
        assert "backend:removed" not in handler._observed_load_success_logged

    def test_model_list_backend_is_not_pruned_when_last_seen_is_stale(self):
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=DualCache())
        handler.observed_load_enabled = True
        deployment = _make_deployment("low-traffic-deployment")
        load_key, _ = handler._get_load_key_for_deployment(deployment)

        handler._sync_observed_load_backends_from_model_list([deployment])
        handler._observed_load_backend_last_seen[load_key] = 1

        backends = handler._get_observed_load_backends_for_sync()

        assert backends == [(load_key, deployment["litellm_params"]["api_base"])]
        assert handler._observed_load_backends == {
            load_key: deployment["litellm_params"]["api_base"]
        }

    def test_request_discovered_backend_is_pruned_when_last_seen_is_stale(self):
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=DualCache())
        handler.observed_load_enabled = True
        handler._observed_load_backends = {"backend:lazy": "http://lazy-node:8000/v1"}
        handler._observed_load_backend_last_seen = {"backend:lazy": 1}
        handler._observed_load_last_error_log = {"backend:lazy": 1}
        handler._observed_load_success_logged = {"backend:lazy": True}

        backends = handler._get_observed_load_backends_for_sync()

        assert backends == []
        assert handler._observed_load_backends == {}
        assert handler._observed_load_backend_last_seen == {}
        assert handler._observed_load_last_error_log == {}
        assert handler._observed_load_success_logged == {}

    def test_removed_model_list_backend_is_not_readded_by_request_registration(self):
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=DualCache())
        handler.observed_load_enabled = True
        active_deployment = _make_deployment("active-deployment")
        removed_deployment = _make_deployment("removed-deployment")
        active_load_key, _ = handler._get_load_key_for_deployment(active_deployment)
        removed_load_key, removed_source = handler._get_load_key_for_deployment(removed_deployment)

        handler._sync_observed_load_backends_from_model_list([active_deployment])
        handler._register_observed_load_backend(
            removed_deployment,
            removed_load_key,
            removed_source,
        )

        assert handler._observed_load_backends == {
            active_load_key: active_deployment["litellm_params"]["api_base"]
        }
        assert removed_load_key not in handler._observed_load_backend_last_seen

    def test_empty_initial_model_list_does_not_block_request_registration(self):
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=DualCache())
        handler.observed_load_enabled = True
        deployment = _make_deployment("request-discovered-deployment")
        load_key, load_key_source = handler._get_load_key_for_deployment(deployment)

        handler._sync_observed_load_backends_from_model_list([])
        handler._register_observed_load_backend(deployment, load_key, load_key_source)

        assert handler._observed_load_model_list_synced is False
        assert handler._observed_load_backends == {
            load_key: deployment["litellm_params"]["api_base"]
        }

    def test_empty_model_list_prunes_after_authoritative_sync(self):
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=DualCache())
        handler.observed_load_enabled = True
        deployment = _make_deployment("removed-deployment")
        load_key, load_key_source = handler._get_load_key_for_deployment(deployment)

        handler._sync_observed_load_backends_from_model_list([deployment])
        handler._sync_observed_load_backends_from_model_list([])
        handler._register_observed_load_backend(deployment, load_key, load_key_source)

        assert handler._observed_load_model_list_synced is True
        assert handler._observed_load_backends == {}
        assert load_key not in handler._observed_load_backend_last_seen

    def test_observed_load_sync_runs_without_redis_lock(self):
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=DualCache())

        assert handler._should_run_observed_load_sync() is True

    def test_shutdown_releases_owned_observed_load_lock(self):
        cache = DualCache()
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=cache)
        redis_client = FakeRedisClient(handler._observed_load_sync_owner)
        cache.redis_cache = FakeRedisCache(redis_client)
        handler.observed_load_enabled = True
        handler._set_observed_load_leader_state(True, "test")

        handler.shutdown_observed_load_sync()

        assert handler.observed_load_enabled is False
        assert handler._observed_load_shutdown_event.is_set()
        assert redis_client.value is None
        assert redis_client.eval_calls[0][1:] == (
            1,
            "test:sticky_lb_weighted:observed_load_sync:lock",
            handler._observed_load_sync_owner,
        )

    def test_shutdown_does_not_release_another_pods_observed_load_lock(self):
        cache = DualCache()
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=cache)
        redis_client = FakeRedisClient("other-owner")
        cache.redis_cache = FakeRedisCache(redis_client)
        handler.observed_load_enabled = True

        handler.shutdown_observed_load_sync()

        assert redis_client.value == "other-owner"


class TestPrometheusMetrics:
    def test_backend_info_metric_exposes_sanitized_load_key_mapping(self):
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=DualCache())
        deployment = _make_deployment("backend-info-deployment")
        deployment["litellm_params"]["api_base"] = (
            "http://user:secret@backend-info-node:8000/v1?api_key=secret"
        )
        load_key, load_key_source = handler._get_load_key_for_deployment(deployment)

        handler._get_request_counts(MG, [deployment])

        assert (
            _metric_sample_value(
                handler._backend_info,
                {
                    "model_group": MG,
                    "deployment_id": "backend-info-deployment",
                    "litellm_model": "openai/gpt-4",
                    "api_base": "http://backend-info-node:8000/v1",
                    "load_key": load_key,
                    "load_key_source": load_key_source,
                },
            )
            == 1
        )

    def test_selection_metrics_reuse_calculated_loads(self):
        handler = StickyLeastBusyWeightedLoggingHandler(
            router_cache=DualCache(), imbalance_threshold=1.5
        )
        deployments = [
            _make_weighted_deployment("metrics-small", 1.0),
            _make_weighted_deployment("metrics-large", 2.0),
        ]

        handler._select_deployment(
            MG,
            deployments,
            {"metrics-small": 10, "metrics-large": 8},
            None,
        )

        assert (
            _metric_sample_value(
                handler._healthy_deployments_count, {"model_group": MG}
            )
            == 2
        )
        assert (
            _metric_sample_value(
                handler._deployment_healthy,
                {"model_group": MG, "deployment_id": "metrics-small"},
            )
            == 1
        )
        assert (
            _metric_sample_value(
                handler._deployment_weight,
                {"model_group": MG, "deployment_id": "metrics-large"},
            )
            == 2
        )
        assert (
            _metric_sample_value(
                handler._normalized_load,
                {"model_group": MG, "deployment_id": "metrics-small"},
            )
            == 10
        )
        assert (
            _metric_sample_value(
                handler._normalized_load,
                {"model_group": MG, "deployment_id": "metrics-large"},
            )
            == 4
        )
        assert _metric_sample_value(handler._reference_load, {"model_group": MG}) == 5.5
        assert (
            _metric_sample_value(handler._threshold_load, {"model_group": MG}) == 8.25
        )

        handler._select_deployment(
            MG,
            [deployments[1]],
            {"metrics-large": 3},
            None,
        )

        assert (
            _metric_sample_value(
                handler._healthy_deployments_count, {"model_group": MG}
            )
            == 1
        )
        assert (
            _metric_sample_value(
                handler._deployment_healthy,
                {"model_group": MG, "deployment_id": "metrics-small"},
            )
            == 0
        )

    def test_negative_count_increments_reset_counter(self):
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=DualCache())
        kwargs = {
            "litellm_params": {
                "metadata": {"model_group": MG},
                "model_info": {"id": "counter-reset-deployment"},
            },
        }

        handler._decrement_request_count(kwargs, callback_type="TEST")

        assert (
            _metric_sample_value(
                handler._counter_resets,
                {"model_group": MG, "deployment_id": "counter-reset-deployment"},
            )
            == 1
        )


class TestNonBlockingIncrement:
    """The increment on the request hot path must never block the event loop."""

    def test_falls_back_to_sync_without_loop(self):
        """Without a running event loop (tests / sync callers), the original
        synchronous Redis increment path is used exactly once."""
        cache = RecordingDualCache()
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=cache)
        dep_id = "sync-fallback-deployment"
        cache_key = handler._get_request_count_cache_key(MG, dep_id)

        result = handler._non_blocking_cache_delta(
            cache_key, 1, handler.cache_ttl, "test", action="increment"
        )

        assert result is not None
        assert cache.increment_calls == [(cache_key, 1)]
        assert cache.async_increment_calls == []

    def test_does_not_block_loop_on_slow_redis(self):
        """With a running loop, a slow async Redis increment must not block the
        caller — the helper returns well before the Redis write completes."""
        cache = RecordingDualCache()
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=cache)
        dep_id = "slow-redis-deployment"
        cache_key = handler._get_request_count_cache_key(MG, dep_id)

        async def main():
            loop = asyncio.get_running_loop()
            started = loop.time()
            # Patch async_increment_cache to simulate a slow Redis round-trip.
            cache.async_increment_calls = []

            async def slow_increment(key, value, **kwargs):
                await asyncio.sleep(2)
                cache.async_increment_calls.append((key, value))
                return 1

            cache.async_increment_cache = slow_increment  # type: ignore[assignment]

            result = handler._non_blocking_cache_delta(
                cache_key, 1, handler.cache_ttl, "test", action="increment"
            )
            elapsed = loop.time() - started

            # Returned immediately with the in-memory value.
            assert result is not None
            assert elapsed < 0.2  # did NOT wait for the 2s Redis write
            # The async write has not completed yet.
            assert cache.async_increment_calls == []

            # Give the scheduled task time to finish.
            await asyncio.sleep(2.5)
            assert cache.async_increment_calls == [(cache_key, 1)]

        asyncio.run(main())

    def test_uses_in_memory_value_immediately(self):
        """With a running loop, the returned value is the in-memory increment
        result, available before the async Redis task completes."""
        cache = RecordingDualCache()
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=cache)
        dep_id = "in-memory-deployment"
        cache_key = handler._get_request_count_cache_key(MG, dep_id)

        async def main():
            # Block the async Redis write so it cannot influence the result.
            pending = asyncio.Event()

            async def blocked_increment(key, value, **kwargs):
                await pending.wait()
                return 1

            cache.async_increment_cache = blocked_increment  # type: ignore[assignment]

            result = handler._non_blocking_cache_delta(
                cache_key, 1, handler.cache_ttl, "test", action="increment"
            )
            assert result == 1  # in-memory value, despite Redis being blocked
            assert cache.increment_calls == []  # sync path NOT taken
            pending.set()  # release the blocked task so the loop can exit

        asyncio.run(main())

    def test_decrement_path_also_non_blocking(self):
        """The sync decrement path (log_success_event/log_failure_event) must
        also defer Redis, so a slow Redis can't block the response callback."""
        cache = RecordingDualCache()
        handler = StickyLeastBusyWeightedLoggingHandler(router_cache=cache)
        dep_id = "decrement-deployment"
        cache_key = handler._get_request_count_cache_key(MG, dep_id)

        async def main():
            pending = asyncio.Event()

            async def blocked_delta(key, value, **kwargs):
                await pending.wait()
                return 0

            cache.async_increment_cache = blocked_delta  # type: ignore[assignment]

            result = handler._non_blocking_cache_delta(
                cache_key, -1, handler.cache_ttl, "test", action="decrement"
            )
            assert result == -1  # in-memory value, despite Redis being blocked
            assert cache.increment_calls == []  # sync path NOT taken
            pending.set()

        asyncio.run(main())
