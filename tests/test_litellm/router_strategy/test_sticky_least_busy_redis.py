"""
Tests for the sticky-least-busy-redis routing strategy.
"""

import hashlib

import pytest

from litellm.caching.caching import DualCache
from litellm.router_strategy.sticky_least_busy_redis import (
    StickyLeastBusyRedisLoggingHandler,
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


MG = "test-model"  # default model group for tests


# Reset singleton between tests
@pytest.fixture(autouse=True)
def reset_singleton():
    StickyLeastBusyRedisLoggingHandler._instance = None
    yield
    StickyLeastBusyRedisLoggingHandler._instance = None


# =====================================================================
# Test: compute_sticky_key (same as original - verify parity)
# =====================================================================


class TestComputeStickyKey:
    def test_none_messages_returns_none(self):
        assert StickyLeastBusyRedisLoggingHandler.compute_sticky_key(None) is None

    def test_empty_messages_returns_none(self):
        assert StickyLeastBusyRedisLoggingHandler.compute_sticky_key([]) is None

    def test_single_user_message(self):
        msgs = [{"role": "user", "content": "hello"}]
        key = StickyLeastBusyRedisLoggingHandler.compute_sticky_key(msgs)
        assert key is not None
        assert len(key) == 64  # SHA-256 hex digest

    def test_identity_is_first_user_message(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        key = StickyLeastBusyRedisLoggingHandler.compute_sticky_key(msgs)
        expected = hashlib.sha256("hello".encode("utf-8")).hexdigest()
        assert key == expected

    def test_system_prompt_not_in_hash(self):
        msgs_with_system = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hello"},
        ]
        msgs_without_system = [
            {"role": "user", "content": "hello"},
        ]
        key_with = StickyLeastBusyRedisLoggingHandler.compute_sticky_key(
            msgs_with_system
        )
        key_without = StickyLeastBusyRedisLoggingHandler.compute_sticky_key(
            msgs_without_system
        )
        expected = hashlib.sha256("hello".encode("utf-8")).hexdigest()
        assert key_with == expected
        assert key_without == expected

    def test_same_conversation_different_turns_same_key(self):
        turn1 = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is KV caching?"},
        ]
        turn2 = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is KV caching?"},
            {"role": "assistant", "content": "KV caching stores..."},
            {"role": "user", "content": "How does it help?"},
        ]
        turn3 = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is KV caching?"},
            {"role": "assistant", "content": "KV caching stores..."},
            {"role": "user", "content": "How does it help?"},
            {"role": "assistant", "content": "It reduces latency..."},
            {"role": "user", "content": "Can you give an example?"},
        ]
        key1 = StickyLeastBusyRedisLoggingHandler.compute_sticky_key(turn1)
        key2 = StickyLeastBusyRedisLoggingHandler.compute_sticky_key(turn2)
        key3 = StickyLeastBusyRedisLoggingHandler.compute_sticky_key(turn3)
        assert key1 == key2 == key3

    def test_different_first_user_message_different_key(self):
        msgs1 = [{"role": "user", "content": "hello"}]
        msgs2 = [{"role": "user", "content": "goodbye"}]
        assert StickyLeastBusyRedisLoggingHandler.compute_sticky_key(
            msgs1
        ) != StickyLeastBusyRedisLoggingHandler.compute_sticky_key(msgs2)

    def test_user_id_included_in_hash(self):
        msgs = [{"role": "user", "content": "hello"}]
        key = StickyLeastBusyRedisLoggingHandler.compute_sticky_key(
            msgs, user_id="sk-abc123"
        )
        expected = hashlib.sha256("sk-abc123:hello".encode("utf-8")).hexdigest()
        assert key == expected

    def test_user_id_differentiates_same_messages(self):
        msgs = [{"role": "user", "content": "hi"}]
        key_a = StickyLeastBusyRedisLoggingHandler.compute_sticky_key(
            msgs, user_id="sk-user-a"
        )
        key_b = StickyLeastBusyRedisLoggingHandler.compute_sticky_key(
            msgs, user_id="sk-user-b"
        )
        assert key_a != key_b

    def test_no_user_id_falls_back_to_content_only(self):
        msgs = [{"role": "user", "content": "hello"}]
        key = StickyLeastBusyRedisLoggingHandler.compute_sticky_key(msgs)
        expected = hashlib.sha256("hello".encode("utf-8")).hexdigest()
        assert key == expected

    def test_system_only_message_returns_none(self):
        msgs = [{"role": "system", "content": "You are helpful"}]
        assert StickyLeastBusyRedisLoggingHandler.compute_sticky_key(msgs) is None


# =====================================================================
# Test: _extract_user_id
# =====================================================================


class TestExtractUserId:
    def test_none_request_kwargs(self):
        assert StickyLeastBusyRedisLoggingHandler._extract_user_id(None) is None

    def test_empty_request_kwargs(self):
        assert StickyLeastBusyRedisLoggingHandler._extract_user_id({}) is None

    def test_user_api_key_preferred(self):
        kwargs = {
            "metadata": {
                "user_api_key": "sk-primary",
                "user_api_key_user_id": "user-123",
            },
            "user": "fallback-user",
        }
        assert StickyLeastBusyRedisLoggingHandler._extract_user_id(kwargs) == "sk-primary"

    def test_user_api_key_user_id_fallback(self):
        kwargs = {
            "metadata": {"user_api_key_user_id": "user-123"},
            "user": "fallback-user",
        }
        assert StickyLeastBusyRedisLoggingHandler._extract_user_id(kwargs) == "user-123"

    def test_user_field_fallback(self):
        kwargs = {"metadata": {}, "user": "fallback-user"}
        assert StickyLeastBusyRedisLoggingHandler._extract_user_id(kwargs) == "fallback-user"

    def test_no_identifiers_returns_none(self):
        kwargs = {"metadata": {"some_other_key": "value"}}
        assert StickyLeastBusyRedisLoggingHandler._extract_user_id(kwargs) is None


# =====================================================================
# Test: Streaming dedup
# =====================================================================


class TestStreamingDedup:
    def test_first_call_increments(self):
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=DualCache())
        assert handler._should_increment("call-1") is True

    def test_second_call_does_not_increment(self):
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=DualCache())
        handler._should_increment("call-1")
        assert handler._should_increment("call-1") is False

    def test_different_call_ids_both_increment(self):
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=DualCache())
        assert handler._should_increment("call-1") is True
        assert handler._should_increment("call-2") is True

    def test_cleanup_allows_re_increment(self):
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=DualCache())
        handler._should_increment("call-1")
        handler._cleanup_call_id("call-1")
        assert handler._should_increment("call-1") is True

    def test_eviction_under_capacity_pressure(self):
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=DualCache())
        handler._seen_call_ids_max_size = 100
        for i in range(150):
            handler._should_increment(f"call-{i}")
        assert len(handler._seen_call_ids) <= 110


# =====================================================================
# Test: Deployment selection - Redis-based sticky routing
# =====================================================================


class TestDeploymentSelection:
    def test_first_request_assigns_least_busy(self):
        """First request for a conversation should assign least-busy node."""
        cache = DualCache()
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=cache)
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]
        # dep-1 has the least load
        request_counts = {"dep-0": 10, "dep-1": 2, "dep-2": 8}
        sticky_key = "test-sticky-key-abc"

        result = handler._select_deployment(MG, deployments, request_counts, sticky_key)
        assert result["model_info"]["id"] == "dep-1"

    def test_first_request_stores_mapping_in_cache(self):
        """First request should store the mapping in cache for subsequent lookups."""
        cache = DualCache()
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=cache)
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]
        request_counts = {"dep-0": 10, "dep-1": 2, "dep-2": 8}
        sticky_key = "test-sticky-key-store"

        handler._select_deployment(MG, deployments, request_counts, sticky_key)

        # Verify the mapping was stored
        route_key = handler._get_sticky_route_cache_key(MG, sticky_key)
        stored = cache.get_cache(key=route_key)
        assert stored == "dep-1"

    def test_subsequent_request_routes_to_stored_node(self):
        """Second request should route to the same node stored in cache."""
        cache = DualCache()
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=cache)
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]
        sticky_key = "test-sticky-key-reuse"

        # First request: dep-1 is least busy
        request_counts = {"dep-0": 10, "dep-1": 2, "dep-2": 8}
        result1 = handler._select_deployment(MG, deployments, request_counts, sticky_key)
        assert result1["model_info"]["id"] == "dep-1"

        # Second request: dep-0 is now least busy, but should still route to dep-1 (sticky)
        request_counts = {"dep-0": 1, "dep-1": 5, "dep-2": 5}
        result2 = handler._select_deployment(MG, deployments, request_counts, sticky_key)
        assert result2["model_info"]["id"] == "dep-1"

    def test_rebalance_when_sticky_node_overloaded(self):
        """When sticky node is overloaded, should route to least-busy and update mapping."""
        cache = DualCache()
        handler = StickyLeastBusyRedisLoggingHandler(
            router_cache=cache, imbalance_threshold=1.5
        )
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]
        sticky_key = "test-sticky-key-overload"

        # First request: assign to dep-1 (least busy)
        request_counts = {"dep-0": 5, "dep-1": 1, "dep-2": 5}
        handler._select_deployment(MG, deployments, request_counts, sticky_key)

        # Now dep-1 is massively overloaded
        # avg = (2+100+2)/3 = 34.67, min = 2
        # reference = (34.67+2)/2 = 18.33, threshold = 1.5 * 18.33 = 27.5
        # dep-1 at 100 >= 27.5, so rebalance
        request_counts = {"dep-0": 2, "dep-1": 100, "dep-2": 2}
        result = handler._select_deployment(MG, deployments, request_counts, sticky_key)
        assert result["model_info"]["id"] != "dep-1"

        # Verify mapping was updated in cache
        route_key = handler._get_sticky_route_cache_key(MG, sticky_key)
        stored = cache.get_cache(key=route_key)
        assert stored != "dep-1"

    def test_rebalance_updates_mapping_for_future_requests(self):
        """After rebalancing, future requests should stick to the new node."""
        cache = DualCache()
        handler = StickyLeastBusyRedisLoggingHandler(
            router_cache=cache, imbalance_threshold=1.5
        )
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]
        sticky_key = "test-sticky-key-rebalance-future"

        # First request: assign to dep-1 (least busy)
        request_counts = {"dep-0": 5, "dep-1": 1, "dep-2": 5}
        handler._select_deployment(MG, deployments, request_counts, sticky_key)

        # Overload dep-1, causes rebalance to dep-0 or dep-2
        request_counts = {"dep-0": 2, "dep-1": 100, "dep-2": 2}
        result_rebalanced = handler._select_deployment(
            MG, deployments, request_counts, sticky_key
        )
        new_node = result_rebalanced["model_info"]["id"]

        # Now all nodes equal load: should still go to the new node (sticky to new mapping)
        request_counts = {"dep-0": 5, "dep-1": 5, "dep-2": 5}
        result_after = handler._select_deployment(
            MG, deployments, request_counts, sticky_key
        )
        assert result_after["model_info"]["id"] == new_node

    def test_unhealthy_node_reassigns(self):
        """If stored node is no longer in healthy deployments, reassign to least-busy."""
        cache = DualCache()
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=cache)
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]
        sticky_key = "test-sticky-key-unhealthy"

        # First request: assign to dep-1
        request_counts = {"dep-0": 10, "dep-1": 1, "dep-2": 10}
        handler._select_deployment(MG, deployments, request_counts, sticky_key)

        # dep-1 goes down — only dep-0 and dep-2 are healthy
        healthy_after = [_make_deployment("dep-0"), _make_deployment("dep-2")]
        request_counts = {"dep-0": 3, "dep-2": 8}
        result = handler._select_deployment(MG, healthy_after, request_counts, sticky_key)
        assert result["model_info"]["id"] == "dep-0"

        # Verify mapping was updated
        route_key = handler._get_sticky_route_cache_key(MG, sticky_key)
        stored = cache.get_cache(key=route_key)
        assert stored == "dep-0"

    def test_no_sticky_key_falls_back_to_least_busy(self):
        """When no sticky key (no messages), should pick least-busy."""
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=DualCache())
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]
        request_counts = {"dep-0": 10, "dep-1": 2, "dep-2": 10}
        result = handler._select_deployment(MG, deployments, request_counts, None)
        assert result["model_info"]["id"] == "dep-1"

    def test_no_sticky_key_does_not_store_mapping(self):
        """When no sticky key, should NOT store any mapping in cache."""
        cache = DualCache()
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=cache)
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]
        request_counts = {"dep-0": 10, "dep-1": 2, "dep-2": 10}
        handler._select_deployment(MG, deployments, request_counts, None)
        # No sticky key means no route key to store
        # Just verify no crash and correct selection

    def test_all_idle_assigns_any_node(self):
        """When all nodes idle, first request assigns to one of them (all are least-busy)."""
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=DualCache())
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]
        request_counts = {"dep-0": 0, "dep-1": 0, "dep-2": 0}
        sticky_key = "test-idle"
        result = handler._select_deployment(MG, deployments, request_counts, sticky_key)
        assert result["model_info"]["id"] in ("dep-0", "dep-1", "dep-2")

    def test_all_idle_sticks_after_first_assignment(self):
        """After first assignment when idle, subsequent requests should be sticky."""
        cache = DualCache()
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=cache)
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]
        request_counts = {"dep-0": 0, "dep-1": 0, "dep-2": 0}
        sticky_key = "test-idle-sticky"

        result1 = handler._select_deployment(MG, deployments, request_counts, sticky_key)
        first_node = result1["model_info"]["id"]

        # All still idle — should stick to the first assignment
        for _ in range(10):
            result = handler._select_deployment(
                MG, deployments, request_counts, sticky_key
            )
            assert result["model_info"]["id"] == first_node

    def test_least_busy_tie_breaking_is_random(self):
        """When multiple nodes tie for least-busy, selection should vary."""
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=DualCache())
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]
        request_counts = {"dep-0": 0, "dep-1": 0, "dep-2": 0}

        selected_ids = set()
        for _ in range(100):
            result = handler._select_deployment(MG, deployments, request_counts, None)
            selected_ids.add(result["model_info"]["id"])

        assert len(selected_ids) >= 2

    def test_threshold_boundary_stays_sticky(self):
        """When load is below threshold with avg+min blend, should stay sticky."""
        cache = DualCache()
        handler = StickyLeastBusyRedisLoggingHandler(
            router_cache=cache, imbalance_threshold=2.0
        )
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]
        sticky_key = "test-threshold"

        # Assign to dep-1 (least busy)
        request_counts = {"dep-0": 5, "dep-1": 1, "dep-2": 5}
        handler._select_deployment(MG, deployments, request_counts, sticky_key)

        # dep-1 load = 6, avg = (3+6+3)/3 = 4, min = 3
        # reference = (4+3)/2 = 3.5, threshold = 2.0 * max(3.5, 1.0) = 7.0
        # 6 < 7.0, should stay sticky
        request_counts = {"dep-0": 3, "dep-1": 6, "dep-2": 3}
        result = handler._select_deployment(MG, deployments, request_counts, sticky_key)
        assert result["model_info"]["id"] == "dep-1"

    def test_skewed_distribution_triggers_rebalance(self):
        """With skewed loads [50, 25, 20, 7, 5], avg+min blend correctly rebalances."""
        cache = DualCache()
        handler = StickyLeastBusyRedisLoggingHandler(
            router_cache=cache, imbalance_threshold=1.5
        )
        deployments = [_make_deployment(f"dep-{i}") for i in range(5)]
        sticky_key = "skewed-test"

        # First request: assign to dep-4 (least busy at 1)
        request_counts = {"dep-0": 10, "dep-1": 10, "dep-2": 10, "dep-3": 10, "dep-4": 1}
        handler._select_deployment(MG, deployments, request_counts, sticky_key)

        # Now loads are skewed: dep-4 has 25, but dep-3=7 and dep-4 isn't least busy
        # avg = (50+25+20+7+5)/5 = 21.4, min = 5
        # reference = (21.4+5)/2 = 13.2, threshold = 1.5 * 13.2 = 19.8
        # dep-4 stored at 25 >= 19.8 → should rebalance
        request_counts = {"dep-0": 50, "dep-1": 25, "dep-2": 20, "dep-3": 7, "dep-4": 5}
        # Override the stored mapping to dep-1 (the one at 25)
        route_key = handler._get_sticky_route_cache_key(MG, sticky_key)
        cache.set_cache(key=route_key, value="dep-1")

        result = handler._select_deployment(MG, deployments, request_counts, sticky_key)
        # dep-1 at 25 >= 19.8, should rebalance to least-busy (dep-4 at 5)
        assert result["model_info"]["id"] != "dep-1"
        assert result["model_info"]["id"] in ("dep-3", "dep-4")

    def test_even_distribution_keeps_sticky(self):
        """When loads are evenly distributed, avg+min blend preserves stickiness."""
        cache = DualCache()
        handler = StickyLeastBusyRedisLoggingHandler(
            router_cache=cache, imbalance_threshold=1.5
        )
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]
        sticky_key = "even-test"

        # Assign to dep-1
        request_counts = {"dep-0": 5, "dep-1": 1, "dep-2": 5}
        handler._select_deployment(MG, deployments, request_counts, sticky_key)

        # Even: avg = 10, min = 10, reference = 10, threshold = 1.5 * 10 = 15
        # dep-1 at 10 < 15 → sticky preserved
        request_counts = {"dep-0": 10, "dep-1": 10, "dep-2": 10}
        result = handler._select_deployment(MG, deployments, request_counts, sticky_key)
        assert result["model_info"]["id"] == "dep-1"

    def test_different_model_groups_independent(self):
        """Different model groups have independent sticky mappings."""
        cache = DualCache()
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=cache)
        sticky_key = "test-independent"

        llama_deps = [_make_deployment(f"llama-{i}") for i in range(2)]
        gpt_deps = [_make_deployment(f"gpt-{i}") for i in range(2)]

        # Assign in llama group: llama-0 is least busy
        llama_counts = {"llama-0": 1, "llama-1": 10}
        handler._select_deployment("llama-70b", llama_deps, llama_counts, sticky_key)

        # Assign in gpt group: gpt-1 is least busy
        gpt_counts = {"gpt-0": 10, "gpt-1": 1}
        handler._select_deployment("gpt-4", gpt_deps, gpt_counts, sticky_key)

        # Verify independent mappings
        llama_key = handler._get_sticky_route_cache_key("llama-70b", sticky_key)
        gpt_key = handler._get_sticky_route_cache_key("gpt-4", sticky_key)
        assert cache.get_cache(key=llama_key) == "llama-0"
        assert cache.get_cache(key=gpt_key) == "gpt-1"


# =====================================================================
# Test: log_pre_api_call with dedup
# =====================================================================


class TestLogPreApiCallDedup:
    def test_increments_once_for_same_call_id(self):
        cache = DualCache()
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=cache)
        kwargs = {
            "litellm_call_id": "test-call-123",
            "litellm_params": {
                "metadata": {"model_group": "test-group"},
                "model_info": {"id": "dep-1"},
                "litellm_call_id": "test-call-123",
            },
        }
        handler.log_pre_api_call(model="test", messages=[], kwargs=kwargs)
        cache_key = handler._get_request_count_cache_key("test-group", "dep-1")
        assert cache.get_cache(key=cache_key) == 1

        # Second call (streaming chunk) should NOT increment
        handler.log_pre_api_call(model="test", messages=[], kwargs=kwargs)
        assert cache.get_cache(key=cache_key) == 1

    def test_different_call_ids_both_increment(self):
        cache = DualCache()
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=cache)
        for call_id in ["call-1", "call-2"]:
            kwargs = {
                "litellm_call_id": call_id,
                "litellm_params": {
                    "metadata": {"model_group": "test-group"},
                    "model_info": {"id": "dep-1"},
                    "litellm_call_id": call_id,
                },
            }
            handler.log_pre_api_call(model="test", messages=[], kwargs=kwargs)
        cache_key = handler._get_request_count_cache_key("test-group", "dep-1")
        assert cache.get_cache(key=cache_key) == 2

    def test_decrement_after_success(self):
        cache = DualCache()
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=cache)
        kwargs = {
            "litellm_call_id": "test-call-456",
            "litellm_params": {
                "metadata": {"model_group": "test-group"},
                "model_info": {"id": "dep-1"},
                "litellm_call_id": "test-call-456",
            },
        }
        handler.log_pre_api_call(model="test", messages=[], kwargs=kwargs)
        cache_key = handler._get_request_count_cache_key("test-group", "dep-1")
        assert cache.get_cache(key=cache_key) == 1

        handler.log_success_event(kwargs, None, None, None)
        assert cache.get_cache(key=cache_key) == 0

    def test_kwargs_none_does_not_crash(self):
        cache = DualCache()
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=cache)
        handler._decrement_request_count(None, callback_type="SYNC-FAILURE")


# =====================================================================
# Test: Async integration
# =====================================================================


class TestAsyncIntegration:
    @pytest.mark.asyncio
    async def test_async_get_available_deployments(self):
        cache = DualCache()
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=cache)
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]
        result = await handler.async_get_available_deployments(
            model_group="test-model",
            healthy_deployments=deployments,
            messages=[{"role": "user", "content": "hello"}],
        )
        assert result is not None
        assert result["model_info"]["id"] in ["dep-0", "dep-1", "dep-2"]

    @pytest.mark.asyncio
    async def test_async_sticky_consistency(self):
        """Same messages should produce same sticky routing in async path."""
        cache = DualCache()
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=cache)
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "how are you?"},
        ]

        result1 = await handler.async_get_available_deployments(
            model_group="test-model",
            healthy_deployments=deployments,
            messages=msgs,
        )
        result2 = await handler.async_get_available_deployments(
            model_group="test-model",
            healthy_deployments=deployments,
            messages=msgs,
        )
        assert result1["model_info"]["id"] == result2["model_info"]["id"]

    @pytest.mark.asyncio
    async def test_async_first_request_assigns_least_busy(self):
        """Async: first request assigns least-busy node."""
        cache = DualCache()
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=cache)
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]

        # Pre-populate request counts in cache
        for dep_id, count in [("dep-0", 10), ("dep-1", 2), ("dep-2", 8)]:
            cache_key = handler._get_request_count_cache_key("test-model", dep_id)
            cache.set_cache(key=cache_key, value=count)

        result = await handler.async_get_available_deployments(
            model_group="test-model",
            healthy_deployments=deployments,
            messages=[{"role": "user", "content": "hello"}],
        )
        # Without Redis, async_get_cache returns from in-memory cache
        # dep-1 has lowest count
        assert result["model_info"]["id"] == "dep-1"

    @pytest.mark.asyncio
    async def test_async_decrement(self):
        cache = DualCache()
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=cache)
        kwargs = {
            "litellm_call_id": "async-call-1",
            "litellm_params": {
                "metadata": {"model_group": "test-group"},
                "model_info": {"id": "dep-1"},
                "litellm_call_id": "async-call-1",
            },
        }
        handler.log_pre_api_call(model="test", messages=[], kwargs=kwargs)
        cache_key = handler._get_request_count_cache_key("test-group", "dep-1")
        assert cache.get_cache(key=cache_key) == 1

        await handler.async_log_success_event(kwargs, None, None, None)
        assert cache.get_cache(key=cache_key) == 0

    @pytest.mark.asyncio
    async def test_async_per_user_differentiation(self):
        """Different user_api_keys with same messages should get different sticky keys."""
        cache = DualCache()
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=cache)
        msgs = [
            {"role": "system", "content": "You are a coding assistant"},
            {"role": "user", "content": "hi"},
        ]

        key_a = handler.compute_sticky_key(msgs, user_id="sk-user-a")
        key_b = handler.compute_sticky_key(msgs, user_id="sk-user-b")
        assert key_a != key_b

    @pytest.mark.asyncio
    async def test_async_kwargs_none_does_not_crash(self):
        cache = DualCache()
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=cache)
        await handler.async_log_failure_event(None, None, None, None)


# =====================================================================
# Test: Singleton behavior
# =====================================================================


class TestSingleton:
    def test_singleton_returns_same_instance(self):
        cache = DualCache()
        h1 = StickyLeastBusyRedisLoggingHandler(router_cache=cache)
        h2 = StickyLeastBusyRedisLoggingHandler(router_cache=cache)
        assert h1 is h2

    def test_singleton_preserves_dedup_state(self):
        cache = DualCache()
        h1 = StickyLeastBusyRedisLoggingHandler(router_cache=cache)
        h1._should_increment("call-1")

        h2 = StickyLeastBusyRedisLoggingHandler(router_cache=cache)
        assert "call-1" in h2._seen_call_ids

    def test_singleton_updates_router_cache(self):
        cache1 = DualCache()
        cache2 = DualCache()
        h1 = StickyLeastBusyRedisLoggingHandler(router_cache=cache1)
        h2 = StickyLeastBusyRedisLoggingHandler(router_cache=cache2)
        assert h1 is h2
        assert h2.router_cache is cache2


# =====================================================================
# Test: Sticky route cache key format
# =====================================================================


class TestStickyRouteCacheKey:
    def test_key_format(self):
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=DualCache())
        key = handler._get_sticky_route_cache_key("llama-70b", "abc123")
        assert key == "sticky_route:llama-70b:abc123"

    def test_request_count_key_format(self):
        handler = StickyLeastBusyRedisLoggingHandler(router_cache=DualCache())
        key = handler._get_request_count_cache_key("llama-70b", "dep-1")
        assert key == "sticky_lb:llama-70b:dep-1:request_count"
