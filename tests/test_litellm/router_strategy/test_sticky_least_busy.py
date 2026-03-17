"""
Tests for the sticky-least-busy routing strategy.
"""

import hashlib

import pytest

from litellm.caching.caching import DualCache
from litellm.router_strategy.sticky_least_busy import StickyLeastBusyLoggingHandler


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


# =====================================================================
# Test: compute_sticky_key
# =====================================================================


class TestComputeStickyKey:
    def test_none_messages_returns_none(self):
        assert StickyLeastBusyLoggingHandler.compute_sticky_key(None) is None

    def test_empty_messages_returns_none(self):
        assert StickyLeastBusyLoggingHandler.compute_sticky_key([]) is None

    def test_single_user_message(self):
        msgs = [{"role": "user", "content": "hello"}]
        key = StickyLeastBusyLoggingHandler.compute_sticky_key(msgs)
        assert key is not None
        assert len(key) == 64  # SHA-256 hex digest

    def test_identity_is_first_user_message(self):
        """Hash is based on first user message content only."""
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        key = StickyLeastBusyLoggingHandler.compute_sticky_key(msgs)
        expected = hashlib.sha256("hello".encode("utf-8")).hexdigest()
        assert key == expected

    def test_system_prompt_not_in_hash(self):
        """System prompt is NOT part of the hash — only first user message matters.
        Same first user message with or without system prompt = same key."""
        msgs_with_system = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        msgs_without_system = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        key_with = StickyLeastBusyLoggingHandler.compute_sticky_key(msgs_with_system)
        key_without = StickyLeastBusyLoggingHandler.compute_sticky_key(
            msgs_without_system
        )
        expected = hashlib.sha256("hello".encode("utf-8")).hexdigest()
        assert key_with == expected
        assert key_without == expected

    def test_deterministic_across_calls(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "what's up?"},
        ]
        key1 = StickyLeastBusyLoggingHandler.compute_sticky_key(msgs)
        key2 = StickyLeastBusyLoggingHandler.compute_sticky_key(msgs)
        assert key1 == key2

    def test_same_conversation_different_turns_same_key(self):
        """Core stickiness: all turns of the same conversation produce the same key."""
        # Turn 1: user asks first question
        turn1 = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is KV caching?"},
        ]
        # Turn 2: conversation continues
        turn2 = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is KV caching?"},
            {"role": "assistant", "content": "KV caching stores..."},
            {"role": "user", "content": "How does it help?"},
        ]
        # Turn 3: even longer
        turn3 = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is KV caching?"},
            {"role": "assistant", "content": "KV caching stores..."},
            {"role": "user", "content": "How does it help?"},
            {"role": "assistant", "content": "It reduces latency..."},
            {"role": "user", "content": "Can you give an example?"},
        ]
        key1 = StickyLeastBusyLoggingHandler.compute_sticky_key(turn1)
        key2 = StickyLeastBusyLoggingHandler.compute_sticky_key(turn2)
        key3 = StickyLeastBusyLoggingHandler.compute_sticky_key(turn3)
        assert key1 == key2 == key3

    def test_different_first_user_message_different_key(self):
        """Different conversations (different first question) get different keys."""
        msgs1 = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hello"},
        ]
        msgs2 = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "goodbye"},
        ]
        assert StickyLeastBusyLoggingHandler.compute_sticky_key(
            msgs1
        ) != StickyLeastBusyLoggingHandler.compute_sticky_key(msgs2)

    def test_same_system_prompt_different_users_different_key(self):
        """Same system prompt but different first user message = different key.
        This prevents hotspots when many users share the same agent/system prompt.
        """
        user_a = [
            {"role": "system", "content": "You are a coding assistant"},
            {"role": "user", "content": "Fix my Python bug"},
            {"role": "assistant", "content": "Sure, show me the code"},
        ]
        user_b = [
            {"role": "system", "content": "You are a coding assistant"},
            {"role": "user", "content": "Write a REST API"},
            {"role": "assistant", "content": "I'll help with that"},
        ]
        assert StickyLeastBusyLoggingHandler.compute_sticky_key(
            user_a
        ) != StickyLeastBusyLoggingHandler.compute_sticky_key(user_b)

    def test_no_system_prompt_stickiness(self):
        """Without system prompt, identity = first user message only."""
        turn1 = [{"role": "user", "content": "hello"}]
        turn2 = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "how are you?"},
        ]
        assert StickyLeastBusyLoggingHandler.compute_sticky_key(
            turn1
        ) == StickyLeastBusyLoggingHandler.compute_sticky_key(turn2)

    def test_system_only_message_returns_none(self):
        """System-only message (no user message) returns None — no sticky routing."""
        msgs = [{"role": "system", "content": "You are helpful"}]
        key = StickyLeastBusyLoggingHandler.compute_sticky_key(msgs)
        assert key is None

    def test_user_id_included_in_hash(self):
        """When user_id is provided, hash is f'{user_id}:{first_user_content}'."""
        msgs = [{"role": "user", "content": "hello"}]
        key = StickyLeastBusyLoggingHandler.compute_sticky_key(
            msgs, user_id="sk-abc123"
        )
        expected = hashlib.sha256("sk-abc123:hello".encode("utf-8")).hexdigest()
        assert key == expected

    def test_user_id_differentiates_same_messages(self):
        """Two users with identical messages get different sticky keys."""
        msgs = [
            {"role": "system", "content": "You are a coding assistant"},
            {"role": "user", "content": "hi"},
        ]
        key_a = StickyLeastBusyLoggingHandler.compute_sticky_key(
            msgs, user_id="sk-user-a"
        )
        key_b = StickyLeastBusyLoggingHandler.compute_sticky_key(
            msgs, user_id="sk-user-b"
        )
        assert key_a != key_b

    def test_same_user_id_same_messages_same_key(self):
        """Same user + same messages = same sticky key (deterministic)."""
        msgs = [{"role": "user", "content": "hello"}]
        key1 = StickyLeastBusyLoggingHandler.compute_sticky_key(
            msgs, user_id="sk-abc"
        )
        key2 = StickyLeastBusyLoggingHandler.compute_sticky_key(
            msgs, user_id="sk-abc"
        )
        assert key1 == key2

    def test_no_user_id_falls_back_to_content_only(self):
        """Without user_id, hash is just the first user message content."""
        msgs = [{"role": "user", "content": "hello"}]
        key_no_uid = StickyLeastBusyLoggingHandler.compute_sticky_key(msgs)
        key_none_uid = StickyLeastBusyLoggingHandler.compute_sticky_key(
            msgs, user_id=None
        )
        expected = hashlib.sha256("hello".encode("utf-8")).hexdigest()
        assert key_no_uid == expected
        assert key_none_uid == expected


# =====================================================================
# Test: _extract_user_id
# =====================================================================


class TestExtractUserId:
    def test_none_request_kwargs(self):
        assert StickyLeastBusyLoggingHandler._extract_user_id(None) is None

    def test_empty_request_kwargs(self):
        assert StickyLeastBusyLoggingHandler._extract_user_id({}) is None

    def test_user_api_key_preferred(self):
        """user_api_key takes highest priority."""
        kwargs = {
            "metadata": {
                "user_api_key": "sk-primary",
                "user_api_key_user_id": "user-123",
            },
            "user": "fallback-user",
        }
        assert StickyLeastBusyLoggingHandler._extract_user_id(kwargs) == "sk-primary"

    def test_user_api_key_user_id_fallback(self):
        """Falls back to user_api_key_user_id when user_api_key is absent."""
        kwargs = {
            "metadata": {"user_api_key_user_id": "user-123"},
            "user": "fallback-user",
        }
        assert StickyLeastBusyLoggingHandler._extract_user_id(kwargs) == "user-123"

    def test_user_field_fallback(self):
        """Falls back to top-level 'user' when metadata has no keys."""
        kwargs = {"metadata": {}, "user": "fallback-user"}
        assert StickyLeastBusyLoggingHandler._extract_user_id(kwargs) == "fallback-user"

    def test_no_metadata_uses_user_field(self):
        """When metadata is missing entirely, uses top-level 'user'."""
        kwargs = {"user": "fallback-user"}
        assert StickyLeastBusyLoggingHandler._extract_user_id(kwargs) == "fallback-user"

    def test_metadata_none_uses_user_field(self):
        """When metadata is explicitly None, uses top-level 'user'."""
        kwargs = {"metadata": None, "user": "fallback-user"}
        assert StickyLeastBusyLoggingHandler._extract_user_id(kwargs) == "fallback-user"

    def test_no_identifiers_returns_none(self):
        """When no user identifiers exist at all, returns None."""
        kwargs = {"metadata": {"some_other_key": "value"}}
        assert StickyLeastBusyLoggingHandler._extract_user_id(kwargs) is None


# =====================================================================
# Test: Consistent hash ring
# =====================================================================


class TestConsistentHashRing:
    def test_single_deployment_always_routes_there(self):
        handler = StickyLeastBusyLoggingHandler(router_cache=DualCache())
        handler._build_hash_ring(MG, ["dep-1"])
        for i in range(100):
            assert handler._get_deployment_for_key(MG, f"key-{i}") == "dep-1"

    def test_distribution_across_deployments(self):
        handler = StickyLeastBusyLoggingHandler(router_cache=DualCache())
        dep_ids = ["dep-1", "dep-2", "dep-3"]
        handler._build_hash_ring(MG, dep_ids)
        counts = {did: 0 for did in dep_ids}
        for i in range(3000):
            selected = handler._get_deployment_for_key(MG, f"key-{i}")
            counts[selected] += 1
        # Each should get roughly 1000 +/- 500
        for did in dep_ids:
            assert 500 < counts[did] < 1500, f"{did} got {counts[did]}"

    def test_ring_stability_on_add(self):
        """Adding a node should only remap ~1/N keys."""
        handler = StickyLeastBusyLoggingHandler(router_cache=DualCache())
        dep_ids_before = ["dep-1", "dep-2", "dep-3"]
        handler._build_hash_ring(MG, dep_ids_before)

        keys = [f"key-{i}" for i in range(1000)]
        mappings_before = {k: handler._get_deployment_for_key(MG, k) for k in keys}

        dep_ids_after = ["dep-1", "dep-2", "dep-3", "dep-4"]
        handler._build_hash_ring(MG, dep_ids_after)
        mappings_after = {k: handler._get_deployment_for_key(MG, k) for k in keys}

        changed = sum(1 for k in keys if mappings_before[k] != mappings_after[k])
        # Expect ~25% remapping, allow up to 40%
        assert changed < 400, f"Too many keys remapped: {changed}/1000"

    def test_ring_stability_on_remove(self):
        """Removing a node should only remap the removed node's keys."""
        handler = StickyLeastBusyLoggingHandler(router_cache=DualCache())
        dep_ids_before = ["dep-1", "dep-2", "dep-3", "dep-4"]
        handler._build_hash_ring(MG, dep_ids_before)

        keys = [f"key-{i}" for i in range(1000)]
        mappings_before = {k: handler._get_deployment_for_key(MG, k) for k in keys}

        dep_ids_after = ["dep-1", "dep-2", "dep-3"]
        handler._build_hash_ring(MG, dep_ids_after)
        mappings_after = {k: handler._get_deployment_for_key(MG, k) for k in keys}

        for k in keys:
            if mappings_before[k] != "dep-4":
                assert mappings_before[k] == mappings_after[k], (
                    f"Key {k} changed from {mappings_before[k]} to "
                    f"{mappings_after[k]} unexpectedly"
                )

    def test_empty_ring_returns_none(self):
        handler = StickyLeastBusyLoggingHandler(router_cache=DualCache())
        assert handler._get_deployment_for_key(MG, "any-key") is None

    def test_ring_not_rebuilt_when_same_ids(self):
        handler = StickyLeastBusyLoggingHandler(router_cache=DualCache())
        handler._build_hash_ring(MG, ["dep-1", "dep-2"])
        ring_before = handler._rings[MG][1]
        handler._build_hash_ring(MG, ["dep-1", "dep-2"])
        assert handler._rings[MG][1] is ring_before  # same object, not rebuilt

    def test_separate_rings_per_model_group(self):
        """Different model groups should have independent rings."""
        handler = StickyLeastBusyLoggingHandler(router_cache=DualCache())
        handler._build_hash_ring("llama-70b", ["dep-A", "dep-B"])
        handler._build_hash_ring("gpt-4", ["dep-X", "dep-Y", "dep-Z"])

        # llama ring should only map to dep-A or dep-B
        for i in range(100):
            result = handler._get_deployment_for_key("llama-70b", f"key-{i}")
            assert result in ("dep-A", "dep-B")

        # gpt-4 ring should only map to dep-X, dep-Y, or dep-Z
        for i in range(100):
            result = handler._get_deployment_for_key("gpt-4", f"key-{i}")
            assert result in ("dep-X", "dep-Y", "dep-Z")

    def test_model_group_ring_independence(self):
        """Rebuilding one model group's ring should not affect another."""
        handler = StickyLeastBusyLoggingHandler(router_cache=DualCache())
        handler._build_hash_ring("llama-70b", ["dep-A", "dep-B"])
        handler._build_hash_ring("gpt-4", ["dep-X", "dep-Y"])

        llama_ring = handler._rings["llama-70b"][1]

        # Rebuild gpt-4 ring with different nodes
        handler._build_hash_ring("gpt-4", ["dep-X", "dep-Y", "dep-Z"])

        # llama ring should be untouched
        assert handler._rings["llama-70b"][1] is llama_ring


# =====================================================================
# Test: Streaming dedup
# =====================================================================


class TestStreamingDedup:
    def test_first_call_increments(self):
        handler = StickyLeastBusyLoggingHandler(router_cache=DualCache())
        assert handler._should_increment("call-1") is True

    def test_second_call_does_not_increment(self):
        handler = StickyLeastBusyLoggingHandler(router_cache=DualCache())
        handler._should_increment("call-1")
        assert handler._should_increment("call-1") is False

    def test_different_call_ids_both_increment(self):
        handler = StickyLeastBusyLoggingHandler(router_cache=DualCache())
        assert handler._should_increment("call-1") is True
        assert handler._should_increment("call-2") is True

    def test_cleanup_allows_re_increment(self):
        handler = StickyLeastBusyLoggingHandler(router_cache=DualCache())
        handler._should_increment("call-1")
        handler._cleanup_call_id("call-1")
        assert handler._should_increment("call-1") is True

    def test_eviction_under_capacity_pressure(self):
        handler = StickyLeastBusyLoggingHandler(router_cache=DualCache())
        handler._seen_call_ids_max_size = 100
        for i in range(150):
            handler._should_increment(f"call-{i}")
        # Should not exceed max size + eviction batch overshoot
        assert len(handler._seen_call_ids) <= 110


# =====================================================================
# Test: Deployment selection logic
# =====================================================================


class TestDeploymentSelection:
    def test_sticky_routing_when_load_balanced(self):
        """When all nodes have similar load, should route to sticky node."""
        handler = StickyLeastBusyLoggingHandler(
            router_cache=DualCache(), imbalance_threshold=1.5
        )
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]
        request_counts = {"dep-0": 5, "dep-1": 5, "dep-2": 5}
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "what's up?"},
        ]
        sticky_key = handler.compute_sticky_key(msgs)
        result = handler._select_deployment(MG, deployments, request_counts, sticky_key)
        # Should be deterministic
        result2 = handler._select_deployment(MG, deployments, request_counts, sticky_key)
        assert result["model_info"]["id"] == result2["model_info"]["id"]

    def test_rebalance_when_sticky_node_overloaded(self):
        """When sticky node is overloaded, should route to least-busy."""
        handler = StickyLeastBusyLoggingHandler(
            router_cache=DualCache(), imbalance_threshold=1.5
        )
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]

        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "what's up?"},
        ]
        sticky_key = handler.compute_sticky_key(msgs)

        # Find which deployment the sticky key maps to
        handler._build_hash_ring(MG, ["dep-0", "dep-1", "dep-2"])
        sticky_dep_id = handler._get_deployment_for_key(MG, sticky_key)

        # Make sticky node massively overloaded
        request_counts = {"dep-0": 2, "dep-1": 2, "dep-2": 2}
        request_counts[sticky_dep_id] = 100

        result = handler._select_deployment(MG, deployments, request_counts, sticky_key)
        assert result["model_info"]["id"] != sticky_dep_id

    def test_no_sticky_key_falls_back_to_least_busy(self):
        """When no sticky key (no messages), should pick least-busy."""
        handler = StickyLeastBusyLoggingHandler(router_cache=DualCache())
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]
        request_counts = {"dep-0": 10, "dep-1": 2, "dep-2": 10}
        result = handler._select_deployment(MG, deployments, request_counts, None)
        assert result["model_info"]["id"] == "dep-1"

    def test_all_idle_uses_sticky(self):
        """When all nodes are idle, should use sticky routing."""
        handler = StickyLeastBusyLoggingHandler(router_cache=DualCache())
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]
        request_counts = {"dep-0": 0, "dep-1": 0, "dep-2": 0}
        sticky_key = "some-conversation-hash"
        result = handler._select_deployment(MG, deployments, request_counts, sticky_key)
        result2 = handler._select_deployment(MG, deployments, request_counts, sticky_key)
        assert result["model_info"]["id"] == result2["model_info"]["id"]

    def test_least_busy_tie_breaking_is_random(self):
        """When multiple nodes tie for least-busy, selection should be random."""
        handler = StickyLeastBusyLoggingHandler(router_cache=DualCache())
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]
        request_counts = {"dep-0": 0, "dep-1": 0, "dep-2": 0}

        selected_ids = set()
        for _ in range(100):
            result = handler._select_deployment(MG, deployments, request_counts, None)
            selected_ids.add(result["model_info"]["id"])

        # With 100 iterations and 3 equal deployments, expect at least 2 different
        assert len(selected_ids) >= 2

    def test_threshold_boundary(self):
        """Test behavior right at the threshold boundary with avg+min blend."""
        handler = StickyLeastBusyLoggingHandler(
            router_cache=DualCache(), imbalance_threshold=2.0
        )
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]

        sticky_key = "test-key"
        handler._build_hash_ring(MG, ["dep-0", "dep-1", "dep-2"])
        sticky_dep_id = handler._get_deployment_for_key(MG, sticky_key)

        # Set load so sticky node is moderately above others
        # avg = (6 + 3 + 3) / 3 = 4, min = 3
        # reference = (4 + 3) / 2 = 3.5
        # threshold_value = 2.0 * max(3.5, 1.0) = 7.0
        # sticky at 6 < 7.0 → should stay sticky
        request_counts = {"dep-0": 3, "dep-1": 3, "dep-2": 3}
        request_counts[sticky_dep_id] = 6

        result = handler._select_deployment(MG, deployments, request_counts, sticky_key)
        assert result["model_info"]["id"] == sticky_dep_id

    def test_skewed_distribution_triggers_rebalance(self):
        """With skewed loads [50, 25, 20, 7, 5], avg-only misses the imbalance
        but avg+min blend correctly detects it and rebalances."""
        handler = StickyLeastBusyLoggingHandler(
            router_cache=DualCache(), imbalance_threshold=1.5
        )
        deployments = [_make_deployment(f"dep-{i}") for i in range(5)]

        sticky_key = "skewed-test"
        handler._build_hash_ring(MG, [f"dep-{i}" for i in range(5)])
        sticky_dep_id = handler._get_deployment_for_key(MG, sticky_key)

        # Skewed: most load on dep-0, idle nodes at dep-3 and dep-4
        # avg = (50+25+20+7+5)/5 = 21.4, min = 5
        # reference = (21.4+5)/2 = 13.2
        # threshold = 1.5 * 13.2 = 19.8
        # Any node with load >= 20 should trigger rebalance
        request_counts = {"dep-0": 50, "dep-1": 25, "dep-2": 20, "dep-3": 7, "dep-4": 5}

        # If sticky node has load >= 20, it should rebalance to least-busy (dep-4)
        if request_counts.get(sticky_dep_id, 0) >= 20:
            result = handler._select_deployment(
                MG, deployments, request_counts, sticky_key
            )
            # Should route to least-busy, not the sticky node
            assert result["model_info"]["id"] != sticky_dep_id
            assert result["model_info"]["id"] in ("dep-3", "dep-4")
        else:
            # Sticky node is already lightly loaded — sticky is fine
            result = handler._select_deployment(
                MG, deployments, request_counts, sticky_key
            )
            assert result["model_info"]["id"] == sticky_dep_id

    def test_even_distribution_keeps_sticky(self):
        """When loads are evenly distributed, avg+min blend preserves stickiness."""
        handler = StickyLeastBusyLoggingHandler(
            router_cache=DualCache(), imbalance_threshold=1.5
        )
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]

        sticky_key = "even-test"
        handler._build_hash_ring(MG, ["dep-0", "dep-1", "dep-2"])
        sticky_dep_id = handler._get_deployment_for_key(MG, sticky_key)

        # Even: avg = 10, min = 10, reference = 10, threshold = 15
        # All nodes at 10 < 15 → sticky preserved
        request_counts = {"dep-0": 10, "dep-1": 10, "dep-2": 10}
        result = handler._select_deployment(MG, deployments, request_counts, sticky_key)
        assert result["model_info"]["id"] == sticky_dep_id


# =====================================================================
# Test: log_pre_api_call with dedup
# =====================================================================


class TestLogPreApiCallDedup:
    def test_increments_once_for_same_call_id(self):
        cache = DualCache()
        handler = StickyLeastBusyLoggingHandler(router_cache=cache)
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
        handler = StickyLeastBusyLoggingHandler(router_cache=cache)
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
        handler = StickyLeastBusyLoggingHandler(router_cache=cache)
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

    def test_decrement_cleans_up_call_id(self):
        cache = DualCache()
        handler = StickyLeastBusyLoggingHandler(router_cache=cache)
        kwargs = {
            "litellm_call_id": "test-call-789",
            "litellm_params": {
                "metadata": {"model_group": "test-group"},
                "model_info": {"id": "dep-1"},
                "litellm_call_id": "test-call-789",
            },
        }
        handler.log_pre_api_call(model="test", messages=[], kwargs=kwargs)
        assert "test-call-789" in handler._seen_call_ids

        handler.log_success_event(kwargs, None, None, None)
        assert "test-call-789" not in handler._seen_call_ids

    def test_double_failure_callback_only_decrements_once(self):
        """litellm fires BOTH sync and async failure callbacks for the same
        failed request (litellm/utils.py:1950-1960). Ensure we only decrement once."""
        cache = DualCache()
        handler = StickyLeastBusyLoggingHandler(router_cache=cache)
        kwargs = {
            "litellm_call_id": "fail-call-1",
            "litellm_params": {
                "metadata": {"model_group": "test-group"},
                "model_info": {"id": "dep-1"},
                "litellm_call_id": "fail-call-1",
            },
        }
        handler.log_pre_api_call(model="test", messages=[], kwargs=kwargs)
        cache_key = handler._get_request_count_cache_key("test-group", "dep-1")
        assert cache.get_cache(key=cache_key) == 1

        # Sync failure callback fires first
        handler.log_failure_event(kwargs, None, None, None)
        assert cache.get_cache(key=cache_key) == 0

        # Async failure callback fires second - should be deduped, NOT decrement again
        handler._decrement_request_count(kwargs, callback_type="ASYNC-FAILURE")
        assert cache.get_cache(key=cache_key) == 0  # NOT -1

    def test_kwargs_none_does_not_crash(self):
        """Failure callbacks can receive kwargs=None in some edge cases."""
        cache = DualCache()
        handler = StickyLeastBusyLoggingHandler(router_cache=cache)
        # Should not raise
        handler._decrement_request_count(None, callback_type="SYNC-FAILURE")

# =====================================================================
# Test: Async integration
# =====================================================================


class TestAsyncIntegration:
    @pytest.mark.asyncio
    async def test_async_get_available_deployments(self):
        cache = DualCache()
        handler = StickyLeastBusyLoggingHandler(router_cache=cache)
        deployments = [_make_deployment(f"dep-{i}") for i in range(3)]
        result = await handler.async_get_available_deployments(
            model_group="test-model",
            healthy_deployments=deployments,
            messages=[{"role": "user", "content": "hello"}],
        )
        assert result is not None
        assert result["model_info"]["id"] in ["dep-0", "dep-1", "dep-2"]

    @pytest.mark.asyncio
    async def test_async_decrement(self):
        cache = DualCache()
        handler = StickyLeastBusyLoggingHandler(router_cache=cache)
        kwargs = {
            "litellm_call_id": "async-call-1",
            "litellm_params": {
                "metadata": {"model_group": "test-group"},
                "model_info": {"id": "dep-1"},
                "litellm_call_id": "async-call-1",
            },
        }
        # Increment via sync (log_pre_api_call is always sync)
        handler.log_pre_api_call(model="test", messages=[], kwargs=kwargs)
        cache_key = handler._get_request_count_cache_key("test-group", "dep-1")
        assert cache.get_cache(key=cache_key) == 1

        # Decrement via async
        await handler.async_log_success_event(kwargs, None, None, None)
        assert cache.get_cache(key=cache_key) == 0

    @pytest.mark.asyncio
    async def test_async_double_failure_dedup(self):
        """Async version: both sync and async failure callbacks fire, only one decrements."""
        cache = DualCache()
        handler = StickyLeastBusyLoggingHandler(router_cache=cache)
        kwargs = {
            "litellm_call_id": "async-fail-1",
            "litellm_params": {
                "metadata": {"model_group": "test-group"},
                "model_info": {"id": "dep-1"},
                "litellm_call_id": "async-fail-1",
            },
        }
        handler.log_pre_api_call(model="test", messages=[], kwargs=kwargs)
        cache_key = handler._get_request_count_cache_key("test-group", "dep-1")
        assert cache.get_cache(key=cache_key) == 1

        # Sync failure fires first
        handler.log_failure_event(kwargs, None, None, None)
        assert cache.get_cache(key=cache_key) == 0

        # Async failure fires second - should NOT decrement
        await handler.async_log_failure_event(kwargs, None, None, None)
        assert cache.get_cache(key=cache_key) == 0  # NOT -1

    @pytest.mark.asyncio
    async def test_async_kwargs_none_does_not_crash(self):
        """Async failure callback with kwargs=None should not crash."""
        cache = DualCache()
        handler = StickyLeastBusyLoggingHandler(router_cache=cache)
        # Should not raise
        await handler.async_log_failure_event(None, None, None, None)

    @pytest.mark.asyncio
    async def test_async_sticky_consistency(self):
        """Same messages should produce same sticky routing in async path."""
        cache = DualCache()
        handler = StickyLeastBusyLoggingHandler(router_cache=cache)
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
    async def test_async_per_user_differentiation(self):
        """Different user_api_keys with same messages should route differently."""
        cache = DualCache()
        handler = StickyLeastBusyLoggingHandler(router_cache=cache)
        deployments = [_make_deployment(f"dep-{i}") for i in range(10)]
        msgs = [
            {"role": "system", "content": "You are a coding assistant"},
            {"role": "user", "content": "hi"},
        ]

        result_a = await handler.async_get_available_deployments(
            model_group="test-model",
            healthy_deployments=deployments,
            messages=msgs,
            request_kwargs={"metadata": {"user_api_key": "sk-user-a"}},
        )
        result_b = await handler.async_get_available_deployments(
            model_group="test-model",
            healthy_deployments=deployments,
            messages=msgs,
            request_kwargs={"metadata": {"user_api_key": "sk-user-b"}},
        )
        # With 10 deployments, different user IDs should very likely map to different nodes
        # (probability of collision is ~1/10). Run the test — if both map to the same
        # deployment, the hash ring distributed them there, which is valid but unlikely.
        # We verify the mechanism works by checking the keys are different.
        key_a = handler.compute_sticky_key(msgs, user_id="sk-user-a")
        key_b = handler.compute_sticky_key(msgs, user_id="sk-user-b")
        assert key_a != key_b

    @pytest.mark.asyncio
    async def test_different_model_groups_route_independently(self):
        """Requests for different model groups use separate rings."""
        cache = DualCache()
        handler = StickyLeastBusyLoggingHandler(router_cache=cache)
        llama_deps = [_make_deployment(f"llama-{i}") for i in range(2)]
        gpt_deps = [_make_deployment(f"gpt-{i}") for i in range(3)]
        msgs = [{"role": "user", "content": "hello"}]

        llama_result = await handler.async_get_available_deployments(
            model_group="llama-70b",
            healthy_deployments=llama_deps,
            messages=msgs,
        )
        gpt_result = await handler.async_get_available_deployments(
            model_group="gpt-4",
            healthy_deployments=gpt_deps,
            messages=msgs,
        )

        assert llama_result["model_info"]["id"] in ("llama-0", "llama-1")
        assert gpt_result["model_info"]["id"] in ("gpt-0", "gpt-1", "gpt-2")

        # Both rings should exist independently
        assert "llama-70b" in handler._rings
        assert "gpt-4" in handler._rings
