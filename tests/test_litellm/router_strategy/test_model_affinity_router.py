"""
Unit tests for Model Affinity (Session Pinning) router.

All tests are self-contained — no LLM calls, no external services.
"""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.router_strategy.model_affinity_router.model_affinity_router import (
    ModelAffinityRouter,
    _LocalAffinityCache,
)
from litellm.types.router import ModelAffinityConfig


# ---------------------------------------------------------------------------
# _LocalAffinityCache tests
# ---------------------------------------------------------------------------


class TestLocalAffinityCache:
    def _cache(self, max_size=100, ttl=60.0) -> _LocalAffinityCache:
        return _LocalAffinityCache(max_size=max_size, ttl=ttl)

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_set_and_get(self):
        cache = self._cache()
        self._run(cache.set("s1", "gpt-4o"))
        assert self._run(cache.get("s1")) == "gpt-4o"

    def test_miss_returns_none(self):
        cache = self._cache()
        assert self._run(cache.get("nonexistent")) is None

    def test_delete_removes_entry(self):
        cache = self._cache()
        self._run(cache.set("s1", "gpt-4o"))
        self._run(cache.delete("s1"))
        assert self._run(cache.get("s1")) is None

    def test_delete_missing_key_is_noop(self):
        cache = self._cache()
        self._run(cache.delete("nonexistent"))  # should not raise

    def test_ttl_expiry(self):
        cache = self._cache(ttl=0.05)  # 50 ms TTL
        self._run(cache.set("s1", "gpt-4o"))
        assert self._run(cache.get("s1")) == "gpt-4o"
        time.sleep(0.1)  # outlast the TTL
        assert self._run(cache.get("s1")) is None  # expired

    def test_lru_eviction(self):
        """When max_size is exceeded the oldest entry is evicted."""
        cache = self._cache(max_size=2)
        self._run(cache.set("a", "m1"))
        self._run(cache.set("b", "m2"))
        self._run(cache.set("c", "m3"))  # should evict "a"
        assert self._run(cache.get("a")) is None  # evicted
        assert self._run(cache.get("b")) == "m2"
        assert self._run(cache.get("c")) == "m3"

    def test_get_promotes_to_mru(self):
        """Accessing an entry should move it to MRU so it is not evicted first."""
        cache = self._cache(max_size=2)
        self._run(cache.set("a", "m1"))
        self._run(cache.set("b", "m2"))
        # access "a" so it becomes MRU
        self._run(cache.get("a"))
        self._run(cache.set("c", "m3"))  # should evict "b", not "a"
        assert self._run(cache.get("a")) == "m1"
        assert self._run(cache.get("b")) is None  # evicted

    def test_overwrite_refreshes_ttl(self):
        """Re-setting a key should reset its TTL."""
        cache = self._cache(ttl=0.05)
        self._run(cache.set("s1", "gpt-4o"))
        time.sleep(0.03)
        self._run(cache.set("s1", "gpt-4o"))  # refresh
        time.sleep(0.03)
        # 60 ms have passed in total, but TTL reset after 30 ms so still alive
        assert self._run(cache.get("s1")) == "gpt-4o"


# ---------------------------------------------------------------------------
# ModelAffinityRouter tests
# ---------------------------------------------------------------------------


def _make_affinity_router(ttl=600, max_sessions=1000) -> ModelAffinityRouter:
    config = ModelAffinityConfig(
        enabled=True, ttl=ttl, max_sessions=max_sessions, storage="local"
    )
    return ModelAffinityRouter(config=config)


class TestModelAffinityRouter:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_new_session_is_unpinned(self):
        r = _make_affinity_router()
        assert self._run(r.get_pinned_model("new-session")) is None

    def test_pin_and_retrieve(self):
        r = _make_affinity_router()
        self._run(r.pin_model("s1", "claude-sonnet"))
        assert self._run(r.get_pinned_model("s1")) == "claude-sonnet"

    def test_clear_session(self):
        r = _make_affinity_router()
        self._run(r.pin_model("s1", "gpt-4o"))
        self._run(r.clear_session("s1"))
        assert self._run(r.get_pinned_model("s1")) is None

    def test_multiple_sessions_are_independent(self):
        r = _make_affinity_router()
        self._run(r.pin_model("s1", "gpt-4o"))
        self._run(r.pin_model("s2", "claude-sonnet"))
        assert self._run(r.get_pinned_model("s1")) == "gpt-4o"
        assert self._run(r.get_pinned_model("s2")) == "claude-sonnet"

    def test_pin_can_be_updated(self):
        r = _make_affinity_router()
        self._run(r.pin_model("s1", "gpt-4o"))
        self._run(r.pin_model("s1", "claude-sonnet"))  # overwrite
        assert self._run(r.get_pinned_model("s1")) == "claude-sonnet"


# ---------------------------------------------------------------------------
# Router integration tests
# ---------------------------------------------------------------------------


class TestRouterAffinityIntegration:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def _make_router(self, **affinity_kwargs):
        import litellm

        return litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-4o",
                    "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"},
                },
                {
                    "model_name": "claude-sonnet",
                    "litellm_params": {
                        "model": "anthropic/claude-sonnet-4-20250514",
                        "api_key": "fake",
                    },
                    "routing_preferences": [
                        {
                            "name": "code_generation",
                            "description": "code programming debugging python function",
                        }
                    ],
                },
            ],
            content_routing={
                "enabled": True,
                "classifier": "rule_based",
                "default_model": "gpt-4o",
                "confidence_threshold": 0.0,
            },
            model_affinity={
                "enabled": True,
                "ttl": 600,
                "max_sessions": 1000,
                "storage": "local",
                **affinity_kwargs,
            },
        )

    def test_router_initializes_affinity_router(self):
        router = self._make_router()
        assert router.model_affinity_router is not None

    def test_first_request_pins_content_routed_model(self):
        """First request with session header should pin the content-routed model."""
        router = self._make_router()
        request_kwargs = {
            "metadata": {
                "headers": {"x-model-affinity": "session-abc"},
            }
        }
        messages = [{"role": "user", "content": "write a python function to sort a list"}]
        result = self._run(
            router.async_pre_routing_hook(
                model="gpt-4o",
                request_kwargs=request_kwargs,
                messages=messages,
            )
        )
        # Content routing selected claude-sonnet; affinity should have pinned it
        assert result is not None
        assert result.model == "claude-sonnet"
        decision = request_kwargs["metadata"].get("model_affinity_decision")
        assert decision is not None
        assert decision["status"] == "new"
        assert decision["model"] == "claude-sonnet"

        # Second request should be served from pin, bypassing content routing
        request_kwargs2 = {
            "metadata": {
                "headers": {"x-model-affinity": "session-abc"},
            }
        }
        messages2 = [{"role": "user", "content": "what is the weather today?"}]
        result2 = self._run(
            router.async_pre_routing_hook(
                model="gpt-4o",
                request_kwargs=request_kwargs2,
                messages=messages2,
            )
        )
        assert result2 is not None
        assert result2.model == "claude-sonnet"  # pinned, despite conversation prompt
        decision2 = request_kwargs2["metadata"].get("model_affinity_decision")
        assert decision2["status"] == "pinned"

    def test_different_sessions_are_independent(self):
        """Two sessions with different IDs must not interfere."""
        router = self._make_router()

        # Session A: code prompt → claude-sonnet
        kw_a = {"metadata": {"headers": {"x-model-affinity": "session-a"}}}
        self._run(
            router.async_pre_routing_hook(
                model="gpt-4o",
                request_kwargs=kw_a,
                messages=[{"role": "user", "content": "write a python function"}],
            )
        )

        # Session B: different ID → gets its own independent routing
        kw_b = {"metadata": {"headers": {"x-model-affinity": "session-b"}}}
        result_b = self._run(
            router.async_pre_routing_hook(
                model="gpt-4o",
                request_kwargs=kw_b,
                messages=[{"role": "user", "content": "write a python function"}],
            )
        )
        # Both session-a and session-b exist independently
        pinned_a = self._run(router.model_affinity_router.get_pinned_model("session-a"))
        pinned_b = self._run(router.model_affinity_router.get_pinned_model("session-b"))
        assert pinned_a is not None
        assert pinned_b is not None

    def test_no_session_header_bypasses_affinity(self):
        """Requests without X-Model-Affinity should route normally (no pin created)."""
        router = self._make_router()
        kw = {
            "metadata": {
                "headers": {},  # no affinity header
            }
        }
        result = self._run(
            router.async_pre_routing_hook(
                model="gpt-4o",
                request_kwargs=kw,
                messages=[{"role": "user", "content": "write a python function"}],
            )
        )
        # Content routing still runs normally
        assert result is not None
        assert result.model == "claude-sonnet"
        # But no pin was created
        assert "model_affinity_decision" not in kw["metadata"]

    def test_affinity_disabled_leaves_router_none(self):
        import litellm

        router = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-4o",
                    "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"},
                }
            ],
            model_affinity={"enabled": False},
        )
        assert router.model_affinity_router is None

    def test_affinity_without_config_leaves_router_none(self):
        import litellm

        router = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-4o",
                    "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"},
                }
            ],
        )
        assert router.model_affinity_router is None

    def test_specific_deployment_bypasses_affinity(self):
        """Affinity must not activate when specific_deployment=True."""
        router = self._make_router()
        # Pre-pin a session
        self._run(router.model_affinity_router.pin_model("s1", "claude-sonnet"))

        kw = {"metadata": {"headers": {"x-model-affinity": "s1"}}}
        result = self._run(
            router.async_pre_routing_hook(
                model="gpt-4o",
                request_kwargs=kw,
                messages=[{"role": "user", "content": "hello"}],
                specific_deployment=True,
            )
        )
        # specific_deployment=True skips both affinity and content routing
        assert result is None
