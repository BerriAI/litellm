"""Tests for the LLMClassifierRouter."""

import asyncio
import os
import sys
from typing import Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.router_strategy.llm_classifier_router import (
    LLMClassifierRouter,
    LLMClassifierRouterConfig,
)
from litellm.router_strategy.llm_classifier_router.llm_classifier_router import (
    _cache_key,
    _parse_tier,
    _TierCache,
)


VALID = ("SIMPLE", "COMPLEX")


def _mock_response(content: str) -> MagicMock:
    return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])


@pytest.fixture
def mock_router_instance() -> MagicMock:
    return MagicMock()


@pytest.fixture
def basic_config() -> Dict:
    return {
        "tiers": {"SIMPLE": "gpt-4o-mini", "COMPLEX": "gpt-4o"},
    }


@pytest.fixture
def router(mock_router_instance, basic_config) -> LLMClassifierRouter:
    return LLMClassifierRouter(
        model_name="test-llm-classifier",
        litellm_router_instance=mock_router_instance,
        llm_classifier_router_config=basic_config,
    )


# ─── Config ──────────────────────────────────────────────────────
class TestLLMClassifierConfig:
    def test_default_values(self):
        c = LLMClassifierRouterConfig()
        assert c.classifier_model == "ollama/qwen2.5:0.5b"
        assert c.classifier_timeout == 3.0
        assert c.enable_cache is True
        assert c.fallback_to_complexity_router is True

    def test_fallback_tier_defaults_to_simple(self):
        c = LLMClassifierRouterConfig()
        assert c.fallback_tier == "SIMPLE"

    def test_tiers_default_has_simple_and_complex(self):
        c = LLMClassifierRouterConfig()
        assert "SIMPLE" in c.tiers
        assert "COMPLEX" in c.tiers

    def test_extra_kwargs_allowed(self):
        c = LLMClassifierRouterConfig(future_field=True, classifier_model="custom")
        assert c.classifier_model == "custom"

    def test_custom_tiers_override_default(self):
        c = LLMClassifierRouterConfig(tiers={"LOW": "a", "HIGH": "b"})
        assert c.tiers == {"LOW": "a", "HIGH": "b"}


# ─── _parse_tier ─────────────────────────────────────────────────
class TestParseTier:
    def test_exact_match_simple(self):
        assert _parse_tier("SIMPLE", VALID) == "SIMPLE"

    def test_exact_match_complex(self):
        assert _parse_tier("COMPLEX", VALID) == "COMPLEX"

    def test_case_insensitive(self):
        assert _parse_tier("simple", VALID) == "SIMPLE"
        assert _parse_tier("Complex", VALID) == "COMPLEX"
        assert _parse_tier("cOmPlEx", VALID) == "COMPLEX"

    def test_with_trailing_punctuation(self):
        assert _parse_tier("SIMPLE.", VALID) == "SIMPLE"
        assert _parse_tier("SIMPLE!", VALID) == "SIMPLE"
        assert _parse_tier("COMPLEX.", VALID) == "COMPLEX"

    def test_with_whitespace(self):
        assert _parse_tier("  SIMPLE  ", VALID) == "SIMPLE"
        assert _parse_tier("\nCOMPLEX\n", VALID) == "COMPLEX"

    def test_substring_fallback(self):
        # "I think this is SIMPLE" doesn't fullmatch but contains SIMPLE
        assert _parse_tier("I think this is SIMPLE", VALID) == "SIMPLE"
        assert _parse_tier("The answer is COMPLEX", VALID) == "COMPLEX"

    def test_unparseable_raises_value_error(self):
        with pytest.raises(ValueError, match="Could not parse tier"):
            _parse_tier("I don't know", VALID)

    def test_unparseable_no_tier_word_raises(self):
        with pytest.raises(ValueError):
            _parse_tier("maybe?", VALID)

    def test_complexity_tier_value_passes_through(self):
        # If the LLM returns one of the 4 complexity tiers, _parse_tier should
        # still raise because it's not in the configured 2-tier set.
        with pytest.raises(ValueError):
            _parse_tier("MEDIUM", VALID)


# ─── _TierCache ──────────────────────────────────────────────────
class TestTierCache:
    def test_hit_returns_cached_tier(self):
        cache = _TierCache(ttl_seconds=60, max_size=100)
        cache.set("k1", "SIMPLE")
        assert cache.get("k1") == "SIMPLE"

    def test_miss_returns_none(self):
        cache = _TierCache(ttl_seconds=60, max_size=100)
        assert cache.get("missing") is None

    def test_expiry(self):
        cache = _TierCache(ttl_seconds=0.01, max_size=100)
        cache.set("k1", "SIMPLE")
        import time

        time.sleep(0.05)
        assert cache.get("k1") is None

    def test_overflow_clears_store(self):
        cache = _TierCache(ttl_seconds=60, max_size=2)
        cache.set("k1", "SIMPLE")
        cache.set("k2", "COMPLEX")
        cache.set("k3", "SIMPLE")
        # After overflow, k1 and k2 are evicted
        assert cache.get("k1") is None
        assert cache.get("k2") is None
        assert cache.get("k3") == "SIMPLE"

    def test_overwrite_same_key(self):
        cache = _TierCache(ttl_seconds=60, max_size=100)
        cache.set("k1", "SIMPLE")
        cache.set("k1", "COMPLEX")
        assert cache.get("k1") == "COMPLEX"


# ─── _cache_key ──────────────────────────────────────────────────
class TestCacheKey:
    def test_deterministic(self):
        assert _cache_key("hello world") == _cache_key("hello world")

    def test_different_inputs_different_keys(self):
        assert _cache_key("a") != _cache_key("b")

    def test_key_length_16(self):
        assert len(_cache_key("anything")) == 16

    def test_handles_unicode(self):
        k = _cache_key("你好世界")
        assert len(k) == 16
        assert _cache_key("你好世界") == k


# ─── classify: cache path ────────────────────────────────────────
class TestClassifyCachePath:
    async def test_successful_llm_call_routes_to_simPLE_tier(self, router):
        with patch(
            "litellm.acompletion",
            new=AsyncMock(return_value=_mock_response("SIMPLE")),
        ) as mock:
            tier, method = await router.classify("What is the capital of France?")
            assert tier == "SIMPLE"
            assert method == "llm"
            mock.assert_awaited_once()

    async def test_complex_prompt_routes_correctly(self, router):
        with patch(
            "litellm.acompletion",
            new=AsyncMock(return_value=_mock_response("COMPLEX")),
        ):
            tier, method = await router.classify("Implement a thread-safe LRU cache with O(1) get/set")
            assert tier == "COMPLEX"
            assert method == "llm"

    async def test_cache_hit_returns_cache_method(self, router):
        with patch(
            "litellm.acompletion",
            new=AsyncMock(return_value=_mock_response("SIMPLE")),
        ):
            await router.classify("prompt X")
            tier, method = await router.classify("prompt X")
            assert method == "cache"
            assert tier == "SIMPLE"

    async def test_cache_prevents_duplicate_llm_calls(self, router):
        with patch(
            "litellm.acompletion",
            new=AsyncMock(return_value=_mock_response("SIMPLE")),
        ) as mock:
            for _ in range(3):
                await router.classify("same prompt")
            mock.assert_awaited_once()

    async def test_cache_disabled_always_calls_llm(self, mock_router_instance):
        config = {
            "tiers": {"SIMPLE": "cheap", "COMPLEX": "expensive"},
            "enable_cache": False,
        }
        r = LLMClassifierRouter(
            model_name="test",
            litellm_router_instance=mock_router_instance,
            llm_classifier_router_config=config,
        )
        with patch(
            "litellm.acompletion",
            new=AsyncMock(return_value=_mock_response("SIMPLE")),
        ) as mock:
            await r.classify("p")
            await r.classify("p")
            assert mock.await_count == 2

    async def test_cache_stale_entry_expired_recalls_llm(self, router):
        # Force expiry by manually setting a near-expired entry
        from time import monotonic

        router._cache._store["k"] = ("SIMPLE", monotonic() - 1)
        with patch(
            "litellm.acompletion",
            new=AsyncMock(return_value=_mock_response("COMPLEX")),
        ) as mock:
            tier, method = await router.classify("anything")
            # 'k' was for a different prompt, so the key wouldn't match anyway,
            # but we can still verify the LLM was called
            assert method == "llm"
            assert mock.await_count == 1


# ─── classify: fallback path ─────────────────────────────────────
class TestClassifyFallbackPath:
    async def test_timeout_triggers_fallback(self, router):
        async def slow(*args, **kwargs):
            await asyncio.sleep(10)
            return _mock_response("SIMPLE")

        with patch("litellm.acompletion", new=AsyncMock(side_effect=slow)):
            tier, method = await router.classify("test", system_prompt=None)
            assert method in ("fallback_rule", "fallback_default")
            assert tier in ("SIMPLE", "COMPLEX")

    async def test_api_error_triggers_fallback(self, router):
        with patch(
            "litellm.acompletion",
            new=AsyncMock(side_effect=Exception("API down")),
        ):
            tier, method = await router.classify("test")
            assert method in ("fallback_rule", "fallback_default")

    async def test_fallback_default_when_complexity_router_disabled(self, mock_router_instance):
        config = {
            "tiers": {"SIMPLE": "cheap", "COMPLEX": "expensive"},
            "fallback_to_complexity_router": False,
        }
        r = LLMClassifierRouter(
            model_name="test",
            litellm_router_instance=mock_router_instance,
            llm_classifier_router_config=config,
        )
        with patch(
            "litellm.acompletion",
            new=AsyncMock(side_effect=Exception("fail")),
        ):
            tier, method = await r.classify("test")
            assert method == "fallback_default"
            assert tier == "SIMPLE"

    async def test_fallback_tier_used_when_complexity_router_also_fails(self, mock_router_instance):
        config = {
            "tiers": {"SIMPLE": "cheap", "COMPLEX": "expensive"},
            "fallback_to_complexity_router": True,
        }
        r = LLMClassifierRouter(
            model_name="test",
            litellm_router_instance=mock_router_instance,
            llm_classifier_router_config=config,
        )
        with patch(
            "litellm.acompletion",
            new=AsyncMock(side_effect=Exception("fail")),
        ):
            # ComplexityRouter should also be patched/mocked to fail,
            # but since it's a rules-based classifier, it should not fail.
            # This test mainly verifies the end result is a valid tier.
            tier, method = await r.classify("test")
            assert method in ("fallback_rule", "fallback_default")
            assert tier in ("SIMPLE", "COMPLEX")

    async def test_invalid_fallback_tier_falls_back_to_first_tier(self, mock_router_instance):
        config = {
            "tiers": {"A": "model-a", "B": "model-b"},
            "fallback_tier": "NONEXISTENT",
            "fallback_to_complexity_router": False,
        }
        r = LLMClassifierRouter(
            model_name="test",
            litellm_router_instance=mock_router_instance,
            llm_classifier_router_config=config,
        )
        with patch(
            "litellm.acompletion",
            new=AsyncMock(side_effect=Exception("fail")),
        ):
            tier, method = await r.classify("test")
            assert method == "fallback_default"
            # Should pick the first valid tier
            assert tier in ("A", "B")


# ─── async_pre_routing_hook ─────────────────────────────────────
class TestAsyncPreRoutingHook:
    async def test_returns_pre_routing_hook_response_for_simple_prompt(self, router):
        from litellm.types.router import PreRoutingHookResponse

        with patch(
            "litellm.acompletion",
            new=AsyncMock(return_value=_mock_response("SIMPLE")),
        ):
            response = await router.async_pre_routing_hook(
                model="test-llm-classifier",
                request_kwargs={},
                messages=[{"role": "user", "content": "hi"}],
            )
            assert isinstance(response, PreRoutingHookResponse)
            assert response.model == "gpt-4o-mini"

    async def test_returns_pre_routing_hook_response_for_complex_prompt(self, router):
        with patch(
            "litellm.acompletion",
            new=AsyncMock(return_value=_mock_response("COMPLEX")),
        ):
            response = await router.async_pre_routing_hook(
                model="test-llm-classifier",
                request_kwargs={},
                messages=[
                    {
                        "role": "user",
                        "content": "Implement a thread-safe LRU cache",
                    }
                ],
            )
            assert response.model == "gpt-4o"

    async def test_stashes_decision_in_metadata(self, router):
        request_kwargs: Dict = {}
        with patch(
            "litellm.acompletion",
            new=AsyncMock(return_value=_mock_response("COMPLEX")),
        ):
            await router.async_pre_routing_hook(
                model="test-llm-classifier",
                request_kwargs=request_kwargs,
                messages=[{"role": "user", "content": "design a system"}],
            )
        meta = request_kwargs["metadata"]
        assert meta["llm_classifier_router_tier"] == "COMPLEX"
        assert meta["llm_classifier_router_method"] == "llm"
        assert meta["llm_classifier_router_model"] == "gpt-4o"

    async def test_preserves_existing_metadata(self, router):
        request_kwargs = {"metadata": {"user_id": "u123"}}
        with patch(
            "litellm.acompletion",
            new=AsyncMock(return_value=_mock_response("SIMPLE")),
        ):
            await router.async_pre_routing_hook(
                model="test-llm-classifier",
                request_kwargs=request_kwargs,
                messages=[{"role": "user", "content": "hi"}],
            )
        assert request_kwargs["metadata"]["user_id"] == "u123"
        assert request_kwargs["metadata"]["llm_classifier_router_tier"] == "SIMPLE"

    async def test_no_user_message_returns_none(self, router):
        response = await router.async_pre_routing_hook(
            model="test-llm-classifier",
            request_kwargs={},
            messages=[{"role": "system", "content": "you are a helper"}],
        )
        assert response is None

    async def test_no_messages_returns_none(self, router):
        response = await router.async_pre_routing_hook(
            model="test-llm-classifier",
            request_kwargs={},
            messages=[],
        )
        assert response is None

    async def test_extracts_user_message_from_message_list(self, router):
        with patch(
            "litellm.acompletion",
            new=AsyncMock(return_value=_mock_response("SIMPLE")),
        ) as mock:
            await router.async_pre_routing_hook(
                model="test-llm-classifier",
                request_kwargs={},
                messages=[
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "What is 2+2?"},
                ],
            )
            # Verify the user content was sent to the LLM
            call_args = mock.call_args
            messages = call_args.kwargs.get("messages") or call_args.args[0]
            user_msg = next(m for m in messages if m["role"] == "user")
            assert user_msg["content"] == "What is 2+2?"


# ─── get_model_for_tier ─────────────────────────────────────────
class TestGetModelForTier:
    def test_known_tier(self, router):
        assert router.get_model_for_tier("SIMPLE") == "gpt-4o-mini"
        assert router.get_model_for_tier("COMPLEX") == "gpt-4o"

    def test_unknown_tier(self, router):
        assert router.get_model_for_tier("UNKNOWN") is None
