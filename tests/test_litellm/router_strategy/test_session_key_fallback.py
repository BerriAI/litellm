"""
Unit tests for Auto Router v2 session_key_fallback derivation (Issue #34766).

Tests cover:
- Session resolution with explicit session_id.
- Fallback to prompt_cache_key when configured.
- Fallback to prefix_hash when configured.
- Normal operation with "none" (default behavior unchanged).
- Deployment affinity with session_key_fallback.
- Adaptive router session key resolution with fallback metadata.
"""

import hashlib
from unittest.mock import AsyncMock

import pytest

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.router_strategy.adaptive_router.hooks import _resolve_session_key
from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter
from litellm.router_strategy.complexity_router.config import ComplexityRouterConfig


@pytest.fixture
def mock_router_instance():
    class _MockRouter:
        def __init__(self):
            self.cache = DualCache()
            self.model_list = [
                {
                    "model_name": "gpt-4o-mini",
                    "litellm_params": {"model": "openai/gpt-4o-mini", "input_cost_per_token": 0.0},
                    "model_info": {},
                },
                {
                    "model_name": "gpt-4o",
                    "litellm_params": {"model": "openai/gpt-4o", "input_cost_per_token": 0.0},
                    "model_info": {},
                },
                {
                    "model_name": "claude-sonnet-4-20250514",
                    "litellm_params": {"model": "anthropic/claude-sonnet-4-20250514", "input_cost_per_token": 0.0},
                    "model_info": {},
                },
                {
                    "model_name": "o1-preview",
                    "litellm_params": {"model": "openai/o1-preview", "input_cost_per_token": 0.0},
                    "model_info": {},
                },
            ]
            self.model_name_to_deployment_indices = {
                "gpt-4o-mini": [0],
                "gpt-4o": [1],
                "claude-sonnet-4-20250514": [2],
                "o1-preview": [3],
            }

    return _MockRouter()


@pytest.fixture
def base_config():
    return {
        "tiers": {
            "SIMPLE": "gpt-4o-mini",
            "MEDIUM": "gpt-4o",
            "COMPLEX": "claude-sonnet-4-20250514",
            "REASONING": "o1-preview",
        },
        "tier_boundaries": {
            "simple_medium": 0.25,
            "medium_complex": 0.50,
            "complex_reasoning": 0.75,
        },
        "session_affinity": True,
        "session_affinity_ttl_seconds": 3600,
    }


class TestSessionKeyFallback:
    SIMPLE_MESSAGE = [{"role": "user", "content": "Hello!"}]
    REASONING_MESSAGE = [
        {"role": "system", "content": "You are a mathematics tutor."},
        {
            "role": "user",
            "content": "Let's think step by step and reason through this problem carefully.",
        },
    ]

    def test_config_default_fallback_is_none(self):
        cfg = ComplexityRouterConfig(tiers={"SIMPLE": "gpt-4o-mini"})
        assert cfg.session_key_fallback == "none"

    def test_config_supports_valid_fallback_modes(self):
        cfg_cache = ComplexityRouterConfig(tiers={"SIMPLE": "gpt-4o-mini"}, session_key_fallback="prompt_cache_key")
        assert cfg_cache.session_key_fallback == "prompt_cache_key"

        cfg_prefix = ComplexityRouterConfig(tiers={"SIMPLE": "gpt-4o-mini"}, session_key_fallback="prefix_hash")
        assert cfg_prefix.session_key_fallback == "prefix_hash"

        cfg_none = ComplexityRouterConfig(tiers={"SIMPLE": "gpt-4o-mini"}, session_key_fallback="none")
        assert cfg_none.session_key_fallback == "none"

    @pytest.mark.asyncio
    async def test_explicit_session_id_takes_precedence_over_fallback(self, mock_router_instance, base_config):
        """When an explicit session_id is provided, it must be used directly, ignoring session_key_fallback."""
        config = {**base_config, "session_key_fallback": "prefix_hash"}
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=config,
        )

        request_kwargs = {
            "metadata": {"session_id": "my-explicit-session", "user_api_key_hash": "key123"},
            "prompt_cache_key": "cache-key-should-be-ignored",
        }

        resolved_id = router._resolve_session_id(
            request_kwargs=request_kwargs,
            resolved_messages=self.REASONING_MESSAGE,
            model="test-router",
        )
        assert resolved_id == "my-explicit-session"

        # Pre routing hook pins model under explicit session id
        res1 = await router.async_pre_routing_hook(
            model="test-router",
            request_kwargs=request_kwargs,
            messages=self.REASONING_MESSAGE,
        )
        assert res1 is not None
        assert res1.model == "o1-preview"

        # Turn 2: simple message under same explicit session_id should hit cache pin
        req2_kwargs = {"metadata": {"session_id": "my-explicit-session", "user_api_key_hash": "key123"}}
        res2 = await router.async_pre_routing_hook(
            model="test-router",
            request_kwargs=req2_kwargs,
            messages=self.SIMPLE_MESSAGE,
        )
        assert res2 is not None
        assert res2.model == "o1-preview"
        assert res2.routing_decision["cause"] == "session_affinity_pin"

    @pytest.mark.asyncio
    async def test_fallback_to_prompt_cache_key(self, mock_router_instance, base_config):
        """When session_id is absent and session_key_fallback='prompt_cache_key', use prompt_cache_key."""
        config = {**base_config, "session_key_fallback": "prompt_cache_key"}
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=config,
        )

        request_kwargs = {
            "prompt_cache_key": "custom-prompt-cache-key-999",
            "metadata": {"user_api_key_hash": "key123"},
        }

        resolved_id = router._resolve_session_id(
            request_kwargs=request_kwargs,
            resolved_messages=self.REASONING_MESSAGE,
            model="test-router",
        )
        assert resolved_id == "custom-prompt-cache-key-999"
        assert request_kwargs["metadata"]["session_id"] == "custom-prompt-cache-key-999"

        # Pre routing hook turn 1
        res1 = await router.async_pre_routing_hook(
            model="test-router",
            request_kwargs=request_kwargs,
            messages=self.REASONING_MESSAGE,
        )
        assert res1 is not None
        assert res1.model == "o1-preview"

        # Pre routing hook turn 2 with same prompt_cache_key
        req2_kwargs = {
            "prompt_cache_key": "custom-prompt-cache-key-999",
            "metadata": {"user_api_key_hash": "key123"},
        }
        res2 = await router.async_pre_routing_hook(
            model="test-router",
            request_kwargs=req2_kwargs,
            messages=self.SIMPLE_MESSAGE,
        )
        assert res2 is not None
        assert res2.model == "o1-preview"
        assert res2.routing_decision["cause"] == "session_affinity_pin"

    @pytest.mark.asyncio
    async def test_fallback_prompt_cache_key_from_extra_body_or_headers(self, mock_router_instance, base_config):
        """prompt_cache_key in extra_body or headers is also resolved."""
        config = {**base_config, "session_key_fallback": "prompt_cache_key"}
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=config,
        )

        req_extra_body = {"extra_body": {"prompt_cache_key": "extra-body-key"}}
        resolved = router._resolve_session_id(req_extra_body)
        assert resolved == "extra-body-key"

        req_headers = {"headers": {"x-prompt-cache-key": "header-key"}}
        resolved_hdr = router._resolve_session_id(req_headers)
        assert resolved_hdr == "header-key"

    @pytest.mark.asyncio
    async def test_fallback_prompt_cache_key_missing_returns_none(self, mock_router_instance, base_config):
        """When prompt_cache_key is absent and fallback='prompt_cache_key', returns None (reclassifies)."""
        config = {**base_config, "session_key_fallback": "prompt_cache_key"}
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=config,
        )

        request_kwargs = {"metadata": {}}
        resolved = router._resolve_session_id(request_kwargs, resolved_messages=self.SIMPLE_MESSAGE)
        assert resolved is None

    @pytest.mark.asyncio
    async def test_fallback_to_prefix_hash(self, mock_router_instance, base_config):
        """When session_id is absent and session_key_fallback='prefix_hash', derive SHA256 prefix hash."""
        config = {**base_config, "session_key_fallback": "prefix_hash"}
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=config,
        )

        request_kwargs = {
            "metadata": {"user_api_key_hash": "user_hash_abc"},
        }

        # Expected SHA256 computation:
        # user_api_key_hash : model_group : first_system_msg : first_user_msg
        system_text = "You are a mathematics tutor."
        user_text = "Let's think step by step and reason through this problem carefully."
        expected_payload = f"litellm-session-key:user_hash_abc:test-router:{system_text}:{user_text}"
        expected_hash = hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()

        derived_id = router._resolve_session_id(
            request_kwargs=request_kwargs,
            resolved_messages=self.REASONING_MESSAGE,
            model="test-router",
        )
        assert derived_id == expected_hash
        assert request_kwargs["metadata"]["session_id"] == expected_hash

        # Turn 1: Classifies as REASONING (o1-preview) and pins it under derived prefix hash
        res1 = await router.async_pre_routing_hook(
            model="test-router",
            request_kwargs=request_kwargs,
            messages=self.REASONING_MESSAGE,
        )
        assert res1 is not None
        assert res1.model == "o1-preview"

        # Turn 2: Follow-up message in the same multi-turn conversation
        turn2_messages = [
            {"role": "system", "content": "You are a mathematics tutor."},
            {
                "role": "user",
                "content": "Let's think step by step and reason through this problem carefully.",
            },
            {"role": "assistant", "content": "Here is step 1..."},
            {"role": "user", "content": "Thanks! Can you summarize step 1 in one line?"},
        ]
        turn2_kwargs = {
            "metadata": {"user_api_key_hash": "user_hash_abc"},
        }

        turn2_derived_id = router._resolve_session_id(
            request_kwargs=turn2_kwargs,
            resolved_messages=turn2_messages,
            model="test-router",
        )
        # Prefix hash MUST match Turn 1 because initial system & user message are identical
        assert turn2_derived_id == expected_hash

        res2 = await router.async_pre_routing_hook(
            model="test-router",
            request_kwargs=turn2_kwargs,
            messages=turn2_messages,
        )
        assert res2 is not None
        assert res2.model == "o1-preview"
        assert res2.routing_decision["cause"] == "session_affinity_pin"

    @pytest.mark.asyncio
    async def test_fallback_prefix_hash_different_conversations_segregate(self, mock_router_instance, base_config):
        """Different initial prompts or different users produce different prefix hashes and route independently."""
        config = {**base_config, "session_key_fallback": "prefix_hash"}
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=config,
        )

        conv_a_kwargs = {"metadata": {"user_api_key_hash": "user_a"}}
        conv_b_kwargs = {"metadata": {"user_api_key_hash": "user_b"}}

        res_a = await router.async_pre_routing_hook(
            model="test-router",
            request_kwargs=conv_a_kwargs,
            messages=self.REASONING_MESSAGE,
        )
        res_b = await router.async_pre_routing_hook(
            model="test-router",
            request_kwargs=conv_b_kwargs,
            messages=self.SIMPLE_MESSAGE,
        )

        assert res_a.model == "o1-preview"
        assert res_b.model == "gpt-4o-mini"
        assert conv_a_kwargs["metadata"]["session_id"] != conv_b_kwargs["metadata"]["session_id"]

    @pytest.mark.asyncio
    async def test_fallback_prefix_hash_empty_messages_returns_none(self, mock_router_instance, base_config):
        """When no message content is present, prefix_hash cannot be derived and returns None."""
        config = {**base_config, "session_key_fallback": "prefix_hash"}
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=config,
        )
        resolved = router._resolve_session_id(request_kwargs={}, resolved_messages=[])
        assert resolved is None

    @pytest.mark.asyncio
    async def test_default_none_behavior_unchanged(self, mock_router_instance, base_config):
        """When session_key_fallback is 'none' (default), no fallback key is derived."""
        config = {**base_config, "session_key_fallback": "none"}
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=config,
        )

        cache = AsyncMock()
        mock_router_instance.cache = cache

        # Without session_id, no cache lookup or cache set happens
        res = await router.async_pre_routing_hook(
            model="test-router",
            request_kwargs={},
            messages=self.SIMPLE_MESSAGE,
        )
        assert res.model == "gpt-4o-mini"
        cache.async_get_cache.assert_not_called()
        cache.async_set_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_deployment_affinity_uses_fallback_session_id(self, mock_router_instance, base_config):
        """When deployment_affinity is active and fallback derives session_id, deployment pin TTL is added."""
        config = {
            **base_config,
            "session_affinity": False,
            "deployment_affinity": True,
            "session_key_fallback": "prompt_cache_key",
        }
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=config,
        )

        request_kwargs = {"prompt_cache_key": "cache-dep-123"}
        res = await router.async_pre_routing_hook(
            model="test-router",
            request_kwargs=request_kwargs,
            messages=self.SIMPLE_MESSAGE,
        )
        assert res is not None
        assert res.session_affinity_ttl_seconds == 3600
        assert request_kwargs["metadata"]["session_id"] == "cache-dep-123"

    def test_adaptive_router_hooks_resolve_session_key_with_fallback_metadata(self):
        """Adaptive router post call hook picks up session_id populated in metadata by fallback."""
        kwargs = {
            "metadata": {"session_id": "fallback-derived-session-id-123"},
        }
        key = _resolve_session_key(kwargs)
        assert key == "fallback-derived-session-id-123"
