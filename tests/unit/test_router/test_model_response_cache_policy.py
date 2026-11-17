from unittest.mock import AsyncMock, patch

import pytest

from litellm import Router
from litellm.router_rust.invoke.endpoints import _resolve_deployment_cache_policy


def test_model_cache_policy_disabled_overrides_global():
    """If model_info specifies response_cache=False, caching is disabled even when global caching is True."""
    deployment = {
        "model_name": "uncached-model",
        "litellm_params": {"model": "openai/gpt-4o"},
        "model_info": {"response_cache": False},
    }
    kwargs = {}
    caching = _resolve_deployment_cache_policy(deployment, global_caching=True, kwargs=kwargs)
    assert caching is False
    assert kwargs.get("cache") == {"no-cache": True, "no-store": True}


def test_model_cache_policy_disabled_cannot_be_bypassed_by_client():
    """Client cannot re-enable caching if admin disabled it on the model."""
    deployment = {
        "model_name": "strictly-uncached",
        "litellm_params": {"model": "openai/gpt-4o"},
        "model_info": {"response_cache": False},
    }
    kwargs = {"cache": {"no-cache": False, "no-store": False}}
    caching = _resolve_deployment_cache_policy(deployment, global_caching=True, kwargs=kwargs)
    assert caching is False
    assert kwargs["cache"]["no-cache"] is True
    assert kwargs["cache"]["no-store"] is True


def test_model_cache_policy_enabled_overrides_global_disabled():
    """If model_info specifies response_cache=True, caching is enabled even when global caching is False."""
    deployment = {
        "model_name": "cached-model",
        "litellm_params": {"model": "openai/gpt-4o"},
        "model_info": {"response_cache": True},
    }
    kwargs = {}
    caching = _resolve_deployment_cache_policy(deployment, global_caching=False, kwargs=kwargs)
    assert caching is True


def test_model_cache_policy_ttl():
    """If model_info specifies cache_ttl, it is applied when no request-level TTL is provided."""
    deployment = {
        "model_name": "long-cached",
        "litellm_params": {"model": "openai/gpt-4o"},
        "model_info": {"cache_ttl": 3600},
    }
    kwargs = {}
    caching = _resolve_deployment_cache_policy(deployment, global_caching=True, kwargs=kwargs)
    assert caching is True
    assert kwargs["cache"]["ttl"] == 3600


def test_model_cache_policy_preserves_client_ttl():
    """If client specifies custom TTL, client TTL is preserved over model default TTL."""
    deployment = {
        "model_name": "long-cached",
        "litellm_params": {"model": "openai/gpt-4o"},
        "model_info": {"cache_ttl": 3600},
    }
    kwargs = {"cache": {"ttl": 60}}
    caching = _resolve_deployment_cache_policy(deployment, global_caching=True, kwargs=kwargs)
    assert caching is True
    assert kwargs["cache"]["ttl"] == 60


def test_model_cache_policy_in_litellm_params():
    """Cache policy configured under litellm_params is also respected."""
    deployment = {
        "model_name": "uncached-model",
        "litellm_params": {
            "model": "openai/gpt-4o",
            "response_cache": False,
            "cache_ttl": 120,
        },
        "model_info": {},
    }
    kwargs = {}
    caching = _resolve_deployment_cache_policy(deployment, global_caching=True, kwargs=kwargs)
    assert caching is False
    assert kwargs["cache"]["no-cache"] is True
    assert kwargs["cache"]["no-store"] is True
    assert kwargs["cache"]["ttl"] == 120


def test_model_cache_policy_unconfigured_inherits_global():
    """When no model-level policy is configured, global caching and kwargs are untouched."""
    deployment = {
        "model_name": "default-model",
        "litellm_params": {"model": "openai/gpt-4o"},
        "model_info": {},
    }
    kwargs = {}
    caching = _resolve_deployment_cache_policy(deployment, global_caching=True, kwargs=kwargs)
    assert caching is True
    assert "cache" not in kwargs


@pytest.mark.asyncio
async def test_router_routes_with_per_model_caching():
    """Router attempt_call applies model-level caching policy during acompletion."""
    model_list = [
        {
            "model_name": "cached-docs",
            "litellm_params": {"model": "openai/gpt-4o", "mock_response": "docs answer"},
            "model_info": {"response_cache": True, "cache_ttl": 3600},
        },
        {
            "model_name": "live-sql",
            "litellm_params": {"model": "openai/gpt-4o", "mock_response": "live answer"},
            "model_info": {"response_cache": False},
        },
    ]

    router = Router(model_list=model_list, cache_responses=True)
    try:
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = {"id": "1", "choices": [{"message": {"content": "ok"}}]}

            # Call cached-docs: caching should be True and cache.ttl should be 3600
            await router.acompletion(model="cached-docs", messages=[{"role": "user", "content": "hi"}])
            assert mock_acompletion.call_count == 1
            call_kw = mock_acompletion.call_args.kwargs
            assert call_kw["caching"] is True
            assert call_kw["cache"]["ttl"] == 3600

            # Call live-sql: caching should be False, no-cache and no-store True
            await router.acompletion(model="live-sql", messages=[{"role": "user", "content": "query"}])
            assert mock_acompletion.call_count == 2
            call_kw2 = mock_acompletion.call_args.kwargs
            assert call_kw2["caching"] is False
            assert call_kw2["cache"]["no-cache"] is True
            assert call_kw2["cache"]["no-store"] is True
    finally:
        router.retire()


def test_client_caching_true_cannot_bypass_disabled_response_cache():
    """Client cannot pass caching=True to bypass model response_cache=False."""
    deployment = {
        "model_name": "strictly-uncached",
        "litellm_params": {"model": "openai/gpt-4o"},
        "model_info": {"response_cache": False},
    }
    kwargs = {"caching": True}
    caching = _resolve_deployment_cache_policy(deployment, global_caching=True, kwargs=kwargs)
    assert caching is False
    assert kwargs["caching"] is False
    assert kwargs["cache"]["no-cache"] is True
    assert kwargs["cache"]["no-store"] is True


def test_client_caching_false_can_opt_out_on_cached_model():
    """Client can pass caching=False to opt out of cache on a cached model."""
    deployment = {
        "model_name": "cached-model",
        "litellm_params": {"model": "openai/gpt-4o"},
        "model_info": {"response_cache": True},
    }
    kwargs = {"caching": False}
    caching = _resolve_deployment_cache_policy(deployment, global_caching=True, kwargs=kwargs)
    assert caching is False
    assert kwargs["caching"] is False


def test_string_boolean_and_numeric_ttl_coercion():
    """String 'false' and numeric string TTLs are coerced properly."""
    deployment_false = {
        "model_name": "string-false-model",
        "litellm_params": {"model": "openai/gpt-4o"},
        "model_info": {"response_cache": "false"},
    }
    kwargs = {}
    assert _resolve_deployment_cache_policy(deployment_false, global_caching=True, kwargs=kwargs) is False
    assert kwargs["cache"]["no-cache"] is True
    assert kwargs["cache"]["no-store"] is True

    deployment_ttl = {
        "model_name": "string-ttl-model",
        "litellm_params": {"model": "openai/gpt-4o"},
        "model_info": {"response_cache": "true", "cache_ttl": "7200"},
    }
    kwargs_ttl = {}
    assert _resolve_deployment_cache_policy(deployment_ttl, global_caching=False, kwargs=kwargs_ttl) is True
    assert kwargs_ttl["cache"]["ttl"] == 7200


def test_cache_as_boolean_in_model_info_or_litellm_params():
    """cache: False or cache: True directly in model_info or litellm_params is respected."""
    deployment = {
        "model_name": "cache-field-false",
        "litellm_params": {"model": "openai/gpt-4o"},
        "model_info": {"cache": False},
    }
    kwargs = {}
    assert _resolve_deployment_cache_policy(deployment, global_caching=True, kwargs=kwargs) is False
    assert kwargs["cache"]["no-cache"] is True
    assert kwargs["cache"]["no-store"] is True

    deployment_true = {
        "model_name": "cache-field-true",
        "litellm_params": {"model": "openai/gpt-4o", "cache": True},
        "model_info": {},
    }
    kwargs_true = {}
    assert _resolve_deployment_cache_policy(deployment_true, global_caching=False, kwargs=kwargs_true) is True


def test_zero_or_negative_cache_ttl_disables_caching():
    """A TTL of 0 or negative disables caching to avoid Redis invalid expiration errors."""
    deployment = {
        "model_name": "zero-ttl-model",
        "litellm_params": {"model": "openai/gpt-4o"},
        "model_info": {"cache_ttl": 0},
    }
    kwargs = {}
    caching = _resolve_deployment_cache_policy(deployment, global_caching=True, kwargs=kwargs)
    assert caching is False
    assert kwargs["cache"]["no-cache"] is True
    assert kwargs["cache"]["no-store"] is True


def test_fine_grained_no_store_and_no_cache_flags():
    """Individual no-store and no-cache flags in model cache config are preserved in kwargs."""
    deployment_no_store = {
        "model_name": "no-store-model",
        "litellm_params": {"model": "openai/gpt-4o"},
        "model_info": {"cache": {"no-store": True}},
    }
    kwargs = {}
    _resolve_deployment_cache_policy(deployment_no_store, global_caching=True, kwargs=kwargs)
    assert kwargs["cache"]["no-store"] is True

    deployment_no_cache = {
        "model_name": "no-cache-model",
        "litellm_params": {"model": "openai/gpt-4o"},
        "model_info": {"cache": {"no-cache": True}},
    }
    kwargs_nc = {}
    _resolve_deployment_cache_policy(deployment_no_cache, global_caching=True, kwargs=kwargs_nc)
    assert kwargs_nc["cache"]["no-cache"] is True


@pytest.mark.asyncio
async def test_model_fallbacks_with_different_cache_policies():
    """Fallbacks isolate caching kwargs per attempt across uncached and cached endpoints."""
    import litellm

    model_list = [
        {
            "model_name": "primary-uncached",
            "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"},
            "model_info": {"response_cache": False},
        },
        {
            "model_name": "fallback-cached",
            "litellm_params": {"model": "azure/gpt-4o", "api_key": "fake"},
            "model_info": {"response_cache": True, "cache_ttl": 60},
        },
    ]

    router = Router(
        model_list=model_list,
        fallbacks=[{"primary-uncached": ["fallback-cached"]}],
        cache_responses=True,
        num_retries=0,
    )
    try:
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.side_effect = [
                litellm.RateLimitError(
                    message="rate limit", response=None, llm_provider="openai", model="openai/gpt-4o"
                ),
                {"id": "2", "choices": [{"message": {"content": "ok"}}]},
            ]

            await router.acompletion(model="primary-uncached", messages=[{"role": "user", "content": "hi"}])
            assert mock_acompletion.call_count == 2

            # Attempt 1: primary-uncached (caching=False, no-cache=True, no-store=True)
            kw1 = mock_acompletion.call_args_list[0].kwargs
            assert kw1["caching"] is False
            assert kw1["cache"]["no-cache"] is True
            assert kw1["cache"]["no-store"] is True

            # Attempt 2: fallback-cached (caching=True, ttl=60)
            kw2 = mock_acompletion.call_args_list[1].kwargs
            assert kw2["caching"] is True
            assert kw2["cache"]["ttl"] == 60
            assert "no-cache" not in kw2["cache"]
    finally:
        router.retire()


@pytest.mark.asyncio
async def test_model_aliases_with_cache_policy():
    """Model aliases resolve to target deployments and apply their caching policy."""
    model_list = [
        {
            "model_name": "target-model",
            "litellm_params": {"model": "openai/gpt-4o"},
            "model_info": {"response_cache": True, "cache_ttl": 500},
        },
    ]

    router = Router(
        model_list=model_list,
        model_group_alias={"chat-alias": "target-model"},
        cache_responses=False,
    )
    try:
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = {"id": "1", "choices": [{"message": {"content": "ok"}}]}

            await router.acompletion(model="chat-alias", messages=[{"role": "user", "content": "hi"}])
            assert mock_acompletion.call_count == 1
            call_kw = mock_acompletion.call_args.kwargs
            assert call_kw["caching"] is True
            assert call_kw["cache"]["ttl"] == 500
    finally:
        router.retire()


def test_sync_text_completion_respects_cache_policy():
    """Synchronous router.text_completion applies model cache policy."""
    from unittest.mock import MagicMock

    model_list = [
        {
            "model_name": "uncached-text",
            "litellm_params": {"model": "openai/gpt-3.5-turbo-instruct"},
            "model_info": {"response_cache": False},
        },
    ]

    router = Router(model_list=model_list, cache_responses=True)
    try:
        with patch("litellm.text_completion", new_callable=MagicMock) as mock_text:
            mock_text.return_value = {"id": "1", "choices": []}
            router.text_completion(model="uncached-text", prompt="test prompt")
            assert mock_text.call_count == 1
            kw = mock_text.call_args.kwargs
            assert kw["caching"] is False
            assert kw["cache"]["no-cache"] is True
            assert kw["cache"]["no-store"] is True
    finally:
        router.retire()


@pytest.mark.asyncio
async def test_streaming_respects_response_cache_disabled():
    """Streaming acompletion properly enforces caching=False and no-store=True."""
    model_list = [
        {
            "model_name": "streaming-uncached",
            "litellm_params": {"model": "openai/gpt-4o"},
            "model_info": {"response_cache": False},
        },
    ]

    router = Router(model_list=model_list, cache_responses=True)
    try:
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = AsyncMock()

            await router.acompletion(
                model="streaming-uncached",
                messages=[{"role": "user", "content": "stream"}],
                stream=True,
            )
            assert mock_acompletion.call_count == 1
            call_kw = mock_acompletion.call_args.kwargs
            assert call_kw["caching"] is False
            assert call_kw["cache"]["no-cache"] is True
            assert call_kw["cache"]["no-store"] is True
    finally:
        router.retire()
