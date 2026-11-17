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
