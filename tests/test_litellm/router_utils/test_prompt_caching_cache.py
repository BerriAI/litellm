"""
Unit tests for PromptCachingCache affinity TTL handling.

The affinity binding (cacheable_prefix_hash -> model_id) must live for as long
as the provider keeps the prefix warm. A request marked with the 1-hour
ephemeral cache (`cache_control: {"type": "ephemeral", "ttl": "1h"}`) should
keep the affinity for ~3600s, while the default 5-minute cache keeps 300s.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath("../../.."))

from litellm.caching.dual_cache import DualCache
from litellm.router_utils.prompt_caching_cache import PromptCachingCache


def _messages_with_ttl(ttl=None):
    """Build messages whose cacheable prefix carries the given cache_control ttl."""
    cache_control = {"type": "ephemeral"}
    if ttl is not None:
        cache_control["ttl"] = ttl
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant. " * 400,
                    "cache_control": cache_control,
                },
            ],
        },
        {"role": "user", "content": "hi"},
    ]


def test_get_cache_control_ttl_one_hour():
    """`ttl: "1h"` in the cacheable prefix maps to 3600 seconds."""
    ttl = PromptCachingCache.get_cache_control_ttl(_messages_with_ttl("1h"), None)
    assert ttl == 3600


def test_get_cache_control_ttl_default_when_unset():
    """No ttl declared falls back to the default 300 seconds."""
    ttl = PromptCachingCache.get_cache_control_ttl(_messages_with_ttl(None), None)
    assert ttl == 300


def test_get_cache_control_ttl_explicit_five_minutes():
    """`ttl: "5m"` keeps the default 300 seconds."""
    ttl = PromptCachingCache.get_cache_control_ttl(_messages_with_ttl("5m"), None)
    assert ttl == 300


def test_add_model_id_uses_one_hour_ttl():
    """add_model_id stores the affinity with a 3600s ttl for the 1h cache."""
    mock_cache = MagicMock(spec=DualCache)
    prompt_cache = PromptCachingCache(cache=mock_cache)

    prompt_cache.add_model_id(
        model_id="deployment-a",
        messages=_messages_with_ttl("1h"),
        tools=None,
    )

    assert mock_cache.set_cache.call_count == 1
    assert mock_cache.set_cache.call_args.kwargs["ttl"] == 3600


def test_add_model_id_defaults_to_five_minutes():
    """add_model_id keeps the 300s default when no ttl is declared."""
    mock_cache = MagicMock(spec=DualCache)
    prompt_cache = PromptCachingCache(cache=mock_cache)

    prompt_cache.add_model_id(
        model_id="deployment-a",
        messages=_messages_with_ttl(None),
        tools=None,
    )

    assert mock_cache.set_cache.call_count == 1
    assert mock_cache.set_cache.call_args.kwargs["ttl"] == 300


@pytest.mark.asyncio
async def test_async_add_model_id_uses_one_hour_ttl():
    """async_add_model_id stores the affinity with a 3600s ttl for the 1h cache."""
    mock_cache = MagicMock(spec=DualCache)
    mock_cache.async_set_cache = AsyncMock()
    prompt_cache = PromptCachingCache(cache=mock_cache)

    await prompt_cache.async_add_model_id(
        model_id="deployment-a",
        messages=_messages_with_ttl("1h"),
        tools=None,
    )

    assert mock_cache.async_set_cache.call_count == 1
    assert mock_cache.async_set_cache.call_args.kwargs["ttl"] == 3600


@pytest.mark.asyncio
async def test_async_add_model_id_defaults_to_five_minutes():
    """async_add_model_id keeps the 300s default when no ttl is declared."""
    mock_cache = MagicMock(spec=DualCache)
    mock_cache.async_set_cache = AsyncMock()
    prompt_cache = PromptCachingCache(cache=mock_cache)

    await prompt_cache.async_add_model_id(
        model_id="deployment-a",
        messages=_messages_with_ttl(None),
        tools=None,
    )

    assert mock_cache.async_set_cache.call_count == 1
    assert mock_cache.async_set_cache.call_args.kwargs["ttl"] == 300
