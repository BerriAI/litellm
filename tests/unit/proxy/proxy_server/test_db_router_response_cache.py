"""Regression coverage for response caching on DB-backed router creation."""

import asyncio
import importlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import litellm
import pytest

from litellm.proxy.proxy_server import ProxyConfig


def _db_model() -> MagicMock:
    record = MagicMock()
    record.model_id = "db-id-1"
    record.model_name = "fake-model"
    record.litellm_params = {"model": "fake-model"}
    record.model_info = {"id": "db-id-1"}
    record.created_by = "default_user_id"
    record.created_at = None
    record.updated_at = None
    record.updated_by = None
    return record


@pytest.mark.asyncio
async def test_db_router_rebuild_serves_same_caller_from_cache_and_isolates_callers(monkeypatch):
    """A DB router rebuild must use response caching without crossing caller boundaries."""
    proxy_config = ProxyConfig()
    fake_model_list = [
        {
            "model_name": "fake-model",
            "litellm_params": {
                "model": "openai/fake-model",
                "api_key": "sk-fake",
                "mock_response": "cached response",
            },
        }
    ]
    cache = litellm.Cache()
    monkeypatch.setattr(litellm, "cache", cache)
    cache_write_complete = asyncio.Event()
    original_async_add_cache = cache.async_add_cache

    async def tracked_async_add_cache(*args: Any, **kwargs: Any) -> Any:
        result = await original_async_add_cache(*args, **kwargs)
        cache_write_complete.set()
        return result

    monkeypatch.setattr(cache, "async_add_cache", tracked_async_add_cache)
    provider_calls: int = 0
    litellm_main = importlib.import_module("litellm.main")
    original_mock_completion = litellm_main.mock_completion

    def tracked_mock_completion(*args, **kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return original_mock_completion(*args, **kwargs)

    monkeypatch.setattr(litellm_main, "mock_completion", tracked_mock_completion)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-test")
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_model_list", [])
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr(proxy_config, "decrypt_model_list_from_db", lambda new_models: fake_model_list)
    monkeypatch.setattr(proxy_config, "_add_router_settings_from_db_config", AsyncMock())

    await proxy_config._update_llm_router(
        new_models=[_db_model()],
        proxy_logging_obj=MagicMock(),
    )
    router = __import__("litellm.proxy.proxy_server", fromlist=["llm_router"]).llm_router

    # Keep the request body identical: isolation must follow the authenticated
    # caller identity, rather than request-content differences.
    request = [{"role": "user", "content": "cache canary"}]
    same_caller_id = "caller-a"
    other_caller_id = "caller-b"
    first = await router.acompletion(model="fake-model", messages=request, user=same_caller_id)
    await asyncio.wait_for(cache_write_complete.wait(), timeout=5)
    same_caller = await router.acompletion(model="fake-model", messages=request, user=same_caller_id)
    other_caller = await router.acompletion(model="fake-model", messages=request, user=other_caller_id)

    assert first.id == same_caller.id
    assert first.id != other_caller.id
    assert provider_calls == 2


@pytest.mark.asyncio
async def test_db_router_rebuild_keeps_cache_disabled_without_global_cache(monkeypatch):
    """The DB router must not enable response caching when the proxy has no cache."""
    proxy_config = ProxyConfig()
    fake_model_list = [
        {
            "model_name": "fake-model",
            "litellm_params": {"model": "openai/fake-model", "api_key": "sk-fake"},
        }
    ]
    monkeypatch.setattr(litellm, "cache", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-test")
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_model_list", [])
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr(proxy_config, "decrypt_model_list_from_db", lambda new_models: fake_model_list)
    monkeypatch.setattr(proxy_config, "_add_router_settings_from_db_config", AsyncMock())

    await proxy_config._update_llm_router(
        new_models=[_db_model()],
        proxy_logging_obj=MagicMock(),
    )
    router = __import__("litellm.proxy.proxy_server", fromlist=["llm_router"]).llm_router

    assert router.cache_responses is False
