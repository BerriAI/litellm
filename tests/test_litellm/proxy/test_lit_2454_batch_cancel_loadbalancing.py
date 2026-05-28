"""
Regression tests for LIT-2454: cancel_batch missing loadbalancing branch.

When a proxy is configured with `litellm.enable_loadbalancing_on_batch_endpoints=True`,
`create_batch` and `retrieve_batch` route through `llm_router`, but historically
`cancel_batch` did NOT. It fell through to the env-var fallback (`OPENAI_API_KEY`),
causing "openai key error" failures for proxies that only have credentials in
team/deployment params.

These tests pin in:
  1. cancel_batch SCENARIO 1 (encoded batch_id) forwards `model` to litellm.acancel_batch.
  2. cancel_batch SCENARIO 2 routes through llm_router when
     `enable_loadbalancing_on_batch_endpoints=True` (mirrors retrieve_batch).
  3. cancel_batch SCENARIO 3 (no loadbalancing, no encoding) still uses the env-var
     fallback for backward-compat.
  4. Router.acancel_batch with model=None iterates deployments (mirrors aretrieve_batch).
  5. Router.acancel_batch with explicit model uses the single-deployment fast path.
"""

import asyncio
import os
import sys
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.openai_files_endpoints.common_utils import (
    encode_file_id_with_model,
)
from litellm.proxy.proxy_server import app
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router
from litellm.types.utils import LiteLLMBatch


_client = TestClient(app)
_TEAM_KEY = "sk-team-openai-xxxx"


def _make_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "openai-team-wildcard",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": _TEAM_KEY,
                },
                "model_info": {"id": "openai-team-id"},
            },
        ]
    )


def _setup_proxy(monkeypatch, llm_router: Router) -> None:
    proxy_logging_obj = ProxyLogging(
        user_api_key_cache=DualCache(default_in_memory_ttl=1)
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", llm_router)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging_obj
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)


def _make_cancelled_batch_response(batch_id: str) -> LiteLLMBatch:
    return LiteLLMBatch(
        id=batch_id,
        completion_window="24h",
        created_at=1234567890,
        endpoint="/v1/chat/completions",
        input_file_id="file-input",
        object="batch",
        status="cancelled",
    )


# ---------------------------------------------------------------------------
# SCENARIO 1: encoded batch_id forwards model to litellm.acancel_batch
# ---------------------------------------------------------------------------
def test_cancel_batch_encoded_id_forwards_model_kwarg(monkeypatch):
    """Encoded batch_id -> proxy must pass `model` to litellm.acancel_batch
    (parity with retrieve_batch; required for provider-config providers).
    """
    router = _make_router()
    _setup_proxy(monkeypatch, router)

    user_key = UserAPIKeyAuth(api_key="test-key")
    app.dependency_overrides[user_api_key_auth] = lambda: user_key

    encoded = encode_file_id_with_model(
        file_id="batch_abc123", model="openai-team-wildcard", id_type="batch"
    )
    captured: Dict[str, Any] = {}

    async def mock_acancel(**kwargs):
        captured.update(kwargs)
        return _make_cancelled_batch_response("batch_abc123")

    monkeypatch.setattr(litellm, "acancel_batch", mock_acancel)

    try:
        resp = _client.post(
            f"/v1/batches/{encoded}/cancel",
            headers={"Authorization": "Bearer test-key"},
        )
        assert resp.status_code == 200, resp.text
    finally:
        app.dependency_overrides.clear()

    assert captured.get("custom_llm_provider") == "openai"
    assert captured.get("model") == "openai-team-wildcard", (
        "model must be forwarded to litellm.acancel_batch so provider-config "
        "providers (e.g. bedrock) can load BatchesConfig. Got: " + repr(captured)
    )
    assert captured.get("batch_id") == "batch_abc123"
    # Team credentials forwarded, not env var
    assert captured.get("api_key") == _TEAM_KEY


# ---------------------------------------------------------------------------
# SCENARIO 2: loadbalancing routes through Router (LIT-2454 main fix)
# ---------------------------------------------------------------------------
def test_cancel_batch_loadbalancing_routes_through_router(monkeypatch):
    """With enable_loadbalancing_on_batch_endpoints=True and a raw provider
    batch_id (no model encoding), cancel must route through llm_router -- not
    fall through to the env-var fallback (which used OPENAI_API_KEY and broke
    proxies with team-only credentials).
    """
    monkeypatch.setattr(litellm, "enable_loadbalancing_on_batch_endpoints", True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    router = _make_router()
    _setup_proxy(monkeypatch, router)

    user_key = UserAPIKeyAuth(api_key="test-key")
    app.dependency_overrides[user_api_key_auth] = lambda: user_key

    captured_router_call: Dict[str, Any] = {}
    captured_litellm_call: Dict[str, Any] = {}

    async def mock_router_acancel(*args, **kwargs):
        if args:
            captured_router_call["model_positional"] = args[0]
        captured_router_call.update(kwargs)
        return _make_cancelled_batch_response("batch_loadbalanced_xyz")

    monkeypatch.setattr(router, "acancel_batch", mock_router_acancel)

    async def mock_litellm_acancel(**kwargs):
        captured_litellm_call.update(kwargs)
        return _make_cancelled_batch_response("batch_loadbalanced_xyz")

    monkeypatch.setattr(litellm, "acancel_batch", mock_litellm_acancel)

    try:
        resp = _client.post(
            "/v1/batches/batch_loadbalanced_xyz/cancel",
            headers={"Authorization": "Bearer test-key"},
        )
        assert resp.status_code == 200, resp.text
    finally:
        app.dependency_overrides.clear()

    # Critical: cancel was routed through llm_router, NOT through the env fallback.
    assert captured_router_call, (
        "Bug repro: cancel_batch fell through to litellm.acancel_batch env fallback "
        "instead of routing through llm_router."
    )
    assert captured_router_call.get("batch_id") == "batch_loadbalanced_xyz"
    # The env fallback path was NOT taken
    assert not captured_litellm_call, (
        "Env-var fallback (litellm.acancel_batch direct) was hit; cancel should "
        "have gone through llm_router. Got: " + repr(captured_litellm_call)
    )


# ---------------------------------------------------------------------------
# SCENARIO 3: backward-compat -- no loadbalancing, no encoding -> env fallback
# ---------------------------------------------------------------------------
def test_cancel_batch_without_loadbalancing_still_uses_env_fallback(monkeypatch):
    """When enable_loadbalancing_on_batch_endpoints is False (default) and the
    batch_id is not encoded, cancel_batch keeps using the env-var fallback --
    same behaviour as before the LIT-2454 fix. Guards against regression in
    the no-router-no-encoding path.
    """
    monkeypatch.setattr(litellm, "enable_loadbalancing_on_batch_endpoints", False)

    router = _make_router()
    _setup_proxy(monkeypatch, router)

    user_key = UserAPIKeyAuth(api_key="test-key")
    app.dependency_overrides[user_api_key_auth] = lambda: user_key

    captured_litellm_call: Dict[str, Any] = {}

    async def mock_acancel(**kwargs):
        captured_litellm_call.update(kwargs)
        return _make_cancelled_batch_response("batch_simple_xyz")

    monkeypatch.setattr(litellm, "acancel_batch", mock_acancel)

    try:
        resp = _client.post(
            "/v1/batches/batch_simple_xyz/cancel",
            headers={"Authorization": "Bearer test-key"},
        )
        assert resp.status_code == 200, resp.text
    finally:
        app.dependency_overrides.clear()

    # Env fallback path: custom_llm_provider defaulted to "openai"
    assert captured_litellm_call.get("custom_llm_provider") == "openai"
    assert captured_litellm_call.get("batch_id") == "batch_simple_xyz"


# ---------------------------------------------------------------------------
# Router.acancel_batch unit tests
# ---------------------------------------------------------------------------
def test_router_acancel_batch_with_model_uses_fast_path(monkeypatch):
    """Existing behaviour: when model is supplied, acancel_batch goes through
    async_function_with_fallbacks (single-deployment fast path)."""
    router = _make_router()

    called: Dict[str, Any] = {}

    async def mock_fallbacks(*args, **kwargs):
        called.update(kwargs)
        return _make_cancelled_batch_response("batch_zzz")

    monkeypatch.setattr(router, "async_function_with_fallbacks", mock_fallbacks)

    result = asyncio.run(
        router.acancel_batch(model="openai-team-wildcard", batch_id="batch_zzz")
    )

    assert isinstance(result, LiteLLMBatch)
    assert called.get("model") == "openai-team-wildcard"
    assert called.get("batch_id") == "batch_zzz"


def test_router_acancel_batch_model_none_iterates_deployments(monkeypatch):
    """LIT-2454: when model is None, acancel_batch iterates deployments and
    returns the first successful one (mirrors aretrieve_batch)."""
    router = _make_router()

    seen_calls = []

    async def mock_litellm_acancel(**kwargs):
        seen_calls.append(kwargs)
        return _make_cancelled_batch_response("batch_iter_xyz")

    monkeypatch.setattr(litellm, "acancel_batch", mock_litellm_acancel)

    result = asyncio.run(router.acancel_batch(batch_id="batch_iter_xyz"))

    assert isinstance(result, LiteLLMBatch)
    assert result.id == "batch_iter_xyz"
    # The deployment was tried -- confirms iteration path executed
    assert len(seen_calls) >= 1
    assert seen_calls[0].get("batch_id") == "batch_iter_xyz"
    assert seen_calls[0].get("custom_llm_provider") in ("openai", "openai/")
    # Team api_key was sourced from the deployment
    assert seen_calls[0].get("api_key") == _TEAM_KEY


def test_router_acancel_batch_model_none_raises_first_error(monkeypatch):
    """When all deployments fail (e.g. batch not found anywhere), the first
    captured error must propagate so callers see a meaningful failure."""
    router = _make_router()

    class _NotFound(Exception):
        pass

    async def mock_litellm_acancel(**kwargs):
        raise _NotFound("batch not found in this provider")

    monkeypatch.setattr(litellm, "acancel_batch", mock_litellm_acancel)

    with pytest.raises(_NotFound):
        asyncio.run(router.acancel_batch(batch_id="batch_does_not_exist"))
