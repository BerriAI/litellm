"""Behavior pins for proxy_server lifecycle, helpers, and small utilities.

Pins covered:
- ``proxy_startup_event``
- ``proxy_shutdown_event``
- ``_initialize_shared_aiohttp_session``
- ``cleanup_router_config_variables``
- ``save_worker_config``
- ``initialize``
- ``load_from_azure_key_vault``
- ``cost_tracking``
- ``_resolve_typed_dict_type``
- ``_resolve_pydantic_type``
- ``get_litellm_model_info``
- ``run_ollama_serve``
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from typing import List, Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from pydantic import BaseModel
from typing_extensions import TypedDict

import litellm.proxy.proxy_server as ps
from litellm.proxy.proxy_server import (
    _initialize_shared_aiohttp_session,
    _resolve_pydantic_type,
    _resolve_typed_dict_type,
    cleanup_router_config_variables,
    cost_tracking,
    get_litellm_model_info,
    initialize,
    load_from_azure_key_vault,
    proxy_shutdown_event,
    proxy_startup_event,
    run_ollama_serve,
    save_worker_config,
)

from .conftest import normalize

# ---------------------------------------------------------------------------
# cleanup_router_config_variables
# ---------------------------------------------------------------------------


def test_cleanup_router_config_variables_resets_globals(monkeypatch):
    monkeypatch.setattr(ps, "master_key", "sk-sentinel", raising=False)
    monkeypatch.setattr(ps, "user_config_file_path", "/tmp/config.yaml", raising=False)
    monkeypatch.setattr(ps, "user_custom_auth", lambda x: x, raising=False)
    monkeypatch.setattr(ps, "health_check_interval", 42, raising=False)
    monkeypatch.setattr(ps, "prisma_client", MagicMock(), raising=False)

    cleanup_router_config_variables()

    observed = {
        "master_key": ps.master_key,
        "user_config_file_path": ps.user_config_file_path,
        "user_custom_auth": ps.user_custom_auth,
        "health_check_interval": ps.health_check_interval,
        "prisma_client": ps.prisma_client,
    }
    assert normalize(observed) == {
        "master_key": None,
        "user_config_file_path": None,
        "user_custom_auth": None,
        "health_check_interval": None,
        "prisma_client": None,
    }


def test_cleanup_router_config_variables_fails_on_unknown_attr_raises():
    """The function only writes documented globals — accessing a non-existent
    one after cleanup should still raise AttributeError."""
    cleanup_router_config_variables()
    with pytest.raises(AttributeError):
        _ = ps.this_attribute_should_not_exist_xyz


# ---------------------------------------------------------------------------
# proxy_shutdown_event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proxy_shutdown_event_disconnects_prisma_and_resets(monkeypatch):
    fake_prisma = MagicMock()
    fake_prisma.disconnect = AsyncMock()
    monkeypatch.setattr(ps, "prisma_client", fake_prisma, raising=False)
    monkeypatch.setattr(ps, "master_key", "sk-x", raising=False)

    fake_jwt = MagicMock()
    fake_jwt.close = AsyncMock()
    monkeypatch.setattr(ps, "jwt_handler", fake_jwt, raising=False)
    monkeypatch.setattr(ps, "db_writer_client", None, raising=False)

    import litellm

    monkeypatch.setattr(litellm, "cache", None, raising=False)
    monkeypatch.setattr(litellm, "success_callback", [], raising=False)

    await proxy_shutdown_event()

    observed = {
        "disconnect_called": fake_prisma.disconnect.await_count == 1,
        "jwt_closed": fake_jwt.close.await_count == 1,
        "master_key_reset": ps.master_key,
        "prisma_reset": ps.prisma_client,
    }
    assert normalize(observed) == {
        "disconnect_called": True,
        "jwt_closed": True,
        "master_key_reset": None,
        "prisma_reset": None,
    }


@pytest.mark.asyncio
async def test_proxy_shutdown_event_prisma_disconnect_raises_error(monkeypatch):
    fake_prisma = MagicMock()
    fake_prisma.disconnect = AsyncMock(side_effect=RuntimeError("db gone"))
    monkeypatch.setattr(ps, "prisma_client", fake_prisma, raising=False)

    fake_jwt = MagicMock()
    fake_jwt.close = AsyncMock()
    monkeypatch.setattr(ps, "jwt_handler", fake_jwt, raising=False)

    import litellm

    monkeypatch.setattr(litellm, "cache", None, raising=False)
    monkeypatch.setattr(litellm, "success_callback", [], raising=False)

    with pytest.raises(RuntimeError, match="db gone"):
        await proxy_shutdown_event()


# ---------------------------------------------------------------------------
# _initialize_shared_aiohttp_session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_shared_aiohttp_session_returns_client_session():
    from aiohttp import ClientSession

    session = await _initialize_shared_aiohttp_session()
    try:
        observed = {
            "is_client_session": isinstance(session, ClientSession),
            "is_closed": session.closed,
            "has_connector": session.connector is not None,
        }
        assert normalize(observed) == {
            "is_client_session": True,
            "is_closed": False,
            "has_connector": True,
        }
    finally:
        if session is not None:
            await session.close()


@pytest.mark.asyncio
async def test_initialize_shared_aiohttp_session_aiohttp_missing_returns_none_on_failure(
    monkeypatch,
):
    """If aiohttp import fails, the function catches and returns None — no raise."""
    import builtins

    real_import = builtins.__import__

    def _raise_for_aiohttp(name, *args, **kwargs):
        if name == "aiohttp":
            raise ImportError("simulated missing aiohttp")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _raise_for_aiohttp)
    result = await _initialize_shared_aiohttp_session()
    assert result is None


# ---------------------------------------------------------------------------
# save_worker_config
# ---------------------------------------------------------------------------


def test_save_worker_config_writes_json_to_environ(monkeypatch):
    monkeypatch.delenv("WORKER_CONFIG", raising=False)

    save_worker_config(model="gpt-4", config="/tmp/c.yaml", debug=True)

    payload = json.loads(os.environ["WORKER_CONFIG"])
    assert normalize(payload) == {
        "model": "gpt-4",
        "config": "/tmp/c.yaml",
        "debug": True,
    }


def test_save_worker_config_invalid_no_kwargs_yields_empty(monkeypatch):
    monkeypatch.delenv("WORKER_CONFIG", raising=False)

    save_worker_config()
    assert os.environ["WORKER_CONFIG"] == "{}"


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


def test_initialize_signature_is_async_with_expected_params():
    sig = inspect.signature(initialize)
    # Hard-coded so a signature change (param added/removed) trips the gate.
    expected_param_count = 17
    observed = {
        "is_async": inspect.iscoroutinefunction(initialize),
        "param_count": len(sig.parameters),
        "has_model": "model" in sig.parameters,
        "has_config": "config" in sig.parameters,
    }
    assert normalize(observed) == {
        "is_async": True,
        "param_count": expected_param_count,
        "has_model": True,
        "has_config": True,
    }


@pytest.mark.asyncio
async def test_initialize_invalid_unexpected_kwarg_raises_type_error():
    with pytest.raises(TypeError):
        await initialize(this_is_not_a_real_kwarg=True)


# ---------------------------------------------------------------------------
# load_from_azure_key_vault
# ---------------------------------------------------------------------------


def test_load_from_azure_key_vault_disabled_no_side_effect(monkeypatch):
    import litellm

    sentinel_secret_mgr = object()
    monkeypatch.setattr(
        litellm, "secret_manager_client", sentinel_secret_mgr, raising=False
    )

    result = load_from_azure_key_vault(use_azure_key_vault=False)

    observed = {
        "return_value": result,
        "secret_manager_unchanged": litellm.secret_manager_client
        is sentinel_secret_mgr,
        "called_with": False,
    }
    assert normalize(observed) == {
        "return_value": None,
        "secret_manager_unchanged": True,
        "called_with": False,
    }


def test_load_from_azure_key_vault_missing_uri_failure_is_swallowed(monkeypatch):
    """Enabled but AZURE_KEY_VAULT_URI unset / azure libs likely unavailable —
    function catches Exception and does not raise."""
    monkeypatch.delenv("AZURE_KEY_VAULT_URI", raising=False)

    result = load_from_azure_key_vault(use_azure_key_vault=True)
    assert result is None


# ---------------------------------------------------------------------------
# cost_tracking
# ---------------------------------------------------------------------------


def test_cost_tracking_adds_two_callbacks_when_prisma_set(monkeypatch):
    import litellm

    fake_prisma = MagicMock()
    monkeypatch.setattr(ps, "prisma_client", fake_prisma, raising=False)
    monkeypatch.setattr(litellm, "callbacks", [], raising=False)
    monkeypatch.setattr(litellm, "_async_success_callback", [], raising=False)

    before_callbacks = len(litellm.callbacks)
    before_async = len(litellm._async_success_callback)

    cost_tracking()

    observed = {
        "added_to_callbacks": len(litellm.callbacks) - before_callbacks,
        "added_to_async_success": len(litellm._async_success_callback) - before_async,
        "prisma_was_set": True,
    }
    assert normalize(observed) == {
        "added_to_callbacks": 1,
        "added_to_async_success": 1,
        "prisma_was_set": True,
    }


def test_cost_tracking_no_op_when_prisma_missing(monkeypatch):
    """Without a prisma_client cost_tracking is a no-op — not an error."""
    import litellm

    monkeypatch.setattr(ps, "prisma_client", None, raising=False)
    monkeypatch.setattr(litellm, "callbacks", [], raising=False)
    monkeypatch.setattr(litellm, "_async_success_callback", [], raising=False)

    cost_tracking()

    assert litellm.callbacks == []
    assert litellm._async_success_callback == []


# ---------------------------------------------------------------------------
# _resolve_typed_dict_type
# ---------------------------------------------------------------------------


class _SampleTD(TypedDict):
    a: int
    b: str


def test_resolve_typed_dict_type_finds_class_in_optional():
    typ = Optional[_SampleTD]
    result = _resolve_typed_dict_type(typ)

    observed = {
        "input_repr": "Optional[_SampleTD]",
        "result_is_sample_td": result is _SampleTD,
        "result_is_class": isinstance(result, type),
    }
    assert normalize(observed) == {
        "input_repr": "Optional[_SampleTD]",
        "result_is_sample_td": True,
        "result_is_class": True,
    }


def test_resolve_typed_dict_type_invalid_plain_type_returns_none():
    """A non-TypedDict, non-Union input returns None — not an error."""
    assert _resolve_typed_dict_type(int) is None
    assert _resolve_typed_dict_type(str) is None


# ---------------------------------------------------------------------------
# _resolve_pydantic_type
# ---------------------------------------------------------------------------


class _SampleModelA(BaseModel):
    x: int


class _SampleModelB(BaseModel):
    y: str


def test_resolve_pydantic_type_extracts_non_none_args_from_union():
    typ = Union[_SampleModelA, _SampleModelB, None]
    result = _resolve_pydantic_type(typ)

    observed = {
        "result_type": type(result).__name__,
        "result_len": len(result),
        "contains_a": _SampleModelA in result,
        "contains_b": _SampleModelB in result,
    }
    assert normalize(observed) == {
        "result_type": "list",
        "result_len": 2,
        "contains_a": True,
        "contains_b": True,
    }


def test_resolve_pydantic_type_invalid_non_union_non_model_returns_empty():
    """When given a non-Union and non-BaseModel input the function returns [].

    This is the silent-empty fallback path — error-ish by behavior."""
    result = _resolve_pydantic_type(int)
    assert result == []


# ---------------------------------------------------------------------------
# get_litellm_model_info
# ---------------------------------------------------------------------------


def test_get_litellm_model_info_uses_base_model_for_lookup(monkeypatch):
    import litellm

    expected_info = {"max_tokens": 8192, "input_cost_per_token": 0.00003}
    fake_get = MagicMock(return_value=expected_info)
    monkeypatch.setattr(litellm, "get_model_info", fake_get, raising=False)

    model = {
        "model_info": {"base_model": "gpt-4"},
        "litellm_params": {"model": "azure/my-deployment"},
    }
    result = get_litellm_model_info(model=model)

    observed = {
        "called_arg": (
            fake_get.call_args.args[0]
            if fake_get.call_args.args
            else fake_get.call_args.kwargs.get("model")
        ),
        "returned_max_tokens": result.get("max_tokens"),
        "returned_cost": result.get("input_cost_per_token"),
    }
    assert normalize(observed) == {
        "called_arg": "gpt-4",
        "returned_max_tokens": 8192,
        "returned_cost": 0.00003,
    }


def test_get_litellm_model_info_invalid_empty_dict_returns_empty():
    """Empty input means model_to_lookup is None — internal exception is caught
    and the function returns {}."""
    result = get_litellm_model_info(model={})
    assert result == {}


# ---------------------------------------------------------------------------
# run_ollama_serve
# ---------------------------------------------------------------------------


def test_run_ollama_serve_invokes_subprocess_popen(monkeypatch):
    fake_popen = MagicMock()
    monkeypatch.setattr(ps.subprocess, "Popen", fake_popen)

    run_ollama_serve()

    args, kwargs = fake_popen.call_args
    observed = {
        "popen_called": fake_popen.call_count == 1,
        "command": args[0] if args else kwargs.get("args"),
        "has_stdout_kw": "stdout" in kwargs,
        "has_stderr_kw": "stderr" in kwargs,
    }
    assert normalize(observed) == {
        "popen_called": True,
        "command": ["ollama", "serve"],
        "has_stdout_kw": True,
        "has_stderr_kw": True,
    }


def test_run_ollama_serve_popen_failure_is_swallowed(monkeypatch):
    """Popen raising OSError must NOT propagate — function logs and returns."""
    monkeypatch.setattr(
        ps.subprocess, "Popen", MagicMock(side_effect=OSError("no ollama binary"))
    )

    result = run_ollama_serve()
    assert result is None


# ---------------------------------------------------------------------------
# proxy_startup_event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proxy_startup_event_is_async_context_manager_with_expected_signature():
    """proxy_startup_event is the FastAPI lifespan. Verify its surface without
    actually running the heavy init path (DB, Router, OTEL, etc.)."""
    sig = inspect.signature(proxy_startup_event)
    wrapped = getattr(proxy_startup_event, "__wrapped__", None)
    observed = {
        "param_count": len(sig.parameters),
        "has_app_param": "app" in sig.parameters,
        "wrapped_is_async": inspect.iscoroutinefunction(wrapped)
        or inspect.isasyncgenfunction(wrapped),
        "has_asynccontextmanager_wrapper": wrapped is not None,
    }
    assert normalize(observed) == {
        "param_count": 1,
        "has_app_param": True,
        "wrapped_is_async": True,
        "has_asynccontextmanager_wrapper": True,
    }


@pytest.mark.asyncio
async def test_proxy_startup_event_invalid_missing_app_arg_raises():
    """Calling the lifespan with no FastAPI app argument must fail."""
    with pytest.raises(TypeError):
        # Intentionally invoke the underlying async generator function with
        # no arguments — the decorator preserves the missing-arg TypeError.
        async with proxy_startup_event():  # type: ignore[call-arg]
            pass


def test_otel_global_provider_published_after_callback_init():
    """The OTel V2 global-provider publish must run after callback
    initialization in ``proxy_startup_event``.

    Regression for the orphan span: a preset (arize, langfuse, …) builds its
    single folded logger during ``_initialize_startup_logging``. Publishing the
    global ``TracerProvider`` before that ran found no logger and built a second
    generic one whose provider became the global, so the FastAPI server span and
    the preset's gen-ai spans exported through different providers and the LLM
    span was orphaned. The publish (``publish_global_otel_v2_provider``) must
    therefore appear after ``_initialize_startup_logging`` in the lifespan source.
    """
    wrapped = getattr(proxy_startup_event, "__wrapped__", proxy_startup_event)
    source = inspect.getsource(wrapped)
    init_pos = source.find("_initialize_startup_logging(")
    publish_pos = source.find("publish_global_otel_v2_provider(")
    assert init_pos != -1, "callback init call not found in proxy_startup_event"
    assert publish_pos != -1, "OTEL global publish not found in proxy_startup_event"
    assert init_pos < publish_pos, (
        "OTEL global provider is published before callbacks are initialized; a "
        "preset logger will not exist yet and a second generic logger will own "
        "the global provider, orphaning gen-ai spans"
    )
