"""Pin ``ProxyLogging.post_call_failure_hook``, ``_is_proxy_only_llm_api_error``,
and ``_handle_logging_proxy_only_error``."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import AlertType, ProxyErrorTypes
from litellm.proxy.utils import ProxyLogging


@pytest.fixture(autouse=True)
def _clear_caps_cache():
    ProxyLogging._callback_capabilities_cache.clear()
    yield
    ProxyLogging._callback_capabilities_cache.clear()


# ---------------------------------------------------------------------------
# _is_proxy_only_llm_api_error
# ---------------------------------------------------------------------------


def test_is_proxy_only_llm_api_truth_table(proxy_logging):
    """Pin the truth table of ``_is_proxy_only_llm_api_error`` in a single
    snapshot. Covers no-route, non-LLM route, HTTPException on LLM route,
    and auth-error short-circuit."""
    snapshot = {
        "no_route": proxy_logging._is_proxy_only_llm_api_error(
            original_exception=Exception(), route=None
        ),
        "non_llm_route": proxy_logging._is_proxy_only_llm_api_error(
            original_exception=HTTPException(status_code=429, detail="rate"),
            route="/random/path",
        ),
        "http_on_llm_route": proxy_logging._is_proxy_only_llm_api_error(
            original_exception=HTTPException(status_code=429, detail="rate"),
            route="/chat/completions",
        ),
        "auth_short_circuit": proxy_logging._is_proxy_only_llm_api_error(
            original_exception=Exception("auth"),
            error_type=ProxyErrorTypes.auth_error,
            route="/chat/completions",
        ),
    }
    assert snapshot == {
        "no_route": False,
        "non_llm_route": False,
        "http_on_llm_route": True,
        "auth_short_circuit": True,
    }


def test_is_proxy_only_llm_api_missing_exception_raises(proxy_logging):
    """Passing nothing should TypeError on the missing positional kwarg."""
    with pytest.raises(TypeError):
        proxy_logging._is_proxy_only_llm_api_error()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# post_call_failure_hook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_call_failure_hook_no_callbacks_returns_none(
    proxy_logging, make_user_api_key_auth, mock_callbacks_disabled
):
    proxy_logging.alert_types = []
    request_data = {"litellm_call_id": "abc", "model": "m", "messages": []}
    out = await proxy_logging.post_call_failure_hook(
        request_data=request_data,
        original_exception=ValueError("oops"),
        user_api_key_dict=make_user_api_key_auth(),
    )
    snapshot = {
        "out_is_none": out is None,
        "litellm_logging_obj_popped": "litellm_logging_obj" not in request_data,
        "call_id_preserved": request_data["litellm_call_id"] == "abc",
        "first_api_call_start_time_present": "first_api_call_start_time" in request_data,
    }
    assert snapshot == {
        "out_is_none": True,
        "litellm_logging_obj_popped": True,
        "call_id_preserved": True,
        "first_api_call_start_time_present": False,
    }


@pytest.mark.asyncio
async def test_post_call_failure_hook_callback_returns_http_exception(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    transformed = HTTPException(status_code=418, detail="teapot")

    class _Cb(CustomLogger):
        async def async_post_call_failure_hook(self, **kwargs):  # type: ignore[override]
            return transformed

    monkeypatch.setattr(litellm, "callbacks", [_Cb()])
    proxy_logging.alert_types = []
    out = await proxy_logging.post_call_failure_hook(
        request_data={"litellm_call_id": "abc"},
        original_exception=ValueError("oops"),
        user_api_key_dict=make_user_api_key_auth(),
    )
    assert out is transformed


@pytest.mark.asyncio
async def test_post_call_failure_hook_callback_raises_http_exception_first_wins(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    err = HTTPException(status_code=418, detail="raised teapot")

    class _Cb(CustomLogger):
        async def async_post_call_failure_hook(self, **kwargs):  # type: ignore[override]
            raise err

    monkeypatch.setattr(litellm, "callbacks", [_Cb()])
    proxy_logging.alert_types = []
    out = await proxy_logging.post_call_failure_hook(
        request_data={"litellm_call_id": "abc"},
        original_exception=ValueError("oops"),
        user_api_key_dict=make_user_api_key_auth(),
    )
    assert out is err


@pytest.mark.asyncio
async def test_post_call_failure_hook_non_http_exception_in_callback_swallowed(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    class _Cb(CustomLogger):
        async def async_post_call_failure_hook(self, **kwargs):  # type: ignore[override]
            raise RuntimeError("non-http inside cb")

    monkeypatch.setattr(litellm, "callbacks", [_Cb()])
    proxy_logging.alert_types = []
    out = await proxy_logging.post_call_failure_hook(
        request_data={"litellm_call_id": "abc"},
        original_exception=ValueError("oops"),
        user_api_key_dict=make_user_api_key_auth(),
    )
    assert out is None


# ---------------------------------------------------------------------------
# _handle_logging_proxy_only_error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_logging_proxy_only_path_uses_existing_logging_obj(
    proxy_logging, make_user_api_key_auth
):
    logging_obj = MagicMock()
    logging_obj.call_type = "acompletion"
    logging_obj.model_call_details = {}
    logging_obj.async_failure_handler = AsyncMock()

    request_data = {
        "litellm_logging_obj": logging_obj,
        "messages": [{"role": "user", "content": "x"}],
        "model": "m",
        "metadata": {},
    }
    await proxy_logging._handle_logging_proxy_only_error(
        request_data=request_data,
        user_api_key_dict=make_user_api_key_auth(),
        route="/chat/completions",
        original_exception=HTTPException(status_code=429, detail="rate"),
    )
    from litellm.constants import LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL

    snapshot = {
        "input_logged": "messages" in logging_obj.model_call_details,
        "call_type_normalized": logging_obj.call_type,
        "marker_present": logging_obj.model_call_details.get(
            LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL
        )
        is True,
        "async_failure_called": logging_obj.async_failure_handler.called,
    }
    assert snapshot == {
        "input_logged": True,
        "call_type_normalized": "acompletion",
        "marker_present": True,
        "async_failure_called": True,
    }


@pytest.mark.asyncio
async def test_handle_logging_proxy_only_path_skips_for_pass_through(
    proxy_logging, make_user_api_key_auth
):
    from litellm.types.utils import CallTypes

    logging_obj = MagicMock()
    logging_obj.call_type = CallTypes.pass_through.value
    logging_obj.model_call_details = {}
    logging_obj.async_failure_handler = AsyncMock()
    logging_obj.pre_call = MagicMock()
    request_data = {
        "litellm_logging_obj": logging_obj,
        "messages": [{"role": "user"}],
        "model": "m",
    }
    await proxy_logging._handle_logging_proxy_only_error(
        request_data=request_data,
        user_api_key_dict=make_user_api_key_auth(),
        route="/chat/completions",
        original_exception=HTTPException(status_code=429, detail="rate"),
    )
    logging_obj.pre_call.assert_not_called()
    logging_obj.async_failure_handler.assert_not_called()


@pytest.mark.asyncio
async def test_handle_logging_proxy_only_path_no_logging_obj_creates_one(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    fake_logging_obj = MagicMock()
    fake_logging_obj.call_type = "acompletion"
    fake_logging_obj.model_call_details = {}
    fake_logging_obj.async_failure_handler = AsyncMock()

    def fake_function_setup(**kwargs):
        return fake_logging_obj, {}

    monkeypatch.setattr(litellm.utils, "function_setup", fake_function_setup)
    request_data = {"messages": [{"role": "user"}], "model": "m"}
    await proxy_logging._handle_logging_proxy_only_error(
        request_data=request_data,
        user_api_key_dict=make_user_api_key_auth(),
        route="/chat/completions",
        original_exception=HTTPException(status_code=429, detail="rate"),
    )
    assert "litellm_call_id" in request_data
    fake_logging_obj.async_failure_handler.assert_called_once()


@pytest.mark.asyncio
async def test_handle_logging_proxy_only_path_propagates_async_failure_raises(
    proxy_logging, make_user_api_key_auth
):
    logging_obj = MagicMock()
    logging_obj.call_type = "acompletion"
    logging_obj.model_call_details = {}
    logging_obj.async_failure_handler = AsyncMock(side_effect=RuntimeError("boom"))
    request_data = {
        "litellm_logging_obj": logging_obj,
        "messages": [{"role": "user"}],
        "model": "m",
    }
    with pytest.raises(RuntimeError):
        await proxy_logging._handle_logging_proxy_only_error(
            request_data=request_data,
            user_api_key_dict=make_user_api_key_auth(),
            route="/chat/completions",
            original_exception=Exception("x"),
        )
