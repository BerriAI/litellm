"""Pin ``ProxyLogging.pre_call_hook`` and ``process_pre_call_hook_response``."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

import litellm
from litellm.exceptions import RejectedRequestError
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.utils import ProxyLogging


@pytest.fixture(autouse=True)
def _clear_caps_cache():
    ProxyLogging._callback_capabilities_cache.clear()
    yield
    ProxyLogging._callback_capabilities_cache.clear()


# ---------------------------------------------------------------------------
# process_pre_call_hook_response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_pre_call_hook_response_dict_returns_response(proxy_logging):
    out = await proxy_logging.process_pre_call_hook_response(
        response={"messages": [{"x": 1}], "model": "m", "temperature": 0.5},
        data={"original": True},
        call_type="completion",
    )
    assert out == {"messages": [{"x": 1}], "model": "m", "temperature": 0.5}


@pytest.mark.asyncio
async def test_process_pre_call_hook_response_string_completion_raises_rejected(proxy_logging):
    with pytest.raises(RejectedRequestError):
        await proxy_logging.process_pre_call_hook_response(
            response="rejected",
            data={"model": "m"},
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_process_pre_call_hook_response_string_other_call_type_raises_http(proxy_logging):
    with pytest.raises(HTTPException) as info:
        await proxy_logging.process_pre_call_hook_response(
            response="bad",
            data={},
            call_type="embeddings",
        )
    assert info.value.status_code == 400


@pytest.mark.asyncio
async def test_process_pre_call_hook_response_exception_reraises(proxy_logging):
    err = RuntimeError("hook said no")
    with pytest.raises(RuntimeError, match="hook said no"):
        await proxy_logging.process_pre_call_hook_response(
            response=err, data={}, call_type="completion"
        )


@pytest.mark.asyncio
async def test_process_pre_call_hook_response_other_type_returns_data(proxy_logging):
    out = await proxy_logging.process_pre_call_hook_response(
        response=12345, data={"a": 1, "b": 2, "c": 3}, call_type="completion"
    )
    assert out == {"a": 1, "b": 2, "c": 3}


# ---------------------------------------------------------------------------
# pre_call_hook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_call_hook_returns_data_when_no_callbacks(proxy_logging, make_user_api_key_auth, mock_callbacks_disabled):
    data = {"messages": [{"role": "user", "content": "hi"}], "model": "m", "temperature": 0.7}
    proxy_logging.slack_alerting_instance = MagicMock(alerting=None)
    out = await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(),
        data=data,
        call_type="completion",
    )
    assert out is data


@pytest.mark.asyncio
async def test_pre_call_hook_returns_none_for_none_data(proxy_logging, make_user_api_key_auth, mock_callbacks_disabled):
    proxy_logging.slack_alerting_instance = MagicMock(alerting=None)
    out = await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(),
        data=None,
        call_type="completion",
    )
    assert out is None


@pytest.mark.asyncio
async def test_pre_call_hook_invokes_pre_call_override(proxy_logging, make_user_api_key_auth, monkeypatch):
    captured: Dict[str, Any] = {}

    class _Cb(CustomLogger):
        async def async_pre_call_hook(self, **kwargs):  # type: ignore[override]
            captured.update(kwargs)
            return {"messages": [{"x": "modified"}], "model": "m", "temperature": 0.1}

    monkeypatch.setattr(litellm, "callbacks", [_Cb()])
    proxy_logging.slack_alerting_instance = MagicMock(alerting=None)
    out = await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(),
        data={"messages": [{"x": "input"}], "model": "m", "temperature": 0.1},
        call_type="completion",
    )
    snapshot = {
        "out_messages": out["messages"],
        "out_model": out["model"],
        "out_temp": out["temperature"],
        "cb_received_call_type": captured.get("call_type"),
    }
    assert snapshot == {
        "out_messages": [{"x": "modified"}],
        "out_model": "m",
        "out_temp": 0.1,
        "cb_received_call_type": "completion",
    }


@pytest.mark.asyncio
async def test_pre_call_hook_propagates_callback_error_raises(proxy_logging, make_user_api_key_auth, monkeypatch):
    class _BadCb(CustomLogger):
        async def async_pre_call_hook(self, **kwargs):  # type: ignore[override]
            raise RuntimeError("rejected")

    monkeypatch.setattr(litellm, "callbacks", [_BadCb()])
    proxy_logging.slack_alerting_instance = MagicMock(alerting=None)
    with pytest.raises(RuntimeError, match="rejected"):
        await proxy_logging.pre_call_hook(
            user_api_key_dict=make_user_api_key_auth(),
            data={"model": "m"},
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_pre_call_hook_processes_guardrail_metadata_when_no_overrides(proxy_logging, make_user_api_key_auth, mock_callbacks_disabled):
    """Even when no callback overrides exist, ``_process_guardrail_metadata`` runs."""
    data = {"messages": [{"role": "user"}], "model": "m", "metadata": {"guardrails": ["g1"]}}
    proxy_logging.slack_alerting_instance = MagicMock(alerting=None)
    invoked = {}

    def fake_process(d):
        invoked["data"] = d

    proxy_logging._process_guardrail_metadata = fake_process  # type: ignore[assignment]
    out = await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(),
        data=data,
        call_type="completion",
    )
    assert out is data
    assert invoked["data"] is data
