"""Pin ``ProxyLogging.during_call_hook``."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks


@pytest.fixture(autouse=True)
def _clear_caps_cache():
    ProxyLogging._callback_capabilities_cache.clear()
    yield
    ProxyLogging._callback_capabilities_cache.clear()


def _make_guardrail(name="g1", should_run=True, response=None):
    cb = MagicMock(spec=CustomGuardrail)
    cb.__class__ = CustomGuardrail
    cb.guardrail_name = name
    cb.event_hook = GuardrailEventHooks.during_call
    cb.use_native_during_call_hook = False
    cb.should_run_guardrail = MagicMock(return_value=should_run)
    cb.async_moderation_hook = AsyncMock(return_value=response)
    return cb


@pytest.mark.asyncio
async def test_during_call_hook_no_guardrail_fast_path_returns_data(proxy_logging, make_user_api_key_auth, mock_callbacks_disabled):
    data = {"messages": [{"role": "user"}], "model": "m", "temperature": 0.1}
    out = await proxy_logging.during_call_hook(
        data=data,
        user_api_key_dict=make_user_api_key_auth(),
        call_type="completion",
    )
    assert out is data


@pytest.mark.asyncio
async def test_during_call_hook_runs_guardrails_in_parallel(proxy_logging, make_user_api_key_auth, monkeypatch):
    g1 = _make_guardrail("a")
    g2 = _make_guardrail("b")
    monkeypatch.setattr(litellm, "callbacks", [g1, g2])
    data = {"messages": [{"role": "user"}], "model": "m", "temperature": 0.1}
    out = await proxy_logging.during_call_hook(
        data=data,
        user_api_key_dict=make_user_api_key_auth(),
        call_type="completion",
    )
    snapshot = {
        "out_is_data": out is data,
        "a_called": g1.async_moderation_hook.called,
        "b_called": g2.async_moderation_hook.called,
    }
    assert snapshot == {"out_is_data": True, "a_called": True, "b_called": True}


@pytest.mark.asyncio
async def test_during_call_hook_guardrail_skipped_when_should_not_run(proxy_logging, make_user_api_key_auth, monkeypatch):
    g = _make_guardrail("g", should_run=False)
    monkeypatch.setattr(litellm, "callbacks", [g])
    await proxy_logging.during_call_hook(
        data={"model": "m"},
        user_api_key_dict=make_user_api_key_auth(),
        call_type="completion",
    )
    g.async_moderation_hook.assert_not_called()


@pytest.mark.asyncio
async def test_during_call_hook_guardrail_error_raises(proxy_logging, make_user_api_key_auth, monkeypatch):
    g = _make_guardrail("bad")
    g.async_moderation_hook = AsyncMock(side_effect=RuntimeError("blocked"))
    monkeypatch.setattr(litellm, "callbacks", [g])
    with pytest.raises(RuntimeError):
        await proxy_logging.during_call_hook(
            data={"model": "m"},
            user_api_key_dict=make_user_api_key_auth(),
            call_type="completion",
        )
