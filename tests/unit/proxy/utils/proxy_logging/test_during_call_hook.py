"""Pin ``ProxyLogging.during_call_hook``."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypesLiteral


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


@pytest.mark.asyncio
async def test_during_call_block_names_the_blocking_guardrail_in_applied_guardrails(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    g = _make_guardrail("blocker")
    g.async_moderation_hook = AsyncMock(side_effect=HTTPException(status_code=400, detail="blocked"))
    monkeypatch.setattr(litellm, "callbacks", [_make_guardrail("passer"), g])
    data = {"model": "m", "metadata": {}}
    with pytest.raises(HTTPException):
        await proxy_logging.during_call_hook(
            data=data,
            user_api_key_dict=make_user_api_key_auth(),
            call_type="completion",
        )
    assert "blocker" in data["metadata"]["applied_guardrails"]


class _ControlledModerationGuardrail(CustomGuardrail):
    def __init__(
        self,
        name: str,
        event_hook: GuardrailEventHooks,
        scan: Callable[[], Awaitable[None]],
    ) -> None:
        super().__init__(guardrail_name=name, event_hook=event_hook, default_on=True)
        self.scan = scan

    async def async_moderation_hook(
        self,
        data: dict[str, object],
        user_api_key_dict: UserAPIKeyAuth | dict[str, object] | None,
        call_type: CallTypesLiteral,
    ) -> None:
        await self.scan()


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ["call_mcp_tool", "acompletion"])
async def test_unselected_raise_policy_preserves_during_call_failure_without_waiting_for_other_scans(
    proxy_logging: ProxyLogging,
    monkeypatch: pytest.MonkeyPatch,
    call_type: CallTypesLiteral,
) -> None:
    slow_started: Final = asyncio.Event()
    release_slow: Final = asyncio.Event()
    slow_finished: Final = asyncio.Event()
    failure: Final = RuntimeError("scanner unavailable")

    async def slow_scan() -> None:
        slow_started.set()
        try:
            await release_slow.wait()
        finally:
            slow_finished.set()

    async def failing_scan() -> None:
        await slow_started.wait()
        raise failure

    mode: Final = (
        GuardrailEventHooks.during_mcp_call if call_type == "call_mcp_tool" else GuardrailEventHooks.during_call
    )
    unrelated: Final = CustomGuardrail(guardrail_name="unselected", event_hook=mode, default_on=False)
    unrelated.apply_mcp_block_policy("raise")
    monkeypatch.setattr(
        litellm,
        "callbacks",
        [
            _ControlledModerationGuardrail("slow", mode, slow_scan),
            _ControlledModerationGuardrail("failing", mode, failing_scan),
            unrelated,
        ],
    )
    request: Final = asyncio.create_task(
        proxy_logging.during_call_hook(
            data={"model": "mcp-tool-call", "mcp_tool_name": "search", "mcp_arguments": {"query": "hello"}},
            user_api_key_dict=UserAPIKeyAuth(),
            call_type=call_type,
        )
    )
    try:
        with pytest.raises(RuntimeError, match="scanner unavailable") as raised:
            await asyncio.wait_for(request, timeout=5)
        assert raised.value is failure
        assert not slow_finished.is_set()
    finally:
        release_slow.set()
        await slow_finished.wait()
