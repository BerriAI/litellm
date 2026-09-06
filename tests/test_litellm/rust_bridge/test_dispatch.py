from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.exceptions import APIError
from litellm.rust_bridge import bindings
from litellm.rust_bridge.chat_completions import error_handling
from litellm.rust_bridge.dispatch import PROPAGATE, PYTHON_ON_ERROR, adispatch, dispatch
from litellm.rust_bridge.runtime import DispatchResult, Handled, NativeFailed, NativeSkipped, NativeSkipReason


class Declined(Exception):
    pass


class Upstream(Exception):
    pass


@pytest.fixture(autouse=True)
def native_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bindings, "get_native_bridge", lambda: SimpleNamespace(RustBridgeDeclined=Declined, RustUpstreamError=Upstream)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("reason", tuple(NativeSkipReason))
async def test_shared_dispatch_calls_python_once_and_logs_skip(
    asynchronous: bool, reason: NativeSkipReason, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="LiteLLM")
    calls: Final[list[str]] = []

    def native() -> DispatchResult[str]:
        calls.append("native")
        return NativeSkipped(reason, "diagnostic detail")

    async def anative() -> DispatchResult[str]:
        return native()

    def python() -> str:
        calls.append("python")
        return "python response"

    async def apython() -> str:
        return python()

    result: Final = (
        await adispatch(native=anative, python=apython, route="test", errors=PROPAGATE)
        if asynchronous
        else dispatch(native=native, python=python, route="test", errors=PROPAGATE)
    )
    assert result == "python response"
    assert calls == ["native", "python"]
    assert f"Native test skipped ({reason.value}): diagnostic detail" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_native_success_does_not_run_python_even_when_value_is_none(asynchronous: bool) -> None:
    async def native() -> DispatchResult[None]:
        return Handled(None)

    def python() -> str:
        pytest.fail("handled results must not run Python")

    async def apython() -> str:
        return python()

    result: Final = (
        await adispatch(native=native, python=apython, route="test", errors=PYTHON_ON_ERROR)
        if asynchronous
        else dispatch(native=lambda: Handled(None), python=python, route="test", errors=PYTHON_ON_ERROR)
    )
    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("policy", ("chat", "propagate", "python"))
@pytest.mark.parametrize("kind", ("declined", "upstream", "unknown", "unexpected", "missing"))
async def test_declarations_preserve_endpoint_error_behavior(
    monkeypatch: pytest.MonkeyPatch, asynchronous: bool, policy: str, kind: str
) -> None:
    if kind == "missing":
        monkeypatch.setattr(bindings, "get_native_bridge", lambda: None)
    error: Final = (
        Declined("unsupported")
        if kind == "declined"
        else Upstream(429, "rate limited")
        if kind == "upstream"
        else RuntimeError("failed")
    )
    rules: Final = (
        error_handling("anthropic", "model")
        if policy == "chat"
        else PYTHON_ON_ERROR
        if policy == "python"
        else PROPAGATE
    )
    calls: Final[list[str]] = []

    def native() -> DispatchResult[str]:
        if kind == "unexpected":
            raise error
        return NativeFailed(error)

    async def anative() -> DispatchResult[str]:
        return native()

    def python() -> str:
        calls.append("python")
        return "python response"

    async def apython() -> str:
        return python()

    async def run() -> str:
        if asynchronous:
            return await adispatch(native=anative, python=apython, route="chat_completions", errors=rules)
        return dispatch(native=native, python=python, route="chat_completions", errors=rules)

    if policy == "python" or (policy == "chat" and kind in ("declined", "missing")):
        assert await run() == "python response"
        assert calls == ["python"]
    elif policy == "chat" and kind == "upstream":
        with pytest.raises(APIError) as caught:
            await run()
        assert caught.value.status_code == 429
        assert caught.value.model == "model"
        assert caught.value.llm_provider == "anthropic"
        assert caught.value.__cause__ is error
        assert calls == []
    else:
        with pytest.raises(type(error)) as caught_original:
            await run()
        assert caught_original.value is error
        assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_python_failure_is_never_reclassified_as_native_failure(asynchronous: bool) -> None:
    error: Final = RuntimeError("Python failed")
    calls: Final[list[str]] = []

    async def native() -> DispatchResult[str]:
        return NativeSkipped(NativeSkipReason.UNAVAILABLE)

    def python() -> str:
        calls.append("python")
        raise error

    async def apython() -> str:
        return python()

    async def run() -> str:
        if asynchronous:
            return await adispatch(native=native, python=apython, route="test", errors=PYTHON_ON_ERROR)
        return dispatch(
            native=lambda: NativeSkipped(NativeSkipReason.UNAVAILABLE),
            python=python,
            route="test",
            errors=PYTHON_ON_ERROR,
        )

    with pytest.raises(RuntimeError) as caught:
        await run()
    assert caught.value is error
    assert calls == ["python"]


@pytest.mark.asyncio
async def test_cancellation_does_not_run_python() -> None:
    async def native() -> DispatchResult[str]:
        raise asyncio.CancelledError

    async def python() -> str:
        pytest.fail("cancellation must not dispatch Python")

    with pytest.raises(asyncio.CancelledError):
        await adispatch(native=native, python=python, route="test", errors=PYTHON_ON_ERROR)


@pytest.mark.parametrize("status", (0, 401, 403, 429, 500, 503))
def test_chat_upstream_mapping_preserves_status_message_and_context(status: int) -> None:
    error: Final = Upstream(status, "upstream failed")
    with pytest.raises(APIError, match="upstream failed") as caught:
        dispatch(
            native=lambda: NativeFailed(error),
            python=lambda: pytest.fail("upstream errors must not run Python"),
            route="chat_completions",
            errors=error_handling("anthropic", "model"),
        )
    assert caught.value.status_code == (status or 500)
    assert caught.value.model == "model"
    assert caught.value.llm_provider == "anthropic"
    assert caught.value.__cause__ is error
