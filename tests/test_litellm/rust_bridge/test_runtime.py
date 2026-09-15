from __future__ import annotations

import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.exceptions import APIError
from litellm.rust_bridge import bindings, runtime
from litellm.rust_bridge.configuration import ComponentName, ExecutionDecision
from litellm.rust_bridge.errors import RustRouteDeclinedError, RustRouteUnavailableError, RustRouteUnsupportedError
from litellm.rust_bridge.route import ComponentExecution


class RustBridgeDeclined(Exception):
    pass


class RustBridgeUnavailable(Exception):
    pass


class RustUpstreamError(Exception):
    pass


class RustHostCallbackError(Exception):
    pass


@pytest.fixture(autouse=True)
def native_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    native: Final = SimpleNamespace(
        RustBridgeDeclined=RustBridgeDeclined,
        RustBridgeUnavailable=RustBridgeUnavailable,
        RustHostCallbackError=RustHostCallbackError,
        RustUpstreamError=RustUpstreamError,
    )
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)


async def _invoke(
    asynchronous: bool,
    decision: ExecutionDecision,
    native_call: Callable[[], object] | None,
    adapt: Callable[[object], object],
    fallback: Callable[[], object] = lambda: "python",
) -> object:
    execution: Final = ComponentExecution(route_name=ComponentName.MESSAGES, decision=decision)
    context: Final = runtime.BridgeErrorContext(route="messages", provider="anthropic", model="model")
    if not asynchronous:
        return runtime.invoke(
            execution=execution,
            native_call=native_call,
            python_fallback=fallback,
            adapt=adapt,
            context=context,
        )

    async def call() -> object:
        assert native_call is not None
        return native_call()

    async def afallback() -> object:
        return fallback()

    return await runtime.ainvoke(
        execution=execution,
        native_call=call if native_call is not None else None,
        python_fallback=afallback,
        adapt=adapt,
        context=context,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_python_selection_runs_only_the_fallback(asynchronous: bool) -> None:
    calls: Final[list[str]] = []

    def native() -> str:
        calls.append("native")
        return "native"

    result: Final = await _invoke(
        asynchronous,
        ExecutionDecision.PYTHON,
        native,
        str,
        lambda: calls.append("python") or "python",
    )
    assert result == "python"
    assert calls == ["python"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("error", (None, RustBridgeDeclined("unsupported"), RustBridgeUnavailable()))
@pytest.mark.parametrize("required", (False, True))
async def test_unavailable_and_declined_follow_policy(
    asynchronous: bool, error: Exception | None, required: bool
) -> None:
    def fail() -> str:
        assert error is not None
        raise error

    decision: Final = ExecutionDecision.RUST_REQUIRED if required else ExecutionDecision.RUST_WITH_FALLBACK
    native_call: Final = fail if error is not None else None
    if required:
        expected: Final = RustRouteDeclinedError if isinstance(error, RustBridgeDeclined) else RustRouteUnavailableError
        with pytest.raises(expected):
            await _invoke(asynchronous, decision, native_call, str)
        return
    assert await _invoke(asynchronous, decision, native_call, str) == "python"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("decision", (ExecutionDecision.RUST_WITH_FALLBACK, ExecutionDecision.RUST_REQUIRED))
@pytest.mark.parametrize("value", (None, False, 0, "native"))
async def test_native_success_does_not_run_fallback(
    asynchronous: bool, decision: ExecutionDecision, value: None | bool | int | str
) -> None:
    calls: Final[list[str]] = []
    result: Final = await _invoke(
        asynchronous,
        decision,
        lambda: value,
        lambda native: native,
        lambda: calls.append("python") or "python",
    )
    assert result is value
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_upstream_failure_never_falls_back(asynchronous: bool) -> None:
    def fail() -> str:
        raise RustUpstreamError(429, "rate limited")

    with pytest.raises(APIError, match="rate limited") as caught:
        await _invoke(asynchronous, ExecutionDecision.RUST_WITH_FALLBACK, fail, str)
    assert caught.value.status_code == 429


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("error", (RuntimeError("execution failed"), asyncio.CancelledError()))
async def test_execution_failure_never_falls_back(asynchronous: bool, error: BaseException) -> None:
    def fail() -> str:
        raise error

    with pytest.raises(type(error)) as caught:
        await _invoke(asynchronous, ExecutionDecision.RUST_WITH_FALLBACK, fail, str)
    assert caught.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("error", (RustBridgeDeclined("adapt"), RustBridgeUnavailable(), RuntimeError("adapt")))
async def test_adaptation_failure_never_falls_back(asynchronous: bool, error: Exception) -> None:
    def adapt(_value: str) -> str:
        raise error

    with pytest.raises(type(error)) as caught:
        await _invoke(asynchronous, ExecutionDecision.RUST_WITH_FALLBACK, lambda: "native", adapt)
    assert caught.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_host_callback_failure_preserves_its_cause(asynchronous: bool) -> None:
    callback_error: Final = RustBridgeDeclined("callback raised native decline type")

    def fail() -> str:
        wrapped: Final = RustHostCallbackError("callback failed")
        wrapped.__cause__ = callback_error
        raise wrapped

    with pytest.raises(RustBridgeDeclined) as caught:
        await _invoke(asynchronous, ExecutionDecision.RUST_WITH_FALLBACK, fail, str)
    assert caught.value is callback_error


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_unsupported_execution_runs_nothing(asynchronous: bool) -> None:
    with pytest.raises(RustRouteUnsupportedError):
        await _invoke(asynchronous, ExecutionDecision.UNSUPPORTED, lambda: "native", str)
