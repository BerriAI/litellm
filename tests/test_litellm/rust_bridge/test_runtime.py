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
    fallback: Callable[[], object] | None = lambda: "python",
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
        assert fallback is not None
        return fallback()

    async def aadapt(value: object) -> object:
        return adapt(value)

    return await runtime.ainvoke(
        execution=execution,
        native_call=call if native_call is not None else None,
        python_fallback=afallback if fallback is not None else None,
        adapt=aadapt,
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
            await _invoke(asynchronous, decision, native_call, str, None)
        return
    assert await _invoke(asynchronous, decision, native_call, str) == "python"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("native_call", (None, lambda: (_ for _ in ()).throw(RustBridgeDeclined("declined"))))
async def test_optional_failure_runs_python_exactly_once(
    asynchronous: bool,
    native_call: Callable[[], object] | None,
) -> None:
    fallback_calls: Final[list[None]] = []

    result: Final = await _invoke(
        asynchronous,
        ExecutionDecision.RUST_WITH_FALLBACK,
        native_call,
        str,
        lambda: fallback_calls.append(None) or "python",
    )

    assert result == "python"
    assert fallback_calls == [None]


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
        None if decision is ExecutionDecision.RUST_REQUIRED else lambda: calls.append("python") or "python",
    )
    assert result is value
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("decision", (ExecutionDecision.PYTHON, ExecutionDecision.RUST_WITH_FALLBACK))
async def test_python_capability_requires_fallback(asynchronous: bool, decision: ExecutionDecision) -> None:
    native_calls: Final[list[bool]] = []
    with pytest.raises(ValueError, match="declares a Python implementation"):
        await _invoke(asynchronous, decision, lambda: native_calls.append(True), str, None)
    assert native_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_rust_required_rejects_python_fallback(asynchronous: bool) -> None:
    native_calls: Final[list[bool]] = []
    with pytest.raises(ValueError, match="declares no Python implementation"):
        await _invoke(
            asynchronous,
            ExecutionDecision.RUST_REQUIRED,
            lambda: native_calls.append(True),
            str,
            lambda: "python",
        )
    assert native_calls == []


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
async def test_async_adaptation_is_awaited_and_failure_never_falls_back() -> None:
    fallback_calls: Final[list[None]] = []

    async def adapt(_value: object) -> object:
        await asyncio.sleep(0)
        raise RuntimeError("async adapt failed")

    execution: Final = ComponentExecution(
        route_name=ComponentName.MESSAGES,
        decision=ExecutionDecision.RUST_WITH_FALLBACK,
    )

    with pytest.raises(RuntimeError, match="async adapt failed"):
        await runtime.ainvoke(
            execution=execution,
            native_call=lambda: asyncio.sleep(0, result="native"),
            python_fallback=lambda: asyncio.sleep(0, result=fallback_calls.append(None)),
            adapt=adapt,
            context=runtime.BridgeErrorContext(route="messages", provider="anthropic", model="model"),
        )

    assert fallback_calls == []


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


def test_lifecycle_unavailable_during_execution_never_falls_back() -> None:
    fallback_calls: Final[list[None]] = []

    with pytest.raises(RustBridgeUnavailable):
        runtime.invoke_lifecycle(
            execution=ComponentExecution(
                route_name=ComponentName.MESSAGES,
                decision=ExecutionDecision.RUST_WITH_FALLBACK,
            ),
            native_call=lambda: (_ for _ in ()).throw(RustBridgeUnavailable()),
            python_fallback=lambda: fallback_calls.append(None),
        )

    assert fallback_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("error", (RustBridgeDeclined("resume"), RustBridgeUnavailable()))
async def test_async_lifecycle_reserved_error_after_admission_never_falls_back(error: Exception) -> None:
    fallback_calls: Final[list[None]] = []

    async def fail_after_admission() -> object:
        await asyncio.sleep(0)
        raise error

    with pytest.raises(type(error)) as caught:
        await runtime.ainvoke_lifecycle(
            execution=ComponentExecution(
                route_name=ComponentName.MESSAGES,
                decision=ExecutionDecision.RUST_WITH_FALLBACK,
            ),
            native_call=fail_after_admission,
            python_fallback=lambda: asyncio.sleep(0, result=fallback_calls.append(None)),
        )

    assert caught.value is error
    assert fallback_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_unsupported_execution_runs_nothing(asynchronous: bool) -> None:
    with pytest.raises(RustRouteUnsupportedError):
        await _invoke(asynchronous, ExecutionDecision.UNSUPPORTED, lambda: "native", str)
