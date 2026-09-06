from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.exceptions import APIError
from litellm.rust_bridge import bindings
from litellm.rust_bridge.chat_completions import error_handling
from litellm.rust_bridge.dispatch import PROPAGATE, PYTHON_ON_ERROR, anative_first, native_first
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
        await anative_first(native=anative, route="test", errors=lambda: PROPAGATE)(apython)()
        if asynchronous
        else native_first(native=native, route="test", errors=lambda: PROPAGATE)(python)()
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
        await anative_first(native=native, route="test", errors=lambda: PYTHON_ON_ERROR)(apython)()
        if asynchronous
        else native_first(native=lambda: Handled(None), route="test", errors=lambda: PYTHON_ON_ERROR)(python)()
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
            return await anative_first(native=anative, route="chat_completions", errors=lambda: rules)(apython)()
        return native_first(native=native, route="chat_completions", errors=lambda: rules)(python)()

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
            return await anative_first(native=native, route="test", errors=lambda: PYTHON_ON_ERROR)(apython)()
        return native_first(
            native=lambda: NativeSkipped(NativeSkipReason.UNAVAILABLE), route="test", errors=lambda: PYTHON_ON_ERROR
        )(python)()

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
        await anative_first(native=native, route="test", errors=lambda: PYTHON_ON_ERROR)(python)()


@pytest.mark.parametrize("status", (0, 401, 403, 429, 500, 503))
def test_chat_upstream_mapping_preserves_status_message_and_context(status: int) -> None:
    error: Final = Upstream(status, "upstream failed")
    with pytest.raises(APIError, match="upstream failed") as caught:
        native_first(
            native=lambda: NativeFailed(error),
            route="chat_completions",
            errors=lambda: error_handling("anthropic", "model"),
        )(lambda: pytest.fail("upstream errors must not run Python"))()
    assert caught.value.status_code == (status or 500)
    assert caught.value.model == "model"
    assert caught.value.llm_provider == "anthropic"
    assert caught.value.__cause__ is error


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_registered_wrapper_preserves_arguments_and_request_error_context(asynchronous: bool) -> None:
    calls: Final[list[tuple[str, str, str]]] = []

    def native(provider: str, *, model: str) -> DispatchResult[str]:
        calls.append(("native", provider, model))
        return (
            NativeFailed(Upstream(429, "limited"))
            if model == "limited"
            else NativeSkipped(NativeSkipReason.UNAVAILABLE)
        )

    async def anative(provider: str, *, model: str) -> DispatchResult[str]:
        return native(provider, model=model)

    def rules(provider: str, *, model: str):
        return error_handling(provider, model)

    @native_first(native=native, route="chat_completions", errors=rules)
    def execute(provider: str, *, model: str) -> str:
        calls.append(("python", provider, model))
        return model

    @anative_first(native=anative, route="chat_completions", errors=rules)
    async def aexecute(provider: str, *, model: str) -> str:
        calls.append(("python", provider, model))
        return model

    assert (await aexecute("first", model="ok") if asynchronous else execute("first", model="ok")) == "ok"

    async def fail() -> None:
        if asynchronous:
            await aexecute("second", model="limited")
        else:
            execute("second", model="limited")

    with pytest.raises(APIError) as caught:
        await fail()
    assert caught.value.llm_provider == "second"
    assert caught.value.model == "limited"
    assert calls == [("native", "first", "ok"), ("python", "first", "ok"), ("native", "second", "limited")]


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ("native", "unavailable", "failed"))
@pytest.mark.parametrize("failure", ("none", "body", "cleanup", "cancel"))
async def test_context_selection_and_lifetime_are_separate(selection: str, failure: str) -> None:
    from collections.abc import AsyncGenerator
    from contextlib import AbstractAsyncContextManager, asynccontextmanager

    from litellm.rust_bridge.dispatch import anative_context

    events: Final[list[str]] = []
    error: Final = RuntimeError("connection use failed")

    @asynccontextmanager
    async def connection(name: str) -> AsyncGenerator[str, None]:
        events.append(f"{name}:enter")
        try:
            yield name
        finally:
            events.append(f"{name}:exit")
            if failure == "cleanup":
                raise error

    async def native() -> DispatchResult[AbstractAsyncContextManager[str]]:
        events.append("attempt")
        if selection == "failed":
            raise RuntimeError("connect failed")
        if selection == "unavailable":
            return NativeSkipped(NativeSkipReason.UNAVAILABLE)
        return Handled(connection("native"))

    @anative_context(native=native, route="websocket", errors=lambda: PYTHON_ON_ERROR)
    def execute() -> AbstractAsyncContextManager[str]:
        events.append("python")
        return connection("python")

    async def run() -> None:
        async with execute() as name:
            assert name == ("native" if selection == "native" else "python")
            if failure == "body":
                raise error
            if failure == "cancel":
                raise asyncio.CancelledError

    if failure == "none":
        await run()
    elif failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await run()
    else:
        with pytest.raises(RuntimeError) as caught:
            await run()
        assert caught.value is error
    expected: Final = (
        ["attempt", "native:enter", "native:exit"]
        if selection == "native"
        else ["attempt", "python", "python:enter", "python:exit"]
    )
    assert events == expected
