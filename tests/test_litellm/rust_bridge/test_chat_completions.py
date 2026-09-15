from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.rust_bridge import bindings, configuration
from litellm.rust_bridge.chat_completions import lifecycle


class RustBridgeDeclined(Exception):
    pass


class RustBridgeUnavailable(Exception):
    pass


class RustHostCallbackError(Exception):
    pass


class RustUpstreamError(Exception):
    pass


NATIVE_EXCEPTIONS: Final = SimpleNamespace(
    RustBridgeDeclined=RustBridgeDeclined,
    RustBridgeUnavailable=RustBridgeUnavailable,
    RustHostCallbackError=RustHostCallbackError,
    RustUpstreamError=RustUpstreamError,
)


class _RecordingSync:
    def __init__(self, result: object = "native", error: BaseException | None = None) -> None:
        self.result: Final = result
        self.error: Final = error
        self.calls: Final[list[tuple[dict[str, object], tuple[object, ...], dict[str, object], object]]] = []

    def __call__(
        self,
        request: dict[str, object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        host: object,
    ) -> object:
        self.calls.append((request, args, kwargs, host))
        if self.error is not None:
            raise self.error
        return self.result


class _RecordingAsync(_RecordingSync):
    def __call__(
        self,
        request: dict[str, object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        host: object,
    ) -> Awaitable[object]:
        self.calls.append((request, args, kwargs, host))
        if self.error is not None:
            raise self.error

        async def result() -> object:
            return self.result

        return result()


@pytest.fixture(autouse=True)
def reset_bridge(monkeypatch: pytest.MonkeyPatch):
    lifecycle.set_rust_chat_completions(chat_completions=None, achat_completions=None)
    configuration.reset_rust_configuration()
    monkeypatch.setenv("LITELLM_RUST", "1")
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: NATIVE_EXCEPTIONS)
    yield
    lifecycle.set_rust_chat_completions(chat_completions=None, achat_completions=None)
    configuration.reset_rust_configuration()


def _sync_function(model: str, messages: list[object], **kwargs: object) -> object:
    return (model, messages, kwargs)


async def _async_function(model: str, messages: list[object], **kwargs: object) -> object:
    return (model, messages, kwargs)


def test_sync_boundary_enters_native_once_and_preserves_call_shape() -> None:
    rust: Final = _RecordingSync()
    lifecycle.set_rust_chat_completions(chat_completions=rust)
    wrapped: Final = lifecycle.wrap_sync(_sync_function)
    messages: Final[list[object]] = [{"role": "user", "content": "hi"}]

    assert wrapped("anthropic/model", messages, temperature=0.2) == "native"
    assert len(rust.calls) == 1
    request, args, kwargs, _ = rust.calls[0]
    assert args == ("anthropic/model", messages)
    assert kwargs == {"temperature": 0.2}
    assert request["model"] == "anthropic/model"
    assert request["messages"] is messages
    assert inspect.signature(wrapped) == inspect.signature(_sync_function)


def test_decline_calls_captured_python_implementation_once() -> None:
    rust: Final = _RecordingSync(error=RustBridgeDeclined("unsupported"))
    lifecycle.set_rust_chat_completions(chat_completions=rust)
    calls: Final[list[None]] = []

    def python(model: str, messages: list[object]) -> str:
        calls.append(None)
        return f"python:{model}:{len(messages)}"

    wrapped: Final = lifecycle.wrap_sync(python)
    assert wrapped("anthropic/model", []) == "python:anthropic/model:0"
    assert len(rust.calls) == 1
    assert calls == [None]


def test_missing_binding_calls_python_without_native_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Final[list[None]] = []
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: None)

    def python(model: str, messages: list[object]) -> str:
        calls.append(None)
        return model

    assert lifecycle.wrap_sync(python)("anthropic/model", []) == "anthropic/model"
    assert calls == [None]


@pytest.mark.asyncio
async def test_async_decline_is_caught_only_while_obtaining_coroutine() -> None:
    entry_decline: Final = _RecordingAsync(error=RustBridgeDeclined("admission"))
    lifecycle.set_rust_chat_completions(achat_completions=entry_decline)
    wrapped: Final = lifecycle.wrap_async(_async_function)
    python_result: Final = await wrapped("anthropic/model", [])
    assert python_result == ("anthropic/model", [], {})

    execution_error: Final = RustBridgeDeclined("resume")

    def fail_after_admission(
        request: dict[str, object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        host: object,
    ) -> Awaitable[object]:
        async def fail() -> object:
            await asyncio.sleep(0)
            raise execution_error

        return fail()

    lifecycle.set_rust_chat_completions(achat_completions=fail_after_admission)
    with pytest.raises(RustBridgeDeclined) as caught:
        await wrapped("anthropic/model", [])
    assert caught.value is execution_error


def test_streaming_is_declined_by_the_same_native_entry() -> None:
    rust: Final = _RecordingSync(error=RustBridgeDeclined("streaming"))
    lifecycle.set_rust_chat_completions(chat_completions=rust)
    wrapped: Final = lifecycle.wrap_sync(_sync_function)

    assert wrapped("anthropic/model", [], stream=True) == ("anthropic/model", [], {"stream": True})
    assert len(rust.calls) == 1
    assert rust.calls[0][0]["host_facts"] == {"stream": True}
