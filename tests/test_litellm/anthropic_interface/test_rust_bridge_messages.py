from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.rust_bridge import bindings, configuration
from litellm.rust_bridge.messages import lifecycle


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


class RecordingSync:
    def __init__(self, error: BaseException | None = None) -> None:
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
        return "native"


@pytest.fixture(autouse=True)
def reset_bridge(monkeypatch: pytest.MonkeyPatch):
    lifecycle.set_rust_messages(messages=None, amessages=None)
    configuration.reset_rust_configuration()
    monkeypatch.setenv("LITELLM_RUST", "1")
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: NATIVE_EXCEPTIONS)
    yield
    lifecycle.set_rust_messages(messages=None, amessages=None)
    configuration.reset_rust_configuration()


def sync_python(max_tokens: int, messages: list[object], model: str, **kwargs: object) -> object:
    return max_tokens, messages, model, kwargs


async def async_python(max_tokens: int, messages: list[object], model: str, **kwargs: object) -> object:
    return max_tokens, messages, model, kwargs


def test_sync_boundary_enters_native_once_and_preserves_call_shape() -> None:
    rust: Final = RecordingSync()
    lifecycle.set_rust_messages(messages=rust)
    wrapped: Final = lifecycle.wrap_sync(sync_python)
    messages: Final[list[object]] = [{"role": "user", "content": "hi"}]

    assert wrapped(64, messages, "anthropic/model", temperature=0.2) == "native"
    assert len(rust.calls) == 1
    request, args, kwargs, _ = rust.calls[0]
    assert args == (64, messages, "anthropic/model")
    assert kwargs == {"temperature": 0.2}
    assert request["model"] == "anthropic/model"
    assert request["body"] == {"max_tokens": 64, "messages": messages, "temperature": 0.2}
    assert inspect.signature(wrapped) == inspect.signature(sync_python)


def test_advisor_interceptor_request_declines_native_host_ownership() -> None:
    rust: Final = RecordingSync(RustBridgeDeclined("host operations"))
    lifecycle.set_rust_messages(messages=rust)
    wrapped: Final = lifecycle.wrap_sync(sync_python)
    tools: Final[list[object]] = [{"type": "advisor_20260301", "model": "advisor-model"}]

    wrapped(64, [], "anthropic/model", tools=tools)

    request, _, _, _ = rust.calls[0]
    assert request["has_agentic_hook"] is True


def test_decline_calls_captured_python_implementation_once() -> None:
    rust: Final = RecordingSync(RustBridgeDeclined("unsupported"))
    lifecycle.set_rust_messages(messages=rust)
    calls: Final[list[None]] = []

    def python(max_tokens: int, messages: list[object], model: str) -> str:
        calls.append(None)
        return model

    assert lifecycle.wrap_sync(python)(1, [], "anthropic/model") == "anthropic/model"
    assert len(rust.calls) == 1
    assert calls == [None]


@pytest.mark.asyncio
async def test_async_decline_is_caught_only_during_admission() -> None:
    def decline(
        request: dict[str, object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        host: object,
    ) -> Awaitable[object]:
        raise RustBridgeDeclined("admission")

    lifecycle.set_rust_messages(amessages=decline)
    wrapped: Final = lifecycle.wrap_async(async_python)
    assert await wrapped(1, [], "anthropic/model") == (1, [], "anthropic/model", {})

    error: Final = RustBridgeDeclined("resume")

    def fail(
        request: dict[str, object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        host: object,
    ) -> Awaitable[object]:
        async def result() -> object:
            await asyncio.sleep(0)
            raise error

        return result()

    lifecycle.set_rust_messages(amessages=fail)
    with pytest.raises(RustBridgeDeclined) as caught:
        await wrapped(1, [], "anthropic/model")
    assert caught.value is error


def test_missing_binding_calls_python_without_native_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: None)
    assert lifecycle.wrap_sync(sync_python)(1, [], "anthropic/model") == (1, [], "anthropic/model", {})
