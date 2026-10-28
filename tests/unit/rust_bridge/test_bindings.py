import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Final, cast  # noqa: TID251  # narrows the async binding's untyped coroutine

import pytest

from litellm.rust_bridge import bindings
from litellm.rust_bridge.chat_completions import entrypoints as chat_completions
from litellm.rust_bridge.embeddings import entrypoints as embeddings
from litellm.rust_bridge.execution import Complete, Step
from litellm.rust_bridge.messages import entrypoints as messages
from litellm.rust_bridge.ocr import entrypoints as ocr
from litellm.rust_bridge.public_call import NativeCall
from litellm.rust_bridge.responses import entrypoints as responses
from litellm.rust_bridge.transcription import native as transcription


def test_binding_distinguishes_disable_from_reset(monkeypatch) -> None:
    native = SimpleNamespace(route=lambda: "native")
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    binding: bindings.NativeBinding[object] = bindings.NativeBinding("route", validate=lambda value: value)

    assert binding.load() is native.route

    binding.override(None)
    assert binding.load() is None

    replacement = object()
    binding.override(replacement)
    assert binding.load() is replacement

    binding.reset()
    assert binding.load() is native.route


@pytest.mark.parametrize(("value", "expected"), ((3, 3), ("invalid", None), (None, None)))
def test_binding_validates_native_attribute(
    monkeypatch: pytest.MonkeyPatch, value: object, expected: int | None
) -> None:
    native: Final = SimpleNamespace(route=value)
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    binding: Final = bindings.NativeBinding("route", validate=lambda item: item if isinstance(item, int) else None)

    assert binding.load() == expected


EXECUTION_BINDINGS: Final = (
    ("completion", chat_completions.NATIVE_COMPLETION, False),
    ("acompletion", chat_completions.NATIVE_ACOMPLETION, True),
    ("embedding", embeddings.NATIVE_EMBEDDING, False),
    ("aembedding", embeddings.NATIVE_AEMBEDDING, True),
    ("messages", messages.NATIVE_MESSAGES, False),
    ("amessages", messages.NATIVE_AMESSAGES, True),
    ("responses", responses.NATIVE_RESPONSES, False),
    ("aresponses", responses.NATIVE_ARESPONSES, True),
    ("ocr", ocr.NATIVE_OCR, False),
    ("aocr", ocr.NATIVE_AOCR, True),
)

TRANSCRIPTION_BINDINGS: Final = (
    ("transcription", transcription.NATIVE_TRANSCRIPTION),
    ("atranscription", transcription.NATIVE_ATRANSCRIPTION),
)

ROUTE_BINDINGS: Final = tuple((attribute, binding) for attribute, binding, _ in EXECUTION_BINDINGS) + (
    TRANSCRIPTION_BINDINGS
)


@pytest.mark.parametrize(
    ("attribute", "route_binding"), ROUTE_BINDINGS, ids=[attribute for attribute, _ in ROUTE_BINDINGS]
)
def test_route_bindings_reject_non_callable_native_attributes(
    monkeypatch: pytest.MonkeyPatch, attribute: str, route_binding: bindings.NativeBinding[object]
) -> None:
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: SimpleNamespace(**{attribute: "not callable"}))
    route_binding.reset()
    assert route_binding.load() is None
    route_binding.reset()


@pytest.mark.parametrize(
    ("attribute", "route_binding"), TRANSCRIPTION_BINDINGS, ids=[attribute for attribute, _ in TRANSCRIPTION_BINDINGS]
)
def test_transcription_bindings_expose_the_native_callable_itself(
    monkeypatch: pytest.MonkeyPatch, attribute: str, route_binding: bindings.NativeBinding[object]
) -> None:
    def native_route() -> None:
        pass

    monkeypatch.setattr(bindings, "get_native_bridge", lambda: SimpleNamespace(**{attribute: native_route}))
    route_binding.reset()
    assert route_binding.load() is native_route
    route_binding.reset()


@dataclass(frozen=True, slots=True)
class _EchoingExecution:
    """Completes on its first resume with the value it was resumed with."""

    def send(self, value: object) -> Step:
        return Complete(("completed with", value))

    def throw(self, error: BaseException) -> Step:
        raise error

    def close(self) -> None:
        pass


@pytest.mark.parametrize(
    ("attribute", "route_binding", "asynchronous"),
    EXECUTION_BINDINGS,
    ids=[attribute for attribute, _, _ in EXECUTION_BINDINGS],
)
def test_execution_bindings_start_the_native_execution_and_return_its_result(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    route_binding: bindings.NativeBinding[object],
    asynchronous: bool,
) -> None:
    call: Final = NativeCall(args=(), kwargs={}, base={})
    received: list[NativeCall] = []

    def native_route(incoming: NativeCall) -> _EchoingExecution:
        received.append(incoming)
        return _EchoingExecution()

    monkeypatch.setattr(bindings, "get_native_bridge", lambda: SimpleNamespace(**{attribute: native_route}))
    route_binding.reset()
    bound: Final = route_binding.load()
    assert callable(bound)

    outcome: Final = bound(call)
    result: Final = (
        asyncio.run(cast("Coroutine[None, None, object]", outcome))  # cast-ok: async bindings return a coroutine
        if asynchronous
        else outcome
    )
    route_binding.reset()

    assert received == [call]
    assert result == ("completed with", None)
