from types import SimpleNamespace
from typing import Final

import pytest

from litellm.rust_bridge import bindings
from litellm.rust_bridge.chat_completions import entrypoints as chat_completions
from litellm.rust_bridge.messages import entrypoints as messages
from litellm.rust_bridge.ocr import entrypoints as ocr
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


ROUTE_BINDINGS: Final = (
    ("completion", chat_completions.NATIVE_COMPLETION),
    ("acompletion", chat_completions.NATIVE_ACOMPLETION),
    ("messages", messages.NATIVE_MESSAGES),
    ("amessages", messages.NATIVE_AMESSAGES),
    ("responses", responses.NATIVE_RESPONSES),
    ("aresponses", responses.NATIVE_ARESPONSES),
    ("ocr", ocr.NATIVE_OCR),
    ("aocr", ocr.NATIVE_AOCR),
    ("transcription", transcription.NATIVE_TRANSCRIPTION),
    ("atranscription", transcription.NATIVE_ATRANSCRIPTION),
)


@pytest.mark.parametrize(
    ("attribute", "route_binding"), ROUTE_BINDINGS, ids=[attribute for attribute, _ in ROUTE_BINDINGS]
)
def test_route_bindings_only_accept_callable_native_attributes(
    monkeypatch: pytest.MonkeyPatch, attribute: str, route_binding: bindings.NativeBinding[object]
) -> None:
    def native_route() -> None:
        pass

    monkeypatch.setattr(bindings, "get_native_bridge", lambda: SimpleNamespace(**{attribute: "not callable"}))
    route_binding.reset()
    assert route_binding.load() is None

    monkeypatch.setattr(bindings, "get_native_bridge", lambda: SimpleNamespace(**{attribute: native_route}))
    route_binding.reset()
    assert route_binding.load() is native_route
    route_binding.reset()
