from types import SimpleNamespace
from typing import Final

import pytest

from litellm.rust_bridge import _native, bindings
from litellm.rust_bridge.chat_completions import entrypoints as chat_completions
from litellm.rust_bridge.embeddings import entrypoints as embeddings
from litellm.rust_bridge.messages import entrypoints as messages
from litellm.rust_bridge.ocr import entrypoints as ocr
from litellm.rust_bridge.responses import entrypoints as responses
from litellm.rust_bridge.transcription import native as transcription


def test_binding_distinguishes_disable_from_reset() -> None:
    native = SimpleNamespace(route=lambda: "native")
    binding: bindings.NativeBinding[object] = bindings.NativeBinding(
        "route", validate=lambda value: value, source=native
    )

    assert binding.load() is native.route

    binding.override(None)
    assert binding.load() is None

    replacement = object()
    binding.override(replacement)
    assert binding.load() is replacement

    binding.reset()
    assert binding.load() is native.route


@pytest.mark.parametrize(("value", "expected"), ((3, 3), ("invalid", None), (None, None)))
def test_binding_validates_native_attribute(value: object, expected: int | None) -> None:
    native: Final = SimpleNamespace(route=value)
    binding: Final = bindings.NativeBinding(
        "route", validate=lambda item: item if isinstance(item, int) else None, source=native
    )

    assert binding.load() == expected


ROUTE_BINDINGS: Final = (
    ("completion", chat_completions.NATIVE_COMPLETION),
    ("acompletion", chat_completions.NATIVE_ACOMPLETION),
    ("embedding", embeddings.NATIVE_EMBEDDING),
    ("aembedding", embeddings.NATIVE_AEMBEDDING),
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
def test_route_bindings_resolve_the_packaged_native_route(
    attribute: str, route_binding: bindings.NativeBinding[object]
) -> None:
    route_binding.reset()
    assert route_binding.load() is getattr(_native, attribute)
