from types import SimpleNamespace
from typing import Final

import pytest

from litellm.rust_bridge import bindings


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
