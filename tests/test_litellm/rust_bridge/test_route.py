from __future__ import annotations

from types import ModuleType
from typing import Final

import pytest

from litellm.rust_bridge import configuration
from litellm.rust_bridge.bindings import BINDING_UNSET
from litellm.rust_bridge.configuration import RouteName
from litellm.rust_bridge.route import NativeRoute


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _unexpected_load() -> ModuleType:
    raise AssertionError("disabled route loaded the extension")


def test_disabled_route_does_not_load_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    configuration.reset_rust_configuration()
    route: Final = NativeRoute(RouteName.MESSAGES)
    binding: Final = route.bind("messages", validate=_string, module_loader=_unexpected_load)
    assert route.select(binding) is None


def test_binding_discovery_validation_and_override() -> None:
    module: Final = ModuleType("fake_native")
    setattr(module, "messages", "native")
    route: Final = NativeRoute(RouteName.MESSAGES)
    binding: Final = route.bind("messages", validate=_string, module_loader=lambda: module)
    assert binding.load() == "native"
    binding.configure("override")
    binding.configure(BINDING_UNSET)
    assert binding.load() == "override"
    binding.override(None)
    assert binding.load() is None
    binding.configure(None)
    assert binding.load() == "native"
    setattr(module, "messages", 42)
    assert binding.load() is None


def test_missing_native_is_unavailable() -> None:
    route: Final = NativeRoute(RouteName.OCR)
    binding: Final = route.bind("ocr", validate=_string, module_loader=lambda: None)
    assert binding.load() is None


def test_explicit_enablement_loads_optional_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "1")
    configuration.reset_rust_configuration()
    module: Final = ModuleType("fake_native")
    setattr(module, "messages", "native")
    route: Final = NativeRoute(RouteName.MESSAGES)
    binding: Final = route.bind("messages", validate=_string, module_loader=lambda: module)
    assert route.select(binding) == "native"
