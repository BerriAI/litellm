from __future__ import annotations

import builtins
from collections.abc import Generator
from types import ModuleType
from typing import Final

import pytest

from litellm.rust_bridge import loader


@pytest.fixture(autouse=True)
def reset_loader_cache() -> Generator[None]:
    loader.reset_native_bridge_cache()
    yield
    loader.reset_native_bridge_cache()


def test_absent_extension_is_cached_until_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__
    attempts = 0

    def unavailable_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        nonlocal attempts
        if name == "litellm.rust_bridge" and "_native" in fromlist:
            attempts += 1
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", unavailable_import)

    assert loader.get_native_bridge() is None
    assert loader.get_native_bridge() is None
    assert attempts == 1
    loader.reset_native_bridge_cache()
    assert loader.get_native_bridge() is None
    assert attempts == 2


@pytest.mark.parametrize(
    ("native", "expected"),
    (
        pytest.param(None, False, id="unavailable"),
        pytest.param(ModuleType("litellm.rust_bridge._native"), True, id="available"),
    ),
)
def test_native_bridge_available(
    monkeypatch: pytest.MonkeyPatch,
    native: ModuleType | None,
    expected: bool,
) -> None:
    monkeypatch.setattr(loader, "get_native_bridge", lambda: native)

    assert loader.native_bridge_available() is expected


@pytest.mark.parametrize(
    ("ready_endpoints", "expected"),
    (
        pytest.param(None, False, id="missing-registry"),
        pytest.param({"messages"}, False, id="mutable-registry"),
        pytest.param(frozenset(), False, id="unregistered"),
        pytest.param(frozenset({"messages"}), True, id="registered"),
    ),
)
def test_native_route_requires_explicit_readiness_registry(
    monkeypatch: pytest.MonkeyPatch,
    ready_endpoints: object,
    expected: bool,
) -> None:
    native: Final = ModuleType("litellm.rust_bridge._native")
    if ready_endpoints is not None:
        native.ready_endpoints = ready_endpoints
    monkeypatch.setattr(loader, "get_native_bridge", lambda: native)

    assert loader.native_route_ready("messages") is expected
