from __future__ import annotations

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


@pytest.mark.parametrize(
    ("ready_endpoints", "expected"),
    (
        pytest.param(None, False, id="missing-registry"),
        pytest.param({"messages"}, False, id="mutable-registry"),
        pytest.param(frozenset(), False, id="unregistered"),
        pytest.param({"messages": frozenset()}, True, id="registered"),
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


def test_native_route_requires_declared_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    native: Final = ModuleType("litellm.rust_bridge._native")
    native.ready_endpoints = {"messages": frozenset({"callbacks"})}
    monkeypatch.setattr(loader, "get_native_bridge", lambda: native)

    assert loader.native_route_ready("messages", frozenset({"callbacks"}))
    assert not loader.native_route_ready("messages", frozenset({"streaming_callbacks"}))
