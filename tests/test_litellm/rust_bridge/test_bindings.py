from types import SimpleNamespace
from typing import Final

import pytest

from litellm.rust_bridge import bindings


@pytest.mark.parametrize(
    ("native", "expected"),
    (
        pytest.param(None, None, id="extension-unavailable"),
        pytest.param(SimpleNamespace(), None, id="attribute-missing"),
        pytest.param(SimpleNamespace(route="invalid"), None, id="attribute-invalid"),
        pytest.param(SimpleNamespace(route=3), 3, id="attribute-valid"),
    ),
)
def test_binding_loads_only_valid_native_attributes(
    monkeypatch: pytest.MonkeyPatch,
    native: object | None,
    expected: int | None,
) -> None:
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    monkeypatch.setattr(bindings, "native_route_ready", lambda _route: native is not None)
    binding: Final = bindings.NativeBinding(
        "test", "route", validate=lambda value: value if isinstance(value, int) else None
    )

    assert binding.load() == expected


@pytest.mark.parametrize(
    "value",
    (
        pytest.param(None, id="missing"),
        pytest.param(3, id="integer"),
        pytest.param("not callable", id="string"),
    ),
)
def test_callable_binding_rejects_non_callables(monkeypatch: pytest.MonkeyPatch, value: object) -> None:
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: SimpleNamespace(route=value))
    monkeypatch.setattr(bindings, "native_route_ready", lambda _route: True)
    binding: Final[bindings.NativeBinding[object]] = bindings.NativeBinding.callable("test", "route")

    assert binding.load() is None


def test_callable_binding_resolves_override_and_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    native_route: Final = lambda: "native"
    replacement: Final = lambda: "replacement"
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: SimpleNamespace(route=native_route))
    monkeypatch.setattr(bindings, "native_route_ready", lambda _route: True)
    binding: Final[bindings.NativeBinding[object]] = bindings.NativeBinding.callable("test", "route")

    assert binding.load() is native_route
    binding.override(None)
    assert binding.load() is None
    binding.override(replacement)
    assert binding.load() is replacement
    binding.reset()
    assert binding.load() is native_route


def test_unregistered_callable_cannot_activate_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    native_route: Final = lambda: "native"
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: SimpleNamespace(route=native_route))
    monkeypatch.setattr(bindings, "native_route_ready", lambda _route: False)
    binding: Final[bindings.NativeBinding[object]] = bindings.NativeBinding.callable("test", "route")

    assert binding.load() is None


class _Declined(Exception):
    pass


class _Upstream(Exception):
    pass


@pytest.mark.parametrize(
    ("native", "expected"),
    (
        pytest.param(None, None, id="extension-unavailable"),
        pytest.param(SimpleNamespace(), None, id="exceptions-missing"),
        pytest.param(
            SimpleNamespace(RustBridgeDeclined=_Declined(), RustUpstreamError=_Upstream),
            None,
            id="declined-not-type",
        ),
        pytest.param(
            SimpleNamespace(RustBridgeDeclined=_Declined, RustUpstreamError=_Upstream()),
            None,
            id="upstream-not-type",
        ),
        pytest.param(
            SimpleNamespace(RustBridgeDeclined=_Declined, RustUpstreamError=_Upstream),
            (_Declined, _Upstream),
            id="valid-exceptions",
        ),
    ),
)
def test_native_exception_types_require_both_exception_classes(
    monkeypatch: pytest.MonkeyPatch,
    native: object | None,
    expected: tuple[type[BaseException], type[BaseException]] | None,
) -> None:
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)

    assert bindings.native_exception_types() == expected
