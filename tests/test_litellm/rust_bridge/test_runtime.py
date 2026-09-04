from __future__ import annotations

from types import SimpleNamespace

import pytest

from litellm.exceptions import APIError
from litellm.rust_bridge import bindings, runtime


class RustBridgeDeclined(Exception):
    pass


class RustUpstreamError(Exception):
    pass


@pytest.fixture(autouse=True)
def native_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    native = SimpleNamespace(
        RustBridgeDeclined=RustBridgeDeclined,
        RustUpstreamError=RustUpstreamError,
    )
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)


def context() -> runtime.BridgeErrorContext:
    return runtime.BridgeErrorContext(provider="anthropic", model="model")


def enabled(*, request_override: bool | None = None) -> bool:
    return request_override is not False


def test_disabled_route_does_not_load_or_call_rust() -> None:
    bridge = runtime.RustBridge[object](
        route="messages",
        load=lambda: pytest.fail("disabled route must not load Rust"),
        enabled=enabled,
    )

    value = bridge.invoke(
        call=lambda _binding: pytest.fail("disabled route must not call Rust"),
        fallback=lambda: "python",
        adapt=str,
        context=context(),
        request_override=False,
    )

    assert value == "python"


def test_unavailable_route_hands_off_to_python() -> None:
    bridge = runtime.RustBridge[object](route="messages", load=lambda: None, enabled=enabled)

    value = bridge.invoke(
        call=lambda _binding: pytest.fail("unavailable route must not call Rust"),
        fallback=lambda: "python",
        adapt=str,
        context=context(),
    )

    assert value == "python"


def test_decline_hands_off_to_python() -> None:
    calls: list[str] = []

    def decline(_binding: object) -> object:
        calls.append("rust")
        raise RustBridgeDeclined("unsupported")

    bridge = runtime.RustBridge(route="messages", load=object, enabled=enabled)
    value = bridge.invoke(
        call=decline,
        fallback=lambda: calls.append("python") or "fallback",
        adapt=str,
        context=context(),
    )

    assert value == "fallback"
    assert calls == ["rust", "python"]


def test_upstream_failure_never_hands_off() -> None:
    def fail(_binding: object) -> object:
        raise RustUpstreamError(429, "rate limited")

    bridge = runtime.RustBridge(route="messages", load=object, enabled=enabled)
    with pytest.raises(APIError, match="rate limited") as caught:
        bridge.invoke(
            call=fail,
            fallback=lambda: pytest.fail("fallback must not run"),
            adapt=str,
            context=context(),
        )

    assert caught.value.status_code == 429


def test_unknown_failure_never_hands_off() -> None:
    bridge = runtime.RustBridge(route="messages", load=object, enabled=enabled)

    with pytest.raises(RuntimeError, match="unknown"):
        bridge.invoke(
            call=lambda _binding: (_ for _ in ()).throw(RuntimeError("unknown")),
            fallback=lambda: pytest.fail("fallback must not run"),
            adapt=str,
            context=context(),
        )


@pytest.mark.asyncio
async def test_async_route_handles_native_success() -> None:
    async def native(_binding: object) -> int:
        return 3

    async def fallback() -> str:
        pytest.fail("fallback must not run")

    bridge = runtime.RustBridge(route="messages", load=object, enabled=enabled)
    value = await bridge.ainvoke(
        call=native,
        fallback=fallback,
        adapt=str,
        context=context(),
    )

    assert value == "3"


def test_required_route_rejects_unavailable_bridge() -> None:
    bridge = runtime.RustBridge[object](route="messages", load=lambda: None, enabled=enabled)

    with pytest.raises(RuntimeError, match="unavailable"):
        bridge.require(
            call=lambda _binding: pytest.fail("unavailable route must not call Rust"),
            adapt=str,
            context=context(),
        )


def test_native_endpoint_resolves_overrides_and_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    def native_sync() -> str:
        return "native"

    async def native_async() -> str:
        return "native async"

    native = SimpleNamespace(
        sync_route=native_sync,
        async_route=native_async,
        RustBridgeDeclined=RustBridgeDeclined,
        RustUpstreamError=RustUpstreamError,
    )
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    endpoint: runtime.RustEndpoint[object, object] = runtime.RustEndpoint.native(
        route="test",
        sync="sync_route",
        asynchronous="async_route",
        enabled=enabled,
    )

    assert endpoint.sync.load() is native_sync
    assert endpoint.asynchronous.load() is native_async

    replacement = object()
    endpoint.override(sync=replacement, asynchronous=None)
    assert endpoint.sync.load() is replacement
    assert endpoint.asynchronous.load() is None

    endpoint.reset()
    assert endpoint.sync.load() is native_sync
    assert endpoint.asynchronous.load() is native_async
