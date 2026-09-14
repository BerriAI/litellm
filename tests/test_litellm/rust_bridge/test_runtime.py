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
    return runtime.BridgeErrorContext(route="messages", provider="anthropic", model="model")


def test_invoke_tags_native_decline_before_running_fallback() -> None:
    calls: list[str] = []

    def decline() -> object:
        calls.append("rust")
        raise RustBridgeDeclined("unsupported")

    value = runtime.invoke(
        native_call=decline,
        fallback=lambda: calls.append("python") or "fallback",
        adapt=str,
        mode=runtime.FallbackMode.PYTHON,
        context=context(),
    )

    assert value == "fallback"
    assert calls == ["rust", "python"]


def test_invoke_translates_upstream_without_fallback() -> None:
    def fail() -> object:
        raise RustUpstreamError(429, "rate limited")

    with pytest.raises(APIError, match="rate limited") as caught:
        runtime.invoke(
            native_call=fail,
            fallback=lambda: pytest.fail("fallback must not run"),
            adapt=str,
            mode=runtime.FallbackMode.PYTHON,
            context=context(),
        )

    assert caught.value.status_code == 429


@pytest.mark.asyncio
async def test_ainvoke_handles_native_success() -> None:
    async def native() -> int:
        return 3

    async def fallback() -> str:
        pytest.fail("fallback must not run")

    assert (
        await runtime.ainvoke(
            native_call=native,
            fallback=fallback,
            adapt=str,
            mode=runtime.FallbackMode.PYTHON,
            context=context(),
        )
        == "3"
    )


def test_required_mode_rejects_unavailable_bridge() -> None:
    with pytest.raises(RuntimeError, match="is unavailable"):
        runtime.invoke(
            native_call=None,
            fallback=lambda: pytest.fail("fallback must not run"),
            adapt=str,
            mode=runtime.FallbackMode.RUST_REQUIRED,
            context=context(),
        )
