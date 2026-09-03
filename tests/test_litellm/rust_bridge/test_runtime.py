from __future__ import annotations

from types import SimpleNamespace
from typing import Final

import openai
import pytest

from litellm.exceptions import (
    APIError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
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

    with pytest.raises(RateLimitError, match="rate limited") as caught:
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


class TestUpstreamStatusClassification:
    """The router's cooldown, retry, and fallback logic keys on litellm exception
    types, so a Rust-surfaced upstream status must raise the same exception the
    Python path would"""

    @pytest.mark.parametrize(
        ("status", "expected"),
        (
            (400, BadRequestError),
            (401, AuthenticationError),
            (403, AuthenticationError),
            (404, NotFoundError),
            (408, Timeout),
            (422, BadRequestError),
            (429, RateLimitError),
            (500, ServiceUnavailableError),
            (503, ServiceUnavailableError),
            (0, APIError),
        ),
    )
    def test_each_status_raises_its_litellm_exception(self, status: int, expected: type[Exception]) -> None:
        def fail() -> object:
            raise RustUpstreamError(status, f"{status}: upstream said no")

        with pytest.raises(expected, match="upstream said no") as caught:
            runtime.invoke(
                native_call=fail,
                fallback=lambda: pytest.fail("fallback must not run"),
                adapt=str,
                mode=runtime.FallbackMode.PYTHON,
                context=context(),
            )

        assert caught.value.__cause__ is not None
        assert isinstance(caught.value.__cause__, RustUpstreamError)

    def test_the_timeout_marker_raises_litellm_timeout(self) -> None:
        def hang() -> object:
            raise RustUpstreamError(408, "upstream request timed out: 30s elapsed")

        with pytest.raises(Timeout) as caught:
            runtime.invoke(
                native_call=hang,
                fallback=lambda: pytest.fail("fallback must not run"),
                adapt=str,
                mode=runtime.FallbackMode.PYTHON,
                context=context(),
            )

        assert caught.value.status_code == 408
        assert isinstance(caught.value.__cause__, RustUpstreamError)
        assert "timed out" in str(caught.value)

    def test_the_cause_chain_preserves_the_rust_error(self) -> None:
        cause: Final = RustUpstreamError(500, "500: boom")

        def fail() -> object:
            raise cause

        with pytest.raises(ServiceUnavailableError) as caught:
            runtime.invoke(
                native_call=fail,
                fallback=lambda: pytest.fail("fallback must not run"),
                adapt=str,
                mode=runtime.FallbackMode.PYTHON,
                context=context(),
            )

        assert caught.value.__cause__ is cause

    def test_a_rust_429_is_classified_the_way_router_cooldown_expects(self) -> None:
        from litellm.router_utils.cooldown_handlers import _is_cooldown_required

        def fail() -> object:
            raise RustUpstreamError(429, "429: slow down")

        with pytest.raises(RateLimitError) as caught:
            runtime.invoke(
                native_call=fail,
                fallback=lambda: pytest.fail("fallback must not run"),
                adapt=str,
                mode=runtime.FallbackMode.PYTHON,
                context=context(),
            )

        error: Final = caught.value
        assert isinstance(error, openai.RateLimitError), "type-based retry and fallback gates must see it"
        assert error.status_code == 429
        assert _is_cooldown_required(None, "deployment", error.status_code) is True
