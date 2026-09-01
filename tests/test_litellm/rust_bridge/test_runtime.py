from __future__ import annotations

from collections.abc import Callable

import pytest

from litellm.exceptions import APIError
from litellm.rust_bridge import runtime
from litellm.rust_bridge.runtime import (
    BridgeErrorContext,
    CoreEngine,
    FallbackMode,
    RustDeclined,
    RustHandled,
    RustUnavailable,
)


class Declined(Exception):
    pass


class Upstream(Exception):
    pass


class Native:
    RustBridgeDeclined = Declined
    RustUpstreamError = Upstream


CONTEXT = BridgeErrorContext(route="messages", provider="anthropic", model="claude")


@pytest.fixture(autouse=True)
def native_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "get_native_bridge", lambda: Native)


@pytest.mark.parametrize(
    ("native_call", "expected", "source", "fallback_calls"),
    [
        (lambda: "native", "adapted native", CoreEngine.RUST, 0),
        (None, "fallback", CoreEngine.PYTHON, 1),
        (lambda: (_ for _ in ()).throw(Declined("not sent")), "fallback", CoreEngine.PYTHON, 1),
    ],
)
def test_invoke_uses_fallback_only_when_safe(
    native_call: Callable[[], str] | None,
    expected: str,
    source: CoreEngine,
    fallback_calls: int,
) -> None:
    calls = 0

    def fallback() -> str:
        nonlocal calls
        calls += 1
        return "fallback"

    result = runtime.invoke(
        native_call=native_call,
        fallback=fallback,
        adapt=lambda value: f"adapted {value}",
        mode=FallbackMode.PYTHON,
        context=CONTEXT,
    )

    assert result.value == expected
    assert result.source is source
    assert calls == fallback_calls


def test_invoke_converts_upstream_error_without_fallback() -> None:
    fallback_calls = 0

    def fallback() -> str:
        nonlocal fallback_calls
        fallback_calls += 1
        return "fallback"

    with pytest.raises(APIError) as exc_info:
        runtime.invoke(
            native_call=lambda: (_ for _ in ()).throw(Upstream(429, "rate limited")),
            fallback=fallback,
            adapt=str,
            mode=FallbackMode.PYTHON,
            context=CONTEXT,
        )

    assert exc_info.value.status_code == 429
    assert "rate limited" in str(exc_info.value)
    assert exc_info.value.headers == {"x-litellm-core": "rust", "x-litellm-rust": "true"}
    assert fallback_calls == 0


def test_invoke_propagates_unknown_error_without_fallback() -> None:
    with pytest.raises(ValueError, match="bug"):
        runtime.invoke(
            native_call=lambda: (_ for _ in ()).throw(ValueError("bug")),
            fallback=lambda: pytest.fail("fallback must not run"),
            adapt=str,
            mode=FallbackMode.PYTHON,
            context=CONTEXT,
        )


@pytest.mark.parametrize(
    "native_call",
    [None, lambda: (_ for _ in ()).throw(Declined("unsupported"))],
)
def test_rust_required_route_never_falls_back(
    native_call: Callable[[], str] | None,
) -> None:
    with pytest.raises(RuntimeError):
        runtime.invoke(
            native_call=native_call,
            fallback=lambda: pytest.fail("fallback must not run"),
            adapt=str,
            mode=FallbackMode.RUST_REQUIRED,
            context=CONTEXT,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, Declined("not sent")])
async def test_ainvoke_success_or_safe_fallback(failure: Exception | None) -> None:
    fallback_calls = 0

    async def native_call() -> str:
        if failure is not None:
            raise failure
        return "native"

    async def fallback() -> str:
        nonlocal fallback_calls
        fallback_calls += 1
        return "fallback"

    result = await runtime.ainvoke(
        native_call=native_call,
        fallback=fallback,
        adapt=lambda value: f"adapted {value}",
        mode=FallbackMode.PYTHON,
        context=CONTEXT,
    )

    assert result.value == ("fallback" if failure else "adapted native")
    assert result.source is (CoreEngine.PYTHON if failure else CoreEngine.RUST)
    assert fallback_calls == (1 if failure else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [Upstream(503, "unavailable"), ValueError("bug")])
async def test_ainvoke_never_falls_back_after_unsafe_failure(failure: Exception) -> None:
    async def native_call() -> str:
        raise failure

    async def fallback() -> str:
        pytest.fail("fallback must not run")

    expected = APIError if isinstance(failure, Upstream) else ValueError
    with pytest.raises(expected):
        await runtime.ainvoke(
            native_call=native_call,
            fallback=fallback,
            adapt=str,
            mode=FallbackMode.PYTHON,
            context=CONTEXT,
        )


@pytest.mark.parametrize(
    ("native_call", "expected_type"),
    [
        (lambda: "native", RustHandled),
        (lambda: (_ for _ in ()).throw(Declined("unsupported")), RustDeclined),
        (None, RustUnavailable),
    ],
)
def test_attempt_classifies_bridge_control_flow(
    native_call: Callable[[], str] | None,
    expected_type: type[RustHandled[str]] | type[RustDeclined] | type[RustUnavailable],
) -> None:
    result = runtime.attempt(native_call=native_call, adapt=str, context=CONTEXT)

    assert isinstance(result, expected_type)
    if isinstance(result, RustDeclined):
        assert result.reason == "unsupported"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_type"),
    [
        (None, RustHandled),
        (Declined("unsupported"), RustDeclined),
    ],
)
async def test_aattempt_classifies_bridge_control_flow(
    failure: Exception | None,
    expected_type: type[RustHandled[str]] | type[RustDeclined],
) -> None:
    async def native_call() -> str:
        if failure is not None:
            raise failure
        return "native"

    result = await runtime.aattempt(native_call=native_call, adapt=str, context=CONTEXT)

    assert isinstance(result, expected_type)
    if isinstance(result, RustDeclined):
        assert result.reason == "unsupported"


@pytest.mark.asyncio
async def test_aattempt_classifies_unavailable_bridge() -> None:
    result = await runtime.aattempt(native_call=None, adapt=str, context=CONTEXT)

    assert isinstance(result, RustUnavailable)


def test_execution_hidden_params_overwrites_reserved_provider_headers() -> None:
    hidden_params = runtime.execution_hidden_params(
        {
            "additional_headers": {
                "X-LiteLLM-Core": "rust",
                "x-litellm-rust": "true",
                "x-provider": "kept",
            }
        },
        CoreEngine.PYTHON,
    )

    assert hidden_params == {
        "core_engine": "python",
        "additional_headers": {"x-provider": "kept", "x-litellm-core": "python"},
    }
