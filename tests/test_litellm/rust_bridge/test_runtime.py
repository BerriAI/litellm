from __future__ import annotations

from types import SimpleNamespace

import httpx
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

    assert value.value == "fallback"
    assert value.source is runtime.CoreEngine.PYTHON
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
    assert caught.value.headers == {
        "x-litellm-core": "rust",
        "x-litellm-rust": "true",
    }


@pytest.mark.asyncio
async def test_ainvoke_handles_native_success() -> None:
    async def native() -> int:
        return 3

    async def fallback() -> str:
        pytest.fail("fallback must not run")

    result = await runtime.ainvoke(
        native_call=native,
        fallback=fallback,
        adapt=str,
        mode=runtime.FallbackMode.PYTHON,
        context=context(),
    )
    assert result.value == "3"
    assert result.source is runtime.CoreEngine.RUST


def test_execution_hidden_params_overwrites_reserved_provider_headers() -> None:
    hidden_params = runtime.execution_hidden_params(
        {
            "provider": "anthropic",
            "additional_headers": {
                "request-id": "req-1",
                "X-LiteLLM-Core": "spoofed",
                "x-litellm-rust": "spoofed",
            },
        },
        runtime.CoreEngine.PYTHON,
    )

    assert hidden_params == {
        "provider": "anthropic",
        "core_engine": "python",
        "additional_headers": {
            "request-id": "req-1",
            "x-litellm-core": "python",
        },
    }


def test_required_mode_rejects_unavailable_bridge() -> None:
    with pytest.raises(RuntimeError, match="is unavailable"):
        runtime.invoke(
            native_call=None,
            fallback=lambda: pytest.fail("fallback must not run"),
            adapt=str,
            mode=runtime.FallbackMode.RUST_REQUIRED,
            context=context(),
        )


@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
@pytest.mark.asyncio
async def test_upstream_error_adapter_preserves_response_without_fallback(asynchronous: bool) -> None:
    request = httpx.Request("POST", "https://example.com/ocr")

    def provider_error(status: int, message: str) -> Exception:
        return httpx.HTTPStatusError(
            message,
            request=request,
            response=httpx.Response(status, request=request),
        )

    error_context = runtime.BridgeErrorContext(
        route="ocr", provider="mistral", model="model", upstream_error=provider_error
    )

    def fail() -> str:
        raise RustUpstreamError(429, '{"message":"rate limited"}')

    async def afail() -> str:
        return fail()

    def fallback() -> str:
        pytest.fail("provider failure must not execute Python fallback")

    async def afallback() -> str:
        return fallback()

    async def invoke() -> None:
        if asynchronous:
            await runtime.ainvoke(
                native_call=afail,
                fallback=afallback,
                adapt=runtime.identity,
                mode=runtime.FallbackMode.PYTHON,
                context=error_context,
            )
        else:
            runtime.invoke(
                native_call=fail,
                fallback=fallback,
                adapt=runtime.identity,
                mode=runtime.FallbackMode.PYTHON,
                context=error_context,
            )

    with pytest.raises(httpx.HTTPStatusError) as caught:
        await invoke()

    assert str(caught.value) == '{"message":"rate limited"}'
    assert caught.value.response.status_code == 429
    assert caught.value.request is request
    assert isinstance(caught.value.__cause__, RustUpstreamError)
    assert caught.value.headers == {"x-litellm-core": "rust", "x-litellm-rust": "true"}
