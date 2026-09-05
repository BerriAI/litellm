"""Tests for the optional Rust-backed Anthropic Messages path."""

import importlib

import httpx
import pytest
from pydantic import ValidationError

import litellm
from litellm.rust_bridge import configuration
from litellm.rust_bridge.request import NativeMessagesRequest, NativeRequestContext

rust_messages = importlib.import_module("litellm.rust_bridge.messages")
rust_bridge_loader = importlib.import_module("litellm.rust_bridge.loader")

FAKE_MESSAGES_RESPONSE: dict[str, object] = {
    "id": "msg_123",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5-20250929",
    "content": [{"type": "text", "text": "hello world"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 3},
}

REQUEST_BODY: dict[str, object] = {
    "model": "claude-sonnet-4-5",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "hi"}],
}


class RecordingMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        request: NativeMessagesRequest,
        *,
        options: object,
        context: NativeRequestContext,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "model": request.model,
                "body": request.body,
                "api_key": options.api_key,
                "api_base": options.api_base,
                "custom_llm_provider": options.custom_llm_provider,
                "extra_headers": options.extra_headers,
                "timeout_seconds": options.timeout_seconds,
            }
        )
        return dict(FAKE_MESSAGES_RESPONSE)


class RecordingAsyncMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def __call__(
        self,
        request: NativeMessagesRequest,
        *,
        options: object,
        context: NativeRequestContext,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "model": request.model,
                "body": request.body,
                "api_key": options.api_key,
                "api_base": options.api_base,
                "custom_llm_provider": options.custom_llm_provider,
                "extra_headers": options.extra_headers,
                "timeout_seconds": options.timeout_seconds,
            }
        )
        return dict(FAKE_MESSAGES_RESPONSE)


class ExplodingAsyncMessages:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(
        self, request: NativeMessagesRequest, *, options: object, context: NativeRequestContext
    ) -> dict[str, object]:
        self.calls += 1
        raise AssertionError("bridge must not be called")


class RaisingAsyncMessages:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(
        self, request: NativeMessagesRequest, *, options: object, context: NativeRequestContext
    ) -> dict[str, object]:
        self.calls += 1
        raise RuntimeError("upstream request failed with status 400: bad request")


@pytest.fixture(autouse=True)
def _reset_rust_flag():
    rust_messages.set_rust_messages(messages=None, amessages=None, decline=None)
    configuration.reset_rust_configuration()
    rust_bridge_loader._cached_bridge = rust_bridge_loader._BRIDGE_SENTINEL
    rust_messages.set_rust_messages(
        decline=lambda model, custom_llm_provider, **features: (
            "unsupported feature"
            if any(features.get(key) for key in ("stream", "has_agentic_hook", "has_custom_client"))
            or features.get("request_format") == "native"
            else None
        )
    )
    yield
    rust_messages.set_rust_messages(messages=None, amessages=None, decline=None)
    configuration.reset_rust_configuration()
    rust_bridge_loader._cached_bridge = rust_bridge_loader._BRIDGE_SENTINEL


def test_load_rust_messages_returns_injected_impl():
    bridge = RecordingMessages()
    litellm.rust(True)
    rust_messages.set_rust_messages(messages=bridge)
    assert rust_messages.load_rust_messages() is bridge


def test_bare_rust_still_toggles_ocr():
    from litellm.rust_bridge.ocr import rust_ocr_enabled

    litellm.rust(True)
    assert rust_ocr_enabled() is True

    litellm.rust(False)
    assert rust_ocr_enabled() is False


def test_load_rust_amessages_returns_injected_impl():
    bridge = RecordingAsyncMessages()
    litellm.rust(True)
    rust_messages.set_rust_messages(amessages=bridge)
    assert rust_messages.load_rust_amessages() is bridge


def test_messages_wrapper_returns_none_when_bridge_absent(monkeypatch):
    monkeypatch.setattr(
        importlib.import_module("litellm.rust_bridge.bindings"),
        "get_native_bridge",
        lambda: None,
    )
    litellm.rust(True)
    assert rust_messages.load_rust_messages() is None
    result = rust_messages.messages(
        model="claude",
        body=REQUEST_BODY,
        api_key="k",
        api_base="b",
        custom_llm_provider="azure_ai",
        extra_headers={},
        timeout=30.0,
    )
    assert result is None


def test_messages_wrapper_forwards_args_and_converts_timeout():
    bridge = RecordingMessages()
    litellm.rust(True)
    rust_messages.set_rust_messages(messages=bridge)

    response = rust_messages.messages(
        model="claude-sonnet-4-5",
        body=REQUEST_BODY,
        api_key="sk-azure",
        api_base="https://resource.services.ai.azure.com/anthropic",
        custom_llm_provider="azure_ai",
        extra_headers={"anthropic-beta": "token-efficient-tools-2025-02-19"},
        timeout=httpx.Timeout(600.0, read=42.0),
    )

    assert response == FAKE_MESSAGES_RESPONSE
    assert bridge.calls[0] == {
        "model": "claude-sonnet-4-5",
        "body": REQUEST_BODY,
        "api_key": "sk-azure",
        "api_base": "https://resource.services.ai.azure.com/anthropic",
        "custom_llm_provider": "azure_ai",
        "extra_headers": {"anthropic-beta": "token-efficient-tools-2025-02-19"},
        "timeout_seconds": 42.0,
    }


@pytest.mark.asyncio
async def test_amessages_wrapper_forwards_args():
    bridge = RecordingAsyncMessages()
    litellm.rust(True)
    rust_messages.set_rust_messages(amessages=bridge)

    response = await rust_messages.amessages(
        model="claude-sonnet-4-5",
        body=REQUEST_BODY,
        api_key="sk-azure",
        api_base="https://resource.services.ai.azure.com/anthropic",
        custom_llm_provider="azure_ai",
        extra_headers=None,
        timeout=12.5,
    )

    assert response == FAKE_MESSAGES_RESPONSE
    assert bridge.calls[0]["model"] == "claude-sonnet-4-5"
    assert bridge.calls[0]["timeout_seconds"] == 12.5


class PythonMessages:
    def __init__(self) -> None:
        self.calls = 0

    def anthropic_messages_handler(self, **kwargs: object) -> dict[str, object]:
        self.calls += 1
        return {**FAKE_MESSAGES_RESPONSE, "id": "python"}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("provider", ["anthropic", "azure_ai", "openai"])
@pytest.mark.asyncio
async def test_public_messages_routes_provider_acceptance(monkeypatch, asynchronous, provider):
    bridge = RecordingMessages()
    async_bridge = RecordingAsyncMessages()
    litellm.rust(True)
    rust_messages.set_rust_messages(messages=bridge, amessages=async_bridge)
    if asynchronous:
        response = await litellm.anthropic.messages.acreate(
            model=f"{provider}/test-model",
            max_tokens=64,
            messages=[{"role": "user", "content": "hi"}],
            api_key="key",
            api_base="https://example.test",
        )
    else:
        response = litellm.anthropic.messages.create(
            model=f"{provider}/test-model",
            max_tokens=64,
            messages=[{"role": "user", "content": "hi"}],
            api_key="key",
            api_base="https://example.test",
        )
    assert response["id"] == "msg_123"
    assert response["_hidden_params"]["additional_headers"]["x-litellm-rust"] == "true"
    calls = async_bridge.calls if asynchronous else bridge.calls
    assert len(calls) == 1
    assert calls[0]["custom_llm_provider"] == provider
    assert calls[0]["body"]["max_tokens"] == 64


@pytest.mark.parametrize("condition", ["disabled", "declined", "missing_binding", "missing_preflight", "stream"])
def test_public_messages_fallback_once(monkeypatch, condition):
    module = importlib.import_module("litellm.llms.anthropic.experimental_pass_through.messages.handler")
    python = PythonMessages()
    monkeypatch.setattr(module, "base_llm_http_handler", python)
    bridge = RecordingMessages()
    litellm.rust(condition != "disabled")
    rust_messages.set_rust_messages(messages=bridge)
    if condition == "declined":
        rust_messages.set_rust_messages(decline=lambda model, custom_llm_provider, **features: "unsupported provider")
    elif condition == "missing_binding":
        rust_messages._MESSAGES.sync.override(None)
    elif condition == "missing_preflight":
        rust_messages._PREFLIGHT.override(None)
    litellm.anthropic.messages.create(
        model="anthropic/test-model",
        max_tokens=64,
        messages=[{"role": "user", "content": "hi"}],
        api_key="key",
        stream=condition == "stream",
    )
    assert python.calls == 1
    assert bridge.calls == []


@pytest.mark.parametrize("response", [{}, {"content": "invalid"}])
def test_public_messages_invalid_response_does_not_fallback(monkeypatch, response):
    module = importlib.import_module("litellm.llms.anthropic.experimental_pass_through.messages.handler")
    python = PythonMessages()
    monkeypatch.setattr(module, "base_llm_http_handler", python)
    litellm.rust(True)
    rust_messages.set_rust_messages(messages=lambda request, *, context, callback_adapter=None: response)
    with pytest.raises(ValidationError):
        litellm.anthropic.messages.create(
            model="anthropic/test-model",
            max_tokens=64,
            messages=[{"role": "user", "content": "hi"}],
            api_key="key",
        )
    assert python.calls == 0


def test_public_messages_preserves_headers_and_optional_parameters(monkeypatch):
    handler = importlib.import_module("litellm.llms.anthropic.experimental_pass_through.messages.handler")
    monkeypatch.setattr(handler, "is_reasoning_auto_summary_enabled", lambda: True)
    configuration.rust(True)
    native = RecordingMessages()
    rust_messages.set_rust_messages(messages=native)
    response = litellm.anthropic.messages.create(
        model="anthropic/claude-sonnet-4-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=2048,
        thinking={"type": "enabled", "budget_tokens": 1024},
        temperature=0.8,
        additional_drop_params=["temperature"],
        api_key="key",
        headers={"x-source": "forwarded"},
        extra_headers={"x-source": "extra"},
        provider_specific_header={
            "custom_llm_provider": "anthropic",
            "extra_headers": {"x-source": "scoped"},
        },
    )
    assert response["id"] == "msg_123"
    assert native.calls[0]["extra_headers"]["x-source"] == "scoped"
    assert native.calls[0]["body"]["thinking"]["display"] == "summarized"
    assert "temperature" not in native.calls[0]["body"]
