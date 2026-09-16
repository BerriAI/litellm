"""Tests for the Rust chat completions bridge.

The native callables are dependency-injected through
``set_rust_chat_completions`` rather than patched, so these run without the
compiled extension present.
"""

from __future__ import annotations

import pytest

from litellm.rust_bridge import configuration
from litellm.rust_bridge import chat_completions as bridge
from litellm.types.utils import ModelResponse

RUST_RESPONSE = {
    "created": 1_700_000_000,
    "model": "claude-sonnet-4-5-20260101",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello from rust"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 11,
        "completion_tokens": 4,
        "total_tokens": 15,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "cache_creation_tokens": 0,
            "text_tokens": 11,
        },
    },
}

MESSAGES = [{"role": "user", "content": "hi"}]


class _FakeDeclined(Exception):
    """Stands in for the native `RustBridgeDeclined`."""


class _FakeUpstream(Exception):
    """Stands in for the native `RustUpstreamError`; args are (status, message)."""


class _FakeNative:
    RustBridgeDeclined = _FakeDeclined
    RustUpstreamError = _FakeUpstream


def _fake_native_bridge(monkeypatch):
    """Expose the bridge's exception classes without the compiled extension."""
    monkeypatch.setattr(bridge, "get_native_bridge", lambda: _FakeNative())


def _hide_native_bridge(monkeypatch):
    """Simulate a wheel built without the compiled extension.

    There is no injection seam for "the .so is absent", so the loader itself is
    replaced; every other case here uses `set_rust_chat_completions`.
    """
    monkeypatch.setattr(bridge, "get_native_bridge", lambda: None)


@pytest.fixture(autouse=True)
def reset_bridge(monkeypatch):
    """Every test starts with no injected callables, and leaves none behind."""
    bridge.set_rust_chat_completions(chat_completions=None, achat_completions=None, decline=None)
    configuration.reset_rust_configuration()
    monkeypatch.setenv("LITELLM_RUST", "1")
    yield
    bridge.set_rust_chat_completions(chat_completions=None, achat_completions=None, decline=None)
    configuration.reset_rust_configuration()


class _RecordingDecline:
    """A stand-in for the native gate that records what it was asked."""

    def __init__(self, reason: str | None = None):
        self.reason = reason
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.reason


class _RecordingCall:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result if result is not None else dict(RUST_RESPONSE)
        self.error = error
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class _RecordingAsyncCall(_RecordingCall):
    async def __call__(self, **kwargs):
        return _RecordingCall.__call__(self, **kwargs)


def _accepts(**overrides) -> bool:
    kwargs = {
        "model": "claude-sonnet-4-5",
        "messages": MESSAGES,
        "optional_params": {"max_tokens": 16},
        "custom_llm_provider": "anthropic",
        "litellm_params": {},
        "stream": None,
    }
    kwargs.update(overrides)
    return bridge.rust_chat_completions_accepts(**kwargs)


class TestGate:
    @pytest.mark.parametrize("custom_llm_provider", ("anthropic", "bedrock", "openai", None))
    def test_the_python_only_route_never_consults_the_core(self, custom_llm_provider):
        gate = _RecordingDecline()
        bridge.set_rust_chat_completions(decline=gate)
        configuration.rust(True)

        assert _accepts(custom_llm_provider=custom_llm_provider) is False
        assert _accepts(custom_llm_provider=custom_llm_provider, stream=True) is False
        assert gate.calls == []


def _call_kwargs(model_response: ModelResponse) -> dict:
    return {
        "model": "claude-sonnet-4-5",
        "messages": MESSAGES,
        "optional_params": {"max_tokens": 16},
        "model_response": model_response,
        "api_key": "sk-test",
        "api_base": None,
        "custom_llm_provider": "anthropic",
        "extra_headers": {},
        "timeout": 30.0,
        "on_response": lambda _rust_response: None,
    }


class TestSyncCall:
    def test_builds_a_model_response_and_stamps_the_rust_header(self):
        native = _RecordingCall()
        bridge.set_rust_chat_completions(chat_completions=native)
        model_response = ModelResponse()
        original_id = model_response.id

        result = bridge.chat_completions(**_call_kwargs(model_response))

        assert result is not None
        assert result.choices[0].message.content == "hello from rust"
        assert result.choices[0].finish_reason == "stop"
        assert result.model == "claude-sonnet-4-5-20260101"
        assert result.usage.prompt_tokens == 11
        assert result.usage.completion_tokens == 4
        assert result.usage.total_tokens == 15
        assert result._hidden_params["additional_headers"] == {"x-litellm-rust": "true"}
        assert result.id == original_id, "the rust path must keep the chatcmpl id litellm already minted"

    def test_passes_the_timeout_through_as_seconds(self):
        native = _RecordingCall()
        bridge.set_rust_chat_completions(chat_completions=native)
        bridge.chat_completions(**_call_kwargs(ModelResponse()))
        assert native.calls[0]["timeout_seconds"] == 30.0

    def test_falls_back_when_the_bridge_is_unavailable(self, monkeypatch):
        _hide_native_bridge(monkeypatch)
        assert bridge.chat_completions(**_call_kwargs(ModelResponse())) is None

    def test_falls_back_when_the_core_declines_before_calling_the_provider(self, monkeypatch):
        _fake_native_bridge(monkeypatch)
        bridge.set_rust_chat_completions(chat_completions=_RecordingCall(error=_FakeDeclined("streaming")))
        assert bridge.chat_completions(**_call_kwargs(ModelResponse())) is None


class TestAsyncCall:
    @pytest.mark.asyncio
    async def test_builds_a_model_response(self):
        bridge.set_rust_chat_completions(achat_completions=_RecordingAsyncCall())
        result = await bridge.achat_completions(**_call_kwargs(ModelResponse()))
        assert result is not None
        assert result.choices[0].message.content == "hello from rust"
        assert result._hidden_params["additional_headers"] == {"x-litellm-rust": "true"}

    @pytest.mark.asyncio
    async def test_falls_back_when_the_bridge_is_unavailable(self, monkeypatch):
        _hide_native_bridge(monkeypatch)
        assert await bridge.achat_completions(**_call_kwargs(ModelResponse())) is None

    @pytest.mark.asyncio
    async def test_falls_back_when_the_core_declines_before_calling_the_provider(self, monkeypatch):
        _fake_native_bridge(monkeypatch)
        bridge.set_rust_chat_completions(achat_completions=_RecordingAsyncCall(error=_FakeDeclined("streaming")))
        assert await bridge.achat_completions(**_call_kwargs(ModelResponse())) is None


class TestAsyncFallbackWrapper:
    @pytest.mark.asyncio
    async def test_returns_the_rust_response_without_running_the_fallback(self):
        bridge.set_rust_chat_completions(achat_completions=_RecordingAsyncCall())
        ran = []

        async def fallback():
            ran.append(True)
            return "python"

        result = await bridge.achat_completions_or_fallback(**_call_kwargs(ModelResponse()), python_fallback=fallback)
        assert result.choices[0].message.content == "hello from rust"
        assert ran == []

    @pytest.mark.asyncio
    async def test_runs_the_fallback_when_the_core_declines(self, monkeypatch):
        _fake_native_bridge(monkeypatch)
        bridge.set_rust_chat_completions(achat_completions=_RecordingAsyncCall(error=_FakeDeclined("streaming")))

        async def fallback():
            return "python"

        result = await bridge.achat_completions_or_fallback(**_call_kwargs(ModelResponse()), python_fallback=fallback)
        assert result == "python"

    @pytest.mark.asyncio
    async def test_runs_the_fallback_when_the_bridge_is_unavailable(self, monkeypatch):
        _hide_native_bridge(monkeypatch)

        async def fallback():
            return "python"

        result = await bridge.achat_completions_or_fallback(**_call_kwargs(ModelResponse()), python_fallback=fallback)
        assert result == "python"


class TestFailureClassification:
    """A failure the provider already saw must not be retried on the Python
    path: it would bill the customer for the same work twice."""

    @pytest.fixture(autouse=True)
    def _native_exceptions(self, monkeypatch):
        _fake_native_bridge(monkeypatch)

    def test_a_decline_falls_back_because_nothing_was_sent(self):
        bridge.set_rust_chat_completions(chat_completions=_RecordingCall(error=_FakeDeclined("streaming")))
        assert bridge.chat_completions(**_call_kwargs(ModelResponse())) is None

    def test_an_upstream_failure_is_surfaced_with_its_status(self):
        from litellm.exceptions import APIError

        bridge.set_rust_chat_completions(chat_completions=_RecordingCall(error=_FakeUpstream(429, "429: rate limited")))
        with pytest.raises(APIError) as raised:
            bridge.chat_completions(**_call_kwargs(ModelResponse()))
        assert raised.value.status_code == 429
        assert "rate limited" in str(raised.value)

    def test_a_transport_failure_with_no_response_surfaces_as_a_500(self):
        from litellm.exceptions import APIError

        bridge.set_rust_chat_completions(chat_completions=_RecordingCall(error=_FakeUpstream(0, "connection reset")))
        with pytest.raises(APIError) as raised:
            bridge.chat_completions(**_call_kwargs(ModelResponse()))
        assert raised.value.status_code == 500

    def test_an_unrecognized_error_is_not_swallowed(self):
        bridge.set_rust_chat_completions(chat_completions=_RecordingCall(error=RuntimeError("something else")))
        with pytest.raises(RuntimeError):
            bridge.chat_completions(**_call_kwargs(ModelResponse()))

    @pytest.mark.asyncio
    async def test_the_async_wrapper_does_not_fall_back_on_an_upstream_failure(self):
        from litellm.exceptions import APIError

        bridge.set_rust_chat_completions(achat_completions=_RecordingAsyncCall(error=_FakeUpstream(500, "500: boom")))
        ran = []

        async def fallback():
            ran.append(True)
            return "python"

        with pytest.raises(APIError):
            await bridge.achat_completions_or_fallback(**_call_kwargs(ModelResponse()), python_fallback=fallback)
        assert ran == [], "a request the provider already served must not be re-issued"

    @pytest.mark.asyncio
    async def test_the_async_wrapper_falls_back_on_a_decline(self):
        bridge.set_rust_chat_completions(
            achat_completions=_RecordingAsyncCall(error=_FakeDeclined("blank message text"))
        )

        async def fallback():
            return "python"

        result = await bridge.achat_completions_or_fallback(**_call_kwargs(ModelResponse()), python_fallback=fallback)
        assert result == "python"
