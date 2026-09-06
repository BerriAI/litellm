"""Tests for the Rust chat completions bridge.

The native callables are dependency-injected through
``set_rust_chat_completions`` rather than patched, so these run without the
compiled extension present.
"""

from __future__ import annotations

import pytest

import litellm
from litellm.rust_bridge import bindings, configuration
from litellm.rust_bridge import chat_completions as bridge
from litellm.rust_bridge.request import (
    NativeBedrockOptions,
    NativeRequestCapabilities,
    NativeRequestContext,
    anthropic_options,
)
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
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: _FakeNative())


def _hide_native_bridge(monkeypatch):
    """Simulate a wheel built without the compiled extension.

    There is no injection seam for "the .so is absent", so the loader itself is
    replaced; every other case here uses `set_rust_chat_completions`.
    """
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: None)


@pytest.fixture(autouse=True)
def reset_bridge(monkeypatch):
    """Every test starts with no injected callables, and leaves none behind."""
    bridge.set_rust_chat_completions(chat_completions=None, achat_completions=None)
    configuration.reset_rust_configuration()
    monkeypatch.setenv("LITELLM_RUST", "1")
    yield
    bridge.set_rust_chat_completions(chat_completions=None, achat_completions=None)
    configuration.reset_rust_configuration()


class _RecordingCall:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result if result is not None else dict(RUST_RESPONSE)
        self.error = error
        self.calls: list[dict] = []

    def __call__(self, request, *, options, context, callback_adapter=None):
        self.calls.append({"request": request, "options": options, "context": context})
        if self.error is not None:
            raise self.error
        return self.result


class _RecordingAsyncCall(_RecordingCall):
    async def __call__(self, request, *, options, context, callback_adapter=None):
        return _RecordingCall.__call__(
            self,
            request,
            options=options,
            context=context,
            callback_adapter=callback_adapter,
        )


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
        assert native.calls[0]["options"].timeout_seconds == 30.0

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
        from litellm.exceptions import RateLimitError

        bridge.set_rust_chat_completions(chat_completions=_RecordingCall(error=_FakeUpstream(429, "429: rate limited")))
        with pytest.raises(RateLimitError) as raised:
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
        from litellm.exceptions import InternalServerError

        bridge.set_rust_chat_completions(achat_completions=_RecordingAsyncCall(error=_FakeUpstream(500, "500: boom")))
        ran = []

        async def fallback():
            ran.append(True)
            return "python"

        with pytest.raises(InternalServerError):
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


@pytest.mark.asyncio
async def test_missing_native_exception_types_does_not_authorize_python_fallback(monkeypatch):
    _hide_native_bridge(monkeypatch)
    bridge.set_rust_chat_completions(
        chat_completions=_RecordingCall(error=RuntimeError("connection failed")),
        achat_completions=_RecordingAsyncCall(error=RuntimeError("connection failed")),
    )

    with pytest.raises(RuntimeError, match="connection failed"):
        bridge.chat_completions(**_call_kwargs(ModelResponse()))

    async def fallback():
        pytest.fail("unknown failure must not retry through Python")

    with pytest.raises(RuntimeError, match="connection failed"):
        await bridge.achat_completions_or_fallback(**_call_kwargs(ModelResponse()), python_fallback=fallback)


def test_provider_credentials_are_separate_from_chat_body_params():
    native = _RecordingCall()
    bridge.set_rust_chat_completions(chat_completions=native)
    configuration.rust(True)
    kwargs = _call_kwargs(ModelResponse())
    kwargs["optional_params"] = {
        "max_tokens": 32,
    }
    kwargs["bedrock"] = NativeBedrockOptions(
        aws_access_key_id="test-access-key",
        aws_secret_access_key="test-secret-key",
    )
    bridge.chat_completions(**kwargs)
    request = native.calls[0]["request"]
    options = native.calls[0]["options"]
    assert request.optional_params == {"max_tokens": 32}
    assert options.bedrock.aws_access_key_id == "test-access-key"
    assert options.bedrock.aws_secret_access_key == "test-secret-key"


def test_provider_payload_extensions_cross_the_boundary_without_partitioning():
    native = _RecordingCall()
    bridge.set_rust_chat_completions(chat_completions=native)
    configuration.rust(True)
    extensions = {
        "vendor_object": {"nested": None},
        "vendor_array": [1, "two", False],
        "vendor_scalar": 0.25,
        "extra_body": {"temperature": 0.2, "config": {"replacement": True}},
    }

    kwargs = _call_kwargs(ModelResponse())
    kwargs["optional_params"] = extensions
    bridge.chat_completions(**kwargs)

    assert native.calls[0]["request"].optional_params == extensions


def test_typed_capability_and_provider_metadata_facts_are_isolated():
    context = NativeRequestContext(
        capabilities=NativeRequestCapabilities(
            stream=True,
            has_agentic_hook=True,
            has_custom_client=True,
            request_format="native",
        )
    )
    anthropic = anthropic_options({"metadata": {"user_id": "user-123", "ignored": object()}})

    assert context.capabilities.request_format == "native"
    assert context.capabilities.has_agentic_hook is True
    assert anthropic.user_id == "user-123"
    assert anthropic.has_user_id is True
    assert anthropic_options({"metadata": {"user_id": object()}}).has_user_id is True
    assert anthropic_options({"metadata": {"user_id": None}}).has_user_id is False



@pytest.mark.parametrize("provider", ["anthropic", "bedrock", "openai"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_public_completion_discovers_any_provider(provider, asynchronous):
    native = _RecordingCall()
    anative = _RecordingAsyncCall()
    bridge.set_rust_chat_completions(chat_completions=native, achat_completions=anative)
    kwargs = {
        "model": f"{provider}/test-model",
        "messages": MESSAGES,
        "api_key": "key",
        "max_tokens": 16,
        "num_retries": 0,
    }
    response = await litellm.acompletion(**kwargs) if asynchronous else litellm.completion(**kwargs)
    assert response.choices[0].message.content == "hello from rust"
    calls = anative.calls if asynchronous else native.calls
    assert len(calls) == 1
    assert calls[0]["options"].custom_llm_provider == provider
    assert calls[0]["request"].messages == MESSAGES
    assert len(native.calls) + len(anative.calls) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", ["decline", "unavailable", "error", "malformed", "cancelled"])
@pytest.mark.asyncio
async def test_public_completion_fallback_contract(monkeypatch, asynchronous, failure):
    import asyncio
    import importlib

    from litellm.integrations.custom_logger import CustomLogger

    class Recorder(CustomLogger):
        def __init__(self):
            self.pre = 0
            self.post = 0

        def log_pre_api_call(self, model, messages, kwargs):
            self.pre += 1

        def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
            self.post += 1

    recorder = Recorder()
    monkeypatch.setattr(litellm, "input_callback", [recorder])
    _fake_native_bridge(monkeypatch)
    python_calls = []

    def python(ctx):
        python_calls.append(ctx)

        def finish():
            ctx.logging.pre_call(input=MESSAGES, api_key="key", additional_args={})
            ctx.logging.post_call(input=MESSAGES, api_key="key", original_response="python")
            return ModelResponse(choices=[{"message": {"role": "assistant", "content": "python"}}])

        async def afinish():
            return finish()

        return afinish() if ctx.acompletion else finish()

    monkeypatch.setattr(importlib.import_module("litellm.main"), "_complete_python", python)
    native_error = {
        "decline": _FakeDeclined("request unsupported"),
        "error": RuntimeError("execution failed"),
        "cancelled": asyncio.CancelledError(),
    }.get(failure)
    native = _RecordingCall(result={} if failure == "malformed" else None, error=native_error)
    anative = _RecordingAsyncCall(result={} if failure == "malformed" else None, error=native_error)
    bridge.set_rust_chat_completions(
        chat_completions=native,
        achat_completions=anative,
    )
    if failure == "unavailable":
        bridge._CHAT.sync.override(None)
        bridge._CHAT.asynchronous.override(None)

    async def run():
        kwargs = {"model": "openai/test-model", "messages": MESSAGES, "api_key": "key", "num_retries": 0}
        return await litellm.acompletion(**kwargs) if asynchronous else litellm.completion(**kwargs)

    if failure in {"error", "malformed", "cancelled"}:
        with pytest.raises(asyncio.CancelledError if failure == "cancelled" else Exception):
            await run()
        assert python_calls == []
    else:
        result = await run()
        assert result.choices[0].message.content == "python"
        assert len(python_calls) == 1
        assert recorder.pre == 1
        assert recorder.post == 1
