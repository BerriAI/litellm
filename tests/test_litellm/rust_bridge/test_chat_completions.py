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
from litellm.rust_bridge.request import NativeChatCompletionsRequest, NativeRequestContext, NativeRequestOptions
from litellm.rust_bridge.runtime import Handled, NativeFailed, NativeSkipped
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
        self.contexts: list[NativeRequestContext] = []

    def __call__(
        self,
        request: NativeChatCompletionsRequest,
        *,
        options: NativeRequestOptions,
        context: NativeRequestContext,
    ):
        kwargs = {
            "model": request.model,
            "messages": request.messages,
            "optional_params": request.optional_params,
            "api_key": options.api_key,
            "api_base": options.api_base,
            "custom_llm_provider": options.custom_llm_provider,
            "extra_headers": options.extra_headers,
            "timeout_seconds": options.timeout_seconds,
        }
        self.calls.append(kwargs)
        self.contexts.append(context)
        if self.error is not None:
            raise self.error
        return self.result


class _RecordingAsyncCall(_RecordingCall):
    async def __call__(
        self,
        request: NativeChatCompletionsRequest,
        *,
        options: NativeRequestOptions,
        context: NativeRequestContext,
    ):
        return _RecordingCall.__call__(self, request, options=options, context=context)


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
    def test_declines_when_the_deployment_did_not_opt_in(self, monkeypatch):
        monkeypatch.delenv("LITELLM_RUST", raising=False)
        gate = _RecordingDecline()
        bridge.set_rust_chat_completions(decline=gate)
        assert _accepts(litellm_params={}) is False
        assert _accepts(litellm_params=None) is False
        assert gate.calls == [], "the gate must not be consulted before opt-in"

    def test_accepts_when_the_deployment_opted_in_and_the_core_agrees(self, monkeypatch):
        monkeypatch.setenv("LITELLM_RUST", "1")
        gate = _RecordingDecline()
        bridge.set_rust_chat_completions(decline=gate)
        assert _accepts() is True
        assert gate.calls[0]["model"] == "claude-sonnet-4-5"
        assert gate.calls[0]["custom_llm_provider"] == "anthropic"

    def test_process_enable_applies_without_request_override(self):
        bridge.set_rust_chat_completions(decline=_RecordingDecline())
        configuration.rust(True)

        assert _accepts(litellm_params={}) is True

    def test_the_env_var_opts_in_without_a_per_model_flag(self, monkeypatch):
        monkeypatch.setenv("LITELLM_RUST", "true")
        bridge.set_rust_chat_completions(decline=_RecordingDecline())
        assert _accepts(litellm_params={}) is True

    def test_declines_streaming_and_providers_off_the_path(self, monkeypatch):
        monkeypatch.setenv("LITELLM_RUST", "1")
        gate = _RecordingDecline()
        bridge.set_rust_chat_completions(decline=gate)
        assert _accepts(stream=True) is False
        assert _accepts(custom_llm_provider="openai") is False
        assert _accepts(custom_llm_provider=None) is False
        assert gate.calls == []

    def test_declines_an_anthropic_request_carrying_a_litellm_metadata_user_id(self, monkeypatch):
        """`AnthropicConfig.transform_request` copies a valid `user_id` into the Messages body.

        It does that inside the function the Rust route replaces, and the core is
        handed `optional_params` only, so accepting here would send the request
        to Anthropic with the abuse-detection attribution silently missing.
        """
        monkeypatch.setenv("LITELLM_RUST", "1")
        gate = _RecordingDecline()
        bridge.set_rust_chat_completions(decline=gate)
        assert _accepts(litellm_params={"metadata": {"user_id": "u-123"}}) is False
        assert gate.calls == [], "the core must not be consulted for a request it cannot see the key of"

        # Bedrock's Converse transform reads no `user_id`, and an Anthropic request
        # whose metadata carries none is one Python would not attribute either.
        assert (
            _accepts(
                custom_llm_provider="bedrock",
                model="bedrock/us-east-1/anthropic.claude-v2",
                litellm_params={"metadata": {"user_id": "u-123"}},
            )
            is True
        )
        assert _accepts(litellm_params={"metadata": {"trace_id": "t-1"}}) is True
        assert _accepts(litellm_params={"metadata": {"user_id": None}}) is True
        assert _accepts(litellm_params={"metadata": None}) is True

    def test_declines_a_bedrock_request_while_the_proxy_owns_request_metadata(self, monkeypatch):
        """`AmazonConverseConfig` resolves proxy-owned `requestMetadata` onto the
        Converse body from `litellm_params`, and owning that field also means
        evicting a caller-supplied one. The core can do neither, so an operator
        who armed `bedrock_request_metadata_fields` keeps the Python path.
        """
        monkeypatch.setenv("LITELLM_RUST", "1")
        gate = _RecordingDecline()
        bridge.set_rust_chat_completions(decline=gate)
        bedrock = {
            "custom_llm_provider": "bedrock",
            "model": "bedrock/us-east-1/anthropic.claude-v2",
        }

        monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", ["user_api_key_team_id"])
        assert _accepts(**bedrock) is False
        assert gate.calls == [], "the core must not be consulted for a field it cannot write"
        assert _accepts() is True, "arming Bedrock attribution must not decline Anthropic"

        monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", None)
        assert _accepts(**bedrock) is True, "the decline follows the operator's opt-in alone"

    def test_declines_when_the_core_declines(self, monkeypatch):
        monkeypatch.setenv("LITELLM_RUST", "1")
        bridge.set_rust_chat_completions(decline=_RecordingDecline("streaming"))
        assert _accepts() is False

    def test_declines_when_the_bridge_is_unavailable(self, monkeypatch):
        monkeypatch.setenv("LITELLM_RUST", "1")
        _hide_native_bridge(monkeypatch)
        assert _accepts() is False

    def test_declines_when_the_gate_itself_raises(self, monkeypatch):
        monkeypatch.setenv("LITELLM_RUST", "1")

        def exploding(**_kwargs):
            raise RuntimeError("boom")

        bridge.set_rust_chat_completions(decline=exploding)
        assert _accepts() is False


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

        assert isinstance(result, Handled)
        result = result.value
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

    def test_preserves_execution_and_client_capabilities(self):
        native = _RecordingCall()
        bridge.set_rust_chat_completions(chat_completions=native)
        bridge.chat_completions(
            **_call_kwargs(ModelResponse()),
            stream=True,
            has_custom_client=True,
        )
        assert native.contexts[0].capabilities.execution_mode == "sync"
        assert native.contexts[0].capabilities.stream is True
        assert native.contexts[0].capabilities.has_custom_client is True

    def test_reports_unavailable_bridge(self, monkeypatch):
        _hide_native_bridge(monkeypatch)
        assert isinstance(bridge.chat_completions(**_call_kwargs(ModelResponse())), NativeSkipped)

    def test_reports_native_decline_to_orchestration(self, monkeypatch):
        _fake_native_bridge(monkeypatch)
        bridge.set_rust_chat_completions(chat_completions=_RecordingCall(error=_FakeDeclined("streaming")))
        assert isinstance(bridge.chat_completions(**_call_kwargs(ModelResponse())), NativeFailed)


class TestAsyncCall:
    @pytest.mark.asyncio
    async def test_builds_a_model_response(self):
        bridge.set_rust_chat_completions(achat_completions=_RecordingAsyncCall())
        result = await bridge.achat_completions(**_call_kwargs(ModelResponse()))
        assert isinstance(result, Handled)
        result = result.value
        assert result.choices[0].message.content == "hello from rust"
        assert result._hidden_params["additional_headers"] == {"x-litellm-rust": "true"}

    @pytest.mark.asyncio
    async def test_reports_unavailable_bridge(self, monkeypatch):
        _hide_native_bridge(monkeypatch)
        assert isinstance(await bridge.achat_completions(**_call_kwargs(ModelResponse())), NativeSkipped)

    @pytest.mark.asyncio
    async def test_reports_native_decline_to_orchestration(self, monkeypatch):
        _fake_native_bridge(monkeypatch)
        bridge.set_rust_chat_completions(achat_completions=_RecordingAsyncCall(error=_FakeDeclined("streaming")))
        assert isinstance(await bridge.achat_completions(**_call_kwargs(ModelResponse())), NativeFailed)
