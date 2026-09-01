"""Tests for the Rust chat completions bridge.

The native callables are dependency-injected through
``set_rust_chat_completions`` rather than patched, so these run without the
compiled extension present.
"""

from __future__ import annotations

import pytest

import litellm
from litellm.rust_bridge import chat_completions as bridge
from litellm.rust_bridge import configuration, runtime
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
    monkeypatch.setattr(runtime, "get_native_bridge", lambda: _FakeNative())


def _hide_native_bridge(monkeypatch):
    """Simulate a wheel built without the compiled extension.

    There is no injection seam for "the .so is absent", so the loader itself is
    replaced; every other case here uses `set_rust_chat_completions`.
    """
    monkeypatch.setattr(runtime, "get_native_bridge", lambda: None)


@pytest.fixture(autouse=True)
def reset_bridge():
    """Every test starts with no injected callables, and leaves none behind."""
    bridge.set_rust_chat_completions(
        chat_completions=None,
        achat_completions=None,
        decline=lambda **_kwargs: None,
    )
    configuration.reset_rust_configuration()
    yield
    bridge.set_rust_chat_completions(chat_completions=None, achat_completions=None, decline=None)
    configuration.reset_rust_configuration()


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


def _should_attempt(**overrides) -> bool:
    kwargs = {
        "model": "claude-sonnet-4-5",
        "messages": MESSAGES,
        "optional_params": {"max_tokens": 16},
        "custom_llm_provider": "anthropic",
        "litellm_params": {"rust": True},
        "stream": None,
    }
    kwargs.update(overrides)
    kwargs.pop("asynchronous", None)
    return bridge.rust_chat_completions_accepts(**kwargs)


class TestEligibility:
    def test_declines_when_the_deployment_did_not_opt_in(self, monkeypatch):
        monkeypatch.delenv("LITELLM_RUST", raising=False)
        bridge.set_rust_chat_completions(chat_completions=_RecordingCall())
        assert _should_attempt(litellm_params={}) is False
        assert _should_attempt(litellm_params=None) is False
        assert _should_attempt(litellm_params={"rust": False}) is False

    def test_attempts_when_the_deployment_opted_in(self, monkeypatch):
        monkeypatch.delenv("LITELLM_RUST", raising=False)
        bridge.set_rust_chat_completions(chat_completions=_RecordingCall())
        assert _should_attempt() is True

    def test_the_env_var_opts_in_without_a_per_model_flag(self, monkeypatch):
        monkeypatch.setenv("LITELLM_RUST", "true")
        bridge.set_rust_chat_completions(chat_completions=_RecordingCall())
        assert _should_attempt(litellm_params={}) is True

    def test_explicit_false_overrides_the_process_switch(self):
        bridge.set_rust_chat_completions(chat_completions=_RecordingCall())
        configuration.use_litellm_rust(True)

        assert _should_attempt(litellm_params={"rust": False}) is False

    def test_declines_streaming_and_providers_off_the_path(self, monkeypatch):
        monkeypatch.delenv("LITELLM_RUST", raising=False)
        bridge.set_rust_chat_completions(chat_completions=_RecordingCall())
        assert _should_attempt(stream=True) is False
        assert _should_attempt(custom_llm_provider="openai") is False
        assert _should_attempt(custom_llm_provider=None) is False

    def test_declines_an_anthropic_request_carrying_a_litellm_metadata_user_id(self, monkeypatch):
        """`AnthropicConfig.transform_request` copies a valid `user_id` into the Messages body.

        It does that inside the function the Rust route replaces, and the core is
        handed `optional_params` only, so accepting here would send the request
        to Anthropic with the abuse-detection attribution silently missing.
        """
        monkeypatch.delenv("LITELLM_RUST", raising=False)
        bridge.set_rust_chat_completions(chat_completions=_RecordingCall())
        assert _should_attempt(litellm_params={"rust": True, "metadata": {"user_id": "u-123"}}) is False

        # Bedrock's Converse transform reads no `user_id`, and an Anthropic request
        # whose metadata carries none is one Python would not attribute either.
        assert (
            _should_attempt(
                custom_llm_provider="bedrock",
                litellm_params={"rust": True, "metadata": {"user_id": "u-123"}},
            )
            is True
        )
        assert _should_attempt(litellm_params={"rust": True, "metadata": {"trace_id": "t-1"}}) is True
        assert _should_attempt(litellm_params={"rust": True, "metadata": {"user_id": None}}) is True
        assert _should_attempt(litellm_params={"rust": True, "metadata": None}) is True

    def test_declines_a_bedrock_request_while_the_proxy_owns_request_metadata(self, monkeypatch):
        """`AmazonConverseConfig` resolves proxy-owned `requestMetadata` onto the
        Converse body from `litellm_params`, and owning that field also means
        evicting a caller-supplied one. The core can do neither, so an operator
        who armed `bedrock_request_metadata_fields` keeps the Python path.
        """
        monkeypatch.delenv("LITELLM_RUST", raising=False)
        bridge.set_rust_chat_completions(chat_completions=_RecordingCall())
        bedrock = {
            "custom_llm_provider": "bedrock",
        }

        monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", ["user_api_key_team_id"])
        assert _should_attempt(**bedrock) is False
        assert _should_attempt() is True, "arming Bedrock attribution must not decline Anthropic"

        monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", None)
        assert _should_attempt(**bedrock) is True, "the decline follows the operator's opt-in alone"

    def test_declines_when_the_bridge_is_unavailable(self, monkeypatch):
        monkeypatch.delenv("LITELLM_RUST", raising=False)
        bridge.set_rust_chat_completions(decline=None)
        _hide_native_bridge(monkeypatch)
        assert _should_attempt() is False

    def test_checks_the_native_capability_gate(self, monkeypatch):
        _hide_native_bridge(monkeypatch)
        bridge.set_rust_chat_completions(decline=None)
        assert _should_attempt() is False
        bridge.set_rust_chat_completions(decline=lambda **_kwargs: None)
        assert _should_attempt() is True


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


def _sync_call_kwargs(model_response: ModelResponse) -> dict:
    return {**_call_kwargs(model_response), "python_fallback": lambda: "python"}


async def _async_python_fallback() -> object:
    return "python"


def _async_call_kwargs(model_response: ModelResponse) -> dict:
    return {**_call_kwargs(model_response), "python_fallback": _async_python_fallback}


class TestSyncCall:
    def test_builds_a_model_response_and_stamps_the_rust_header(self):
        native = _RecordingCall()
        bridge.set_rust_chat_completions(chat_completions=native)
        model_response = ModelResponse()
        original_id = model_response.id

        result = bridge.chat_completions_or_fallback(**_sync_call_kwargs(model_response))

        assert result is not None
        assert result.choices[0].message.content == "hello from rust"
        assert result.choices[0].finish_reason == "stop"
        assert result.model == "claude-sonnet-4-5-20260101"
        assert result.usage.prompt_tokens == 11
        assert result.usage.completion_tokens == 4
        assert result.usage.total_tokens == 15
        assert result._hidden_params["core_engine"] == "rust"
        assert result._hidden_params["additional_headers"] == {
            "x-litellm-core": "rust",
            "x-litellm-rust": "true",
        }
        assert result.id == original_id, "the rust path must keep the chatcmpl id litellm already minted"

    def test_passes_the_timeout_through_as_seconds(self):
        native = _RecordingCall()
        bridge.set_rust_chat_completions(chat_completions=native)
        bridge.chat_completions_or_fallback(**_sync_call_kwargs(ModelResponse()))
        assert native.calls[0]["timeout_seconds"] == 30.0

    def test_falls_back_when_the_bridge_is_unavailable(self, monkeypatch):
        _hide_native_bridge(monkeypatch)
        assert bridge.chat_completions_or_fallback(**_sync_call_kwargs(ModelResponse())) == "python"

    def test_falls_back_when_the_core_declines_before_calling_the_provider(self, monkeypatch):
        _fake_native_bridge(monkeypatch)
        bridge.set_rust_chat_completions(chat_completions=_RecordingCall(error=_FakeDeclined("streaming")))
        assert bridge.chat_completions_or_fallback(**_sync_call_kwargs(ModelResponse())) == "python"

    def test_model_response_fallback_is_stamped_python(self, monkeypatch):
        _fake_native_bridge(monkeypatch)
        bridge.set_rust_chat_completions(chat_completions=_RecordingCall(error=_FakeDeclined("unsupported")))
        fallback_response = ModelResponse()

        result = bridge.chat_completions_or_fallback(
            **_call_kwargs(ModelResponse()),
            python_fallback=lambda: fallback_response,
        )

        assert result is fallback_response
        assert result._hidden_params["core_engine"] == "python"
        assert result._hidden_params["additional_headers"] == {"x-litellm-core": "python"}


class TestAsyncCall:
    @pytest.mark.asyncio
    async def test_builds_a_model_response(self):
        bridge.set_rust_chat_completions(achat_completions=_RecordingAsyncCall())
        result = await bridge.achat_completions_or_fallback(**_async_call_kwargs(ModelResponse()))
        assert result is not None
        assert result.choices[0].message.content == "hello from rust"
        assert result._hidden_params["core_engine"] == "rust"
        assert result._hidden_params["additional_headers"] == {
            "x-litellm-core": "rust",
            "x-litellm-rust": "true",
        }

    @pytest.mark.asyncio
    async def test_falls_back_when_the_bridge_is_unavailable(self, monkeypatch):
        _hide_native_bridge(monkeypatch)
        assert await bridge.achat_completions_or_fallback(**_async_call_kwargs(ModelResponse())) == "python"

    @pytest.mark.asyncio
    async def test_falls_back_when_the_core_declines_before_calling_the_provider(self, monkeypatch):
        _fake_native_bridge(monkeypatch)
        bridge.set_rust_chat_completions(achat_completions=_RecordingAsyncCall(error=_FakeDeclined("streaming")))
        assert await bridge.achat_completions_or_fallback(**_async_call_kwargs(ModelResponse())) == "python"


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
        assert bridge.chat_completions_or_fallback(**_sync_call_kwargs(ModelResponse())) == "python"

    def test_an_upstream_failure_is_surfaced_with_its_status(self):
        from litellm.exceptions import APIError

        bridge.set_rust_chat_completions(chat_completions=_RecordingCall(error=_FakeUpstream(429, "429: rate limited")))
        with pytest.raises(APIError) as raised:
            bridge.chat_completions_or_fallback(**_sync_call_kwargs(ModelResponse()))
        assert raised.value.status_code == 429
        assert "rate limited" in str(raised.value)

    def test_a_transport_failure_with_no_response_surfaces_as_a_500(self):
        from litellm.exceptions import APIError

        bridge.set_rust_chat_completions(chat_completions=_RecordingCall(error=_FakeUpstream(0, "connection reset")))
        with pytest.raises(APIError) as raised:
            bridge.chat_completions_or_fallback(**_sync_call_kwargs(ModelResponse()))
        assert raised.value.status_code == 500

    def test_an_unrecognized_error_is_not_swallowed(self):
        bridge.set_rust_chat_completions(chat_completions=_RecordingCall(error=RuntimeError("something else")))
        with pytest.raises(RuntimeError):
            bridge.chat_completions_or_fallback(**_sync_call_kwargs(ModelResponse()))

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
