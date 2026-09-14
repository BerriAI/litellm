from __future__ import annotations

from typing import Final

import pytest

import litellm
from litellm.rust_bridge import chat_completions as bridge
from litellm.rust_bridge import configuration
from litellm.types.utils import ModelResponse

native = pytest.importorskip("litellm.rust_bridge._native")

RUST_RESPONSE: Final = {
    "created": 1_700_000_000,
    "model": "claude-sonnet-4-5-20260101",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello from rust"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
}
MESSAGES: Final = [{"role": "user", "content": "hi"}]

_FakeDeclined = native.RustBridgeDeclined
_FakeUpstream = native.RustUpstreamError


class _FakeNative:
    RustBridgeDeclined = _FakeDeclined
    RustUpstreamError = _FakeUpstream


class _RecordingCall:
    def __init__(self, result: object = RUST_RESPONSE, error: Exception | None = None) -> None:
        self.result: Final = result
        self.error: Final = error
        self.calls: Final[list[dict[str, object]]] = []

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        on_request: Final = kwargs["on_request"]
        assert callable(on_request)
        on_request()
        return self.result


class _RecordingAsyncCall(_RecordingCall):
    async def __call__(self, **kwargs: object) -> object:
        return super().__call__(**kwargs)


@pytest.fixture(autouse=True)
def reset_bridge(monkeypatch: pytest.MonkeyPatch):
    bridge.set_rust_chat_completions(chat_completions=None, achat_completions=None)
    configuration.reset_rust_configuration()
    monkeypatch.setenv("LITELLM_RUST", "1")
    yield
    bridge.set_rust_chat_completions(chat_completions=None, achat_completions=None)
    configuration.reset_rust_configuration()


def _fake_native_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("litellm.rust_bridge.bindings.get_native_bridge", lambda: _FakeNative())


def _call_kwargs(model_response: ModelResponse, fallback: object = "python") -> dict[str, object]:
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
        "python_fallback": lambda: fallback,
    }


def test_sync_native_entrypoint_runs_once_and_logs_once() -> None:
    events: Final[list[str]] = []
    native_call: Final = _RecordingCall()
    bridge.set_rust_chat_completions(chat_completions=native_call)
    kwargs: Final = _call_kwargs(ModelResponse())
    kwargs.update({"on_request": lambda: events.append("pre"), "on_response": lambda _value: events.append("post")})

    result: Final = bridge.chat_completions(**kwargs)

    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "hello from rust"
    assert result._hidden_params["additional_headers"] == {"x-litellm-rust": "true"}
    assert len(native_call.calls) == 1
    assert events == ["pre", "post"]


def test_decline_has_no_logging_effect_and_runs_one_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_native_bridge(monkeypatch)
    events: Final[list[str]] = []
    native_call: Final = _RecordingCall(error=_FakeDeclined("unsupported"))
    bridge.set_rust_chat_completions(chat_completions=native_call)
    kwargs: Final = _call_kwargs(ModelResponse(), fallback="python")
    kwargs.update({"on_request": lambda: events.append("pre"), "on_response": lambda _value: events.append("post")})

    assert bridge.chat_completions(**kwargs) == "python"
    assert len(native_call.calls) == 1
    assert events == []


def test_streaming_uses_python_without_loading_native() -> None:
    native_call: Final = _RecordingCall()
    bridge.set_rust_chat_completions(chat_completions=native_call)
    kwargs: Final = _call_kwargs(ModelResponse())
    kwargs["stream"] = True
    assert bridge.chat_completions(**kwargs) == "python"
    assert native_call.calls == []


def test_host_facts_reach_the_single_native_call(monkeypatch: pytest.MonkeyPatch) -> None:
    native_call: Final = _RecordingCall()
    bridge.set_rust_chat_completions(chat_completions=native_call)
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", ["team_id"])
    kwargs: Final = _call_kwargs(ModelResponse())
    kwargs["litellm_params"] = {"metadata": {"user_id": "u-1"}}
    bridge.chat_completions(**kwargs)
    assert native_call.calls[0]["host_facts"] == {
        "stream": False,
        "anthropic_user_id": True,
        "bedrock_metadata_owned": True,
    }


def test_upstream_and_adaptation_failures_never_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_native_bridge(monkeypatch)
    fallback_calls: Final[list[bool]] = []
    bridge.set_rust_chat_completions(chat_completions=_RecordingCall(error=_FakeUpstream(429, "rate limited")))
    kwargs: Final = _call_kwargs(ModelResponse())
    kwargs["python_fallback"] = lambda: fallback_calls.append(True)
    with pytest.raises(litellm.APIError, match="rate limited"):
        bridge.chat_completions(**kwargs)
    assert fallback_calls == []

    bridge.set_rust_chat_completions(chat_completions=_RecordingCall())
    kwargs["on_response"] = lambda _value: (_ for _ in ()).throw(RuntimeError("adapt failed"))
    with pytest.raises(RuntimeError, match="adapt failed"):
        bridge.chat_completions(**kwargs)
    assert fallback_calls == []


@pytest.mark.asyncio
async def test_async_native_and_fallback_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    native_call: Final = _RecordingAsyncCall()
    bridge.set_rust_chat_completions(achat_completions=native_call)

    async def fallback() -> str:
        return "python"

    kwargs: Final = _call_kwargs(ModelResponse())
    kwargs["python_fallback"] = fallback
    result: Final = await bridge.achat_completions(**kwargs)
    assert isinstance(result, ModelResponse)
    assert len(native_call.calls) == 1

    configuration.rust(False)
    assert await bridge.achat_completions(**kwargs) == "python"
