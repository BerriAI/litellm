from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from typing import Final

import pytest

import litellm
from litellm.rust_bridge import bindings, configuration
from litellm.rust_bridge import chat_completions as bridge
from litellm.rust_bridge.callbacks import OneShotCallbackHandle
from litellm.types.utils import ModelResponse

RUST_RESPONSE: Final[dict[str, object]] = {
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
MESSAGES: Final[tuple[dict[str, str], ...]] = ({"role": "user", "content": "hi"},)
OPTIONAL_PARAMS: Final[dict[str, object]] = {"max_tokens": 16}
REQUEST_CONTEXT: Final = bridge.NativeChatContext(
    metadata={"user_id": "user-1"},
    litellm_metadata={"trace_id": "trace-1"},
    request_metadata_fields=("user_api_key_team_id",),
)


@pytest.fixture(autouse=True)
def reset_bridge(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    configuration.reset_rust_configuration()
    bridge.set_rust_chat_completions(sync=None, asynchronous=None)
    yield
    configuration.reset_rust_configuration()
    bridge.set_rust_chat_completions(sync=None, asynchronous=None)


class RecordingCall:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        *,
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: Mapping[str, object] | None,
        timeout_seconds: float | None,
        request_context: bridge.NativeChatContext,
        callback_adapter: OneShotCallbackHandle,
    ) -> Mapping[str, object]:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "optional_params": optional_params,
                "api_key": api_key,
                "api_base": api_base,
                "custom_llm_provider": custom_llm_provider,
                "extra_headers": extra_headers,
                "timeout_seconds": timeout_seconds,
                "request_context": request_context,
            }
        )
        callback_adapter.pre_call(
            {
                "provider": custom_llm_provider or "anthropic",
                "model": model,
                "call_id": "native-test",
                "trace_id": None,
                "attempt": 1,
                "started_at": 1.0,
                "request": {"model": model, "messages": messages},
                "api_base": api_base or "",
                "headers": extra_headers or {},
            }
        )
        callback_adapter.post_call(
            {
                "provider": custom_llm_provider or "anthropic",
                "model": model,
                "call_id": "native-test",
                "trace_id": None,
                "attempt": 1,
                "started_at": 1.0,
                "response": RUST_RESPONSE,
                "status_code": 200,
                "headers": {},
                "ended_at": 2.0,
            }
        )
        return RUST_RESPONSE


class RecordingAsyncCall(RecordingCall):
    async def __call__(
        self,
        *,
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: Mapping[str, object] | None,
        timeout_seconds: float | None,
        request_context: bridge.NativeChatContext,
        callback_adapter: OneShotCallbackHandle,
    ) -> Mapping[str, object]:
        return super().__call__(
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout_seconds=timeout_seconds,
            request_context=request_context,
            callback_adapter=callback_adapter,
        )


class RecordingCallbacks:
    def __init__(self, responses: list[Mapping[str, object]]) -> None:
        self.responses: Final = responses

    def pre_call(self, _payload: object, /) -> None:
        return None

    def post_call(self, payload: object, /) -> None:
        event: Final = payload if isinstance(payload, Mapping) else {}
        response: Final = event.get("response")
        if isinstance(response, Mapping):
            self.responses.append(response)

    def error(self, _payload: object, /) -> None:
        return None


def call_kwargs(
    model_response: ModelResponse,
    observed: list[Mapping[str, object]],
    *,
    provider: str = "anthropic",
    optional_params: Mapping[str, object] | None = None,
    eligible: bool = True,
) -> dict[str, object]:
    request: Final = bridge.NativeChatCompletionsRequest(
        model="claude-sonnet-4-5",
        messages=MESSAGES,
        optional_params=OPTIONAL_PARAMS if optional_params is None else optional_params,
        api_key="sk-test",
        api_base=None,
        custom_llm_provider=provider,
        extra_headers={"x-request-id": "req-1"},
        timeout=30.0,
        request_context=REQUEST_CONTEXT,
        callback_adapter=RecordingCallbacks(observed),
    )
    return {
        "prepare": lambda: request,
        "model": "claude-sonnet-4-5",
        "model_response": model_response,
        "provider": provider,
        "request_override": True,
        "eligible": eligible,
    }


def assert_model_response(result: object, original_id: str) -> None:
    assert isinstance(result, ModelResponse)
    assert result.id == original_id
    assert result.model == "claude-sonnet-4-5-20260101"
    assert result.choices[0].message.content == "hello from rust"
    assert result.usage.prompt_tokens == 11
    assert result._hidden_params["additional_headers"] == {"x-litellm-rust": "true"}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", (pytest.param("sync", id="sync"), pytest.param("async", id="async")))
async def test_dispatch_builds_model_response_and_forwards_exact_request(mode: str) -> None:
    observed: list[Mapping[str, object]] = []
    model_response: Final = ModelResponse()
    if mode == "sync":
        native: RecordingCall = RecordingCall()
        bridge.set_rust_chat_completions(sync=native)
        result: Final = bridge.dispatch_chat_completions(
            **call_kwargs(model_response, observed),
            python_fallback=lambda: "python",
        )
    else:
        native = RecordingAsyncCall()
        bridge.set_rust_chat_completions(asynchronous=native)

        async def fallback() -> str:
            return "python"

        result = await bridge.adispatch_chat_completions(
            **call_kwargs(model_response, observed),
            python_fallback=fallback,
        )

    assert_model_response(result, model_response.id)
    assert native.calls == [
        {
            "model": "claude-sonnet-4-5",
            "messages": MESSAGES,
            "optional_params": OPTIONAL_PARAMS,
            "api_key": "sk-test",
            "api_base": None,
            "custom_llm_provider": "anthropic",
            "extra_headers": {"x-request-id": "req-1"},
            "timeout_seconds": 30.0,
            "request_context": REQUEST_CONTEXT,
        }
    ]
    assert observed == [RUST_RESPONSE]


@pytest.mark.asyncio
async def test_async_dispatch_falls_back_once_when_binding_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: None)
    fallback_calls = 0

    async def fallback() -> str:
        nonlocal fallback_calls
        fallback_calls += 1
        return "python"

    result: Final = await bridge.adispatch_chat_completions(
        **call_kwargs(ModelResponse(), []),
        python_fallback=fallback,
    )

    assert result == "python"
    assert fallback_calls == 1


@pytest.mark.parametrize(
    ("provider", "optional_params"),
    (
        pytest.param("openai", OPTIONAL_PARAMS, id="provider-policy"),
        pytest.param("anthropic", {**OPTIONAL_PARAMS, "stream": True}, id="stream-policy"),
    ),
)
def test_provider_and_stream_compatibility_are_deferred_to_rust(
    provider: str,
    optional_params: Mapping[str, object],
) -> None:
    native: Final = RecordingCall()
    bridge.set_rust_chat_completions(sync=native)
    kwargs: Final = call_kwargs(ModelResponse(), [], provider=provider, optional_params=optional_params)

    bridge.dispatch_chat_completions(**kwargs, python_fallback=lambda: "python")

    assert len(native.calls) == 1


def test_custom_python_client_stays_on_python_path() -> None:
    native: Final = RecordingCall()
    bridge.set_rust_chat_completions(sync=native)
    kwargs: Final = call_kwargs(ModelResponse(), [], eligible=False)

    result: Final = bridge.dispatch_chat_completions(**kwargs, python_fallback=lambda: "python")

    assert result == "python"
    assert native.calls == []


def test_public_chat_passes_provider_visible_context_to_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    native: Final = RecordingCall()
    bridge.set_rust_chat_completions(sync=native)
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", ["user_api_key_team_id"])

    result: Final = litellm.completion(
        model="anthropic/claude-sonnet-4-5",
        messages=list(MESSAGES),
        max_tokens=16,
        metadata={"user_id": "user-1"},
        litellm_metadata={"trace_id": "trace-1"},
        rust=True,
    )

    assert result.choices[0].message.content == "hello from rust"
    context: Final = native.calls[0]["request_context"]
    assert context == REQUEST_CONTEXT


@pytest.mark.asyncio
async def test_public_async_chat_dispatches_before_the_python_provider_switch() -> None:
    native: Final = RecordingAsyncCall()
    bridge.set_rust_chat_completions(asynchronous=native)

    result: Final = await litellm.acompletion(
        model="anthropic/claude-sonnet-4-5",
        messages=list(MESSAGES),
        max_tokens=16,
        rust=True,
    )

    assert result.choices[0].message.content == "hello from rust"
    assert len(native.calls) == 1
