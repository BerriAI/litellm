from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from typing import Final

import pytest

import litellm
from litellm.rust_bridge import bindings, configuration
from litellm.rust_bridge import chat_completions as bridge
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


@pytest.fixture(autouse=True)
def reset_bridge(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    configuration.reset_rust_configuration()
    bridge.set_rust_chat_completions(chat_completions=None, achat_completions=None, decline=None)
    yield
    configuration.reset_rust_configuration()
    bridge.set_rust_chat_completions(chat_completions=None, achat_completions=None, decline=None)


class RecordingDecline:
    def __init__(self, reason: str | None = None) -> None:
        self.reason: Final = reason
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object] | None,
        custom_llm_provider: str | None,
    ) -> str | None:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "optional_params": optional_params,
                "custom_llm_provider": custom_llm_provider,
            }
        )
        return self.reason


class RecordingCall:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object] | None,
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: Mapping[str, object] | None,
        timeout_seconds: float | None,
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
            }
        )
        return RUST_RESPONSE


class RecordingAsyncCall(RecordingCall):
    async def __call__(
        self,
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object] | None,
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: Mapping[str, object] | None,
        timeout_seconds: float | None,
    ) -> Mapping[str, object]:
        return super().__call__(
            model,
            messages,
            optional_params,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            timeout_seconds,
        )


def accepts(
    *,
    model: str = "claude-sonnet-4-5",
    messages: Sequence[object] = MESSAGES,
    optional_params: Mapping[str, object] = OPTIONAL_PARAMS,
    custom_llm_provider: str | None = "anthropic",
    litellm_params: Mapping[str, object] | None = None,
    stream: object = None,
) -> bool:
    return bridge.rust_chat_completions_accepts(
        model=model,
        messages=messages,
        optional_params=optional_params,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        stream=stream,
    )


@pytest.mark.parametrize(
    ("enablement", "expected"),
    (
        pytest.param("request-true", True, id="request-true"),
        pytest.param("request-false", False, id="request-false-over-process"),
        pytest.param("process", True, id="process-override"),
        pytest.param("environment", True, id="environment"),
        pytest.param("disabled", False, id="disabled"),
    ),
)
def test_gate_forwards_enablement_to_preflight(
    monkeypatch: pytest.MonkeyPatch,
    enablement: str,
    expected: bool,
) -> None:
    gate: Final = RecordingDecline()
    bridge.set_rust_chat_completions(decline=gate)
    if enablement in {"request-false", "process"}:
        configuration.rust(True)
    if enablement == "environment":
        monkeypatch.setenv("LITELLM_RUST", "true")
    litellm_params: Final = (
        {"rust": True} if enablement == "request-true" else {"rust": False} if enablement == "request-false" else {}
    )

    assert accepts(litellm_params=litellm_params) is expected
    assert gate.calls == (
        [
            {
                "model": "claude-sonnet-4-5",
                "messages": MESSAGES,
                "optional_params": OPTIONAL_PARAMS,
                "custom_llm_provider": "anthropic",
            }
        ]
        if expected
        else []
    )


@pytest.mark.parametrize(
    ("provider", "litellm_params", "stream", "bedrock_metadata_fields"),
    (
        pytest.param("openai", {"rust": True}, None, None, id="unsupported-provider"),
        pytest.param(None, {"rust": True}, None, None, id="missing-provider"),
        pytest.param("anthropic", {"rust": True}, True, None, id="streaming"),
        pytest.param(
            "anthropic",
            {"rust": True, "metadata": {"user_id": "u-123"}},
            None,
            None,
            id="anthropic-user-id",
        ),
        pytest.param(
            "bedrock",
            {"rust": True},
            None,
            ["user_api_key_team_id"],
            id="bedrock-owned-request-metadata",
        ),
    ),
)
def test_gate_short_circuits_python_only_requests(
    monkeypatch: pytest.MonkeyPatch,
    provider: str | None,
    litellm_params: Mapping[str, object],
    stream: object,
    bedrock_metadata_fields: list[str] | None,
) -> None:
    gate: Final = RecordingDecline()
    bridge.set_rust_chat_completions(decline=gate)
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", bedrock_metadata_fields)

    assert (
        accepts(
            custom_llm_provider=provider,
            litellm_params=litellm_params,
            stream=stream,
        )
        is False
    )
    assert gate.calls == []


@pytest.mark.parametrize(
    ("provider", "litellm_params"),
    (
        pytest.param("anthropic", {"rust": True}, id="anthropic"),
        pytest.param("anthropic", {"rust": True, "metadata": {"trace_id": "t-1"}}, id="metadata-unrelated"),
        pytest.param("anthropic", {"rust": True, "metadata": {"user_id": None}}, id="user-id-none"),
        pytest.param("anthropic", {"rust": True, "metadata": None}, id="metadata-none"),
        pytest.param("bedrock", {"rust": True}, id="bedrock-unowned-metadata"),
    ),
)
def test_gate_accepts_requests_supported_by_preflight(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    litellm_params: Mapping[str, object],
) -> None:
    gate: Final = RecordingDecline()
    bridge.set_rust_chat_completions(decline=gate)
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", None)

    assert accepts(custom_llm_provider=provider, litellm_params=litellm_params) is True
    assert gate.calls == [
        {
            "model": "claude-sonnet-4-5",
            "messages": MESSAGES,
            "optional_params": OPTIONAL_PARAMS,
            "custom_llm_provider": provider,
        }
    ]


def call_kwargs(model_response: ModelResponse, observed: list[Mapping[str, object]]) -> dict[str, object]:
    return {
        "model": "claude-sonnet-4-5",
        "messages": MESSAGES,
        "optional_params": OPTIONAL_PARAMS,
        "model_response": model_response,
        "api_key": "sk-test",
        "api_base": None,
        "custom_llm_provider": "anthropic",
        "extra_headers": {"x-request-id": "req-1"},
        "timeout": 30.0,
        "on_response": observed.append,
        "request_override": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", (pytest.param("sync", id="sync"), pytest.param("async", id="async")))
async def test_call_builds_model_response_and_forwards_exact_request(mode: str) -> None:
    observed: list[Mapping[str, object]] = []
    model_response: Final = ModelResponse()
    original_id: Final = model_response.id
    if mode == "sync":
        native: RecordingCall = RecordingCall()
        bridge.set_rust_chat_completions(chat_completions=native)
        result: Final = bridge.chat_completions(**call_kwargs(model_response, observed))
    else:
        native = RecordingAsyncCall()
        bridge.set_rust_chat_completions(achat_completions=native)
        result = await bridge.achat_completions(**call_kwargs(model_response, observed))

    assert result is not None
    assert result.id == original_id
    assert result.model == "claude-sonnet-4-5-20260101"
    assert result.choices[0].message.content == "hello from rust"
    assert result.choices[0].finish_reason == "stop"
    assert (
        result.usage.prompt_tokens,
        result.usage.completion_tokens,
        result.usage.total_tokens,
    ) == (11, 4, 15)
    assert result._hidden_params["additional_headers"] == {"x-litellm-rust": "true"}
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
        }
    ]
    assert observed == [RUST_RESPONSE]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("native_available", "expected", "expected_fallback_calls"),
    (
        pytest.param(True, "hello from rust", 0, id="native-success"),
        pytest.param(False, "python", 1, id="bridge-unavailable"),
    ),
)
async def test_async_fallback_wrapper_selects_exactly_one_path(
    monkeypatch: pytest.MonkeyPatch,
    native_available: bool,
    expected: str,
    expected_fallback_calls: int,
) -> None:
    observed: list[Mapping[str, object]] = []
    fallback_calls = 0
    if native_available:
        bridge.set_rust_chat_completions(achat_completions=RecordingAsyncCall())
    else:
        monkeypatch.setattr(bindings, "get_native_bridge", lambda: None)

    async def fallback() -> str:
        nonlocal fallback_calls
        fallback_calls += 1
        return "python"

    result: Final = await bridge.achat_completions_or_fallback(
        **call_kwargs(ModelResponse(), observed),
        python_fallback=fallback,
    )
    actual: Final = result.choices[0].message.content if isinstance(result, ModelResponse) else result

    assert actual == expected
    assert fallback_calls == expected_fallback_calls
    assert observed == ([RUST_RESPONSE] if native_available else [])
