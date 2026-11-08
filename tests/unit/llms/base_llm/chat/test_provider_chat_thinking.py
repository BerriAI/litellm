import json
from itertools import chain
from typing import Final, Mapping, cast

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, JsonValue
from respx import MockRouter

import litellm
from litellm import get_llm_provider
from litellm.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
)
from litellm.main import stream_chunk_builder
from litellm.utils import get_optional_params

from tests.unit.llms.base_llm.chat.test_provider_chat_translation import (
    _BY_ID,
    _Case,
    _aws_frame,
    _call as _provider_call,
    _request_body,
)

_THINKING_BUDGET: Final = 16000
_THINKING: Final[JsonValue] = {"type": "enabled", "budget_tokens": _THINKING_BUDGET}

_ANTHROPIC: Final = _BY_ID["anthropic_sonnet45"]
_BEDROCK_HAIKU: Final = _BY_ID["bedrock_converse_haiku"]
_BEDROCK_SONNET: Final = _BY_ID["bedrock_converse_anthropic_thinking"]

_THINKING_CASES: Final = (_ANTHROPIC, _BEDROCK_SONNET)
_RESPONSE_FORMAT_CASES: Final = (_ANTHROPIC, _BEDROCK_HAIKU)

_JSON_PREFIX: Final = '{"agent_doing": "researching '
_JSON_SUFFIX: Final = 'home automation"}'
_JSON_CONTENT: Final = _JSON_PREFIX + _JSON_SUFFIX
_REASONING: Final = "reasoning here"
_SIGNATURE: Final = "sig-1"


class _RFormat(BaseModel):
    model_config = ConfigDict(frozen=True)
    question: str
    answer: str


_JSON_SCHEMA_ARGS: Final[Mapping[str, JsonValue]] = {
    "messages": [
        {"role": "system", "content": "Summarize the agent's thinking into short descriptions."},
        {"role": "user", "content": "Here is the input data."},
    ],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "final_output",
            "strict": True,
            "schema": {
                "properties": {"agent_doing": {"title": "Agent Doing", "type": "string"}},
                "required": ["agent_doing"],
                "title": "ThinkingStep",
                "type": "object",
                "additionalProperties": False,
            },
        },
    },
}

_THINKING_MESSAGES: Final[Mapping[str, JsonValue]] = {
    "messages": [{"role": "user", "content": "Generate 5 question + answer pairs"}],
}


def _case_id(case: _Case) -> str:
    return case["id"]


def _call(case: _Case, **extra: JsonValue) -> object:
    return _provider_call(case, extra)


def _anthropic_sse(events: tuple[Mapping[str, JsonValue], ...]) -> str:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


_ANTHROPIC_START: Final[Mapping[str, JsonValue]] = {
    "type": "message_start",
    "message": {
        "id": "msg_offline",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": [],
        "stop_reason": None,
        "usage": {"input_tokens": 10, "output_tokens": 1},
    },
}
_ANTHROPIC_END: Final[tuple[Mapping[str, JsonValue], ...]] = (
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
    {"type": "message_stop"},
)


def _anthropic_json_stream() -> str:
    return _anthropic_sse(
        (
            _ANTHROPIC_START,
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": _JSON_PREFIX}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": _JSON_SUFFIX}},
            {"type": "content_block_stop", "index": 0},
            *_ANTHROPIC_END,
        )
    )


def _anthropic_thinking_stream() -> str:
    return _anthropic_sse(
        (
            _ANTHROPIC_START,
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": _REASONING}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": _SIGNATURE}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "done"}},
            {"type": "content_block_stop", "index": 1},
            *_ANTHROPIC_END,
        )
    )


_CONVERSE_USAGE: Final[Mapping[str, JsonValue]] = {"usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}}


def _converse_stream(frames: tuple[tuple[str, Mapping[str, JsonValue]], ...]) -> bytes:
    return b"".join(_aws_frame(event_type, payload) for event_type, payload in frames)


def _converse_json_stream() -> bytes:
    return _converse_stream(
        (
            ("messageStart", {"role": "assistant"}),
            ("contentBlockDelta", {"delta": {"text": _JSON_PREFIX}, "contentBlockIndex": 0}),
            ("contentBlockDelta", {"delta": {"text": _JSON_SUFFIX}, "contentBlockIndex": 0}),
            ("contentBlockStop", {"contentBlockIndex": 0}),
            ("messageStop", {"stopReason": "end_turn"}),
            ("metadata", _CONVERSE_USAGE),
        )
    )


def _converse_thinking_stream() -> bytes:
    return _converse_stream(
        (
            ("messageStart", {"role": "assistant"}),
            ("contentBlockDelta", {"delta": {"reasoningContent": {"text": _REASONING}}, "contentBlockIndex": 0}),
            ("contentBlockDelta", {"delta": {"reasoningContent": {"signature": _SIGNATURE}}, "contentBlockIndex": 0}),
            ("contentBlockStop", {"contentBlockIndex": 0}),
            ("contentBlockDelta", {"delta": {"text": "done"}, "contentBlockIndex": 1}),
            ("contentBlockStop", {"contentBlockIndex": 1}),
            ("messageStop", {"stopReason": "end_turn"}),
            ("metadata", _CONVERSE_USAGE),
        )
    )


def _non_stream_response(case: _Case) -> httpx.Response:
    if case["shape"] == "anthropic":
        return httpx.Response(
            200,
            json={
                "id": "msg_offline",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5-20250929",
                "content": [
                    {"type": "thinking", "thinking": _REASONING, "signature": _SIGNATURE},
                    {"type": "text", "text": _JSON_CONTENT},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )
    return httpx.Response(
        200,
        json={
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"reasoningContent": {"reasoningText": {"text": _REASONING, "signature": _SIGNATURE}}},
                        {"text": _JSON_CONTENT},
                    ],
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        },
    )


def _stream_response(case: _Case, *, thinking: bool) -> httpx.Response:
    if case["shape"] == "anthropic":
        return httpx.Response(
            200,
            content=_anthropic_thinking_stream() if thinking else _anthropic_json_stream(),
            headers={"content-type": "text/event-stream"},
        )
    return httpx.Response(
        200,
        content=_converse_thinking_stream() if thinking else _converse_json_stream(),
        headers={"content-type": "application/vnd.amazon.eventstream"},
    )


def _thinking_param(case: _Case, body: Mapping[str, JsonValue]) -> JsonValue:
    if case["shape"] == "anthropic":
        return body["thinking"]
    return cast(Mapping[str, JsonValue], body["additionalModelRequestFields"])["thinking"]


def _max_tokens(case: _Case, body: Mapping[str, JsonValue]) -> JsonValue:
    if case["shape"] == "anthropic":
        return body["max_tokens"]
    return cast(Mapping[str, JsonValue], body["inferenceConfig"])["maxTokens"]


def _json_schema_title(case: _Case, body: Mapping[str, JsonValue]) -> str:
    if case["shape"] == "anthropic":
        output_format: Final = cast(Mapping[str, JsonValue], body["output_format"])
        assert output_format["type"] == "json_schema"
        return cast(str, cast(Mapping[str, JsonValue], output_format["schema"])["title"])
    text_format: Final = cast(
        Mapping[str, JsonValue],
        cast(Mapping[str, JsonValue], body["outputConfig"])["textFormat"],
    )
    assert text_format["type"] == "json_schema"
    json_schema: Final = cast(
        Mapping[str, JsonValue], cast(Mapping[str, JsonValue], text_format["structure"])["jsonSchema"]
    )
    return cast(str, json.loads(cast(str, json_schema["schema"]))["title"])


def _has_forced_tool_choice(case: _Case, body: Mapping[str, JsonValue]) -> bool:
    if case["shape"] == "anthropic":
        return body.get("tool_choice") is not None
    return "toolConfig" in body


@pytest.mark.parametrize("case", _RESPONSE_FORMAT_CASES, ids=_case_id)
def test_anthropic_response_format_streaming_vs_non_streaming(case: _Case, respx_mock: MockRouter) -> None:
    stream_route: Final = respx_mock.post(case["stream_url"]).mock(return_value=_stream_response(case, thinking=False))
    chunks: Final = tuple(cast(litellm.CustomStreamWrapper, _call(case, **_JSON_SCHEMA_ARGS, stream=True)))
    built: Final = stream_chunk_builder(chunks=list(chunks))
    stream_body: Final = _request_body(stream_route)

    non_stream_route: Final = respx_mock.post(case["url"]).mock(return_value=_non_stream_response(case))
    non_stream: Final = cast(litellm.ModelResponse, _call(case, **_JSON_SCHEMA_ARGS))
    non_stream_body: Final = _request_body(non_stream_route)

    assert len(chunks) > 1
    assert _json_schema_title(case, stream_body) == "ThinkingStep"
    assert _json_schema_title(case, non_stream_body) == "ThinkingStep"
    assert built is not None
    streamed_json: Final = cast(
        Mapping[str, JsonValue],
        json.loads(cast(str, cast(litellm.ModelResponse, built).choices[0].message.content)),
    )
    non_stream_json: Final = cast(Mapping[str, JsonValue], json.loads(cast(str, non_stream.choices[0].message.content)))
    assert streamed_json == non_stream_json == {"agent_doing": "researching home automation"}


@pytest.mark.parametrize("case", _THINKING_CASES, ids=_case_id)
def test_completion_thinking_with_response_format(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = respx_mock.post(case["url"]).mock(return_value=_non_stream_response(case))
    response: Final = cast(
        litellm.ModelResponse,
        _call(case, thinking=_THINKING, **_THINKING_MESSAGES, response_format=cast(JsonValue, _RFormat)),
    )
    body: Final = _request_body(route)
    assert _thinking_param(case, body) == _THINKING
    assert _json_schema_title(case, body) == "_RFormat"
    assert not _has_forced_tool_choice(case, body)
    assert response.choices[0].message.content == _JSON_CONTENT
    assert response.choices[0].message.reasoning_content == _REASONING


@pytest.mark.parametrize("case", _THINKING_CASES, ids=_case_id)
def test_completion_thinking_with_max_tokens(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = respx_mock.post(case["url"]).mock(return_value=_non_stream_response(case))
    response: Final = cast(
        litellm.ModelResponse,
        _call(case, thinking=_THINKING, **_THINKING_MESSAGES, max_completion_tokens=20000),
    )
    body: Final = _request_body(route)
    assert _max_tokens(case, body) == 20000
    assert _thinking_param(case, body) == _THINKING
    assert response.choices[0].message.content == _JSON_CONTENT


@pytest.mark.parametrize("case", _THINKING_CASES, ids=_case_id)
def test_completion_thinking_without_max_tokens(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = respx_mock.post(case["url"]).mock(return_value=_non_stream_response(case))
    response: Final = cast(litellm.ModelResponse, _call(case, thinking=_THINKING, **_THINKING_MESSAGES))
    body: Final = _request_body(route)
    max_tokens: Final = cast(int, _max_tokens(case, body))
    assert max_tokens == _THINKING_BUDGET + DEFAULT_MAX_TOKENS
    assert max_tokens > _THINKING_BUDGET
    assert _thinking_param(case, body) == _THINKING
    assert response.choices[0].message.content == _JSON_CONTENT


@pytest.mark.parametrize("case", _THINKING_CASES, ids=_case_id)
def test_anthropic_thinking_output_stream(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = respx_mock.post(case["stream_url"]).mock(return_value=_stream_response(case, thinking=True))
    chunks: Final = tuple(
        cast(
            litellm.CustomStreamWrapper,
            _call(
                case,
                thinking=_THINKING,
                messages=[{"role": "user", "content": "Tell me a joke."}],
                stream=True,
            ),
        )
    )
    deltas: Final = tuple(chunk.choices[0].delta for chunk in chunks)
    thinking_deltas: Final = tuple(
        delta
        for delta in deltas
        if isinstance(getattr(delta, "thinking_blocks", None), list)
        and delta.thinking_blocks
        and isinstance(getattr(delta, "reasoning_content", None), str)
    )
    blocks: Final = chain.from_iterable(cast(list[object], delta.thinking_blocks) for delta in thinking_deltas)
    signatures: Final = tuple(cast(Mapping[str, JsonValue], block).get("signature") for block in blocks)
    assert _thinking_param(case, _request_body(route)) == _THINKING
    assert not any(delta.tool_calls for delta in deltas)
    assert "".join(cast(str, delta.reasoning_content) for delta in thinking_deltas) == _REASONING
    assert _SIGNATURE in signatures


@pytest.mark.parametrize("case", _THINKING_CASES, ids=_case_id)
def test_anthropic_reasoning_effort_thinking_translation(case: _Case, respx_mock: MockRouter) -> None:
    model: Final = case["kwargs"].get("model", "")
    _, provider, _, _ = get_llm_provider(model=model)
    optional_params: Final = get_optional_params(model=model, custom_llm_provider=provider, reasoning_effort="high")
    assert optional_params["thinking"] == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    }
    assert "reasoning_effort" not in optional_params

    route: Final = respx_mock.post(case["url"]).mock(return_value=_non_stream_response(case))
    _call(case, reasoning_effort="high", messages=[{"role": "user", "content": "hi"}])
    body: Final = _request_body(route)
    assert _thinking_param(case, body) == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    }
    assert "reasoning_effort" not in json.dumps(body)
    assert _max_tokens(case, body) == DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET + DEFAULT_MAX_TOKENS


@pytest.mark.parametrize("case", _THINKING_CASES, ids=_case_id)
@pytest.mark.parametrize(
    ("effort", "budget"),
    (
        ("low", DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET),
        ("medium", DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET),
    ),
)
def test_reasoning_effort_maps_to_distinct_thinking_budgets(
    case: _Case, effort: str, budget: int, respx_mock: MockRouter
) -> None:
    route: Final = respx_mock.post(case["url"]).mock(return_value=_non_stream_response(case))
    _call(case, reasoning_effort=effort, messages=[{"role": "user", "content": "hi"}])
    body: Final = _request_body(route)
    assert _thinking_param(case, body) == {"type": "enabled", "budget_tokens": budget}
    assert _max_tokens(case, body) == budget + DEFAULT_MAX_TOKENS
