import base64
import json
import struct
import zlib
from typing import Final, Mapping, cast

import httpx
import pytest
import respx
from pydantic import BaseModel, JsonValue
from respx import MockRouter

import litellm
from litellm import get_llm_provider
from litellm.constants import DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET
from litellm.main import stream_chunk_builder
from litellm.utils import get_optional_params

from tests.unit.llms.base_llm.chat.test_provider_chat_translation import (
    _BY_ID,
    _Case,
    _complete,
    _request_body,
    _aws_frame,
)


_THINKING: Final[Mapping[str, JsonValue]] = {"type": "enabled", "budget_tokens": 16000}


def _thinking_kwargs(case: _Case) -> Mapping[str, JsonValue]:
    if case["id"].startswith("bedrock"):
        return {"thinking": _THINKING}
    return {"thinking": _THINKING}


_THINKING_CASES: Final = _BY_ID["anthropic_sonnet45"]
_ANTHROPIC_CASE: Final = _BY_ID["anthropic_sonnet45"]

_BEDROCK_THINKING_CASE: Final = _Case(
    {
        "id": "bedrock_converse_sonnet_thinking",
        "shape": "bedrock_converse",
        "kwargs": {
            "model": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "aws_access_key_id": "AKIAFAKE",
            "aws_secret_access_key": "fakesecret",
            "aws_region_name": "us-east-1",
        },
        "url": "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-4-5-20250929-v1:0/converse",
        "stream_url": "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-4-5-20250929-v1:0/converse-stream",
        "router": False,
        "developer_kept": False,
    }
)


class _RFormat(BaseModel):
    model_config = {"frozen": True}
    question: str
    answer: str


_JSON_CONTENT: Final = '{"agent_doing": "researching home automation"}'


def _anthropic_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_offline",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5-20250929",
            "content": [
                {"type": "thinking", "thinking": "reasoning here", "signature": "sig-1"},
                {"type": "text", "text": _JSON_CONTENT},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )


def _anthropic_sse() -> str:
    events: Final = (
        {"type": "message_start", "message": {"id": "msg_offline", "type": "message", "role": "assistant",
                                            "model": "claude-sonnet-4-5-20250929", "content": [],
                                            "stop_reason": None, "usage": {"input_tokens": 10, "output_tokens": 1}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": _JSON_CONTENT}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    )
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def _anthropic_thinking_sse() -> str:
    events: Final = (
        {"type": "message_start", "message": {"id": "msg_offline", "type": "message", "role": "assistant",
                                            "model": "claude-sonnet-4-5-20250929", "content": [],
                                            "stop_reason": None, "usage": {"input_tokens": 10, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "reasoning here"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig-1"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "done"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    )
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def _converse_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "output": {"message": {"role": "assistant", "content": [
                {"reasoningContent": {"reasoningText": {"text": "reasoning here", "signature": "sig-1"}}},
                {"text": _JSON_CONTENT},
            ]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        },
    )


def _converse_frames(events: tuple[Mapping[str, JsonValue], ...]) -> bytes:
    def frame(payload: Mapping[str, JsonValue]) -> bytes:
        raw: Final = json.dumps(payload).encode()
        event_type: Final = next(iter(payload))
        headers: Final = (
            (":event-type", event_type),
            (":content-type", "application/json"),
            (":message-type", "event"),
        )
        kv_bytes: Final = b"".join(
            len(k).to_bytes(1, "big") + k.encode() + b"\x07" + len(v).to_bytes(2, "big") + v.encode()
            for k, v in headers
        )
        hdr_len: Final = len(kv_bytes)
        prelude: Final = struct.pack(">II", 12 + hdr_len + len(raw) + 4, hdr_len)
        prelude_crc: Final = struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF)
        rest: Final = kv_bytes + raw
        total_crc: Final = struct.pack(">I", zlib.crc32(prelude + prelude_crc + rest) & 0xFFFFFFFF)
        return prelude + prelude_crc + rest + total_crc

    return b"".join(frame(e) for e in events)


def _converse_stream_bytes() -> bytes:
    events: Final = (
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": _JSON_CONTENT}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}}},
    )
    return _converse_frames(events)


def _converse_thinking_stream_bytes() -> bytes:
    events: Final = (
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {"reasoningContent": {}}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "reasoning here"}}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"signature": "sig-1"}}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"contentBlockStart": {"start": {}, "contentBlockIndex": 1}},
        {"contentBlockDelta": {"delta": {"text": "done"}, "contentBlockIndex": 1}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}}},
    )
    return _converse_frames(events)


def _nonstream_canned(case: _Case) -> httpx.Response:
    if case["shape"] == "anthropic":
        return _anthropic_response()
    return _converse_response()


def _stream_canned(case: _Case, *, thinking: bool) -> httpx.Response:
    if case["shape"] == "anthropic":
        return httpx.Response(
            200,
            content=(_anthropic_thinking_sse() if thinking else _anthropic_sse()),
            headers={"content-type": "text/event-stream"},
        )
    return httpx.Response(
        200,
        content=(_converse_thinking_stream_bytes() if thinking else _converse_stream_bytes()),
        headers={"content-type": "application/vnd.amazon.eventstream"},
    )


def _complete_thinking(case: _Case, respx_mock: MockRouter, extra: Mapping[str, JsonValue], *, stream: bool = False, thinking_stream: bool = False):
    route: Final = respx_mock.post(case["stream_url"] if stream else case["url"]).mock(
        return_value=_stream_canned(case, thinking=thinking_stream) if stream else _nonstream_canned(case)
    )
    response: Final = _complete(case, extra)
    return route, response


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


@pytest.mark.parametrize("case", (_ANTHROPIC_CASE, _BEDROCK_THINKING_CASE), ids=lambda c: c["id"])
def test_anthropic_response_format_streaming_vs_non_streaming(case: _Case, respx_mock: MockRouter) -> None:
    stream_route: Final = respx_mock.post(case["stream_url"]).mock(return_value=_stream_canned(case, thinking=False))
    resp_stream: Final = _complete(case, {**_JSON_SCHEMA_ARGS, "stream": True})
    chunks: Final = tuple(resp_stream)
    built: Final = stream_chunk_builder(chunks=list(chunks))
    stream_body: Final = _request_body(stream_route)

    non_route: Final = respx_mock.post(case["url"]).mock(return_value=_nonstream_canned(case))
    resp_non: Final = _complete(case, _JSON_SCHEMA_ARGS)
    non_body: Final = _request_body(non_route)

    for body in (stream_body, non_body):
        if case["shape"] == "anthropic":
            output_format: Final = cast(Mapping[str, JsonValue], body["output_format"])
            assert output_format["type"] == "json_schema"
            assert cast(Mapping[str, JsonValue], output_format["schema"])["title"] == "ThinkingStep"
        else:
            output_config: Final = cast(Mapping[str, JsonValue], body["outputConfig"])
            text_format: Final = cast(Mapping[str, JsonValue], output_config["textFormat"])
            assert text_format["type"] == "json_schema"
    assert (
        json.loads(built.choices[0].message.content).keys()
        == json.loads(resp_non.choices[0].message.content).keys()
        == {"agent_doing"}
    )


def _thinking_request_thinking(case: _Case, body: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    if case["shape"] == "anthropic":
        return cast(Mapping[str, JsonValue], body["thinking"])
    fields: Final = cast(Mapping[str, JsonValue], body["additionalModelRequestFields"])
    return cast(Mapping[str, JsonValue], fields["thinking"])


def _max_tokens_value(case: _Case, body: Mapping[str, JsonValue]) -> int:
    if case["shape"] == "anthropic":
        return cast(int, body["max_tokens"])
    return cast(int, cast(Mapping[str, JsonValue], body["inferenceConfig"])["maxTokens"])


def _response_text(case: _Case, response: object) -> str:
    if case["shape"] == "anthropic" or case["shape"] == "bedrock_converse":
        return response.choices[0].message.content
    return response.choices[0].message.content


@pytest.mark.parametrize("case", (_ANTHROPIC_CASE, _BEDROCK_THINKING_CASE), ids=lambda c: c["id"])
def test_completion_thinking_with_response_format(case: _Case, respx_mock: MockRouter) -> None:
    route: Final = respx_mock.post(case["url"]).mock(return_value=_nonstream_canned(case))
    response: Final = _complete(
        case,
        {
            **_thinking_kwargs(case),
            "messages": [{"role": "user", "content": "Generate 5 question + answer pairs"}],
            "response_format": _RFormat,
        },
    )
    body: Final = _request_body(route)
    thinking: Final = _thinking_request_thinking(case, body)
    assert thinking == {"type": "enabled", "budget_tokens": 16000}
    if case["shape"] == "anthropic":
        assert cast(Mapping[str, JsonValue], body["output_format"])["type"] == "json_schema"
    else:
        assert cast(Mapping[str, JsonValue], cast(Mapping[str, JsonValue], body["outputConfig"])["textFormat"])["type"] == "json_schema"
    assert response.choices[0].message.content == _JSON_CONTENT


def test_completion_thinking_with_max_tokens(respx_mock: MockRouter) -> None:
    case: Final = _ANTHROPIC_CASE
    route: Final = respx_mock.post(case["url"]).mock(return_value=_nonstream_canned(case))
    response: Final = _complete(
        case,
        {
            **_thinking_kwargs(case),
            "messages": [{"role": "user", "content": "Generate 5 question + answer pairs"}],
            "max_completion_tokens": 20000,
        },
    )
    body: Final = _request_body(route)
    assert _max_tokens_value(case, body) == 20000
    assert _thinking_request_thinking(case, body) == {"type": "enabled", "budget_tokens": 16000}
    assert response.choices[0].message.content == _JSON_CONTENT


def test_completion_thinking_without_max_tokens(respx_mock: MockRouter) -> None:
    case: Final = _ANTHROPIC_CASE
    route: Final = respx_mock.post(case["url"]).mock(return_value=_nonstream_canned(case))
    response: Final = _complete(
        case,
        {
            **_thinking_kwargs(case),
            "messages": [{"role": "user", "content": "Generate 5 question + answer pairs"}],
        },
    )
    body: Final = _request_body(route)
    assert _max_tokens_value(case, body) == 20096
    assert _thinking_request_thinking(case, body) == {"type": "enabled", "budget_tokens": 16000}
    assert response.choices[0].message.content == _JSON_CONTENT


@pytest.mark.parametrize("case", (_ANTHROPIC_CASE, _BEDROCK_THINKING_CASE), ids=lambda c: c["id"])
def test_anthropic_thinking_output_stream(case: _Case, respx_mock: MockRouter) -> None:
    respx_mock.post(case["stream_url"]).mock(return_value=_stream_canned(case, thinking=True))
    resp: Final = _complete(
        case,
        {
            **_thinking_kwargs(case),
            "messages": [{"role": "user", "content": "Tell me a joke."}],
            "stream": True,
        },
    )
    reasoning_content_exists: Final[list[bool]] = [False]
    signature_block_exists: Final[list[bool]] = [False]
    tool_call_exists: Final[list[bool]] = [False]
    for chunk in resp:
        if chunk.choices[0].delta.tool_calls:
            tool_call_exists[0] = True
        if (
            getattr(chunk.choices[0].delta, "thinking_blocks", None) is not None
            and getattr(chunk.choices[0].delta, "reasoning_content", None) is not None
            and isinstance(chunk.choices[0].delta.thinking_blocks, list)
            and len(chunk.choices[0].delta.thinking_blocks) > 0
        ):
            reasoning_content_exists[0] = True
            if chunk.choices[0].delta.thinking_blocks[0].get("signature"):
                signature_block_exists[0] = True
    assert tool_call_exists[0] is False
    assert reasoning_content_exists[0] is True
    assert signature_block_exists[0] is True


def test_anthropic_reasoning_effort_thinking_translation() -> None:
    for model in ("anthropic/claude-sonnet-4-5-20250929", "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"):
        _, provider, _, _ = get_llm_provider(model=model)
        optional_params: Final = get_optional_params(
            model=model,
            custom_llm_provider=provider,
            reasoning_effort="high",
        )
        assert optional_params["thinking"] == {
            "type": "enabled",
            "budget_tokens": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
        }
        assert "reasoning_effort" not in optional_params
