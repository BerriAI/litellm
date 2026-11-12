import base64
import json
import math
import os
import uuid
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.resources import files
from itertools import chain
from typing import Final
from urllib.parse import unquote

import anthropic
import openai
import pytest
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.upstream import (
    _aws_event_frame,  # pyright: ignore[reportPrivateUsage]  # the scripted Bedrock peer's only event-stream encoder
)
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter
from redis import Redis

_TOKEN: Final = "synthetic-bedrock-bearer"
_NATIVE_REDACTED: Final = base64.b64encode(b"\x12\x1esynthetic redacted reasoning payload").decode()
_CONVERSE_REDACTED: Final = base64.b64encode(_NATIVE_REDACTED.encode()).decode()
_SIGNATURE: Final = "EqoBCkgIBxABGAIiQsyntheticSignature"
_ANSWER: Final[dict[str, JsonValue]] = {"city": "Tokyo", "population_millions": 14.0}
_THINKING_TOKENS: Final = 87
_EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
_NO_CACHE: Final[dict[str, JsonValue]] = {"no-cache": True}
_THINKING: Final[dict[str, JsonValue]] = {"type": "enabled", "budget_tokens": 1024}
_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "population_millions": {"type": "number"}},
    "required": ["city", "population_millions"],
    "additionalProperties": False,
}
_RESPONSE_FORMAT: Final[dict[str, JsonValue]] = {
    "type": "json_schema",
    "json_schema": {"name": "city_fact", "schema": _SCHEMA},
}
_CONVERSE_USAGE: Final[dict[str, JsonValue]] = {
    "inputTokens": 40,
    "outputTokens": 120,
    "cacheReadInputTokens": 50,
    "totalTokens": 210,
}
_LLAMA_MODEL_ID: Final = "us.meta.llama3-3-70b-instruct-v1:0"
_VLLM_BACKEND: Final = "qwen3-fake-stream-audit"
_VLLM_KEY: Final = "synthetic-hosted-vllm-key"
_AZURE_DEPLOYMENT: Final = "reasoning-deployment"
_AZURE_API_VERSION: Final = "2025-03-01-preview"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_DISCOVERY_PROBE: Final = ("GET", "/v1/models")


def _tool(name: str) -> dict[str, JsonValue]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class _CostEntry(_Frozen):
    litellm_provider: str | None = None
    supports_native_structured_output: bool | None = None
    supports_reasoning: bool | None = None


class _ToolFunction(_Frozen):
    name: str | None = None
    arguments: str | None = None


class _ToolCall(_Frozen):
    index: int = 0
    function: _ToolFunction


class _PromptDetails(_Frozen):
    cached_tokens: int | None = None


class _CompletionDetails(_Frozen):
    reasoning_tokens: int | None = None


class _Usage(_Frozen):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: _PromptDetails | None = None
    completion_tokens_details: _CompletionDetails | None = None


class _Message(_Frozen):
    content: str | None = None
    reasoning_content: str | None = None
    thinking_blocks: tuple[dict[str, JsonValue], ...] | None = None
    tool_calls: tuple[_ToolCall, ...] | None = None


class _Choice(_Frozen):
    message: _Message
    finish_reason: str | None = None


class _Completion(_Frozen):
    choices: tuple[_Choice, ...]
    usage: _Usage


class _StreamChoice(_Frozen):
    delta: _Message
    finish_reason: str | None = None


class _Chunk(_Frozen):
    choices: tuple[_StreamChoice, ...] = ()
    usage: _Usage | None = None


class _ConverseMessage(_Frozen):
    role: str
    content: tuple[dict[str, JsonValue], ...]


class _ConverseRequest(_Frozen):
    messages: tuple[_ConverseMessage, ...]


@dataclass(frozen=True, slots=True)
class _Answer:
    content: str
    reasoning: str
    thinking_blocks: tuple[dict[str, JsonValue], ...]
    tool_calls: tuple[tuple[str, JsonValue], ...]
    finish_reason: str | None
    usage: _Usage | None


_COST_MAP: Final = TypeAdapter(dict[str, _CostEntry])


def _fakes_structured_output_stream(model: str, entry: _CostEntry) -> bool:
    return (
        model.startswith("us.anthropic.claude")
        and entry.litellm_provider == "bedrock_converse"
        and entry.supports_native_structured_output is False
        and entry.supports_reasoning is True
    )


def _claude_without_native_structured_output() -> str:
    entries: Final = _COST_MAP.validate_json(
        (files("litellm") / "model_prices_and_context_window_backup.json").read_bytes()
    )
    return next(model for model, entry in entries.items() if _fakes_structured_output_stream(model, entry))


def _converse_reply(content: Sequence[JsonValue], stop_reason: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "output": {"message": {"role": "assistant", "content": list(content)}},
                "stopReason": stop_reason,
                "usage": _CONVERSE_USAGE,
                "additionalModelResponseFields": {
                    "usage": {"output_tokens_details": {"thinking_tokens": _THINKING_TOKENS}}
                },
                "metrics": {"latencyMs": 1},
            }
        ).encode()
    )


def _reasoned_json_answer(reasoning: str) -> list[JsonValue]:
    return [
        {"reasoningContent": {"reasoningText": {"text": reasoning, "signature": _SIGNATURE}}},
        {"reasoningContent": {"redactedContent": _CONVERSE_REDACTED}},
        {"toolUse": {"toolUseId": "tooluse_json_1", "name": "json_tool_call", "input": _ANSWER}},
    ]


def _is_replay(request: Request) -> bool:
    return any(message.role == "assistant" for message in _ConverseRequest.model_validate_json(request.body).messages)


def _converse_peer(reasoning: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert unquote(request.target).endswith("/converse"), request.target
        if _is_replay(request):
            return _converse_reply([{"text": "replayed"}], "end_turn")
        return _converse_reply(_reasoned_json_answer(reasoning), "tool_use")

    return respond


def _converse_targets(wire: Wire) -> list[str]:
    return [unquote(request.target).rsplit("/", 1)[-1] for request in wire.drain()]


def _claude_model(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(
        model=f"bedrock/converse/{_claude_without_native_structured_output()}",
        api_key=_TOKEN,
        aws_region_name="us-east-1",
        aws_bedrock_runtime_endpoint=wire.url,
    )


def _openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def _parsed_arguments(arguments: str | None) -> JsonValue:
    return _JSON_OBJECT.validate_json(arguments or "{}")


def _completion_answer(completion: _Completion) -> _Answer:
    choice: Final = completion.choices[0]
    return _Answer(
        content=choice.message.content or "",
        reasoning=choice.message.reasoning_content or "",
        thinking_blocks=choice.message.thinking_blocks or (),
        tool_calls=tuple(
            (call.function.name or "", _parsed_arguments(call.function.arguments))
            for call in choice.message.tool_calls or ()
        ),
        finish_reason=choice.finish_reason,
        usage=completion.usage,
    )


def _joined_call(calls: Sequence[_ToolCall], index: int) -> tuple[str, JsonValue]:
    pieces: Final = tuple(call.function for call in calls if call.index == index)
    return (
        "".join(piece.name or "" for piece in pieces),
        _parsed_arguments("".join(piece.arguments or "" for piece in pieces)),
    )


def _streamed_answer(chunks: Sequence[_Chunk]) -> _Answer:
    choices: Final = tuple(chain.from_iterable(chunk.choices for chunk in chunks))
    deltas: Final = tuple(choice.delta for choice in choices)
    calls: Final = tuple(chain.from_iterable(delta.tool_calls or () for delta in deltas))
    finishes: Final = tuple(choice.finish_reason for choice in choices if choice.finish_reason)
    usages: Final = tuple(chunk.usage for chunk in chunks if chunk.usage is not None)
    return _Answer(
        content="".join(delta.content or "" for delta in deltas),
        reasoning="".join(delta.reasoning_content or "" for delta in deltas),
        thinking_blocks=tuple(chain.from_iterable(delta.thinking_blocks or () for delta in deltas)),
        tool_calls=tuple(_joined_call(calls, index) for index in sorted({call.index for call in calls})),
        finish_reason=finishes[-1] if finishes else None,
        usage=usages[-1] if usages else None,
    )


def _chat(gateway: Gateway, body: Mapping[str, JsonValue]) -> _Answer:
    completion: Final = _openai_client(gateway).chat.completions.create(
        model=str(body["model"]),
        messages=[{"role": "user", "content": str(body["question"])}],
        extra_body={key: value for key, value in body.items() if key not in ("model", "question")},
    )
    return _completion_answer(_Completion.model_validate(completion.model_dump(mode="json")))


def _chat_stream(gateway: Gateway, body: Mapping[str, JsonValue]) -> _Answer:
    stream: Final = _openai_client(gateway).chat.completions.create(
        model=str(body["model"]),
        messages=[{"role": "user", "content": str(body["question"])}],
        stream=True,
        stream_options={"include_usage": True},
        extra_body={key: value for key, value in body.items() if key not in ("model", "question")},
    )
    return _streamed_answer(tuple(_Chunk.model_validate(chunk.model_dump(mode="json")) for chunk in stream))


def _unindexed(blocks: Sequence[dict[str, JsonValue]]) -> tuple[dict[str, JsonValue], ...]:
    return tuple({key: value for key, value in block.items() if key != "index"} for block in blocks)


def _json_question(model: str, marker: str, cache: Mapping[str, JsonValue] = _NO_CACHE) -> dict[str, JsonValue]:
    return {
        "model": model,
        "question": f"How many people live in Tokyo, in millions? {marker}",
        "response_format": _RESPONSE_FORMAT,
        "thinking": _THINKING,
        "max_tokens": 2048,
        "cache": dict(cache),
    }


def test_fake_streamed_json_schema_chat_answer_matches_the_non_streamed_answer(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    reasoning: Final = f"Tokyo proper has about 14 million people, marker {marker}."
    with wire_server(_converse_peer(reasoning)) as wire, gateway.scenario() as scenario:
        model: Final = _claude_model(scenario, wire)
        question: Final = _json_question(model, marker)

        answer: Final = _chat(gateway, question)
        streamed: Final = _chat_stream(gateway, question)

        assert _converse_targets(wire) == ["converse", "converse"]
        assert (answer.reasoning, answer.finish_reason, answer.tool_calls) == (reasoning, "stop", ())
        assert _JSON_OBJECT.validate_json(answer.content) == _ANSWER
        assert answer.thinking_blocks == (
            {"type": "thinking", "thinking": reasoning, "signature": _SIGNATURE},
            {"type": "redacted_thinking", "data": _NATIVE_REDACTED},
        )
        assert (streamed.reasoning, streamed.finish_reason, streamed.tool_calls) == (reasoning, "stop", ())
        assert _JSON_OBJECT.validate_json(streamed.content) == _ANSWER
        assert _unindexed(streamed.thinking_blocks) == answer.thinking_blocks
        assert streamed.usage == answer.usage
        assert answer.usage is not None and answer.usage.completion_tokens_details is not None
        assert answer.usage.completion_tokens_details.reasoning_tokens == _THINKING_TOKENS

        replayed: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "thinking": _THINKING,
                "max_tokens": 2048,
                "cache": _NO_CACHE,
                "messages": [
                    {"role": "user", "content": str(question["question"])},
                    {
                        "role": "assistant",
                        "content": streamed.content,
                        "thinking_blocks": list(streamed.thinking_blocks),
                    },
                    {"role": "user", "content": "And Osaka?"},
                ],
            },
        )
        assert replayed.status_code == 200, replayed.text
        sent: Final = _ConverseRequest.model_validate_json(wire.drain()[-1].body)
        assert next(message for message in sent.messages if message.role == "assistant").content == (
            {"reasoningContent": {"reasoningText": {"text": reasoning, "signature": _SIGNATURE}}},
            {"reasoningContent": {"redactedContent": _CONVERSE_REDACTED}},
            {"text": streamed.content},
        )


def _spend_rows(model: str, count: int) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(
            'SELECT spend, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)
        ),
        lambda found: len(found) >= count,
        seconds=70,
    )


def test_fake_streamed_cached_prompt_is_billed_like_the_non_streamed_answer(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_converse_peer(f"Billing check {marker}.")) as wire, gateway.scenario() as scenario:
        model: Final = _claude_model(scenario, wire)
        question: Final = _json_question(model, marker)
        _chat(gateway, question)
        _chat_stream(gateway, question)

        rows: Final = _spend_rows(model, 2)
        assert _converse_targets(wire) == ["converse", "converse"]
        assert len(rows) == 2, rows
        first, second = rows
        assert (first["prompt_tokens"], first["completion_tokens"]) == (
            second["prompt_tokens"],
            second["completion_tokens"],
        )
        assert float(str(first["spend"])) > 0
        assert math.isclose(float(str(second["spend"])), float(str(first["spend"])), rel_tol=1e-9), rows


def _cache_holds(needle: str) -> bool:
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as store:
        return any(needle.encode() in _bytes(store.get(key)) for key in store.scan_iter(count=1000, _type="STRING"))


def _bytes(value: object) -> bytes:
    return value if isinstance(value, bytes) else b""


def test_cached_fake_stream_replays_the_answer_without_reaching_bedrock(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    reasoning: Final = f"Cached reasoning {marker}."
    with wire_server(_converse_peer(reasoning)) as wire, gateway.scenario() as scenario:
        model: Final = _claude_model(scenario, wire)
        question: Final = _json_question(model, marker, cache={})

        first: Final = _chat_stream(gateway, question)
        assert first.reasoning == reasoning
        eventually(lambda: _cache_holds(marker), bool, seconds=30)
        assert _converse_targets(wire) == ["converse"]

        replay: Final = _chat_stream(gateway, question)

        assert _converse_targets(wire) == []
        assert (replay.content, replay.finish_reason, replay.tool_calls) == (first.content, "stop", ())
        assert _unindexed(replay.thinking_blocks) in ((), _unindexed(first.thinking_blocks))


def _responses_reasoning_and_answer(
    events: Sequence[dict[str, JsonValue]],
) -> tuple[list[tuple[str, str, JsonValue]], str, str]:
    lifecycle: Final = [
        (str(event["type"]), str(_JSON_OBJECT.validate_python(event["item"])["type"]), event["output_index"])
        for event in events
        if event["type"] in ("response.output_item.added", "response.output_item.done")
    ]
    summary: Final = "".join(
        str(event["delta"]) for event in events if event["type"] == "response.reasoning_summary_text.delta"
    )
    text: Final = "".join(str(event["delta"]) for event in events if event["type"] == "response.output_text.delta")
    return lifecycle, summary, text


_REASONING_THEN_MESSAGE: Final = [
    ("response.output_item.added", "reasoning", 0),
    ("response.output_item.done", "reasoning", 0),
    ("response.output_item.added", "message", 1),
    ("response.output_item.done", "message", 1),
]


def _raw_responses_events(gateway: Gateway, body: Mapping[str, JsonValue]) -> list[dict[str, JsonValue]]:
    with gateway.client.stream(
        "POST",
        "/v1/responses",
        json={**body, "stream": True, "store": False, "cache": _NO_CACHE},
        headers={"Authorization": f"Bearer {gateway.key}"},
    ) as response:
        assert response.status_code == 200, response.read()
        return [
            _JSON_OBJECT.validate_json(line.removeprefix("data: "))
            for line in response.iter_lines()
            if line.startswith("data: {")
        ]


def _completed_usage(events: Sequence[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    completed: Final = [event for event in events if event["type"] == "response.completed"]
    assert len(completed) == 1, [event["type"] for event in events]
    return _JSON_OBJECT.validate_python(_JSON_OBJECT.validate_python(completed[0]["response"])["usage"])


def test_fake_streamed_json_schema_responses_stream_closes_the_reasoning_before_the_answer(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    reasoning: Final = f"Responses reasoning {marker}."
    with wire_server(_converse_peer(reasoning)) as wire, gateway.scenario() as scenario:
        model: Final = _claude_model(scenario, wire)
        events: Final = _raw_responses_events(
            gateway,
            {
                "model": model,
                "input": f"How many people live in Tokyo, in millions? {marker}",
                "reasoning": {"effort": "high"},
                "max_output_tokens": 2048,
                "text": {"format": {"type": "json_schema", "name": "city_fact", "schema": _SCHEMA}},
            },
        )

        lifecycle, summary, text = _responses_reasoning_and_answer(events)
        assert _converse_targets(wire) == ["converse"]
        assert lifecycle == _REASONING_THEN_MESSAGE, [event["type"] for event in events]
        assert summary == reasoning
        assert _JSON_OBJECT.validate_json(text) == _ANSWER
        usage: Final = _completed_usage(events)
        assert _JSON_OBJECT.validate_python(usage["input_tokens_details"])["cached_tokens"] == 50
        assert _JSON_OBJECT.validate_python(usage["output_tokens_details"])["reasoning_tokens"] == _THINKING_TOKENS


def _anthropic_client(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)


def _block_fields(block: anthropic.types.ContentBlock) -> tuple[str, str, str]:
    match block.type:
        case "thinking":
            return (block.type, block.thinking, block.signature)
        case "redacted_thinking":
            return (block.type, block.data, "")
        case "text":
            return (block.type, block.text, "")
        case _:
            return (block.type, "", "")


def _message_blocks(message: anthropic.types.Message) -> list[tuple[str, str, str]]:
    return [_block_fields(block) for block in message.content]


def test_fake_streamed_json_schema_messages_stream_opens_one_block_per_thinking_block(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    reasoning: Final = f"Messages reasoning {marker}."
    with wire_server(_converse_peer(reasoning)) as wire, gateway.scenario() as scenario:
        model: Final = _claude_model(scenario, wire)
        with _anthropic_client(gateway).messages.stream(
            model=model,
            max_tokens=2048,
            thinking={"type": "enabled", "budget_tokens": 1024},
            messages=[{"role": "user", "content": f"How many people live in Tokyo, in millions? {marker}"}],
            extra_body={"output_format": {"type": "json_schema", "schema": _SCHEMA}, "cache": _NO_CACHE},
        ) as stream:
            final: Final = stream.get_final_message()

        assert _converse_targets(wire) == ["converse"]
        blocks: Final = _message_blocks(final)
        assert blocks[:2] == [
            ("thinking", reasoning, _SIGNATURE),
            ("redacted_thinking", _NATIVE_REDACTED, ""),
        ], blocks
        assert [block[0] for block in blocks[2:]] == ["text"], blocks
        assert _JSON_OBJECT.validate_json(blocks[2][1]) == _ANSWER
        assert final.stop_reason == "end_turn"


def _frame(event_type: str, payload: Mapping[str, JsonValue]) -> bytes:
    return _aws_event_frame(event_type, payload, "sc", "u")


def _converse_stream_with_thinking(reasoning_parts: Sequence[str], answer: str) -> Reply:
    return Reply(
        content_type=_EVENT_STREAM,
        body=b"".join(
            (
                _frame("messageStart", {"role": "assistant"}),
                *(
                    _frame(
                        "contentBlockDelta",
                        {"delta": {"reasoningContent": {"text": part}}, "contentBlockIndex": 0},
                    )
                    for part in reasoning_parts
                ),
                _frame(
                    "contentBlockDelta",
                    {"delta": {"reasoningContent": {"signature": _SIGNATURE}}, "contentBlockIndex": 0},
                ),
                _frame("contentBlockStop", {"contentBlockIndex": 0}),
                _frame("contentBlockDelta", {"delta": {"text": answer}, "contentBlockIndex": 1}),
                _frame("contentBlockStop", {"contentBlockIndex": 1}),
                _frame("messageStop", {"stopReason": "end_turn"}),
                _frame("metadata", {"usage": {"inputTokens": 11, "outputTokens": 9, "totalTokens": 20}}),
            )
        ),
    )


def test_native_converse_stream_thinking_reaches_the_anthropic_sdk_once_with_its_signature(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    parts: Final = ("Native thinking ", f"marker {marker}.")
    answer: Final = "It is sunny in Paris."

    def respond(request: Request) -> Reply:
        assert unquote(request.target).endswith("/converse-stream"), request.target
        return _converse_stream_with_thinking(parts, answer)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _claude_model(scenario, wire)
        with _anthropic_client(gateway).messages.stream(
            model=model,
            max_tokens=2048,
            thinking={"type": "enabled", "budget_tokens": 1024},
            messages=[{"role": "user", "content": f"Weather in Paris? {marker}"}],
            extra_body={"cache": _NO_CACHE},
        ) as stream:
            final: Final = stream.get_final_message()

        assert _converse_targets(wire) == ["converse-stream"]
        assert _message_blocks(final) == [("thinking", "".join(parts), _SIGNATURE), ("text", answer, "")]


def _llama_peer(content: list[JsonValue], stop_reason: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert unquote(request.target).endswith("/converse"), request.target
        return Reply(
            body=json.dumps(
                {
                    "output": {"message": {"role": "assistant", "content": content}},
                    "stopReason": stop_reason,
                    "usage": {"inputTokens": 175, "outputTokens": 23, "totalTokens": 198},
                    "metrics": {"latencyMs": 1},
                }
            ).encode()
        )

    return respond


_PARIS: Final[dict[str, JsonValue]] = {"city": "Paris"}


@pytest.mark.parametrize(
    "content, stop_reason, expected_calls",
    [
        pytest.param(
            [
                {"toolUse": {"toolUseId": "tooluse_weather", "name": "get_weather", "input": _PARIS}},
                {"toolUse": {"toolUseId": "tooluse_time", "name": "get_time", "input": _PARIS}},
            ],
            "tool_use",
            (("get_weather", _PARIS), ("get_time", _PARIS)),
            id="two-tool-use-blocks",
        ),
        pytest.param(
            [{"text": json.dumps({"type": "function", "name": "get_weather", "parameters": _PARIS})}],
            "end_turn",
            (("get_weather", _PARIS),),
            id="tool-call-written-as-text",
        ),
    ],
)
def test_fake_streamed_llama_tool_calls_match_the_non_streamed_answer(
    gateway: Gateway,
    content: list[JsonValue],
    stop_reason: str,
    expected_calls: tuple[tuple[str, JsonValue], ...],
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_llama_peer(content, stop_reason)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/converse/{_LLAMA_MODEL_ID}",
            api_key=_TOKEN,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
        )
        question: Final[dict[str, JsonValue]] = {
            "model": model,
            "question": f"Weather and time in Paris? {marker}",
            "tools": [_tool("get_weather"), _tool("get_time")],
            "cache": _NO_CACHE,
        }

        answer: Final = _chat(gateway, question)
        streamed: Final = _chat_stream(gateway, question)

        assert _converse_targets(wire) == ["converse", "converse"]
        assert (answer.tool_calls, answer.finish_reason) == (expected_calls, "tool_calls")
        assert (streamed.tool_calls, streamed.finish_reason, streamed.content) == (
            answer.tool_calls,
            "tool_calls",
            answer.content,
        )


def _azure_o_series_reply() -> bytes:
    return json.dumps(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": 1,
            "model": "o4-mini",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_weather",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": json.dumps(_PARIS)},
                            },
                            {
                                "id": "call_time",
                                "type": "function",
                                "function": {"name": "get_time", "arguments": json.dumps(_PARIS)},
                            },
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 85,
                "completion_tokens": 1955,
                "total_tokens": 2040,
                "prompt_tokens_details": {"cached_tokens": 64},
                "completion_tokens_details": {"reasoning_tokens": 1920},
            },
        }
    ).encode()


def test_azure_o_series_fake_stream_keeps_every_tool_call_and_token_detail(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(lambda _: Reply(body=_azure_o_series_reply())) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"azure/o_series/{_AZURE_DEPLOYMENT}",
            api_base=wire.url,
            api_key="synthetic-azure-key",
            api_version=_AZURE_API_VERSION,
        )
        streamed: Final = _chat_stream(
            gateway,
            {
                "model": model,
                "question": f"Weather and time in Paris? {marker}",
                "tools": [_tool("get_weather"), _tool("get_time")],
                "cache": _NO_CACHE,
            },
        )

        received: Final = wire.drain()
        assert [unquote(request.target).split("?")[0] for request in received] == [
            f"/openai/deployments/{_AZURE_DEPLOYMENT}/chat/completions"
        ]
        assert _JSON_OBJECT.validate_json(received[0].body).get("stream") is not True
        assert (streamed.tool_calls, streamed.finish_reason) == (
            (("get_weather", _PARIS), ("get_time", _PARIS)),
            "tool_calls",
        )
        assert streamed.usage is not None
        assert streamed.usage.prompt_tokens_details == _PromptDetails(cached_tokens=64)
        assert streamed.usage.completion_tokens_details == _CompletionDetails(reasoning_tokens=1920)


def _vllm_frame(identity: str, delta: Mapping[str, JsonValue], finish_reason: str | None = None) -> bytes:
    usage: Final = (
        {"usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42}} if finish_reason else {}
    )
    body: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": _VLLM_BACKEND,
        "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish_reason}],
        **usage,
    }
    return b"data: " + json.dumps(body).encode() + b"\n\n"


def _vllm_sse(deltas: Sequence[Mapping[str, JsonValue]], finish_reason: str) -> Reply:
    identity: Final = f"chatcmpl-audit-{uuid.uuid4().hex}"
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _vllm_frame(identity, {"role": "assistant", "content": ""}),
            *(_vllm_frame(identity, delta) for delta in deltas),
            _vllm_frame(identity, {}, finish_reason),
            b"data: [DONE]\n\n",
        ),
    )


def _is_discovery_probe(request: Request) -> bool:
    return (request.method, request.target) == _DISCOVERY_PROBE


@contextmanager
def _vllm_server(reply: Reply) -> Generator[Wire, None, None]:
    with wire_server(
        lambda request: Reply(body=b'{"object":"list","data":[]}') if _is_discovery_probe(request) else reply
    ) as wire:
        yield wire


def _streamed_chats(wire: Wire) -> list[JsonValue]:
    posted: Final = [request for request in wire.drain() if not _is_discovery_probe(request)]
    assert [(request.method, request.target) for request in posted] == [("POST", "/v1/chat/completions")]
    return [_JSON_OBJECT.validate_json(request.body).get("stream") for request in posted]


@pytest.mark.parametrize(
    "deltas, merged",
    [
        pytest.param(
            ({"reasoning_content": "R1 "}, {"reasoning_content": "R2 ", "content": "A1 "}, {"content": "A2"}),
            "<think>R1 R2 </think>A1 A2",
            id="answer-starts-in-the-last-reasoning-chunk",
        ),
        pytest.param(
            ({"reasoning_content": "R ", "content": "A "}, {"content": "B"}),
            "<think>R </think>A B",
            id="reasoning-and-answer-in-the-first-chunk",
        ),
        pytest.param(
            ({"reasoning_content": "R1 "}, {"reasoning_content": "R2 "}, {"content": "A1 "}, {"content": "A2"}),
            "<think>R1 R2 </think>A1 A2",
            id="reasoning-and-answer-in-separate-chunks",
        ),
    ],
)
def test_merged_reasoning_content_closes_the_think_tag_before_the_answer(
    gateway: Gateway, deltas: tuple[dict[str, JsonValue], ...], merged: str
) -> None:
    with _vllm_server(_vllm_sse(deltas, "stop")) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"hosted_vllm/{_VLLM_BACKEND}",
            api_base=wire.url + "/v1",
            api_key=_VLLM_KEY,
            merge_reasoning_content_in_choices=True,
        )
        streamed: Final = _chat_stream(
            gateway, {"model": model, "question": f"Hello {uuid.uuid4().hex}", "cache": _NO_CACHE}
        )

        assert _streamed_chats(wire) == [True]
        assert (streamed.content, streamed.reasoning, streamed.finish_reason) == (merged, "", "stop")


_VLLM_TOOL_CALL: Final[dict[str, JsonValue]] = {
    "tool_calls": [
        {
            "index": 0,
            "id": "call_bridge_weather",
            "type": "function",
            "function": {"name": "get_weather", "arguments": json.dumps(_PARIS)},
        }
    ]
}


@pytest.mark.parametrize(
    "answer_delta, finish_reason, answer_item",
    [
        pytest.param({"content": "It is sunny."}, "stop", "message", id="text"),
        pytest.param(_VLLM_TOOL_CALL, "tool_calls", "function_call", id="tool-call"),
    ],
)
def test_responses_stream_closes_the_reasoning_item_before_an_answer_from_the_same_chunk(
    gateway: Gateway, answer_delta: dict[str, JsonValue], finish_reason: str, answer_item: str
) -> None:
    marker: Final = uuid.uuid4().hex
    reasoning: Final = f"Bridge reasoning {marker}."
    reply: Final = _vllm_sse(({"reasoning_content": reasoning, **answer_delta},), finish_reason)
    with _vllm_server(reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"hosted_vllm/{_VLLM_BACKEND}",
            api_base=wire.url + "/v1",
            api_key=_VLLM_KEY,
            use_chat_completions_api=True,
        )
        events: Final = _raw_responses_events(
            gateway,
            {
                "model": model,
                "input": f"Weather in Paris? {marker}",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ],
            },
        )

        lifecycle, summary, text = _responses_reasoning_and_answer(events)
        assert _streamed_chats(wire) == [True]
        assert lifecycle == [
            ("response.output_item.added", "reasoning", 0),
            ("response.output_item.done", "reasoning", 0),
            ("response.output_item.added", answer_item, 1),
            ("response.output_item.done", answer_item, 1),
        ], [event["type"] for event in events]
        assert summary == reasoning
        assert text == str(answer_delta.get("content", ""))
