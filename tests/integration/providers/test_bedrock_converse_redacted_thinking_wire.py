import json
from collections.abc import Callable, Mapping
from itertools import chain
from typing import Final
from urllib.parse import unquote

import pytest
from integration._support.client import Gateway
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue

CLAUDE_MODEL_ID: Final = "us.anthropic.claude-sonnet-4-6"
NOVA_MODEL_ID: Final = "us.amazon.nova-pro-v1:0"
APPLICATION_PROFILE_ARN: Final = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"
TOKEN: Final = "synthetic-bedrock-bearer"
REDACTED: Final = "EmwKAhgBEgy3va3pzix/LafPsn4aDFIT2Xlxh0L5L8rLVyIwxtE3rAFBa8cr3qpP"
FIRST_ANSWER: Final = "I looked into it."
FINAL_ANSWER: Final = "redacted thinking replay control"
QUESTION: Final = "What's the weather in Paris?"
FOLLOW_UP: Final = "Reply with the word ok."
THINKING: Final[dict[str, JsonValue]] = {"type": "enabled", "budget_tokens": 1024}
WEATHER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
TOOL_USE: Final[dict[str, JsonValue]] = {
    "toolUseId": "tooluse_weather_1",
    "name": "get_weather",
    "input": {"city": "Paris"},
}
USAGE: Final[dict[str, JsonValue]] = {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15}
EVENT_STREAM: Final = "application/vnd.amazon.eventstream"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class _Message(_Frozen):
    content: str | None = None
    thinking_blocks: tuple[dict[str, JsonValue], ...] = ()
    tool_calls: tuple[dict[str, JsonValue], ...] = ()


class _Choice(_Frozen):
    message: _Message


class _Completion(_Frozen):
    choices: tuple[_Choice, ...]


class _Delta(_Frozen):
    content: str | None = None
    thinking_blocks: tuple[dict[str, JsonValue], ...] | None = None


class _StreamChoice(_Frozen):
    delta: _Delta


class _Chunk(_Frozen):
    choices: tuple[_StreamChoice, ...] = ()


class _ConverseMessage(_Frozen):
    role: str
    content: tuple[dict[str, JsonValue], ...]


class _ConverseRequest(_Frozen):
    messages: tuple[_ConverseMessage, ...]


def _first_turn_content(with_tool: bool) -> list[JsonValue]:
    return [
        {"reasoningContent": {"redactedContent": REDACTED}},
        {"text": FIRST_ANSWER},
        *([{"toolUse": TOOL_USE}] if with_tool else []),
    ]


def _converse_reply(content: list[JsonValue], stop_reason: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "output": {"message": {"role": "assistant", "content": content}},
                "stopReason": stop_reason,
                "usage": USAGE,
                "metrics": {"latencyMs": 1},
            }
        ).encode()
    )


def _frame(event_type: str, payload: Mapping[str, JsonValue]) -> bytes:
    return _aws_event_frame(event_type, payload, "sc", "u")


def _stream_reply(*, first_turn: bool) -> Reply:
    reasoning: Final = (
        (
            _frame(
                "contentBlockDelta",
                {"delta": {"reasoningContent": {"redactedContent": REDACTED}}, "contentBlockIndex": 0},
            ),
            _frame("contentBlockStop", {"contentBlockIndex": 0}),
        )
        if first_turn
        else ()
    )
    text: Final = FIRST_ANSWER if first_turn else FINAL_ANSWER
    return Reply(
        content_type=EVENT_STREAM,
        body=b"".join(
            (
                _frame("messageStart", {"role": "assistant"}),
                *reasoning,
                _frame("contentBlockDelta", {"delta": {"text": text}, "contentBlockIndex": 1}),
                _frame("contentBlockStop", {"contentBlockIndex": 1}),
                _frame("messageStop", {"stopReason": "end_turn"}),
                _frame("metadata", {"usage": USAGE}),
            )
        ),
    )


def _is_replay(request: Request) -> bool:
    return any(message.role == "assistant" for message in _ConverseRequest.model_validate_json(request.body).messages)


def _bedrock_peer(*, with_tool: bool = False) -> Callable[[Request], Reply]:
    """Turn 1 answers with a redacted reasoning block, the way Bedrock does for a safety-redacted Claude turn.
    Turn 2, the one replaying that history, gets a plain answer."""

    def respond(request: Request) -> Reply:
        if unquote(request.target).endswith("/converse-stream"):
            return _stream_reply(first_turn=not _is_replay(request))
        if _is_replay(request):
            return _converse_reply([{"text": FINAL_ANSWER}], "end_turn")
        return _converse_reply(_first_turn_content(with_tool), "tool_use" if with_tool else "end_turn")

    return respond


def _chat(gateway: Gateway, body: Mapping[str, JsonValue]) -> _Message:
    response: Final = gateway.request("POST", "/v1/chat/completions", {**body, "cache": {"no-cache": True}})
    assert response.status_code == 200, response.text
    return _Completion.model_validate_json(response.content).choices[0].message


def _chat_stream(gateway: Gateway, body: Mapping[str, JsonValue]) -> _Message:
    with gateway.client.stream(
        "POST",
        "/v1/chat/completions",
        json={**body, "stream": True, "cache": {"no-cache": True}},
        headers={"Authorization": f"Bearer {gateway.key}"},
    ) as response:
        assert response.status_code == 200, response.read()
        payloads: Final = tuple(
            line[6:] for line in response.iter_lines() if line.startswith("data: ") and line != "data: [DONE]"
        )
    chunks: Final = tuple(_Chunk.model_validate_json(payload) for payload in payloads)
    deltas: Final = tuple(choice.delta for choice in chain.from_iterable(chunk.choices for chunk in chunks))
    return _Message(
        content="".join(delta.content or "" for delta in deltas),
        thinking_blocks=tuple(chain.from_iterable(delta.thinking_blocks or () for delta in deltas)),
    )


def _replayed_assistant_turn(wire: Wire) -> _ConverseMessage:
    received: Final = tuple(_ConverseRequest.model_validate_json(request.body) for request in wire.drain())
    return next(message for message in received[-1].messages if message.role == "assistant")


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
def test_claude_redacted_thinking_round_trips_back_to_converse_as_redacted_content(
    gateway: Gateway, stream: bool
) -> None:
    send: Final = _chat_stream if stream else _chat
    with wire_server(_bedrock_peer()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/converse/{CLAUDE_MODEL_ID}",
            api_key=TOKEN,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
        )
        question: Final[dict[str, JsonValue]] = {"role": "user", "content": QUESTION}
        first: Final = send(gateway, {"model": model, "messages": [question], "thinking": THINKING, "max_tokens": 2048})
        assert first.thinking_blocks == ({"type": "redacted_thinking", "data": REDACTED},)

        second: Final = send(
            gateway,
            {
                "model": model,
                "thinking": THINKING,
                "max_tokens": 2048,
                "messages": [
                    question,
                    {"role": "assistant", "content": first.content, "thinking_blocks": list(first.thinking_blocks)},
                    {"role": "user", "content": FOLLOW_UP},
                ],
            },
        )

        assert second.content == FINAL_ANSWER
        assert _replayed_assistant_turn(wire).content == (
            {"reasoningContent": {"redactedContent": REDACTED}},
            {"text": FIRST_ANSWER},
        )


def test_claude_redacted_thinking_before_a_tool_call_is_replayed_with_the_tool_result(gateway: Gateway) -> None:
    with wire_server(_bedrock_peer(with_tool=True)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/converse/{CLAUDE_MODEL_ID}",
            api_key=TOKEN,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
        )
        question: Final[dict[str, JsonValue]] = {"role": "user", "content": QUESTION}
        first: Final = _chat(
            gateway,
            {"model": model, "messages": [question], "tools": [WEATHER_TOOL], "thinking": THINKING, "max_tokens": 2048},
        )
        assert first.thinking_blocks == ({"type": "redacted_thinking", "data": REDACTED},)
        assert len(first.tool_calls) == 1

        second: Final = _chat(
            gateway,
            {
                "model": model,
                "tools": [WEATHER_TOOL],
                "thinking": THINKING,
                "max_tokens": 2048,
                "messages": [
                    question,
                    {
                        "role": "assistant",
                        "content": first.content,
                        "thinking_blocks": list(first.thinking_blocks),
                        "tool_calls": list(first.tool_calls),
                    },
                    {"role": "tool", "tool_call_id": str(TOOL_USE["toolUseId"]), "content": "18C, sunny"},
                ],
            },
        )

        assert second.content == FINAL_ANSWER
        assert _replayed_assistant_turn(wire).content == (
            {"reasoningContent": {"redactedContent": REDACTED}},
            {"text": FIRST_ANSWER},
            {"toolUse": TOOL_USE},
        )


@pytest.mark.parametrize(
    "target, model_info, replays_redacted",
    [
        pytest.param(NOVA_MODEL_ID, {}, False, id="nova"),
        pytest.param(APPLICATION_PROFILE_ARN, {}, False, id="application-profile-arn-without-base-model"),
        pytest.param(
            APPLICATION_PROFILE_ARN,
            {"base_model": f"bedrock/{CLAUDE_MODEL_ID}"},
            True,
            id="application-profile-arn-with-claude-base-model",
        ),
    ],
)
def test_claude_history_replayed_to_another_converse_target_only_carries_redacted_content_when_it_is_claude(
    gateway: Gateway, target: str, model_info: dict[str, JsonValue], replays_redacted: bool
) -> None:
    """A Router fallback or model switch replays a Claude history to a different Converse target. Nova, Llama,
    DeepSeek and gpt-oss 400 on redactedContent, so only a target known to be Claude gets the block"""
    with wire_server(_bedrock_peer()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"bedrock/converse/{target}",
            api_key=TOKEN,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
            model_info=model_info,
        )

        reply: Final = _chat(
            gateway,
            {
                "model": model,
                "max_tokens": 256,
                "messages": [
                    {"role": "user", "content": QUESTION},
                    {
                        "role": "assistant",
                        "content": FIRST_ANSWER,
                        "thinking_blocks": [{"type": "redacted_thinking", "data": REDACTED}],
                    },
                    {"role": "user", "content": FOLLOW_UP},
                ],
            },
        )

        assert reply.content == FINAL_ANSWER
        redacted: Final = ({"reasoningContent": {"redactedContent": REDACTED}},) if replays_redacted else ()
        assert _replayed_assistant_turn(wire).content == (*redacted, {"text": FIRST_ANSWER})
