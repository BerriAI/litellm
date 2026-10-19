import base64
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
# Synthetic stand-in shaped like the `data` Bedrock Invoke returned for us.anthropic.claude-sonnet-4-6 in
# us-west-2 on 2026-10-14: base64 of a protobuf message, so it opens with "E" + the 0x12 field tag. The
# real value embeds the caller's AWS account id, so it is not committed.
NATIVE_REDACTED: Final = "EmwKAhgBEgy3va3pzix/LafPsn4aDFIT2Xlxh0L5L8rLVyIwxtE3rAFBa8cr3qpP"
CONVERSE_REDACTED: Final = base64.b64encode(NATIVE_REDACTED.encode()).decode()
OPENAI_ENCRYPTED: Final = "litellm_encrypted_reasoning:gAAAAABsynthetic-openai-reasoning=="
ANTHROPIC_MODEL: Final = "claude-sonnet-4-6"
ANTHROPIC_REPLY: Final[dict[str, JsonValue]] = {
    "id": "msg_redacted_replay",
    "type": "message",
    "role": "assistant",
    "model": ANTHROPIC_MODEL,
    "content": [{"type": "text", "text": "redacted thinking replay control"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 11, "output_tokens": 4},
}
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


class _AnthropicMessage(_Frozen):
    role: str
    content: str | tuple[dict[str, JsonValue], ...]


class _AnthropicRequest(_Frozen):
    messages: tuple[_AnthropicMessage, ...]


def _first_turn_content(with_tool: bool) -> list[JsonValue]:
    return [
        {"reasoningContent": {"redactedContent": CONVERSE_REDACTED}},
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
                {"delta": {"reasoningContent": {"redactedContent": CONVERSE_REDACTED}}, "contentBlockIndex": 0},
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
        if request.target == "/v1/messages":
            return Reply(body=json.dumps(ANTHROPIC_REPLY).encode())
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
        streamed_position: Final[dict[str, JsonValue]] = {"index": 0} if stream else {}
        assert first.thinking_blocks == ({"type": "redacted_thinking", "data": NATIVE_REDACTED, **streamed_position},)

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
            {"reasoningContent": {"redactedContent": CONVERSE_REDACTED}},
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
        assert first.thinking_blocks == ({"type": "redacted_thinking", "data": NATIVE_REDACTED},)
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
            {"reasoningContent": {"redactedContent": CONVERSE_REDACTED}},
            {"text": FIRST_ANSWER},
            {"toolUse": TOOL_USE},
        )


_CLAUDE: Final[dict[str, JsonValue]] = {"model": f"bedrock/converse/{CLAUDE_MODEL_ID}"}


def _redacted(data: str) -> dict[str, JsonValue]:
    return {"type": "redacted_thinking", "data": data}


@pytest.mark.parametrize(
    "deployment, history_block, sent_data",
    [
        pytest.param({"model": f"bedrock/converse/{NOVA_MODEL_ID}"}, _redacted(NATIVE_REDACTED), None, id="nova"),
        pytest.param(
            {"model": f"bedrock/converse/{APPLICATION_PROFILE_ARN}"},
            _redacted(NATIVE_REDACTED),
            None,
            id="application-profile-arn-without-base-model",
        ),
        pytest.param(
            {
                "model": f"bedrock/converse/{APPLICATION_PROFILE_ARN}",
                "model_info": {"base_model": f"bedrock/{CLAUDE_MODEL_ID}"},
            },
            _redacted(NATIVE_REDACTED),
            CONVERSE_REDACTED,
            id="application-profile-arn-with-claude-base-model",
        ),
        pytest.param(
            {**_CLAUDE, "model_id": APPLICATION_PROFILE_ARN},
            _redacted(NATIVE_REDACTED),
            CONVERSE_REDACTED,
            id="claude-model-id-override",
        ),
        pytest.param(_CLAUDE, _redacted(NATIVE_REDACTED), CONVERSE_REDACTED, id="claude-native-history"),
        pytest.param(
            _CLAUDE, _redacted(CONVERSE_REDACTED), CONVERSE_REDACTED, id="claude-history-saved-with-converse-wrapping"
        ),
        pytest.param(_CLAUDE, _redacted(OPENAI_ENCRYPTED), None, id="claude-redacted-openai-encrypted-reasoning"),
        pytest.param(
            _CLAUDE,
            {"type": "thinking", "thinking": "Checking the forecast.", "signature": OPENAI_ENCRYPTED},
            None,
            id="claude-thinking-signed-with-openai-encrypted-reasoning",
        ),
    ],
)
def test_redacted_history_replayed_to_a_converse_target_carries_redacted_content_only_when_it_is_claude(
    gateway: Gateway, deployment: dict[str, JsonValue], history_block: dict[str, JsonValue], sent_data: str | None
) -> None:
    """A Router fallback or model switch replays a redacted history to a different Converse target. Non-Claude
    targets 400 on redactedContent, so only a target known to be Claude gets the block, always as base64 of
    Claude's own data. Claude cannot read OpenAI encrypted reasoning, so that block never reaches it"""
    model_info: Final = deployment.get("model_info")
    parameters: Final = {key: value for key, value in deployment.items() if key != "model_info"}
    with wire_server(_bedrock_peer()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model_info=model_info if isinstance(model_info, dict) else None,
            api_key=TOKEN,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
            **parameters,
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
                        "thinking_blocks": [history_block],
                    },
                    {"role": "user", "content": FOLLOW_UP},
                ],
            },
        )

        assert reply.content == FINAL_ANSWER
        redacted: Final = () if sent_data is None else ({"reasoningContent": {"redactedContent": sent_data}},)
        assert _replayed_assistant_turn(wire).content == (*redacted, {"text": FIRST_ANSWER})


@pytest.mark.parametrize("legacy_history", [False, True], ids=["converse-reply", "history-saved-with-converse-wrapping"])
def test_converse_redacted_thinking_replays_to_the_anthropic_api_as_claude_native_data(
    gateway: Gateway, legacy_history: bool
) -> None:
    """A fallback from Claude on Converse to the Anthropic API (or Bedrock Invoke, which builds the same
    messages) must hand Claude its own data back, or Anthropic 400s with "Invalid `data` in `redacted_thinking`"
    """
    with wire_server(_bedrock_peer()) as wire, gateway.scenario() as scenario:
        converse: Final = scenario.model(
            model=f"bedrock/converse/{CLAUDE_MODEL_ID}",
            api_key=TOKEN,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
        )
        anthropic: Final = scenario.model(
            model=f"anthropic/{ANTHROPIC_MODEL}", api_base=wire.url, api_key="synthetic-anthropic-key"
        )
        question: Final[dict[str, JsonValue]] = {"role": "user", "content": QUESTION}
        first: Final = _chat(
            gateway, {"model": converse, "messages": [question], "thinking": THINKING, "max_tokens": 2048}
        )
        history: Final = (_redacted(CONVERSE_REDACTED),) if legacy_history else first.thinking_blocks

        second: Final = _chat(
            gateway,
            {
                "model": anthropic,
                "thinking": THINKING,
                "max_tokens": 2048,
                "messages": [
                    question,
                    {"role": "assistant", "content": first.content, "thinking_blocks": list(history)},
                    {"role": "user", "content": FOLLOW_UP},
                ],
            },
        )

        assert second.content == FINAL_ANSWER
        sent: Final = _AnthropicRequest.model_validate_json(wire.drain()[-1].body)
        assert sent.messages[1].content == (
            {"type": "redacted_thinking", "data": NATIVE_REDACTED},
            {"type": "text", "text": FIRST_ANSWER},
        )
