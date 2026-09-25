import base64
import json
import uuid
from pathlib import Path
from types import MappingProxyType
from typing import Final

import anthropic
import pytest
from integration._support.client import Gateway, Scenario
from integration._support.process import owned_proxy_process
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

EVERYTHING_MODEL: Final = "global.anthropic.claude-fable-5-1"
DISPLAY_ONLY_MODEL: Final = "us.anthropic.claude-sonnet-5"
TOKEN: Final = "synthetic-bedrock-bearer"
BODY: Final = TypeAdapter(dict[str, JsonValue])
MESSAGES: Final = TypeAdapter(list[dict[str, JsonValue]])
BETAS: Final = TypeAdapter(list[str])
EFFORT_BETA: Final = "mid-conversation-output-config-2026-07-01"
TOOL_CHANGES_BETA: Final = "mid-conversation-tool-changes-2026-07-01"
DISPLAY_BETA: Final = "thinking-display-updates-2026-08-18"
EXTENSION_BETAS: Final = frozenset({EFFORT_BETA, TOOL_CHANGES_BETA, DISPLAY_BETA})
# Bedrock InvokeModel in us-east-1 on 2026-09-25 accepted each extension only with its beta, and only on these models
BEDROCK_ACCEPTS: Final = MappingProxyType(
    {EVERYTHING_MODEL: EXTENSION_BETAS, DISPLAY_ONLY_MODEL: frozenset({DISPLAY_BETA})}
)
TOOL_ADDITION: Final = {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "mcp__linear__list_issues"}}
TERSE: Final = {"type": "text", "text": "Answer tersely."}
DEFERRED_TOOL: Final = {
    "name": "mcp__linear__list_issues",
    "description": "List Linear issues",
    "defer_loading": True,
    "input_schema": {"type": "object", "properties": {}},
}
UPDATES: Final = {"type": "adaptive", "display": "updates"}
EFFORT_ONLY: Final = {"role": "system", "content": [], "output_config": {"effort": "low"}}
REJECTED: Final = "Bedrock Invoke rejects the extension: "


def _invoke(model: str) -> str:
    return f"/model/{model}/invoke"


def _invoke_stream(model: str) -> str:
    return f"/model/{model}/invoke-with-response-stream"


def _claude_code_turns(tag: str) -> list[dict[str, JsonValue]]:
    return [
        {"role": "user", "content": f"hello {tag}"},
        {"role": "assistant", "content": "Hi! What can I do for you?"},
        EFFORT_ONLY,
        {"role": "user", "content": "What tools do you have for Linear?"},
        {"role": "system", "content": [TERSE, TOOL_ADDITION]},
    ]


def _first_tag(body: dict[str, JsonValue]) -> str:
    first: Final = MESSAGES.validate_python(body["messages"])[0]["content"]
    text: Final = first if isinstance(first, str) else MESSAGES.validate_python(first)[0]["text"]
    assert isinstance(text, str), first
    return text.rsplit(" ", 1)[-1]


def _extension_betas_used(body: dict[str, JsonValue]) -> frozenset[str]:
    thinking: Final = body.get("thinking")
    messages: Final = MESSAGES.validate_python(body["messages"])
    blocks: Final = [
        block for message in messages if isinstance(message["content"], list) for block in message["content"]
    ]
    return frozenset(
        beta
        for beta, used in (
            (DISPLAY_BETA, isinstance(thinking, dict) and thinking.get("display") == "updates"),
            (EFFORT_BETA, any("output_config" in message for message in messages)),
            (
                TOOL_CHANGES_BETA,
                any(
                    isinstance(block, dict) and block.get("type") in ("tool_addition", "tool_removal")
                    for block in blocks
                ),
            ),
        )
        if used
    )


def _message(tag: str, model: str) -> bytes:
    return json.dumps(
        {
            "id": f"msg_{tag}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": f"answer {tag}"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 11, "output_tokens": 4},
        }
    ).encode()


def _chunk(payload: dict[str, JsonValue]) -> bytes:
    encoded: Final = base64.b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()
    return _aws_event_frame("chunk", {"bytes": encoded}, "", "")


def _stream(tag: str, model: str) -> bytes:
    return (
        _chunk(
            {
                "type": "message_start",
                "message": {
                    "id": f"msg_{tag}",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 11, "output_tokens": 0},
                },
            }
        )
        + _chunk({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
        + _chunk({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": f"answer {tag}"}})
        + _chunk({"type": "content_block_stop", "index": 0})
        + _chunk({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}})
        + _chunk({"type": "message_stop"})
    )


def lenient_bedrock_peer(request: Request) -> Reply:
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    model: Final = request.target.split("/")[2]
    tag: Final = _first_tag(BODY.validate_python(json.loads(request.body)))
    if request.target == _invoke_stream(model):
        return Reply(content_type="application/vnd.amazon.eventstream", chunks=(_stream(tag, model),))
    assert request.target == _invoke(model), request.target
    return Reply(body=_message(tag, model))


def bedrock_peer(request: Request) -> Reply:
    body: Final = BODY.validate_python(json.loads(request.body))
    sent_betas: Final = frozenset(BETAS.validate_python(body.get("anthropic_beta") or []))
    accepted: Final = sent_betas & BEDROCK_ACCEPTS[request.target.split("/")[2]]
    rejected: Final = sorted(_extension_betas_used(body) - accepted)
    if rejected:
        return Reply(status=400, body=json.dumps({"message": REJECTED + ", ".join(rejected)}).encode())
    return lenient_bedrock_peer(request)


def _register(scenario: Scenario, api_base: str, model: str, *, drop_params: bool = False) -> str:
    return scenario.model(
        model=f"bedrock/invoke/{model}",
        api_key=TOKEN,
        aws_region_name="us-east-1",
        api_base=api_base,
        drop_params=drop_params,
    )


def _sent(wire_requests: tuple[Request, ...], target: str) -> dict[str, JsonValue]:
    assert len(wire_requests) == 1, [request.target for request in wire_requests]
    assert wire_requests[0].target == target, wire_requests[0].target
    return BODY.validate_python(json.loads(wire_requests[0].body))


def _betas(sent: dict[str, JsonValue]) -> frozenset[str]:
    return EXTENSION_BETAS & frozenset(BETAS.validate_python(sent.get("anthropic_beta") or []))


def _stream_text(response_text: str) -> str:
    events: Final = tuple(
        json.loads(line.removeprefix("data:")) for line in response_text.splitlines() if line.startswith("data:")
    )
    assert events[-1]["type"] == "message_stop", response_text
    return "".join(event["delta"]["text"] for event in events if event["type"] == "content_block_delta")


def test_extensions_the_model_supports_reach_bedrock_unchanged_with_their_betas(gateway: Gateway) -> None:
    tag: Final = uuid.uuid4().hex
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire.url, EVERYTHING_MODEL)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 300,
                "thinking": UPDATES,
                "tools": [DEFERRED_TOOL],
                "messages": _claude_code_turns(tag),
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["content"] == [{"type": "text", "text": f"answer {tag}"}], response.text
        sent: Final = _sent(wire.drain(), _invoke(EVERYTHING_MODEL))
        assert sent["messages"] == _claude_code_turns(tag), sent
        assert sent["thinking"] == UPDATES, sent
        assert _betas(sent) == EXTENSION_BETAS, sent


def test_extensions_the_model_supports_reach_bedrock_on_the_streaming_invoke_path(gateway: Gateway) -> None:
    tag: Final = uuid.uuid4().hex
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire.url, EVERYTHING_MODEL)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 300,
                "stream": True,
                "thinking": UPDATES,
                "messages": _claude_code_turns(tag),
            },
        )
        assert response.status_code == 200, response.text
        assert _stream_text(response.text) == f"answer {tag}", response.text
        sent: Final = _sent(wire.drain(), _invoke_stream(EVERYTHING_MODEL))
        assert sent["messages"] == _claude_code_turns(tag), sent
        assert _betas(sent) == EXTENSION_BETAS, sent


def test_unsupported_per_message_effort_is_refused_with_an_actionable_error_without_drop_params(
    gateway: Gateway,
) -> None:
    tag: Final = uuid.uuid4().hex
    turns: Final = _claude_code_turns(tag)[:4]
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire.url, DISPLAY_ONLY_MODEL)
        response: Final = gateway.request(
            "POST", "/v1/messages", {"model": model, "max_tokens": 300, "messages": turns}
        )
        assert response.status_code == 400, response.text
        assert "messages[2].output_config" in response.text and "drop_params" in response.text, response.text
        assert DISPLAY_ONLY_MODEL in response.text, response.text
        assert wire.drain() == (), "a refused request must not reach Bedrock"


def test_unsupported_per_message_effort_is_dropped_and_supported_display_kept_with_drop_params(
    gateway: Gateway,
) -> None:
    tag: Final = uuid.uuid4().hex
    turns: Final = _claude_code_turns(tag)[:4]
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire.url, DISPLAY_ONLY_MODEL, drop_params=True)
        client: Final = anthropic.Anthropic(
            api_key=gateway.key, base_url=str(gateway.client.base_url).rstrip("/"), max_retries=0
        )
        message: Final = client.messages.create(
            model=model,
            max_tokens=300,
            thinking=UPDATES,  # pyright: ignore[reportArgumentType]  # Claude Code sends this shape; the SDK types lag
            messages=turns,  # pyright: ignore[reportArgumentType]  # effort-only system messages postdate the SDK types
        )
        assert message.role == "assistant" and message.id == f"msg_{tag}", message
        sent: Final = _sent(wire.drain(), _invoke(DISPLAY_ONLY_MODEL))
        assert sent["messages"] == [turns[0], turns[1], turns[3]], sent
        assert sent["thinking"] == UPDATES, sent
        assert _betas(sent) == {DISPLAY_BETA}, sent


def test_unsupported_tool_changes_are_refused_with_an_actionable_error_without_modify_params(
    gateway: Gateway,
) -> None:
    tag: Final = uuid.uuid4().hex
    turns: Final = [*_claude_code_turns(tag)[:2], *_claude_code_turns(tag)[3:]]
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire.url, DISPLAY_ONLY_MODEL, drop_params=True)
        response: Final = gateway.request(
            "POST", "/v1/messages", {"model": model, "max_tokens": 300, "messages": turns}
        )
        assert response.status_code == 400, response.text
        assert "messages[3].content[1] (type 'tool_addition')" in response.text, response.text
        assert "modify_params" in response.text, response.text
        assert wire.drain() == (), "a refused request must not reach Bedrock"


def test_every_unsupported_extension_is_removed_on_the_streaming_invoke_path_with_both_opt_ins(
    gateway: Gateway, tmp_path: Path
) -> None:
    tag: Final = uuid.uuid4().hex
    turns: Final = [*_claude_code_turns(tag), {"role": "system", "content": [TOOL_ADDITION]}]
    with (
        wire_server(bedrock_peer) as wire,
        owned_proxy_process(gateway, tmp_path, {"LITELLM_MODIFY_PARAMS": "True"}) as owned,
        owned.gateway.scenario() as scenario,
    ):
        model: Final = _register(scenario, wire.url, DISPLAY_ONLY_MODEL, drop_params=True)
        response: Final = owned.gateway.request(
            "POST",
            "/v1/messages",
            {"model": model, "max_tokens": 300, "stream": True, "thinking": UPDATES, "messages": turns},
        )
        assert response.status_code == 200, response.text
        assert _stream_text(response.text) == f"answer {tag}", response.text
        sent: Final = _sent(wire.drain(), _invoke_stream(DISPLAY_ONLY_MODEL))
        assert sent["messages"] == [turns[0], turns[1], turns[3], {"role": "system", "content": [TERSE]}], sent
        assert sent["thinking"] == UPDATES, sent
        assert _betas(sent) == {DISPLAY_BETA}, sent


@pytest.mark.parametrize("display", ["summarized", "omitted", None])
def test_other_thinking_displays_reach_bedrock_unchanged_without_the_updates_beta(
    gateway: Gateway, display: str | None
) -> None:
    tag: Final = uuid.uuid4().hex
    thinking: Final = {"type": "adaptive", **({"display": display} if display is not None else {})}
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire.url, DISPLAY_ONLY_MODEL)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {"model": model, "max_tokens": 300, "thinking": thinking, "messages": [{"role": "user", "content": tag}]},
        )
        assert response.status_code == 200, response.text
        sent: Final = _sent(wire.drain(), _invoke(DISPLAY_ONLY_MODEL))
        assert sent["thinking"] == thinking, sent
        assert _betas(sent) == frozenset(), sent


@pytest.mark.parametrize(
    "output_config", [7, ["high"], "", "x" * 5000], ids=["int", "list", "empty_string", "five_kb_string"]
)
def test_malformed_per_message_output_config_is_dropped_on_every_message(
    gateway: Gateway, output_config: JsonValue
) -> None:
    tag: Final = uuid.uuid4().hex
    turns: Final = [
        {"role": "user", "content": f"read {tag}", "output_config": output_config},
        {"role": "assistant", "content": [TERSE], "output_config": output_config},
        {"role": "user", "content": "continue", "output_config": output_config},
    ]
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire.url, DISPLAY_ONLY_MODEL, drop_params=True)
        response: Final = gateway.request(
            "POST", "/v1/messages", {"model": model, "max_tokens": 300, "messages": turns}
        )
        assert response.status_code == 200, response.text
        sent: Final = _sent(wire.drain(), _invoke(DISPLAY_ONLY_MODEL))
        assert sent["messages"] == [{k: v for k, v in turn.items() if k != "output_config"} for turn in turns], sent


def test_non_string_thinking_display_and_bare_string_blocks_are_forwarded_as_sent(gateway: Gateway) -> None:
    tag: Final = uuid.uuid4().hex
    turns: Final = [
        {"role": "user", "content": f"read {tag}"},
        {"role": "assistant", "content": ["ok"]},
        {"role": "user", "content": "continue"},
    ]
    with wire_server(lenient_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire.url, DISPLAY_ONLY_MODEL)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {"model": model, "max_tokens": 300, "thinking": {"type": "adaptive", "display": 7}, "messages": turns},
        )
        assert response.status_code == 200, response.text
        sent: Final = _sent(wire.drain(), _invoke(DISPLAY_ONLY_MODEL))
        assert sent["thinking"] == {"type": "adaptive", "display": 7}, sent
        assert sent["messages"] == turns, sent


def test_anthropic_direct_deployment_forwards_the_extensions_verbatim(gateway: Gateway) -> None:
    tag: Final = uuid.uuid4().hex

    def anthropic_peer(request: Request) -> Reply:
        assert request.target == "/v1/messages", request.target
        return Reply(body=_message(tag, "claude-sonnet-5"))

    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-5", api_base=wire.url, api_key="synthetic-anthropic-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {"model": model, "max_tokens": 300, "thinking": UPDATES, "messages": _claude_code_turns(tag)},
        )
        assert response.status_code == 200, response.text
        sent: Final = _sent(wire.drain(), "/v1/messages")
        assert sent["messages"] == _claude_code_turns(tag), sent
        assert sent["thinking"] == UPDATES, sent


def test_bedrock_invoke_chat_completions_are_untouched_by_the_messages_sanitizer(gateway: Gateway) -> None:
    tag: Final = uuid.uuid4().hex
    with wire_server(bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _register(scenario, wire.url, DISPLAY_ONLY_MODEL)
        body: Final = gateway.chat(model, text=f"chat control {tag}")
        assert body["choices"][0]["message"]["content"] == f"answer {tag}", body
        sent: Final = _sent(wire.drain(), _invoke(DISPLAY_ONLY_MODEL))
        assert sent["messages"] == [{"role": "user", "content": [{"type": "text", "text": f"chat control {tag}"}]}], (
            sent
        )
