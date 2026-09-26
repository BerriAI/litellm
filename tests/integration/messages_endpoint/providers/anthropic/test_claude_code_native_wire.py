import json
import uuid
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_API_KEY: Final = "synthetic-anthropic-key"
_MODEL: Final = "claude-sonnet-4-5"
_FIXTURE: Final = Path(__file__).with_name("claude_code_request.json")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_CLI_BETA: Final = (
    "claude-code-20250219,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,"
    "context-management-2025-06-27,prompt-caching-scope-2026-01-05"
)


def _sse_frame(event: str, data: JsonValue) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _message_stream(identity: str) -> tuple[bytes, ...]:
    return (
        _sse_frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "model": _MODEL,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 12, "output_tokens": 1},
                },
            },
        ),
        _sse_frame(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        _sse_frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "PONG"}},
        ),
        _sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse_frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 4},
            },
        ),
        _sse_frame("message_stop", {"type": "message_stop"}),
    )


def sse_events(text: str) -> tuple[tuple[str, dict[str, object]], ...]:
    frames: Final = tuple(frame for frame in text.split("\n\n") if frame.strip())
    return tuple(
        (
            next(line.removeprefix("event: ") for line in frame.splitlines() if line.startswith("event: ")),
            json.loads(next(line.removeprefix("data: ") for line in frame.splitlines() if line.startswith("data: "))),
        )
        for frame in frames
    )


@pytest.mark.covers("other.provider_wire.anthropic.claude_code_native_request_survives_and_streams_back")
def test_claude_code_streaming_request_reaches_anthropic_intact_and_streams_back(gateway: Gateway) -> None:
    identity: Final = f"msg_cc_{uuid.uuid4().hex}"
    fixture: Final = _JSON_OBJECT.validate_json(_FIXTURE.read_bytes())
    cli_beta: Final = frozenset(_CLI_BETA.split(","))
    content: Final = list(object_value(fixture["messages"][0])["content"])
    content[0] = {**content[0], "text": f"cache-bust-{uuid.uuid4().hex}"}
    request_body: Final = {**fixture, "messages": [{"role": "user", "content": content}]}

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/messages", request.target
        assert request.headers["x-api-key"] == _API_KEY
        assert request.headers["anthropic-version"] == "2023-06-01"
        upstream_beta: Final = frozenset(request.headers.get("anthropic-beta", "").split(","))
        assert cli_beta <= upstream_beta, request.headers.get("anthropic-beta")
        body: Final = _JSON_OBJECT.validate_json(request.body)
        expected: Final = {**request_body, "model": _MODEL}
        assert body == expected, {
            key: (expected.get(key), body.get(key))
            for key in expected.keys() | body.keys()
            if expected.get(key) != body.get(key)
        }
        return Reply(content_type="text/event-stream", chunks=_message_stream(identity))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**request_body, "model": model},
            params={"beta": "true"},
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "user-agent": "claude-cli/2.1.283 (external, sdk-cli)",
                "x-claude-code-session-id": "00000000-0000-4000-8000-000000000000",
                "x-stainless-arch": "x64",
                "x-stainless-lang": "js",
                "x-stainless-os": "Linux",
                "x-stainless-package-version": "0.112.1",
                "x-stainless-retry-count": "0",
                "x-stainless-runtime": "node",
                "x-stainless-runtime-version": "v26.3.0",
                "x-stainless-timeout": "600",
                "anthropic-beta": _CLI_BETA,
                "anthropic-dangerous-direct-browser-access": "true",
                "anthropic-version": "2023-06-01",
                "x-app": "cli",
                "x-api-key": gateway.key,
            },
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream"), dict(response.headers)
        events: Final = sse_events(response.text)
        assert [event for event, _ in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert events[2][1]["delta"] == {"type": "text_delta", "text": "PONG"}
        assert events[4][1]["delta"]["stop_reason"] == "end_turn"
        assert events[4][1]["usage"]["output_tokens"] == 4
        assert len(wire.drain()) == 1
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (identity,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert rows[0]["prompt_tokens"] == 12 and rows[0]["completion_tokens"] == 4
