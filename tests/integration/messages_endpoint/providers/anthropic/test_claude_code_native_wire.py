import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

_API_KEY: Final = "synthetic-anthropic-key"
_MODEL: Final = "claude-sonnet-4-5"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_CLI_BETA: Final = (
    "claude-code-20250219,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,"
    "context-management-2025-06-27,prompt-caching-scope-2026-01-05"
)
_CACHE: Final = {"type": "ephemeral"}


def _schema(properties: JsonValue, required: tuple[str, ...]) -> dict[str, JsonValue]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _field(description: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"description": description, **extra}


def _tools() -> tuple[dict[str, JsonValue], ...]:
    _MAX: Final = 9007199254740991
    return (
        {
            "name": "Bash",
            "description": "Executes a given bash command and returns its output.",
            "input_schema": _schema(
                {
                    "command": _field("The command to execute", type="string"),
                    "timeout": _field("Optional timeout in milliseconds (max 600000)", type="number"),
                    "description": _field(
                        "Clear, concise description of what this command does in active voice.", type="string"
                    ),
                    "run_in_background": _field("Set to true to run this command in the background.", type="boolean"),
                    "dangerouslyDisableSandbox": _field(
                        "Set this to true to dangerously override sandbox mode and run commands without sandboxing.",
                        type="boolean",
                    ),
                },
                ("command",),
            ),
        },
        {
            "name": "Read",
            "description": "Reads a file from the local filesystem.",
            "input_schema": _schema(
                {
                    "file_path": _field("The absolute path to the file to read", type="string"),
                    "offset": _field("The line number to start reading from.", type="integer", minimum=0, maximum=_MAX),
                    "limit": _field("The number of lines to read.", type="integer", exclusiveMinimum=0, maximum=_MAX),
                    "pages": _field('Page range for PDF files (e.g., "1-5", "3", "10-20").', type="string"),
                },
                ("file_path",),
            ),
        },
        {
            "name": "Edit",
            "description": "Performs exact string replacements in files.",
            "input_schema": _schema(
                {
                    "file_path": _field("The absolute path to the file to modify", type="string"),
                    "old_string": _field("The text to replace", type="string"),
                    "new_string": _field(
                        "The text to replace it with (must be different from old_string)", type="string"
                    ),
                    "replace_all": _field(
                        "Replace all occurrences of old_string (default false)", default=False, type="boolean"
                    ),
                },
                ("file_path", "old_string", "new_string"),
            ),
        },
        {
            "name": "Agent",
            "description": "Launch a new agent to handle complex, multi-step tasks.",
            "input_schema": _schema(
                {
                    "description": _field("A short (3-5 word) description of the task", type="string"),
                    "prompt": _field("The task for the agent to perform", type="string"),
                    "subagent_type": _field("The type of specialized agent to use for this task", type="string"),
                    "model": _field(
                        "Optional model override for this agent.",
                        type="string",
                        enum=["sonnet", "opus", "haiku", "fable"],
                    ),
                    "run_in_background": _field(
                        "Agents run in the background by default; you will be notified when one completes.",
                        type="boolean",
                    ),
                    "isolation": _field("Isolation mode.", type="string", enum=["worktree", "remote"]),
                },
                ("description", "prompt"),
            ),
        },
    )


def _claude_code_request(cache_bust: str) -> dict[str, JsonValue]:
    reminders: Final = (
        f"<system-reminder>\n{cache_bust}\n</system-reminder>",
        "<system-reminder>\nSynthetic model identity reminder.\n</system-reminder>",
        "<system-reminder>\nSynthetic agent types reminder.\n</system-reminder>",
        "<system-reminder>\nSynthetic skills reminder.\n</system-reminder>",
        "<system-reminder>\n<total_tokens>15000000 tokens left</total_tokens>\n</system-reminder>",
        "<system-reminder>\nSynthetic date reminder.\n</system-reminder>",
        "<system-reminder>\nSynthetic attribution reminder.\n</system-reminder>",
    )
    return {
        "model": "",
        "system": [
            {"type": "text", "text": "Synthetic billing header block from a Claude Code request."},
            {"type": "text", "text": "Synthetic agent identity system prompt.", "cache_control": _CACHE},
            {"type": "text", "text": "Synthetic interactive agent instructions.", "cache_control": _CACHE},
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    *[{"type": "text", "text": reminder} for reminder in reminders],
                    {
                        "type": "text",
                        "text": "Reply with exactly the word PONG",
                        "cache_control": _CACHE,
                    },
                ],
            }
        ],
        "tools": list(_tools()),
        "metadata": {
            "user_id": json.dumps(
                {
                    "device_id": "0" * 64,
                    "account_uuid": "",
                    "session_id": "00000000-0000-4000-8000-000000000000",
                }
            )
        },
        "max_tokens": 32000,
        "thinking": {"budget_tokens": 31999, "type": "enabled", "display": "omitted"},
        "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        "stream": True,
    }


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
    request_body: Final = _claude_code_request(f"cache-bust-{uuid.uuid4().hex}")
    cli_beta: Final = frozenset(_CLI_BETA.split(","))

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
