"""Shared Claude Code-shaped request builders and upstream stream fixtures for the /v1/messages contracts."""

import json
from collections.abc import Mapping
from typing import Final

from pydantic import JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
ANTHROPIC_API_KEY: Final = "synthetic-anthropic-key"
OPENAI_API_KEY: Final = "synthetic-openai-key"
OPENAI_BACKEND: Final = "gpt-5.4-mini"
SONNET: Final = "claude-sonnet-4-5"
FABLE: Final = "claude-fable-5-1"
OPUS: Final = "claude-opus-5-5"
CLI_BETA: Final = (
    "claude-code-20250219,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,"
    "context-management-2025-06-27,prompt-caching-scope-2026-01-05"
)
FRONTIER_CLI_BETA: Final = (
    f"{CLI_BETA},mid-conversation-system-2026-04-07,per-turn-control-2026-07-01,"
    "mid-conversation-tool-changes-2026-07-01,effort-2025-11-24"
)
CACHE: Final = {"type": "ephemeral"}
CONTEXT_MANAGEMENT: Final = {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}
THINKING_BUDGET: Final = {"budget_tokens": 31999, "type": "enabled", "display": "omitted"}
THINKING_ADAPTIVE: Final = {"type": "adaptive", "display": "omitted"}
METADATA_USER_ID: Final = json.dumps(
    {
        "device_id": "0" * 64,
        "account_uuid": "",
        "session_id": "00000000-0000-4000-8000-000000000000",
    }
)


def schema(properties: JsonValue, required: tuple[str, ...]) -> dict[str, JsonValue]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def field(description: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"description": description, **extra}


def tools() -> tuple[dict[str, JsonValue], ...]:
    _MAX: Final = 9007199254740991
    return (
        {
            "name": "Bash",
            "description": "Executes a given bash command and returns its output.",
            "input_schema": schema(
                {
                    "command": field("The command to execute", type="string"),
                    "timeout": field("Optional timeout in milliseconds (max 600000)", type="number"),
                    "description": field(
                        "Clear, concise description of what this command does in active voice.", type="string"
                    ),
                    "run_in_background": field("Set to true to run this command in the background.", type="boolean"),
                    "dangerouslyDisableSandbox": field(
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
            "input_schema": schema(
                {
                    "file_path": field("The absolute path to the file to read", type="string"),
                    "offset": field("The line number to start reading from.", type="integer", minimum=0, maximum=_MAX),
                    "limit": field("The number of lines to read.", type="integer", exclusiveMinimum=0, maximum=_MAX),
                    "pages": field('Page range for PDF files (e.g., "1-5", "3", "10-20").', type="string"),
                },
                ("file_path",),
            ),
        },
        {
            "name": "Edit",
            "description": "Performs exact string replacements in files.",
            "input_schema": schema(
                {
                    "file_path": field("The absolute path to the file to modify", type="string"),
                    "old_string": field("The text to replace", type="string"),
                    "new_string": field(
                        "The text to replace it with (must be different from old_string)", type="string"
                    ),
                    "replace_all": field(
                        "Replace all occurrences of old_string (default false)", default=False, type="boolean"
                    ),
                },
                ("file_path", "old_string", "new_string"),
            ),
        },
        {
            "name": "Agent",
            "description": "Launch a new agent to handle complex, multi-step tasks.",
            "input_schema": schema(
                {
                    "description": field("A short (3-5 word) description of the task", type="string"),
                    "prompt": field("The task for the agent to perform", type="string"),
                    "subagent_type": field("The type of specialized agent to use for this task", type="string"),
                    "model": field(
                        "Optional model override for this agent.",
                        type="string",
                        enum=["sonnet", "opus", "haiku", "fable"],
                    ),
                    "run_in_background": field(
                        "Agents run in the background by default; you will be notified when one completes.",
                        type="boolean",
                    ),
                    "isolation": field("Isolation mode.", type="string", enum=["worktree", "remote"]),
                },
                ("description", "prompt"),
            ),
        },
    )


def system_blocks() -> tuple[dict[str, JsonValue], ...]:
    return (
        {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.283.00; cc_entrypoint=sdk-cli;"},
        {"type": "text", "text": "Synthetic agent identity system prompt.", "cache_control": CACHE},
        {"type": "text", "text": "Synthetic interactive agent instructions.", "cache_control": CACHE},
    )


def claude_code_request(cache_bust: str) -> dict[str, JsonValue]:
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
        "system": list(system_blocks()),
        "messages": [
            {
                "role": "user",
                "content": [
                    *[{"type": "text", "text": reminder} for reminder in reminders],
                    {"type": "text", "text": "Reply with exactly the word PONG", "cache_control": CACHE},
                ],
            }
        ],
        "tools": list(tools()),
        "metadata": {"user_id": METADATA_USER_ID},
        "max_tokens": 32000,
        "thinking": dict(THINKING_BUDGET),
        "context_management": dict(CONTEXT_MANAGEMENT),
        "stream": True,
    }


def frontier_request(
    cache_bust: str,
    effort: str,
    max_tokens: int,
    prompt_text: str = "Reply with exactly the word PONG",
    stream: bool = True,
) -> dict[str, JsonValue]:
    return {
        "model": "",
        "system": list(system_blocks()),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"<system-reminder>\n{cache_bust}\n</system-reminder>"},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "# Environment\nSynthetic environment block.",
                        "cache_control": CACHE,
                    }
                ],
            },
        ],
        "tools": list(tools()),
        "metadata": {"user_id": METADATA_USER_ID},
        "max_tokens": max_tokens,
        "thinking": dict(THINKING_ADAPTIVE),
        "context_management": dict(CONTEXT_MANAGEMENT),
        "output_config": {"effort": effort},
        "stream": stream,
    }


def tool_loop_turn2(
    base: dict[str, JsonValue],
    assistant_content: tuple[dict[str, JsonValue], ...],
    tool_results: tuple[tuple[str, JsonValue], ...],
) -> dict[str, JsonValue]:
    return {
        **base,
        "messages": [
            *base["messages"],
            {"role": "assistant", "content": list(assistant_content)},
            {
                "role": "user",
                "content": [
                    {"tool_use_id": tool_use_id, "type": "tool_result", "content": content}
                    for tool_use_id, content in tool_results
                ],
            },
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "<total_tokens>14999970 tokens left</total_tokens>",
                        "cache_control": CACHE,
                    },
                    {
                        "type": "text",
                        "text": "First privately list what you need next; then request every item that doesn't depend on another's result in this one response.",
                    },
                ],
            },
        ],
    }


def cli_headers(key: str, beta: str = CLI_BETA) -> dict[str, str]:
    return {
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
        "anthropic-beta": beta,
        "anthropic-dangerous-direct-browser-access": "true",
        "anthropic-version": "2023-06-01",
        "x-app": "cli",
        "x-api-key": key,
    }


def sse_frame(event: str, data: JsonValue) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def sse_events(text: str) -> tuple[tuple[str, dict[str, object]], ...]:
    frames: Final = tuple(frame for frame in text.split("\n\n") if frame.strip())
    return tuple(
        (
            next(line.removeprefix("event: ") for line in frame.splitlines() if line.startswith("event: ")),
            json.loads(next(line.removeprefix("data: ") for line in frame.splitlines() if line.startswith("data: "))),
        )
        for frame in frames
    )


def _start_usage(usage: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: value for key, value in usage.items() if key != "output_tokens"}


def text_stream(identity: str, model: str, text: str, usage: dict[str, int]) -> tuple[bytes, ...]:
    return (
        sse_frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": _start_usage(usage),
                },
            },
        ),
        sse_frame(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        sse_frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        ),
        sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
        sse_frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": usage["output_tokens"]},
            },
        ),
        sse_frame("message_stop", {"type": "message_stop"}),
    )


def tool_use_stream(
    identity: str,
    model: str,
    thinking: str,
    signature: str,
    tool_calls: tuple[tuple[str, str, JsonValue], ...],
    usage: dict[str, int],
) -> tuple[bytes, ...]:
    frames: list[bytes] = [
        sse_frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": _start_usage(usage),
                },
            },
        ),
        sse_frame(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        ),
        sse_frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": thinking}},
        ),
        sse_frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": signature}},
        ),
        sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ]
    for index, (tool_id, name, tool_input) in enumerate(tool_calls, start=1):
        arguments: Final = json.dumps(tool_input)
        frames += [
            sse_frame(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "tool_use", "id": tool_id, "name": name, "input": {}},
                },
            ),
            sse_frame(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": arguments[: len(arguments) // 2]},
                },
            ),
            sse_frame(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": arguments[len(arguments) // 2 :]},
                },
            ),
            sse_frame("content_block_stop", {"type": "content_block_stop", "index": index}),
        ]
    frames += [
        sse_frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": usage["output_tokens"]},
            },
        ),
        sse_frame("message_stop", {"type": "message_stop"}),
    ]
    return tuple(frames)


def responses_completed(
    identity: str,
    model: str,
    output_items: tuple[dict[str, JsonValue], ...],
    usage: dict[str, int],
    status: str = "completed",
    incomplete_details: JsonValue = None,
) -> bytes:
    return json.dumps(
        {
            "id": f"resp_{identity}",
            "object": "response",
            "created_at": 1789788253,
            "status": status,
            "incomplete_details": incomplete_details,
            "model": model,
            "output": list(output_items),
            "usage": usage,
        }
    ).encode()


def responses_stream(identity: str, model: str, output_items: tuple[dict[str, JsonValue], ...]) -> tuple[bytes, ...]:
    frames: list[bytes] = [
        sse_frame(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": f"resp_{identity}",
                    "object": "response",
                    "status": "in_progress",
                    "model": model,
                    "output": [],
                },
            },
        )
    ]
    for index, item in enumerate(output_items):
        item_id: Final = str(item.get("id", f"item_{index}"))
        frames.append(
            sse_frame(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": {**item, "content": []} if item.get("type") == "message" else item,
                },
            )
        )
        if item.get("type") == "message":
            text: Final = "".join(part.get("text", "") for part in item.get("content", ()) if isinstance(part, dict))
            frames.append(
                sse_frame(
                    "response.output_text.delta",
                    {"type": "response.output_text.delta", "output_index": index, "item_id": item_id, "delta": text},
                )
            )
        if item.get("type") == "reasoning":
            summary_text: Final = "".join(
                str(part.get("text", "")) for part in item.get("summary", ()) if isinstance(part, dict)
            )
            if summary_text:
                frames.append(
                    sse_frame(
                        "response.reasoning_summary_text.delta",
                        {
                            "type": "response.reasoning_summary_text.delta",
                            "output_index": index,
                            "item_id": item_id,
                            "delta": summary_text,
                        },
                    )
                )
        if item.get("type") == "function_call":
            frames.append(
                sse_frame(
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": index,
                        "item_id": item_id,
                        "delta": item.get("arguments", ""),
                    },
                )
            )
        frames.append(
            sse_frame(
                "response.output_item.done",
                {"type": "response.output_item.done", "output_index": index, "item": item},
            )
        )
    frames.append(
        sse_frame(
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": f"resp_{identity}",
                    "object": "response",
                    "status": "completed",
                    "model": model,
                    "output": list(output_items),
                    "usage": {"input_tokens": 41, "output_tokens": 5, "total_tokens": 46},
                },
            },
        )
    )
    return tuple(frames)
