"""The Claude Code CLI, pinned in this suite's package.json, runs one headless
turn against the proxy on a virtual key: it streams over `/v1/messages`, calls
the Bash tool, reads the tool result, and answers. Client side the stream-json
events carry the tool_use, the tool_result, the streamed deltas, and the final
result; proxy side the spend rows carry real spend and `stream: true`."""

from __future__ import annotations

import json
from typing import Final

import pytest

from client_apps_client import ClientAppsClient, unwrap_run
from client_apps_models import (
    ClaudeAssistantEvent,
    ClaudeResultEvent,
    ClaudeStreamEvent,
    ClaudeUserEvent,
    ContentBlockDelta,
    ToolResultBlock,
    ToolUseBlock,
)
from e2e_config import CHEAP_ANTHROPIC_MODEL
from spend_rows import streamed_tool_turn_rows

pytestmark = pytest.mark.e2e

PROMPT: Final = "Use the Bash tool to run the command `echo pong` and report what it printed."


@pytest.mark.covers("other.client_apps.claude_code_cli.tool_turn_streams")
def test_claude_code_cli_streams_a_tool_turn(client: ClientAppsClient, scoped_key: str) -> None:
    events = unwrap_run(
        client.run_claude_code(
            key=scoped_key, model=CHEAP_ANTHROPIC_MODEL, prompt=PROMPT, allowed_tool="Bash(echo pong)"
        )
    )

    tool_uses = [
        block
        for event in events
        if isinstance(event, ClaudeAssistantEvent) and isinstance(event.message.content, list)
        for block in event.message.content
        if isinstance(block, ToolUseBlock)
    ]
    assert [block.name for block in tool_uses] == ["Bash"], events

    tool_results = [
        block
        for event in events
        if isinstance(event, ClaudeUserEvent) and isinstance(event.message.content, list)
        for block in event.message.content
        if isinstance(block, ToolResultBlock)
    ]
    assert len(tool_results) == 1, events
    assert not tool_results[0].is_error, tool_results[0]
    assert "pong" in json.dumps(tool_results[0].content), tool_results[0]

    delta_types = [
        event.event.delta.type
        for event in events
        if isinstance(event, ClaudeStreamEvent) and isinstance(event.event, ContentBlockDelta)
    ]
    assert "input_json_delta" in delta_types, delta_types
    assert "text_delta" in delta_types, delta_types

    results = [event for event in events if isinstance(event, ClaudeResultEvent)]
    assert len(results) == 1, events
    assert results[0].subtype == "success", results[0]
    assert not results[0].is_error, results[0]
    assert results[0].result is not None and "pong" in results[0].result, results[0]

    streamed_tool_turn_rows(client.proxy, scoped_key)
