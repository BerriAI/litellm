"""The Codex CLI, pinned in this suite's package.json, runs one headless
`codex exec --json` turn against the proxy on a virtual key through a custom
model provider (`wire_api = "responses"`, so `/v1/responses`): it runs a shell
command, reads its output, and answers. Client side the JSONL events carry the
completed command execution and the agent message; proxy side the spend rows
carry real spend and `stream: true` (Codex always streams the Responses API)."""

from __future__ import annotations

from typing import Final

import pytest

from client_apps_client import ClientAppsClient, unwrap_run
from client_apps_models import CodexAgentMessage, CodexCommandExecution, CodexItemCompleted, CodexTurnCompleted
from e2e_config import CHEAP_OPENAI_MODEL
from spend_rows import streamed_tool_turn_rows

pytestmark = pytest.mark.e2e

PROMPT: Final = "Run the shell command `echo pong` and reply with exactly what it printed."


@pytest.mark.covers("other.client_apps.codex_cli.tool_turn_streams")
def test_codex_cli_streams_a_tool_turn(client: ClientAppsClient, scoped_key: str) -> None:
    events = unwrap_run(client.run_codex(key=scoped_key, model=CHEAP_OPENAI_MODEL, prompt=PROMPT))

    completed_items = [event.item for event in events if isinstance(event, CodexItemCompleted)]
    commands = [item for item in completed_items if isinstance(item, CodexCommandExecution)]
    assert len(commands) == 1, events
    assert commands[0].status == "completed", commands[0]
    assert commands[0].exit_code == 0, commands[0]
    assert "pong" in commands[0].aggregated_output, commands[0]

    messages = [item for item in completed_items if isinstance(item, CodexAgentMessage)]
    assert len(messages) == 1, events
    assert "pong" in messages[0].text, messages[0]

    turns = [event for event in events if isinstance(event, CodexTurnCompleted)]
    assert len(turns) == 1, events
    assert turns[0].usage.output_tokens > 0, turns[0]

    streamed_tool_turn_rows(client.proxy, scoped_key)
