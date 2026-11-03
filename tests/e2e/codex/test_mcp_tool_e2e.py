from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import pytest
from codex.harness import CodexHarness, McpServer
from codex.setups import SETUPS, ModelSetup
from e2e_config import unique_marker

pytestmark = pytest.mark.e2e

MCP_SERVER: Final = Path(__file__).with_name("mcp_server.py")


@pytest.mark.parametrize("deployment", SETUPS)
class TestCodexMcpTool:
    def test_calls_a_local_mcp_tool_again_on_a_resumed_turn(self, codex: CodexHarness, setup: ModelSetup) -> None:
        secret: Final = codex.home / "secret.txt"
        vault: Final = McpServer(command=sys.executable, args=(str(MCP_SERVER),), env={"SECRET_FILE": str(secret)})
        codex.add_mcp_server("vault", vault)
        workspace: Final = codex.new_workspace(files={})

        first_marker: Final = unique_marker()
        _ = secret.write_text(first_marker)
        first: Final = codex.exec(
            "Call the read_secret tool of the vault MCP server, then reply with exactly what it returned.",
            model=setup.model,
            cwd=workspace,
        )
        assert first.exit_code == 0, first
        assert first.thread_id is not None, first
        assert any(first_marker in call.output for call in first.mcp_calls), first
        assert first_marker in first.answer, first

        second_marker: Final = unique_marker()
        _ = secret.write_text(second_marker)
        second: Final = codex.resume(
            first.thread_id,
            "The secret has changed. Call the read_secret tool again, then reply with exactly what it returned.",
            model=setup.model,
            cwd=workspace,
        )
        assert second.exit_code == 0, second
        assert any(second_marker in call.output for call in second.mcp_calls), second
        assert second_marker in second.answer, second
