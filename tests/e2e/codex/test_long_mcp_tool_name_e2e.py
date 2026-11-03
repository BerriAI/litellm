from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import pytest
from codex.harness import CodexHarness, McpServer
from codex.setups import ModelSetup, with_expected_failure
from e2e_config import unique_marker

pytestmark = pytest.mark.e2e

MCP_SERVER: Final = Path(__file__).with_name("mcp_server.py")
LONG_TOOL_NAME: Final = "read_the_current_secret_value_from_the_vault_that_this_codex_session_was_given"


@pytest.mark.parametrize(
    "deployment",
    with_expected_failure(
        "claude_bedrock",
        reason="LiteLLM passes Codex's namespaced MCP tool name to Bedrock uncapped and Bedrock rejects tool names over"
        " 64 characters, failing every turn (LIT-10289)",
    ),
)
class TestCodexLongMcpToolName:
    def test_calls_an_mcp_tool_whose_name_is_over_64_characters(self, codex: CodexHarness, setup: ModelSetup) -> None:
        marker: Final = unique_marker()
        secret: Final = codex.home / "secret.txt"
        _ = secret.write_text(marker)
        vault: Final = McpServer(
            command=sys.executable,
            args=(str(MCP_SERVER),),
            env={"SECRET_FILE": str(secret), "TOOL_NAME": LONG_TOOL_NAME},
        )
        codex.add_mcp_server("vault", vault)
        run: Final = codex.exec(
            f"Call the {LONG_TOOL_NAME} tool of the vault MCP server, then reply with exactly what it returned.",
            model=setup.model,
            cwd=codex.new_workspace(files={}),
        )
        assert any(marker in call.output for call in run.mcp_calls), run
        assert marker in run.answer, run
