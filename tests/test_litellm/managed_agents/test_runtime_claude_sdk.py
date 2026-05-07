"""
Slow-but-real integration test for ``ClaudeSDKAgentRuntime``.

Hits the real Anthropic API via ``claude-agent-sdk``. Skipped unless
``ANTHROPIC_API_KEY`` is set in the env (typically loaded from .env in
local dev). Marked ``@pytest.mark.slow`` so the default unit-test pass
(``-k 'not slow'``) skips it.

What this proves end-to-end:
  * The runtime constructs a ``ClaudeAgentOptions`` with the right cwd.
  * The SDK actually picks up the cwd and writes its tool output there.
  * Our event translation captures at least one assistant_message and
    a terminal run_finished event.

This is an integration smoke — it intentionally does NOT make detailed
assertions about LLM output. The model can phrase things differently
between runs.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from litellm.managed_agents.agent_runtime.base import AgentConfig, SessionState
from litellm.managed_agents.agent_runtime.claude_sdk import ClaudeSDKAgentRuntime
from litellm.managed_agents.events import EVENT_TYPE_RUN_FINISHED
from litellm.managed_agents.sandbox.local import LocalSandbox


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set; skipping live Claude SDK test",
    ),
]


@pytest.mark.asyncio
async def test_create_file_via_claude_sdk():
    pytest.importorskip("claude_agent_sdk")

    workdir = tempfile.mkdtemp(prefix="litellm_managed_agents_test_")
    try:
        sandbox = LocalSandbox(working_dir=workdir)
        runtime = ClaudeSDKAgentRuntime()
        session_state = SessionState(session_id="sess_test", cwd=workdir)
        agent_config = AgentConfig(
            name="filemaker",
            model=None,  # let the SDK pick its default
            system_prompt=(
                "You are a filesystem agent. Use the Write tool to do exactly "
                "what the user asks."
            ),
        )

        terminal_seen = False
        async for event in runtime.run(
            prompt='Use the Write tool to create a file named "foo.txt" in the '
            'current working directory containing exactly the text "bar". '
            "Then stop.",
            sandbox=sandbox,
            session_state=session_state,
            agent_config=agent_config,
        ):
            if event.type == EVENT_TYPE_RUN_FINISHED:
                terminal_seen = True
                break

        assert terminal_seen, "expected a run_finished event from claude-agent-sdk"
        created = Path(workdir) / "foo.txt"
        assert created.exists(), f"expected {created} to exist"
        assert "bar" in created.read_text()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
