from __future__ import annotations

from typing import Final

import pytest
from codex.harness import CodexHarness
from codex.setups import SETUPS, ModelSetup
from e2e_config import unique_marker

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("deployment", SETUPS)
class TestCodexSubagent:
    def test_a_subagent_reads_the_task_its_parent_hands_it(self, codex: CodexHarness, setup: ModelSetup) -> None:
        codeword: Final = unique_marker()
        run: Final = codex.exec(
            'Spawn exactly one sub-agent with the spawn_agent tool, with fork_turns set to "none" so it does not'
            f" inherit this conversation. Its task: reply with exactly this codeword and nothing else: {codeword}."
            " Wait for its reply, then tell me the exact reply it sent you, verbatim.",
            model=setup.model,
            cwd=codex.new_workspace(files={}),
            enable=("multi_agent_v2",),
            disable=("shell_tool", "unified_exec"),
        )
        assert run.exit_code == 0, run
        assert run.thread_id is not None, run
        subagents: Final = codex.subagents(run.thread_id)
        assert subagents, run
        assert not any(subagent.forked for subagent in subagents), (subagents, run)
        assert any(any(codeword in reply for reply in subagent.replies) for subagent in subagents), (subagents, run)
