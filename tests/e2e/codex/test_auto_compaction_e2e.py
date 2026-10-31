from __future__ import annotations

from typing import Final

import pytest
from codex.harness import CodexHarness
from codex.setups import ModelSetup, with_expected_failure
from e2e_config import unique_marker

pytestmark = pytest.mark.e2e

COMPACT_AFTER_TOKENS: Final = 500
PADDING: Final = "".join(f"line {number}: the quick brown fox jumps over the lazy dog\n" for number in range(400))


@pytest.mark.parametrize(
    "deployment",
    with_expected_failure(
        "claude_bedrock",
        reason="LiteLLM's Responses-to-chat bridge ignores Codex's compaction_trigger item, so Codex gets no"
        " compaction back and fails the turn",
    ),
)
class TestCodexAutoCompaction:
    def test_remembers_a_value_after_the_gateway_compacts_the_session(
        self, codex: CodexHarness, setup: ModelSetup
    ) -> None:
        marker: Final = unique_marker()
        workspace: Final = codex.new_workspace(files={"secret.txt": marker, "padding.txt": PADDING})
        first: Final = codex.exec(
            "Run `cat secret.txt` and remember the value it prints. Then run `cat padding.txt`. Then reply with OK.",
            model=setup.model,
            cwd=workspace,
            compact_after_tokens=COMPACT_AFTER_TOKENS,
        )
        assert first.thread_id is not None, first
        (workspace / "secret.txt").unlink()
        second: Final = codex.resume(
            first.thread_id,
            "What value did secret.txt print earlier? Reply with just that value.",
            model=setup.model,
            cwd=workspace,
            compact_after_tokens=COMPACT_AFTER_TOKENS,
        )
        assert any(compaction.remote for compaction in codex.compactions(first.thread_id)), (first, second)
        assert marker in second.answer, second
