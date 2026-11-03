from __future__ import annotations

from typing import Final

import pytest
from codex.harness import CodexHarness
from codex.setups import COMPACTION_SETUPS, SETUPS, ModelSetup
from e2e_config import unique_marker

pytestmark = pytest.mark.e2e

COMPACT_AFTER_TOKENS: Final = 500
PADDING: Final = "".join(f"line {number}: the quick brown fox jumps over the lazy dog\n" for number in range(400))


class TestCodexUnderTheRealModelName:
    @pytest.mark.parametrize("deployment", SETUPS)
    def test_answers_with_what_a_command_printed(
        self, real_name_codex: CodexHarness, real_name_setup: ModelSetup
    ) -> None:
        marker: Final = unique_marker()
        workspace: Final = real_name_codex.new_workspace(files={"secret.txt": marker})
        run: Final = real_name_codex.exec(
            "Run `cat secret.txt`, then reply with exactly what it printed.", model=real_name_setup.model, cwd=workspace
        )
        assert run.exit_code == 0, run
        assert any(marker in command.output for command in run.commands), run
        assert marker in run.answer, run

    @pytest.mark.parametrize("deployment", COMPACTION_SETUPS)
    def test_remembers_a_value_after_the_gateway_compacts_the_session(
        self, real_name_codex: CodexHarness, real_name_setup: ModelSetup
    ) -> None:
        marker: Final = unique_marker()
        workspace: Final = real_name_codex.new_workspace(files={"secret.txt": marker, "padding.txt": PADDING})
        first: Final = real_name_codex.exec(
            "Run `cat secret.txt` and remember the value it prints. Then run `cat padding.txt`. Then reply with OK.",
            model=real_name_setup.model,
            cwd=workspace,
            compact_after_tokens=COMPACT_AFTER_TOKENS,
        )
        assert first.thread_id is not None, first
        (workspace / "secret.txt").unlink()
        second: Final = real_name_codex.resume(
            first.thread_id,
            "What value did secret.txt print earlier? Reply with just that value.",
            model=real_name_setup.model,
            cwd=workspace,
            compact_after_tokens=COMPACT_AFTER_TOKENS,
        )
        assert any(compaction.remote for compaction in real_name_codex.compactions(first.thread_id)), (first, second)
        assert marker in second.answer, second
