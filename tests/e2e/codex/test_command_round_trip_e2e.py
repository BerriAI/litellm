from __future__ import annotations

from typing import Final

import pytest
from codex.harness import CodexHarness
from codex.setups import SETUPS, ModelSetup
from e2e_config import unique_marker

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("deployment", SETUPS)
class TestCodexCommandRoundTrip:
    def test_answers_with_what_a_command_printed(self, codex: CodexHarness, setup: ModelSetup) -> None:
        marker: Final = unique_marker()
        workspace: Final = codex.new_workspace(files={"secret.txt": marker})
        run: Final = codex.exec(
            "Run `cat secret.txt`, then reply with exactly what it printed.", model=setup.model, cwd=workspace
        )
        assert any(marker in command.output for command in run.commands), run
        assert marker in run.answer, run
