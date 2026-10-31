from __future__ import annotations

from typing import Final

import pytest
from codex.harness import CodexHarness
from codex.setups import SETUPS, ModelSetup
from e2e_config import unique_marker

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("deployment", SETUPS)
class TestCodexLongSession:
    def test_follows_a_chain_of_files_one_command_at_a_time(self, codex: CodexHarness, setup: ModelSetup) -> None:
        second: Final = f"{unique_marker()}.txt"
        third: Final = f"{unique_marker()}.txt"
        marker: Final = f"DONE-{unique_marker()}"
        workspace: Final = codex.new_workspace(files={"start.txt": second, second: third, third: marker})
        run: Final = codex.exec(
            "Run `cat start.txt`. It prints a file name. Run `cat` on that file, one command per step, and keep"
            " following the file names until a command prints a value that starts with DONE-. Reply with that value.",
            model=setup.model,
            cwd=workspace,
        )
        assert len(run.commands) >= 3, run
        assert marker in run.answer, run
