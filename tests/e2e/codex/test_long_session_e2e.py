from __future__ import annotations

from typing import Final

import pytest
from codex.harness import CodexHarness
from codex.setups import SETUPS, ModelSetup
from e2e_config import unique_marker

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("deployment", SETUPS)
class TestCodexLongSession:
    def test_remembers_every_command_output_in_a_chain_of_files(self, codex: CodexHarness, setup: ModelSetup) -> None:
        codes: Final = (unique_marker(), unique_marker(), unique_marker())
        second: Final = f"{unique_marker()}.txt"
        third: Final = f"{unique_marker()}.txt"
        workspace: Final = codex.new_workspace(
            files={
                "start.txt": f"code {codes[0]}\nnext {second}\n",
                second: f"code {codes[1]}\nnext {third}\n",
                third: f"code {codes[2]}\nlast file\n",
            }
        )
        run: Final = codex.exec(
            "Run `cat start.txt`. Each file prints a code and then the name of the next file, until one says last"
            " file. Run `cat` on each next file, one command per step. Reply with the three codes in the order you"
            " read them.",
            model=setup.model,
            cwd=workspace,
        )
        assert len(run.commands) >= 3, run
        assert all(code in run.answer for code in codes), run
