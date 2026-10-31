from __future__ import annotations

from typing import Final

import pytest
from codex.harness import CodexHarness
from codex.setups import SETUPS, ModelSetup
from e2e_config import unique_marker

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("deployment", SETUPS)
class TestCodexFileEdit:
    def test_changes_a_file_with_the_patch_tool(self, codex: CodexHarness, setup: ModelSetup) -> None:
        marker: Final = unique_marker()
        workspace: Final = codex.new_workspace(files={"config.txt": "name = placeholder\n"})
        run: Final = codex.exec(
            f"Use the apply_patch tool to change `placeholder` to `{marker}` in config.txt. Change nothing else.",
            model=setup.model,
            cwd=workspace,
        )
        assert run.patched_files == (workspace / "config.txt",), run
        assert (workspace / "config.txt").read_text() == f"name = {marker}\n", run
