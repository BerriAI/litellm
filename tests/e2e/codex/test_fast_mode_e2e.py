from __future__ import annotations

from typing import Final

import pytest
from codex.harness import CodexHarness
from codex.setups import ModelSetup, with_expected_failure
from e2e_config import unique_marker

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize(
    "deployment",
    with_expected_failure(
        "bedrock_runtime",
        reason="LiteLLM's catalog offers Codex the priority tier for GPT on Bedrock Runtime, which rejects it"
        " with a 400 (LIT-9793)",
    ),
)
class TestCodexFastMode:
    def test_answers_with_what_a_command_printed_under_the_real_model_name(
        self, real_name_codex: CodexHarness, real_name_setup: ModelSetup
    ) -> None:
        marker: Final = unique_marker()
        workspace: Final = real_name_codex.new_workspace(files={"secret.txt": marker})
        run: Final = real_name_codex.exec(
            "Run `cat secret.txt`, then reply with exactly what it printed.",
            model=real_name_setup.model,
            cwd=workspace,
            fast=True,
        )
        assert run.exit_code == 0, run
        assert any(marker in command.output for command in run.commands), run
        assert marker in run.answer, run
