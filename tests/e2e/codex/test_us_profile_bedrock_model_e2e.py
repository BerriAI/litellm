from __future__ import annotations

from typing import Final

import pytest
from codex.harness import CodexHarness
from codex.setups import BEDROCK_RUNTIME_US_PROFILE, ModelSetup
from e2e_config import unique_marker

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("deployment", (BEDROCK_RUNTIME_US_PROFILE,))
class TestCodexUsProfileBedrockModel:
    def test_answers_through_a_us_inference_profile(self, codex: CodexHarness, setup: ModelSetup) -> None:
        marker: Final = unique_marker()
        workspace: Final = codex.new_workspace(files={"secret.txt": marker})
        run: Final = codex.exec(
            "Run `cat secret.txt`, then reply with exactly what it printed.", model=setup.model, cwd=workspace
        )
        assert run.exit_code == 0, run
        assert run.usage is not None and run.usage.output_tokens > 0, run
        assert any(marker in command.output for command in run.commands), run
        assert marker in run.answer, run
