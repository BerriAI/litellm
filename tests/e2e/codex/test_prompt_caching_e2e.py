from __future__ import annotations

from typing import Final

import pytest
from codex.harness import CodexHarness
from codex.setups import ModelSetup, with_expected_failures
from e2e_config import unique_marker

pytestmark = pytest.mark.e2e


SETUPS_WITHOUT_PROMPT_CACHING: Final = with_expected_failures(
    {
        "claude_bedrock": (
            "Codex sends no cache breakpoints, and LiteLLM adds no Bedrock cachePoint for Claude unless the"
            " deployment sets cache_control_injection_points."
        ),
    }
)


@pytest.mark.parametrize("deployment", SETUPS_WITHOUT_PROMPT_CACHING)
class TestCodexPromptCaching:
    def test_reads_the_repeated_prefix_from_the_cache_on_the_next_request(
        self, codex: CodexHarness, setup: ModelSetup
    ) -> None:
        marker: Final = unique_marker()
        workspace: Final = codex.new_workspace(files={"secret.txt": marker})
        run: Final = codex.exec(
            "Run `cat secret.txt`, then reply with exactly what it printed.", model=setup.model, cwd=workspace
        )
        assert run.exit_code == 0, run
        assert marker in run.answer, run
        assert run.usage is not None and run.usage.cached_input_tokens > 0, run
