from __future__ import annotations

from typing import Final

import pytest
from codex.harness import CodexHarness
from codex.setups import BEDROCK_MANTLE_US_PROFILE, ModelSetup, with_expected_failure
from e2e_config import unique_marker

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize(
    "deployment",
    with_expected_failure(
        "bedrock_mantle",
        reason="LiteLLM misses the cost-map row for a us.-prefixed Bedrock Mantle model, so it calls a path Bedrock"
        " answers with an empty 200 and Codex gets no reply (#46247)",
        setups=(BEDROCK_MANTLE_US_PROFILE,),
    ),
)
class TestCodexUsProfileMantleModel:
    def test_answers_through_a_us_inference_profile(self, codex: CodexHarness, setup: ModelSetup) -> None:
        marker: Final = unique_marker()
        workspace: Final = codex.new_workspace(files={"secret.txt": marker})
        run: Final = codex.exec(
            "Run `cat secret.txt`, then reply with exactly what it printed.", model=setup.model, cwd=workspace
        )
        assert run.exit_code == 0, run
        assert run.usage is not None and run.usage.output_tokens > 0, run
        assert marker in run.answer, run
