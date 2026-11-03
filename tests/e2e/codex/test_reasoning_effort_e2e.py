from __future__ import annotations

from typing import Final

import pytest
from codex.harness import CodexHarness
from codex.setups import SETUPS, ModelSetup
from e2e_config import unique_marker

pytestmark = pytest.mark.e2e

HIGH_EFFORT: Final = ("--config", 'model_reasoning_effort="high"')


@pytest.mark.parametrize("deployment", SETUPS)
class TestCodexReasoningEffort:
    def test_reasons_at_high_effort_over_numbers_it_read_through_the_gateway(
        self, codex: CodexHarness, setup: ModelSetup
    ) -> None:
        numbers: Final = tuple(100 + int(unique_marker()[:4], 16) % 900 for _ in range(3))
        workspace: Final = codex.new_workspace(files={"numbers.txt": "\n".join(map(str, numbers)) + "\n"})
        run: Final = codex.exec_with_flags(
            "Run `cat numbers.txt` and no other command, then work out the sum of the squares of the numbers it"
            " printed yourself. Reply with that sum as digits only.",
            model=setup.model,
            cwd=workspace,
            flags=HIGH_EFFORT,
        )
        assert run.exit_code == 0, run
        assert str(sum(number * number for number in numbers)) in run.answer, run
        assert run.usage is not None and run.usage.reasoning_output_tokens > 0, run
