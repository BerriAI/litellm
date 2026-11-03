from __future__ import annotations

from typing import Final

import pytest
from codex.harness import CodexHarness
from codex.setups import SETUPS, ModelSetup
from e2e_config import unique_marker
from pydantic import BaseModel, ConfigDict

pytestmark = pytest.mark.e2e

OUTPUT_SCHEMA: Final = (
    '{"type": "object", "properties": {"marker": {"type": "string"}, "number": {"type": "integer"}},'
    ' "required": ["marker", "number"], "additionalProperties": false}'
)


class StructuredAnswer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    marker: str
    number: int


@pytest.mark.parametrize("deployment", SETUPS)
class TestCodexStructuredOutput:
    def test_answers_in_the_output_schema_with_values_it_read_through_the_gateway(
        self, codex: CodexHarness, setup: ModelSetup
    ) -> None:
        expected: Final = StructuredAnswer(marker=unique_marker(), number=int(unique_marker()[:6], 16))
        workspace: Final = codex.new_workspace(
            files={"marker.txt": f"{expected.marker}\n", "number.txt": f"{expected.number}\n"}
        )
        schema: Final = codex.home / "schema.json"
        _ = schema.write_text(OUTPUT_SCHEMA)
        run: Final = codex.exec_with_flags(
            "Run `cat marker.txt number.txt`. Put the first line it printed in marker and the second in number.",
            model=setup.model,
            cwd=workspace,
            flags=("--output-schema", str(schema)),
        )
        assert run.exit_code == 0, run
        assert StructuredAnswer.model_validate_json(run.answer) == expected, run
