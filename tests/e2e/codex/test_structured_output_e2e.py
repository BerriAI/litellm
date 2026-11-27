from __future__ import annotations

import json
from typing import Final

import pytest
from codex.harness import CodexHarness
from codex.setups import SETUPS, ModelSetup
from e2e_config import unique_marker
from pydantic import TypeAdapter

pytestmark = pytest.mark.e2e

STRUCTURED_ANSWER: Final = TypeAdapter(dict[str, str | int])


def _output_schema(text_key: str, number_key: str) -> str:
    return json.dumps(
        {
            "type": "object",
            "properties": {text_key: {"type": "string"}, number_key: {"type": "integer"}},
            "required": [text_key, number_key],
            "additionalProperties": False,
        }
    )


@pytest.mark.parametrize("deployment", SETUPS)
class TestCodexStructuredOutput:
    def test_answers_in_the_output_schema_with_values_it_read_through_the_gateway(
        self, codex: CodexHarness, setup: ModelSetup
    ) -> None:
        text_key: Final = f"k{unique_marker()[:8]}"
        number_key: Final = f"k{unique_marker()[:8]}"
        marker: Final = unique_marker()
        number: Final = int(unique_marker()[:6], 16)
        workspace: Final = codex.new_workspace(files={"marker.txt": f"{marker}\n", "number.txt": f"{number}\n"})
        schema: Final = codex.home / "schema.json"
        _ = schema.write_text(_output_schema(text_key, number_key))
        run: Final = codex.exec_with_flags(
            "Run `cat marker.txt number.txt`. Put the first line it printed in the string field of your answer and"
            " the second in its integer field.",
            model=setup.model,
            cwd=workspace,
            flags=("--output-schema", str(schema)),
        )
        assert run.exit_code == 0, run
        assert STRUCTURED_ANSWER.validate_json(run.answer, strict=True) == {text_key: marker, number_key: number}, run
