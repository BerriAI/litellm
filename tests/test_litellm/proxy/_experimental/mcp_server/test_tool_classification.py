import json
from pathlib import Path

import pytest

from litellm.proxy._experimental.mcp_server.tool_classification import classify_tool_op

_FIXTURE_PATH: Path = Path(__file__).parent / "fixtures" / "mcp_tool_classification_cases.json"
_CASES: list[dict] = json.loads(_FIXTURE_PATH.read_text())


@pytest.mark.parametrize(
    ("name", "description", "expected"),
    [pytest.param(c["name"], c.get("description"), c["expected"], id=c["name"]) for c in _CASES],
)
def test_classify_tool_op_fixture(name: str, description: str | None, expected: str) -> None:
    assert classify_tool_op(name, description) == expected
