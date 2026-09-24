"""Parity test for the Python port of the dashboard's classifyToolOp.

Both sides read the same fixture (ui/litellm-dashboard/src/utils/
mcpToolCrudClassification.fixture.json), so any drift in regexes or ordering
fails on one side or the other.
"""

import json
from pathlib import Path

import pytest

from litellm.proxy._experimental.mcp_server.tool_op_classification import classify_tool_op

_FIXTURE_PATH: Path = (
    Path(__file__).resolve().parents[5]
    / "ui"
    / "litellm-dashboard"
    / "src"
    / "utils"
    / "mcpToolCrudClassification.fixture.json"
)
_FIXTURE: list[dict[str, str]] = json.loads(_FIXTURE_PATH.read_text())


@pytest.mark.parametrize(
    "row",
    _FIXTURE,
    ids=[f"{row['name']}|{row['description']}" for row in _FIXTURE],
)
def test_classify_tool_op_matches_dashboard(row: dict[str, str]) -> None:
    assert classify_tool_op(row["name"], row["description"]) == row["expected"]
