from typing import Final

import pytest
from pydantic import JsonValue, ValidationError

from litellm.proxy.lens.agent_contract import (
    Checkpoint,
    EvidenceRequest,
    FindingGroups,
    Findings,
    PythonAgentTurn,
    PythonRequest,
)


def test_agent_turn_preserves_tool_order_unicode_and_structured_result() -> None:
    turn: Final = PythonAgentTurn[Findings].model_validate(
        {
            "tools": [
                {"action": "read", "execution_id": "run-1", "span_ids": ["span-2"], "char_start": 4},
                {"action": "python", "code": "print('é終')", "execution_ids": ["run-1"]},
                {"action": "history", "turn_start": 2, "turn_end": 3, "char_start": 500, "char_end": 520},
            ],
            "checkpoint": "Keep the original failed tool response",
            "result": {"findings": []},
        }
    )
    assert tuple(tool.action for tool in turn.tools) == ("read", "python", "history")
    assert isinstance(turn.tools[1], PythonRequest)
    assert turn.tools[1].code == "print('é終')"
    assert isinstance(turn.tools[2], EvidenceRequest)
    assert (turn.tools[2].turn_start, turn.tools[2].turn_end) == (2, 3)
    assert (turn.tools[2].char_start, turn.tools[2].char_end) == (500, 520)
    assert turn.result == Findings(findings=())
    assert PythonAgentTurn[Findings].model_validate_json(turn.model_dump_json()) == turn


@pytest.mark.parametrize(
    "payload",
    (
        {"tools": [{"action": "http", "url": "https://example.com"}]},
        {"tools": [{"action": "python", "code": ""}]},
        {"tools": [{"action": "read", "char_start": -1}]},
        {"tools": [{"action": "history", "turn_start": -1}]},
        {"tools": [{"action": "history", "char_end": -1}]},
        {"tools": [{"action": "read_reviews", "review_phase": "unknown"}]},
        {"tools": [{"action": "read", "unrecognized_scope": "all"}]},
        {"checkpoint": ""},
        {"result": {"findings": [{"title": "Unsupported conclusion"}]}},
    ),
)
def test_model_output_rejects_unsupported_tools_ranges_and_incomplete_findings(payload: JsonValue) -> None:
    with pytest.raises(ValidationError):
        PythonAgentTurn[Findings].model_validate(payload)


def test_consolidation_requires_members_and_checkpoint_requires_notes() -> None:
    with pytest.raises(ValidationError):
        FindingGroups.model_validate({"groups": [{"members": [], "representative": "new:0"}]})
    groups: Final = FindingGroups.model_validate(
        {"groups": [{"members": ["new:0", "saved:1"], "representative": "saved:1"}]}
    )
    assert groups.groups[0].members == ("new:0", "saved:1")
    assert groups.groups[0].representative == "saved:1"
    with pytest.raises(ValidationError):
        Checkpoint(working_notes="")
    assert Checkpoint(working_notes="Resume with the original tool evidence").working_notes
