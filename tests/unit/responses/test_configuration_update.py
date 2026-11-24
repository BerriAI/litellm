from typing import Final

import pytest

from litellm.responses.additional_tools import additional_tools_in_input
from litellm.responses.configuration_update import (
    ReasoningUpdate,
    TextUpdate,
    additional_tools_item,
    configuration_update_item,
)

_EFFORT: Final = ReasoningUpdate(effort="high")
_FORMAT: Final = TextUpdate(format={"type": "json_schema", "name": "answer", "schema": {"type": "object"}})


@pytest.mark.parametrize(
    ("reasoning", "text", "expected"),
    [
        (None, None, None),
        (_EFFORT, None, {"type": "configuration_update", "reasoning": {"effort": "high"}}),
        (None, _FORMAT, {"type": "configuration_update", "text": _FORMAT}),
        (_EFFORT, _FORMAT, {"type": "configuration_update", "reasoning": {"effort": "high"}, "text": _FORMAT}),
    ],
)
def test_configuration_update_item_carries_only_the_changed_settings(
    reasoning: ReasoningUpdate | None, text: TextUpdate | None, expected: object
) -> None:
    assert configuration_update_item(reasoning=reasoning, text=text) == expected


def test_additional_tools_item_is_read_back_by_the_shared_additional_tools_reader() -> None:
    tool: Final = {"type": "function", "name": "get_time", "parameters": {"type": "object"}}

    item: Final = additional_tools_item((tool,))

    assert item == {"type": "additional_tools", "role": "developer", "tools": [tool]}
    assert additional_tools_in_input([item]) == (tool,)
