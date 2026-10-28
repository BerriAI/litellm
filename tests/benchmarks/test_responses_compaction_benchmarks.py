import json
from collections.abc import Mapping
from typing import Final

import pytest

from litellm.responses.litellm_completion_transformation.compaction_items import (
    with_answered_turns_after_newest_compaction,
)

COMPACTION_ITEM: Final[Mapping[str, object]] = {
    "type": "compaction",
    "id": "cmp_1",
    "encrypted_content": json.dumps({"type": "compaction", "content": "Turn one planned Tokyo."}),
}
LONG_TURN_OF_ASSISTANT_TEXT: Final[tuple[Mapping[str, object], ...]] = tuple(
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": f"Step {index}."}]}
    for index in range(5_000)
)
CALL: Final[Mapping[str, object]] = {
    "type": "function_call",
    "call_id": "call_osaka",
    "name": "get_weather",
    "arguments": "{}",
}
OUTPUT: Final[Mapping[str, object]] = {"type": "function_call_output", "call_id": "call_osaka", "output": "Sunny."}


@pytest.mark.benchmark
def test_with_answered_turns_after_newest_compaction_long_turn_split_from_its_output() -> None:
    replayed: Final = with_answered_turns_after_newest_compaction(
        [*LONG_TURN_OF_ASSISTANT_TEXT, CALL, COMPACTION_ITEM, OUTPUT]
    )

    assert replayed == [COMPACTION_ITEM, *LONG_TURN_OF_ASSISTANT_TEXT, CALL, OUTPUT]
