from collections.abc import Mapping
from typing import Final, Literal

from pydantic import JsonValue


def with_leading_system_message(
    request: Mapping[str, JsonValue], *, role: Literal["system", "developer"] = "system"
) -> dict[str, JsonValue]:
    instructions: Final = request["instructions"]
    items: Final = request["input"]
    assert isinstance(instructions, str), request
    assert isinstance(items, list), request
    return {
        **{key: value for key, value in request.items() if key != "instructions"},
        "input": [
            {"type": "message", "role": role, "content": [{"type": "input_text", "text": instructions}]},
            *items,
        ],
    }
