"""A system prompt too long for the Responses `instructions` field goes in `input` instead."""

from collections.abc import Mapping
from typing import Final

from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping

# OpenAI and Azure answer longer `instructions` with 400 string_above_max_length, counting code points (2026-11-19)
RESPONSES_INSTRUCTIONS_MAX_LENGTH: Final = 1_048_576


def _sent(request: Mapping[str, object], key: str) -> object:
    extra_body: Final = as_str_mapping(request.get("extra_body")) or {}
    return extra_body.get(key, request.get(key))


def _without(request: Mapping[str, object], key: str) -> Mapping[str, object]:
    extra_body: Final = as_str_mapping(request.get("extra_body"))
    return {
        **{name: value for name, value in request.items() if name != key},
        **(
            {"extra_body": {name: value for name, value in extra_body.items() if name != key}}
            if extra_body is not None
            else {}
        ),
    }


def with_long_instructions_in_input(request: Mapping[str, object]) -> Mapping[str, object]:
    instructions: Final = _sent(request, "instructions")
    input_items: Final = as_object_list(_sent(request, "input"))
    if (
        not isinstance(instructions, str)
        or len(instructions) <= RESPONSES_INSTRUCTIONS_MAX_LENGTH
        or input_items is None
    ):
        return request
    system_item: Final = {
        "type": "message",
        "role": "system",
        "content": [{"type": "input_text", "text": instructions}],
    }
    moved: Final = {"input": [system_item, *input_items]}
    without_instructions: Final = _without(request, "instructions")
    extra_body: Final = as_str_mapping(without_instructions.get("extra_body"))
    return {
        **without_instructions,
        **moved,
        **({"extra_body": {**extra_body, **moved}} if extra_body is not None and "input" in extra_body else {}),
    }
