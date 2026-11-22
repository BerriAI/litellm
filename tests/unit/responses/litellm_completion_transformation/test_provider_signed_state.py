import json
from functools import reduce
from typing import Final

import pytest

from litellm.constants import REDACTED_BY_LITELM_STRING
from litellm.litellm_core_utils.prompt_templates.factory import THOUGHT_SIGNATURE_SEPARATOR
from litellm.proxy.spend_tracking.spend_tracking_utils import _sanitize_request_body_for_spend_logs_payload
from litellm.responses.litellm_completion_transformation.provider_signed_state import (
    holds_altered_provider_state,
    without_altered_provider_state,
)

_LONG: Final = "Eq0BCkYICxgCKkB" * 2000
_WHOLE_THINKING: Final = {"type": "thinking", "thinking": "Count.", "signature": "Eq0BCkYI=="}
_WHOLE_REDACTED: Final = {"type": "redacted_thinking", "data": "EmwKAhgB=="}


def _cut(value: str) -> str:
    return _sanitize_request_body_for_spend_logs_payload({"text": value})["text"]


@pytest.mark.parametrize(
    "altered_block",
    [
        pytest.param({**_WHOLE_THINKING, "signature": _cut(_LONG)}, id="cut-signature"),
        pytest.param({**_WHOLE_THINKING, "thinking": _cut(_LONG)}, id="cut-signed-text"),
        pytest.param({"type": "redacted_thinking", "data": _cut(_LONG)}, id="cut-redacted-thinking"),
    ],
)
def test_an_altered_thinking_block_is_left_out_and_its_whole_neighbors_stay(altered_block: dict[str, str]):
    message: Final = {
        "role": "assistant",
        "content": "954",
        "thinking_blocks": [_WHOLE_THINKING, altered_block, _WHOLE_REDACTED],
        "provider_specific_fields": {"thinking_blocks": [altered_block]},
    }

    assert holds_altered_provider_state({"choices": [{"message": message}]})
    assert without_altered_provider_state({"choices": [{"message": message}]}) == {
        "choices": [
            {
                "message": {
                    **message,
                    "thinking_blocks": [_WHOLE_THINKING, _WHOLE_REDACTED],
                    "provider_specific_fields": {"thinking_blocks": []},
                }
            }
        ]
    }


@pytest.mark.parametrize("key", ["thought_signature", "response_thought_signature"])
def test_an_altered_thought_signature_is_left_out_and_the_call_it_signed_stays(key: str):
    call: Final = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
    stored: Final = {**call, "provider_specific_fields": {key: _cut(_LONG), "a": 1}}

    assert holds_altered_provider_state(stored)
    assert without_altered_provider_state(stored) == {**call, "provider_specific_fields": {"a": 1}}


def test_altered_text_signatures_are_left_out_of_the_message_they_signed():
    fields: Final = {"thought_signatures": ["EmwKAhgB==", _cut(_LONG)], "grounding": "kept"}

    assert holds_altered_provider_state({"content": "954", "provider_specific_fields": fields})
    assert without_altered_provider_state({"content": "954", "provider_specific_fields": fields}) == {
        "content": "954",
        "provider_specific_fields": {"grounding": "kept"},
    }


def test_an_altered_reasoning_item_keeps_its_summary_without_the_encrypted_content():
    reasoning: Final = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "Counted."}],
        "encrypted_content": _cut(json.dumps([{"type": "thinking", "thinking": "Count.", "signature": _LONG}])),
    }

    assert holds_altered_provider_state([reasoning])
    assert without_altered_provider_state([reasoning]) == [
        {key: value for key, value in reasoning.items() if key != "encrypted_content"}
    ]


@pytest.mark.parametrize(
    "stored",
    [
        pytest.param([_WHOLE_THINKING, _WHOLE_REDACTED], id="whole-blocks"),
        pytest.param({"type": "reasoning", "encrypted_content": _LONG}, id="whole-reasoning-item"),
        pytest.param({"thought_signatures": [_LONG], "thought_signature": _LONG}, id="whole-thought-signatures"),
        pytest.param({"content": _cut(_LONG), "reasoning_content": _cut(_LONG)}, id="cut-unsigned-text"),
        pytest.param([{"type": "thinking", "thinking": _cut(_LONG)}], id="cut-unsigned-thinking"),
        pytest.param({"content": _cut(f"Ids hold {THOUGHT_SIGNATURE_SEPARATOR}. {_LONG}")}, id="cut-text-naming-ids"),
        pytest.param({"type": "web_search_call", "encrypted_content": _cut(_LONG)}, id="cut-state-of-another-item"),
        pytest.param(
            {"id": f"call_1{THOUGHT_SIGNATURE_SEPARATOR}{_LONG}", "type": "function"}, id="whole-signed-call-id"
        ),
        pytest.param({"id": _cut(_LONG), "type": "function"}, id="cut-unsigned-call-id"),
        pytest.param([{**_WHOLE_THINKING, "thinking": f"It holds {_cut(_LONG)} now."}], id="signed-text-quoting-a-cut"),
        pytest.param([{**_WHOLE_THINKING, "thinking": REDACTED_BY_LITELM_STRING}], id="signed-text-naming-a-mask"),
        pytest.param([{"type": ["string", "null"]}, {"type": {"enum": ["thinking"]}}], id="schemas-typed-by-no-name"),
        pytest.param(_cut(_LONG), id="plain-string-input"),
        pytest.param(None, id="nothing-stored"),
    ],
)
def test_whole_state_and_unsigned_text_replay_unchanged(stored: object):
    assert without_altered_provider_state(stored) == stored
    assert not holds_altered_provider_state(stored)


@pytest.mark.parametrize(
    ("item", "loses_a_signature"),
    [
        pytest.param({"id": "", "type": "function"}, True, id="chat-tool-call"),
        pytest.param({"call_id": "", "type": "function_call"}, True, id="function-call"),
        pytest.param({"call_id": "", "type": "custom_tool_call"}, True, id="custom-tool-call"),
        pytest.param({"id": "", "type": "tool_use"}, True, id="tool-use"),
        pytest.param({"call_id": "", "type": "function_call_output", "output": "14"}, False, id="function-call-output"),
        pytest.param({"call_id": "", "type": "apply_patch_call_output"}, False, id="call-output-that-says-nothing"),
        pytest.param({"tool_use_id": "", "type": "tool_result"}, False, id="tool-result"),
        pytest.param({"tool_call_id": "", "role": "tool"}, False, id="chat-tool-output"),
        pytest.param({"id": ""}, False, id="typed-by-no-name"),
    ],
)
@pytest.mark.parametrize("call_id", ["call_1", ""], ids=["id-and-signature", "signature-only"])
@pytest.mark.parametrize("signature", [_LONG, f"A{THOUGHT_SIGNATURE_SEPARATOR}{_LONG}"], ids=["plain", "nested"])
def test_a_cut_signed_call_id_replays_as_the_unsigned_id_it_starts_with(
    item: dict[str, str], loses_a_signature: bool, call_id: str, signature: str
):
    cut_id: Final = _cut(f"{call_id}{THOUGHT_SIGNATURE_SEPARATOR}{signature}")
    stored: Final = {key: value or cut_id for key, value in item.items()}

    assert without_altered_provider_state([stored]) == [{key: value or call_id for key, value in item.items()}]
    assert holds_altered_provider_state([stored]) == loses_a_signature


@pytest.mark.parametrize("key", [0, "a"], ids=["lists", "objects"])
def test_state_nested_past_the_depth_bound_replays_instead_of_failing(key: int | str):
    levels: Final = range(5000)
    nested: Final = reduce(lambda inner, _: [inner] if key == 0 else {key: inner}, levels, "leaf")

    replayed: Final = without_altered_provider_state(nested)

    assert reduce(lambda outer, _: outer[key], levels, replayed) == "leaf"
    assert not holds_altered_provider_state(nested)
