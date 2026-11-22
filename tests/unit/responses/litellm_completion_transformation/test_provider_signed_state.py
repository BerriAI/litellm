import json
from typing import Final

import pytest

from litellm.constants import REDACTED_BY_LITELM_STRING
from litellm.litellm_core_utils.prompt_templates.factory import THOUGHT_SIGNATURE_SEPARATOR
from litellm.proxy.spend_tracking.spend_tracking_utils import _sanitize_request_body_for_spend_logs_payload
from litellm.responses.litellm_completion_transformation.provider_signed_state import without_altered_provider_state

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
        pytest.param({**_WHOLE_THINKING, "thinking": f"Key {REDACTED_BY_LITELM_STRING}."}, id="masked-signed-text"),
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

    replayed: Final = without_altered_provider_state({**call, "provider_specific_fields": {key: _cut(_LONG), "a": 1}})

    assert replayed == {**call, "provider_specific_fields": {"a": 1}}


def test_altered_text_signatures_are_left_out_of_the_message_they_signed():
    fields: Final = {"thought_signatures": ["EmwKAhgB==", _cut(_LONG)], "grounding": "kept"}

    assert without_altered_provider_state({"content": "954", "provider_specific_fields": fields}) == {
        "content": "954",
        "provider_specific_fields": {"grounding": "kept"},
    }


def test_an_altered_reasoning_item_keeps_its_summary_without_the_encrypted_content():
    reasoning: Final = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "Counted."}],
        "encrypted_content": json.dumps([{"type": "thinking", "thinking": "Count.", "signature": _cut(_LONG)}]),
    }

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
        pytest.param({"type": "web_search_call", "encrypted_content": _cut(_LONG)}, id="cut-state-of-another-item"),
        pytest.param(
            {"id": _cut(f"call_1{THOUGHT_SIGNATURE_SEPARATOR}{_LONG}"), "type": "function"}, id="cut-signed-call-id"
        ),
        pytest.param(_cut(_LONG), id="plain-string-input"),
        pytest.param(None, id="nothing-stored"),
    ],
)
def test_whole_state_and_unsigned_text_replay_unchanged(stored: object):
    assert without_altered_provider_state(stored) == stored
