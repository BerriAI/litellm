"""Task-state suppression applies only to chat completion text extraction."""

from litellm.llms.a2a.common_utils import (
    extract_text_from_a2a_response,
    extract_text_from_a2a_response_for_guardrails,
)

_WORKING_STATUS_ONLY: dict = {
    "jsonrpc": "2.0",
    "id": "1",
    "result": {
        "kind": "task",
        "status": {
            "state": "working",
            "message": {"parts": [{"kind": "text", "text": "interim agent text"}]},
        },
    },
}


def test_chat_extractor_suppresses_working_status_message_text():
    assert extract_text_from_a2a_response(_WORKING_STATUS_ONLY) == ""


def test_guardrail_extractor_includes_working_status_message_text():
    assert extract_text_from_a2a_response_for_guardrails(_WORKING_STATUS_ONLY) == "interim agent text"
