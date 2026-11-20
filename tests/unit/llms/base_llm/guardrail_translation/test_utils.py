import json
from typing import Final

import pytest

import litellm
from litellm.llms.base_llm.guardrail_translation.utils import (
    blocked_response_usage,
    blocked_responses_stream_usage,
    effective_skip_system_message_for_guardrail,
    effective_skip_tool_message_for_guardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.lakera_ai_v2 import LakeraAIGuardrail
from litellm.types.llms.openai import ResponseAPIUsage


@pytest.mark.parametrize(
    ("per_guardrail", "global_default", "expected"),
    [
        (True, False, True),
        (False, True, False),
        (None, True, True),
        (None, False, False),
    ],
)
def test_a_per_guardrail_skip_flag_wins_over_the_global_setting(
    monkeypatch: pytest.MonkeyPatch, per_guardrail: bool | None, global_default: bool, expected: bool
):
    monkeypatch.setattr(litellm, "skip_system_message_in_guardrail", global_default)
    monkeypatch.setattr(litellm, "skip_tool_message_in_guardrail", global_default)
    guardrail: Final = LakeraAIGuardrail(
        api_key="lakera-test-key",
        skip_system_message_in_guardrail=per_guardrail,
        skip_tool_message_in_guardrail=per_guardrail,
    )

    assert effective_skip_system_message_for_guardrail(guardrail) is expected
    assert effective_skip_tool_message_for_guardrail(guardrail) is expected


def test_blocked_response_usage_reads_a_message_delta_whose_json_holds_unicode_line_separators() -> None:
    message_start: Final = {"type": "message_start", "message": {"usage": {"input_tokens": 11, "output_tokens": 1}}}
    message_delta: Final = {
        "type": "message_delta",
        "delta": {"stop_reason": "stop_sequence", "stop_sequence": "END\u2028OF\u2029TURN"},
        "usage": {"output_tokens": 7},
    }
    stream: Final = "".join(
        f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        for event in (message_start, message_delta)
    ).encode()

    assert blocked_response_usage([stream]) == {"input_tokens": 11, "output_tokens": 7}


@pytest.mark.parametrize("terminal_event_type", ["response.completed", "response.incomplete"])
def test_blocked_responses_stream_usage_reads_the_terminal_event_of_a_blocked_stream(terminal_event_type: str) -> None:
    upstream_usage: Final = ResponseAPIUsage(input_tokens=14, output_tokens=20, total_tokens=34)
    events_before_the_block: Final = [
        {"type": "response.output_text.delta", "delta": "hi"},
        {"type": terminal_event_type, "response": {"usage": upstream_usage}},
    ]

    assert blocked_responses_stream_usage(events_before_the_block) == upstream_usage
