from collections.abc import Mapping
from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.process import graceful_stop_seconds
from integration._support.provider import SharedProvider
from integration._support.wire import Reply
from integration.translation.case import SSEFrame, TranslationTestCase
from integration.translation.gemini_thoughts import LONG_SUMMARY, thoughts_off, with_thinking_config
from integration.translation.messages.bases.gemini import GEMINI_3_8_FLASH_TEST_CASE, GEMINI_MESSAGES_RESPONSE
from integration.translation.runner import (
    assert_provider_request,
    assert_translation,
    encode_sse_frame,
    parse_sse_frames,
)
from pydantic import JsonValue

__all__ = ("thoughts_off",)

pytestmark = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

GEMINI_3_8_FLASH_THOUGHTS_OFF_THOUGHT_SIGNATURE: Final = (
    "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"
)

GEMINI_3_8_FLASH_THOUGHTS_OFF_RESPONSE: Final[Mapping[str, JsonValue]] = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "text": "I need to inspect `shop/validation.py` next to examine how `validate_quantity` handles zero, negative, and non-integer quantities.",
                    },
                    {
                        "functionCall": {
                            "name": "read_file",
                            "args": {
                                "end_line": 40,
                                "path": "shop/validation.py",
                                "start_line": 1,
                            },
                            "id": "call_412671",
                        },
                        "thoughtSignature": GEMINI_3_8_FLASH_THOUGHTS_OFF_THOUGHT_SIGNATURE,
                    },
                ],
                "role": "model",
            },
            "finishReason": "STOP",
            "index": 0,
            "finishMessage": "Model generated function call(s).",
        },
    ],
    "usageMetadata": {
        "promptTokenCount": 354,
        "candidatesTokenCount": 64,
        "totalTokenCount": 739,
        "promptTokensDetails": [
            {
                "modality": "TEXT",
                "tokenCount": 354,
            },
        ],
        "thoughtsTokenCount": 321,
        "serviceTier": "standard",
    },
    "modelVersion": "gemini-3.8-flash",
    "responseId": "1C3RatW-Jtuc_uMPhaLroA0",
}

GEMINI_3_8_FLASH_THOUGHTS_OFF_STREAM_THOUGHT_SIGNATURE: Final = (
    "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"
)

GEMINI_3_8_FLASH_THOUGHTS_OFF_STREAM_FRAMES: Final = (
    SSEFrame(
        data={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "I need to read `shop/validation.py` next to see how `validate_quantity` currently handles zero, negative, and",
                            },
                        ],
                        "role": "model",
                    },
                    "index": 0,
                },
            ],
            "usageMetadata": {
                "promptTokenCount": 354,
                "candidatesTokenCount": 27,
                "totalTokenCount": 676,
                "promptTokensDetails": [
                    {
                        "modality": "TEXT",
                        "tokenCount": 354,
                    },
                ],
                "thoughtsTokenCount": 295,
                "serviceTier": "standard",
            },
            "modelVersion": "gemini-3.8-flash",
            "responseId": "2C3RavD_A77W-sAP9brxyQk",
        }
    ),
    SSEFrame(
        data={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": " non-integer quantities.",
                            },
                        ],
                        "role": "model",
                    },
                    "index": 0,
                },
            ],
            "usageMetadata": {
                "promptTokenCount": 354,
                "candidatesTokenCount": 32,
                "totalTokenCount": 681,
                "promptTokensDetails": [
                    {
                        "modality": "TEXT",
                        "tokenCount": 354,
                    },
                ],
                "thoughtsTokenCount": 295,
                "serviceTier": "standard",
            },
            "modelVersion": "gemini-3.8-flash",
            "responseId": "2C3RavD_A77W-sAP9brxyQk",
        }
    ),
    SSEFrame(
        data={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "read_file",
                                    "args": {
                                        "start_line": 1,
                                        "end_line": 50,
                                        "path": "shop/validation.py",
                                    },
                                    "id": "call_654547",
                                },
                                "thoughtSignature": GEMINI_3_8_FLASH_THOUGHTS_OFF_STREAM_THOUGHT_SIGNATURE,
                            },
                        ],
                        "role": "model",
                    },
                    "index": 0,
                },
            ],
            "usageMetadata": {
                "promptTokenCount": 354,
                "candidatesTokenCount": 65,
                "totalTokenCount": 714,
                "promptTokensDetails": [
                    {
                        "modality": "TEXT",
                        "tokenCount": 354,
                    },
                ],
                "thoughtsTokenCount": 295,
                "serviceTier": "standard",
            },
            "modelVersion": "gemini-3.8-flash",
            "responseId": "2C3RavD_A77W-sAP9brxyQk",
        }
    ),
    SSEFrame(
        data={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "",
                            },
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                },
            ],
            "usageMetadata": {
                "promptTokenCount": 354,
                "candidatesTokenCount": 65,
                "totalTokenCount": 714,
                "promptTokensDetails": [
                    {
                        "modality": "TEXT",
                        "tokenCount": 354,
                    },
                ],
                "thoughtsTokenCount": 295,
                "serviceTier": "standard",
            },
            "modelVersion": "gemini-3.8-flash",
            "responseId": "2C3RavD_A77W-sAP9brxyQk",
        }
    ),
)

ADAPTIVE_THINKING: Final[Mapping[str, JsonValue]] = {"type": "adaptive"}

GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE: Final = replace(
    GEMINI_3_8_FLASH_TEST_CASE,
    scenario="thoughts_off",
    litellm_request={**GEMINI_3_8_FLASH_TEST_CASE.litellm_request, "thinking": dict(ADAPTIVE_THINKING)},
    expected_provider_request=with_thinking_config(
        GEMINI_3_8_FLASH_TEST_CASE.expected_provider_request, {"thinkingLevel": "high", "includeThoughts": False}
    ),
    mock_provider_response=GEMINI_3_8_FLASH_THOUGHTS_OFF_RESPONSE,
    expected_litellm_response={
        **GEMINI_MESSAGES_RESPONSE,
        "id": "1C3RatW-Jtuc_uMPhaLroA0",
        "model": "gemini/gemini-3.8-flash",
        "content": [
            {
                "type": "text",
                "text": (
                    "I need to inspect `shop/validation.py` next to examine how `validate_quantity` handles zero, "
                    "negative, and non-integer quantities."
                ),
            },
            {
                "type": "tool_use",
                "id": "call_412671",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 40},
                "provider_specific_fields": {"signature": GEMINI_3_8_FLASH_THOUGHTS_OFF_THOUGHT_SIGNATURE},
            },
        ],
        "usage": {"input_tokens": 354, "output_tokens": 385},
    },
)


def with_thinking(case: TranslationTestCase, scenario: str, field: str, value: JsonValue) -> TranslationTestCase:
    return replace(
        case,
        scenario=scenario,
        litellm_request={**case.litellm_request, "thinking": {**ADAPTIVE_THINKING, field: value}},
    )


THINKING_ASKS: Final = (
    with_thinking(GEMINI_3_8_FLASH_TEST_CASE, "display_summarized", "display", "summarized"),
    with_thinking(GEMINI_3_8_FLASH_TEST_CASE, "display_updates", "display", "updates"),
    with_thinking(GEMINI_3_8_FLASH_TEST_CASE, "display_int", "display", 7),
    with_thinking(GEMINI_3_8_FLASH_TEST_CASE, "display_empty", "display", ""),
    with_thinking(GEMINI_3_8_FLASH_TEST_CASE, "display_list", "display", ["omitted"]),
    with_thinking(GEMINI_3_8_FLASH_TEST_CASE, "display_long", "display", LONG_SUMMARY),
    with_thinking(GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE, "display_omitted", "display", "omitted"),
    with_thinking(GEMINI_3_8_FLASH_TEST_CASE, "summary_auto", "summary", "auto"),
    with_thinking(GEMINI_3_8_FLASH_TEST_CASE, "summary_empty", "summary", ""),
    with_thinking(GEMINI_3_8_FLASH_TEST_CASE, "summary_long", "summary", LONG_SUMMARY),
    with_thinking(GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE, "summary_int", "summary", 7),
    with_thinking(GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE, "summary_list", "summary", ["auto"]),
    with_thinking(GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE, "summary_null", "summary", None),
)


def test_gemini_keeps_thinking_and_billing_reasoning_without_thought_text_when_the_gateway_turns_it_off(
    thoughts_off: Gateway, provider: SharedProvider
) -> None:
    assert_translation(GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE, thoughts_off, provider)


@pytest.mark.parametrize("case", THINKING_ASKS, ids=tuple(case.id for case in THINKING_ASKS))
def test_a_thinking_display_or_summary_ask_brings_thought_text_back_over_the_gateway_default(
    case: TranslationTestCase, thoughts_off: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, thoughts_off, provider)


def test_display_omitted_alone_keeps_thought_text_on_the_default_gateway(
    gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(
        with_thinking(GEMINI_3_8_FLASH_TEST_CASE, "display_omitted_default", "display", "omitted"), gateway, provider
    )


def test_a_streamed_turn_keeps_thinking_and_billing_reasoning_without_thought_text(
    thoughts_off: Gateway, provider: SharedProvider
) -> None:
    provider.expect(
        Reply(
            content_type="text/event-stream",
            chunks=tuple(encode_sse_frame(frame) for frame in GEMINI_3_8_FLASH_THOUGHTS_OFF_STREAM_FRAMES),
        )
    )
    response: Final = thoughts_off.request(
        "POST", "/v1/messages", {**GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE.litellm_request, "stream": True}
    )
    assert_provider_request(
        provider.received(),
        "/models/gemini-3.8-flash:streamGenerateContent?alt=sse",
        GEMINI_3_8_FLASH_TEST_CASE.expected_provider_headers,
        GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE.expected_provider_request,
    )
    assert response.status_code == 200, response.text
    events: Final = tuple(object_value(frame.data) for frame in parse_sse_frames(response.text))
    started: Final = tuple(
        object_value(event["content_block"])["type"] for event in events if event.get("type") == "content_block_start"
    )
    assert started == ("text", "tool_use")
    deltas: Final = tuple(
        object_value(event["delta"]) for event in events if event.get("type") == "content_block_delta"
    )
    assert "".join(str(delta.get("text", "")) for delta in deltas) == (
        "I need to read `shop/validation.py` next to see how `validate_quantity` currently handles zero, negative, "
        "and non-integer quantities."
    )
    assert [object_value(event["usage"]) for event in events if event.get("type") == "message_delta"][-1] == {
        "input_tokens": 354,
        "output_tokens": 360,
    }
