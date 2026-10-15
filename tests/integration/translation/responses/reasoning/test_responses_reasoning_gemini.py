from collections.abc import Mapping
from dataclasses import replace
from typing import Final
from unittest.mock import ANY

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.process import graceful_stop_seconds
from integration._support.provider import SharedProvider
from integration._support.wire import Reply
from integration.translation.case import SSEFrame, TranslationTestCase
from integration.translation.gemini_thoughts import LONG_SUMMARY, thoughts_off, with_thinking_config
from integration.translation.responses.bases.gemini import GEMINI_3_8_FLASH_TEST_CASE, GEMINI_RESPONSES_RESPONSE
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
    "BRZqlxnFX4XfQiS9pxNL"
)

GEMINI_3_8_FLASH_THOUGHTS_OFF_RESPONSE: Final[Mapping[str, JsonValue]] = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "text": "To see how zero, negative, and non-integer quantities are currently handled and rejected by the validation logic, the next file I need to inspect is `shop/validation.py`.",
                    },
                    {
                        "functionCall": {
                            "name": "read_file",
                            "args": {
                                "path": "shop/validation.py",
                                "start_line": 1,
                                "end_line": 50,
                            },
                            "id": "call_584229",
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
        "candidatesTokenCount": 70,
        "totalTokenCount": 790,
        "promptTokensDetails": [
            {
                "modality": "TEXT",
                "tokenCount": 354,
            },
        ],
        "thoughtsTokenCount": 366,
        "serviceTier": "standard",
    },
    "modelVersion": "gemini-3.8-flash",
    "responseId": "2y3RasgqkpyMxw-RxpeQBg",
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
                                "text": "To understand how zero, negative, and non-integer quantities are checked during validation, I need to inspect",
                            },
                        ],
                        "role": "model",
                    },
                    "index": 0,
                },
            ],
            "usageMetadata": {
                "promptTokenCount": 354,
                "candidatesTokenCount": 21,
                "totalTokenCount": 847,
                "promptTokensDetails": [
                    {
                        "modality": "TEXT",
                        "tokenCount": 354,
                    },
                ],
                "thoughtsTokenCount": 472,
                "serviceTier": "standard",
            },
            "modelVersion": "gemini-3.8-flash",
            "responseId": "3S3Rat6pNonj_uMPv9i78Qw",
        }
    ),
    SSEFrame(
        data={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": " `shop/validation.py` next.",
                            },
                        ],
                        "role": "model",
                    },
                    "index": 0,
                },
            ],
            "usageMetadata": {
                "promptTokenCount": 354,
                "candidatesTokenCount": 30,
                "totalTokenCount": 856,
                "promptTokensDetails": [
                    {
                        "modality": "TEXT",
                        "tokenCount": 354,
                    },
                ],
                "thoughtsTokenCount": 472,
                "serviceTier": "standard",
            },
            "modelVersion": "gemini-3.8-flash",
            "responseId": "3S3Rat6pNonj_uMPv9i78Qw",
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
                                        "end_line": 50,
                                        "path": "shop/validation.py",
                                        "start_line": 1,
                                    },
                                    "id": "call_5018",
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
                "candidatesTokenCount": 63,
                "totalTokenCount": 889,
                "promptTokensDetails": [
                    {
                        "modality": "TEXT",
                        "tokenCount": 354,
                    },
                ],
                "thoughtsTokenCount": 472,
                "serviceTier": "standard",
            },
            "modelVersion": "gemini-3.8-flash",
            "responseId": "3S3Rat6pNonj_uMPv9i78Qw",
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
                "candidatesTokenCount": 63,
                "totalTokenCount": 889,
                "promptTokensDetails": [
                    {
                        "modality": "TEXT",
                        "tokenCount": 354,
                    },
                ],
                "thoughtsTokenCount": 472,
                "serviceTier": "standard",
            },
            "modelVersion": "gemini-3.8-flash",
            "responseId": "3S3Rat6pNonj_uMPv9i78Qw",
        }
    ),
)

GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE: Final = replace(
    GEMINI_3_8_FLASH_TEST_CASE,
    scenario="thoughts_off",
    expected_provider_request=with_thinking_config(
        GEMINI_3_8_FLASH_TEST_CASE.expected_provider_request, {"thinkingLevel": "high", "includeThoughts": False}
    ),
    mock_provider_response=GEMINI_3_8_FLASH_THOUGHTS_OFF_RESPONSE,
    expected_litellm_response={
        **GEMINI_RESPONSES_RESPONSE,
        "model": "gemini/gemini-3.8-flash",
        "output": [
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            "To see how zero, negative, and non-integer quantities are currently handled and rejected "
                            "by the validation logic, the next file I need to inspect is `shop/validation.py`."
                        ),
                        "annotations": [],
                    },
                ],
                "phase": None,
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 50}',
                "call_id": "call_584229__thought__" + GEMINI_3_8_FLASH_THOUGHTS_OFF_THOUGHT_SIGNATURE,
                "name": "read_file",
                "type": "function_call",
                "id": "fc_call_584229__thought__" + GEMINI_3_8_FLASH_THOUGHTS_OFF_THOUGHT_SIGNATURE,
                "namespace": None,
                "status": "completed",
                "provider_specific_fields": {"thought_signature": GEMINI_3_8_FLASH_THOUGHTS_OFF_THOUGHT_SIGNATURE},
            },
        ],
        "usage": {
            "input_tokens": 354,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 354,
                "video_tokens": None,
            },
            "output_tokens": 436,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 366, "text_tokens": 70},
            "total_tokens": 790,
            "cost": None,
        },
    },
)


def with_summary(case: TranslationTestCase, scenario: str, summary: JsonValue) -> TranslationTestCase:
    reasoning: Final[Mapping[str, JsonValue]] = {"effort": "high", "summary": summary}
    return replace(
        case,
        scenario=scenario,
        litellm_request={**case.litellm_request, "reasoning": dict(reasoning)},
        expected_litellm_response={**case.expected_litellm_response, "reasoning": dict(reasoning)},
    )


SUMMARY_ASKS: Final = (
    with_summary(GEMINI_3_8_FLASH_TEST_CASE, "summary_auto", "auto"),
    with_summary(GEMINI_3_8_FLASH_TEST_CASE, "summary_concise", "concise"),
    with_summary(GEMINI_3_8_FLASH_TEST_CASE, "summary_empty", ""),
    with_summary(GEMINI_3_8_FLASH_TEST_CASE, "summary_long", LONG_SUMMARY),
    with_summary(GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE, "summary_int", 7),
    with_summary(GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE, "summary_list", ["auto"]),
    with_summary(GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE, "summary_null", None),
)


def test_gemini_keeps_thinking_and_billing_reasoning_without_thought_text_when_the_gateway_turns_it_off(
    thoughts_off: Gateway, provider: SharedProvider
) -> None:
    assert_translation(GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE, thoughts_off, provider)


@pytest.mark.parametrize("case", SUMMARY_ASKS, ids=tuple(case.id for case in SUMMARY_ASKS))
def test_a_reasoning_summary_ask_brings_thought_text_back_over_the_gateway_default(
    case: TranslationTestCase, thoughts_off: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, thoughts_off, provider)


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
        "POST", "/v1/responses", {**GEMINI_3_8_FLASH_TEST_CASE.litellm_request, "stream": True}
    )
    assert_provider_request(
        provider.received(),
        "/models/gemini-3.8-flash:streamGenerateContent?alt=sse",
        GEMINI_3_8_FLASH_TEST_CASE.expected_provider_headers,
        GEMINI_3_8_FLASH_THOUGHTS_OFF_TEST_CASE.expected_provider_request,
    )
    assert response.status_code == 200, response.text
    events: Final = tuple(
        object_value(frame.data) for frame in parse_sse_frames(response.text) if frame.data != "[DONE]"
    )
    assert [event["type"] for event in events if str(event["type"]).startswith("response.reasoning")] == []
    assert "".join(str(event["delta"]) for event in events if event["type"] == "response.output_text.delta") == (
        "To understand how zero, negative, and non-integer quantities are checked during validation, I need to "
        "inspect `shop/validation.py` next."
    )
    completed: Final = object_value(
        next(event["response"] for event in events if event["type"] == "response.completed")
    )
    assert [object_value(item)["type"] for item in list_value(completed["output"])] == ["message", "function_call"]
    assert object_value(completed["usage"])["output_tokens_details"] == {
        "audio_tokens": None,
        "reasoning_tokens": 472,
        "text_tokens": 63,
    }
