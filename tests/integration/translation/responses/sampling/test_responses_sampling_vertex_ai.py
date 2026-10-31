from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.responses.bases.vertex_ai import GEMINI_2_5_FLASH_TEST_CASE
from integration.translation.runner import assert_translation

GENERATION_CONFIG: Final = object_value(GEMINI_2_5_FLASH_TEST_CASE.expected_provider_request["generationConfig"])
USAGE: Final = object_value(GEMINI_2_5_FLASH_TEST_CASE.expected_litellm_response["usage"])

GEMINI_2_5_FLASH_TOP_LOGPROBS_TEST_CASE: Final = replace(
    GEMINI_2_5_FLASH_TEST_CASE,
    scenario="top_logprobs",
    litellm_request={**GEMINI_2_5_FLASH_TEST_CASE.litellm_request, "top_logprobs": 2},
    expected_provider_request={
        **GEMINI_2_5_FLASH_TEST_CASE.expected_provider_request,
        "generationConfig": {**GENERATION_CONFIG, "responseLogprobs": True, "logprobs": 2},
    },
    mock_provider_response={
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "Hello."}]},
                "finishReason": "STOP",
                "avgLogprobs": -2.20536901451851e-06,
                "logprobsResult": {
                    "topCandidates": [
                        {
                            "candidates": [
                                {"token": "Hello", "logProbability": -1.3113013e-06, "tokenId": 9259},
                                {"token": "Hi", "logProbability": -15.149978, "tokenId": 10979},
                            ]
                        },
                        {
                            "candidates": [
                                {"token": ".", "logProbability": -3.0994368e-06, "tokenId": 236761},
                                {"token": "<ctrl100>", "logProbability": -13.81299, "tokenId": 106},
                            ]
                        },
                    ],
                    "chosenCandidates": [
                        {"token": "Hello", "logProbability": -1.3113013e-06, "tokenId": 9259},
                        {"token": ".", "logProbability": -3.0994368e-06, "tokenId": 236761},
                    ],
                },
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 9,
            "candidatesTokenCount": 2,
            "totalTokenCount": 30,
            "trafficType": "ON_DEMAND",
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 9}],
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 2}],
            "thoughtsTokenCount": 19,
        },
        "modelVersion": "gemini-2.5-flash",
        "createTime": "2026-10-31T23:20:20.755722Z",
        "responseId": "tHfmaoqQLrKZ9LsPvuuB-AU",
    },
    expected_litellm_response={
        **GEMINI_2_5_FLASH_TEST_CASE.expected_litellm_response,
        "output": [
            {
                "type": "message",
                "id": GEMINI_2_5_FLASH_TEST_CASE.expected_litellm_response["output"][0]["id"],
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Hello.",
                        "annotations": [],
                        "logprobs": [
                            {
                                "token": "Hello",
                                "logprob": -1.3113013e-06,
                                "bytes": None,
                                "top_logprobs": [
                                    {"token": "Hello", "logprob": -1.3113013e-06, "bytes": None},
                                    {"token": "Hi", "logprob": -15.149978, "bytes": None},
                                ],
                            },
                            {
                                "token": ".",
                                "logprob": -3.0994368e-06,
                                "bytes": None,
                                "top_logprobs": [
                                    {"token": ".", "logprob": -3.0994368e-06, "bytes": None},
                                    {"token": "<ctrl100>", "logprob": -13.81299, "bytes": None},
                                ],
                            },
                        ],
                    }
                ],
                "phase": None,
            }
        ],
        "usage": {
            **USAGE,
            "output_tokens": 21,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 19, "text_tokens": 2},
            "total_tokens": 30,
        },
    },
)


@pytest.mark.parametrize("case", [GEMINI_2_5_FLASH_TOP_LOGPROBS_TEST_CASE], ids=lambda case: case.id)
def test_responses_sampling_vertex_ai(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
