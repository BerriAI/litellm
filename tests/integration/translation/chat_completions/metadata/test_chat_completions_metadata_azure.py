from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.bases.azure import GPT_5_4_TEST_CASE
from integration.translation.runner import assert_translation, translation_params

GPT_5_4_METADATA_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="metadata",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "store": True, "metadata": {"ticket": "123"}},
    expected_provider_request={
        **GPT_5_4_TEST_CASE.expected_provider_request,
        "store": True,
        "metadata": {"ticket": "123"},
    },
    mock_provider_response={
        **GPT_5_4_TEST_CASE.mock_provider_response,
        "created": 1791949855,
        "id": "chatcmpl-EYkIxV64n9Fh02y1VXv778rj2925S",
        "routing": {"serving_pipereplica": "d20261011005103-46bc738d-default-r0-dp0-default"},
        "usage": {
            "completion_tokens": 5,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "latency_checkpoint": {
                "engine_tbt_ms": 10,
                "engine_ttft_ms": 78,
                "engine_ttlt_ms": 136,
                "pre_inference_ms": 195,
                "service_tbt_ms": 11,
                "service_ttft_ms": 308,
                "service_ttlt_ms": 361,
                "user_visible_ttft_ms": 113,
            },
            "prompt_tokens": 19,
            "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
            "total_tokens": 24,
        },
    },
    expected_litellm_response={
        **GPT_5_4_TEST_CASE.expected_litellm_response,
        "id": "chatcmpl-EYkIxV64n9Fh02y1VXv778rj2925S",
        "created": 1791949855,
        "usage": {
            "completion_tokens": 5,
            "prompt_tokens": 19,
            "total_tokens": 24,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
            "latency_checkpoint": {
                "engine_tbt_ms": 10,
                "engine_ttft_ms": 78,
                "engine_ttlt_ms": 136,
                "pre_inference_ms": 195,
                "service_tbt_ms": 11,
                "service_ttft_ms": 308,
                "service_ttlt_ms": 361,
                "user_visible_ttft_ms": 113,
            },
        },
        "routing": {"serving_pipereplica": "d20261011005103-46bc738d-default-r0-dp0-default"},
    },
    known_bug="/v1/chat/completions treats metadata as LiteLLM logging metadata and only forwards it to Azure OpenAI when litellm.enable_preview_features is on, so the provider never receives the caller's metadata",
)


@pytest.mark.parametrize("case", translation_params(GPT_5_4_METADATA_TEST_CASE))
def test_chat_completions_metadata_azure(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
