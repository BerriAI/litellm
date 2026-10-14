from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.responses.bases.openai_responses import GPT_5_6_SOL_TEST_CASE
from integration.translation.runner import assert_translation, translation_params

GPT_5_6_SOL_CONVERSATION_TEST_CASE: Final = replace(
    GPT_5_6_SOL_TEST_CASE,
    scenario="conversation",
    litellm_request={
        **GPT_5_6_SOL_TEST_CASE.litellm_request,
        "conversation": "conv_6acef6c3fea0819796499de845024b7b09e0bfee1dc5e202",
    },
    expected_provider_request={
        **GPT_5_6_SOL_TEST_CASE.expected_provider_request,
        "conversation": "conv_6acef6c3fea0819796499de845024b7b09e0bfee1dc5e202",
    },
    mock_provider_response={
        **GPT_5_6_SOL_TEST_CASE.mock_provider_response,
        "id": "resp_09e0bfee1dc5e202006acefb6620d88197a2bc02e8460f3e8a",
        "created_at": 1791949670,
        "access_programs": None,
        "completed_at": 1791949671,
        "conversation": {"id": "conv_6acef6c3fea0819796499de845024b7b09e0bfee1dc5e202"},
        "output": [
            {
                "id": "msg_09e0bfee1dc5e202006acefb671ac0819791e3e93606cff7e4",
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": "Hello."}],
                "phase": "final_answer",
                "role": "assistant",
            }
        ],
        "usage": {
            "input_tokens": 34,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 40,
        },
    },
    expected_litellm_response={
        **GPT_5_6_SOL_TEST_CASE.expected_litellm_response,
        "created_at": 1791949670,
        "output": [
            {
                "id": "msg_09e0bfee1dc5e202006acefb671ac0819791e3e93606cff7e4",
                "content": [{"annotations": [], "text": "Hello.", "type": "output_text", "logprobs": []}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
                "phase": "final_answer",
            }
        ],
        "usage": {
            "input_tokens": 34,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": None,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 6,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": None},
            "total_tokens": 40,
            "cost": None,
        },
        "access_programs": None,
        "completed_at": 1791949671,
        "conversation": {"id": "conv_6acef6c3fea0819796499de845024b7b09e0bfee1dc5e202"},
    },
    known_bug="/v1/responses drops conversation before the OpenAI request. get_requested_response_api_optional_param keeps only ResponsesAPIOptionalRequestParams keys and conversation is not one of them",
)


@pytest.mark.parametrize("case", translation_params(GPT_5_6_SOL_CONVERSATION_TEST_CASE))
def test_responses_conversation_openai_responses(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
