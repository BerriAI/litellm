from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.responses.bases.openai import GPT_5_4_TEST_CASE
from integration.translation.runner import assert_translation, translation_params

GPT_5_4_CONVERSATION_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="conversation",
    litellm_request={
        **GPT_5_4_TEST_CASE.litellm_request,
        "conversation": "conv_6acef6c3fea0819796499de845024b7b09e0bfee1dc5e202",
    },
    expected_provider_request={
        **GPT_5_4_TEST_CASE.expected_provider_request,
        "conversation": "conv_6acef6c3fea0819796499de845024b7b09e0bfee1dc5e202",
    },
    mock_provider_response={
        **GPT_5_4_TEST_CASE.mock_provider_response,
        "id": "resp_09e0bfee1dc5e202006acef6c4375481979ad43cafcb0b52fb",
        "created_at": 1791948484,
        "access_programs": None,
        "completed_at": 1791948484,
        "conversation": {"id": "conv_6acef6c3fea0819796499de845024b7b09e0bfee1dc5e202"},
        "output": [
            {
                "id": "msg_09e0bfee1dc5e202006acef6c4e4e081978a0e70cdac3f2534",
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": "Hello."}],
                "phase": "final_answer",
                "role": "assistant",
            }
        ],
    },
    expected_litellm_response={
        **GPT_5_4_TEST_CASE.expected_litellm_response,
        "created_at": 1791948484,
        "output": [
            {
                "id": "msg_09e0bfee1dc5e202006acef6c4e4e081978a0e70cdac3f2534",
                "content": [{"annotations": [], "text": "Hello.", "type": "output_text", "logprobs": []}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
                "phase": "final_answer",
            }
        ],
        "access_programs": None,
        "completed_at": 1791948484,
        "conversation": {"id": "conv_6acef6c3fea0819796499de845024b7b09e0bfee1dc5e202"},
    },
    known_bug="/v1/responses drops conversation before the OpenAI request. get_requested_response_api_optional_param keeps only ResponsesAPIOptionalRequestParams keys and conversation is not one of them",
)


@pytest.mark.parametrize("case", translation_params(GPT_5_4_CONVERSATION_TEST_CASE))
def test_responses_conversation_openai(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
