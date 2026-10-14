from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.openai import GPT_5_4_TEST_CASE
from integration.translation.runner import assert_translation, translation_params

GPT_5_4_SERVICE_TIER_TEST_CASE: Final = replace(
    GPT_5_4_TEST_CASE,
    scenario="service_tier",
    litellm_request={**GPT_5_4_TEST_CASE.litellm_request, "service_tier": "auto"},
    expected_provider_request={**GPT_5_4_TEST_CASE.expected_provider_request, "service_tier": "auto"},
    mock_provider_response={
        **GPT_5_4_TEST_CASE.mock_provider_response,
        "id": "resp_0738fe6d3b086cd1006acef6c23a1487d2860e02c53af945a6",
        "created_at": 1791948482,
        "access_programs": None,
        "completed_at": 1791948483,
        "output": [
            {
                "id": "msg_0738fe6d3b086cd1006acef6c310cc87d283ecc2cc1d6ff29c",
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
        "id": "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOm9wZW5haTttb2RlbF9pZDo3NGI3ZDQzZjBmNjQ1MjlkYWY4NWQ4MjRmNjE3ZGYzNDRjZTdhNGI3NjI3YTYzMDMzZGUzMDVmNTE2NTRlZjc2O3Jlc3BvbnNlX2lkOnJlc3BfMDczOGZlNmQzYjA4NmNkMTAwNmFjZWY2YzIzYTE0ODdkMjg2MGUwMmM1M2FmOTQ1YTY=",
    },
)


@pytest.mark.parametrize("case", translation_params(GPT_5_4_SERVICE_TIER_TEST_CASE))
def test_messages_service_tier_openai(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
