from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.anthropic import CLAUDE_SONNET_4_6 as BASE
from integration.translation.runner import run

SIGNATURE_1: Final = (
    "EpECCqgBCBIYAipAivUPApu85FYYe3+cXal8EiJOza7QGqKyekC8vDSn4oyeqGa2CrarO4abiuG7dzBXjmYR8+daw4h50ZjKmak7czIRY2xh"
    "dWRlLXNvbm5ldC00LTY4AEIIdGhpbmtpbmdaJGQwMDgxZjJiLWQ5NjEtNGFhYi05ZTRjLTcxYmU3ZTA0ZTY3MJoBEwoRY2xhdWRlLXNvbm5l"
    "dC00LTaoAY3fhdYGEgwJHsjNkCTlV9k1jWQaDOmeP/z67YtLTojSqCIwhRXGrNzSuGfMD1HqA72lctQCy83Wkr0u8W5lBXXn+MD6WfJGTJqM"
    "1FW7qRmOMOKJKhbheeMpsTs7XvdvsiDQqgM4PAJt4cwgGAE="
)

THINKING_BUDGET: Final = replace(
    BASE,
    scenario="thinking_budget",
    litellm_request={
        **BASE.litellm_request,
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    },
    expected_provider_request={
        **BASE.expected_provider_request,
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    },
    mock_provider_response={
        **BASE.mock_provider_response,
        "id": "msg_011CffzUREgTzMm1dXRqP2LR",
        "content": [
            {"type": "thinking", "thinking": "Hello!", "signature": SIGNATURE_1},
            {"type": "text", "text": "Hello!"},
        ],
        "usage": {
            "input_tokens": 47,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 15,
            "output_tokens_details": {"thinking_tokens": 7},
            "service_tier": "standard",
            "inference_geo": "global",
        },
    },
    expected_litellm_response={
        **BASE.expected_litellm_response,
        "id": "msg_011CffzUREgTzMm1dXRqP2LR",
        "content": [
            {"type": "thinking", "thinking": "Hello!", "signature": SIGNATURE_1},
            {"type": "text", "text": "Hello!"},
        ],
        "usage": {
            "input_tokens": 47,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 15,
            "output_tokens_details": {"thinking_tokens": 7},
            "service_tier": "standard",
            "inference_geo": "global",
        },
    },
)


@pytest.mark.parametrize("case", [THINKING_BUDGET], ids=lambda case: case.id)
def test_messages_reasoning_anthropic(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    run(case, gateway, provider)
