from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.anthropic import CLAUDE_SONNET_4_6_TEST_CASE, CLAUDE_SONNET_5_5_TEST_CASE
from integration.translation.runner import assert_translation

CLAUDE_SONNET_4_6_SERVICE_TIER_RESPONSE_FIELDS: Final = {
    "content": [{"type": "text", "text": "Hello!"}],
    "usage": {
        "input_tokens": 18,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
        "output_tokens": 5,
        "service_tier": "standard",
        "inference_geo": "global",
    },
}

CLAUDE_SONNET_4_6_SERVICE_TIER_TEST_CASE: Final = replace(
    CLAUDE_SONNET_4_6_TEST_CASE,
    scenario="service_tier",
    litellm_request={**CLAUDE_SONNET_4_6_TEST_CASE.litellm_request, "service_tier": "standard_only"},
    expected_provider_request={
        **CLAUDE_SONNET_4_6_TEST_CASE.expected_provider_request,
        "service_tier": "standard_only",
    },
    mock_provider_response={
        **CLAUDE_SONNET_4_6_TEST_CASE.mock_provider_response,
        **CLAUDE_SONNET_4_6_SERVICE_TIER_RESPONSE_FIELDS,
        "id": "msg_011Cg1RG36pCcTLE26htFFqC",
    },
    expected_litellm_response={
        **CLAUDE_SONNET_4_6_TEST_CASE.expected_litellm_response,
        **CLAUDE_SONNET_4_6_SERVICE_TIER_RESPONSE_FIELDS,
        "id": "msg_011Cg1RG36pCcTLE26htFFqC",
    },
)


CLAUDE_SONNET_5_5_SERVICE_TIER_THINKING_SIGNATURE: Final = (
    "CAQSkQcKEAgSGAI4AUIIdGhpbmtpbmcSDEpY9gn98XnFX1pSihoMmijS1UTCmGlkINMoIjAndpfj217OaTruvzuuNVWqe7ISoH6Ju4alxqev"
    "mfITGMNOiuRIUP5U2fMyQhVrJ20qrgaZUyE2dwvMcSaJAXqpHGrHCxxl8Caev2GOUBeBgI9gZVE08HMn22idoM90TAa1NXUZkKmJ+Rw0aQEE"
    "G2r0rvUNY8xN5zFa8bCtNGYXqIC+Sy+kv5Al6J8dc8Gmkh0CX/Sij414Y71xUxNLQtzspQJ6inLuGE53jONMKYy9iW5VKZiD4KDc5GKB180i"
    "qSsPnhQh2PrgJk0btL5IG9WTogBkEolEHP7CuM/dH4AJCKQdbSfrnGtf3+hcJmO/pVceie1PxAdtWEme/BTfuDmsZ+wDDx6RcUCzsoDkQKSi"
    "ISH8A33QjJ4RRsDKCM5/JimPVW97t/AwNGh2eDLg6G/4lyAhnU0FddZtG5mgjv+dMeeYFMmdEkKDibaV+URx6OlqhRlLbVoAGQNlfX0+3Mi0"
    "t4fbJbuu6KrlH0JZ2hVcrGdUE7C8Sf+MIiZVss2HStotGydIdvBIqs8WWXQmZxK3ym8E30zBOFCNdBJ7ZAifEsiteBxxebxxuWDKiplF02id"
    "0Y0mEg0V+oY8DeNyK5yvfGXKZy3C1ejSBAJgCw5xtNZykQpivNMz+5dDcmD6HLNgo/irB7pJ+/HkqfAeAGv9IGQc/sOWv4rfz1et+SMHR42t"
    "D+4KDajydnrafn+eRJqDsEpf5B06/1QGA+YDhKwXp/QEJPGpqGyoAo3ZCs99Xt8cWdLqwiSEEfopw78xSWGEhbf9eGrPNtWm+0Zunl11NRSc"
    "pmhoUusiE3CibPiegC+xS80Dh9ffKozaq9KT6A12xWs+KErvKCHFze28o1Qlyl/bXwIX7OjWj38QWTDiV+pJ/JiiVsZHZzlTeZP9hLyzoZxu"
    "Kz0nYPgxd56wGmDBvK/ReamE4B0avh6R25zlVw7XakbNZabWFDlrMolvm5hMqZ/c4RumSrkOtBRqrCoTwaUnWVv+Ge9bv9IGDAIvqpbSgn+0"
    "fgxzRkeCP7nxqDuD05XM5Tg4pUV4b+20PT5KqPCDfkYiMrRsfLLGk++ZX9NIXiocrWT8pKaJrMBiPVN/u+l8RZhpJdzEs1dBJZMsxSAJqu2E"
    "PHM6nXt2hR0PpoYo4vsuVdiwiQXC9Jb6ZgQjGAE="
)

CLAUDE_SONNET_5_5_SERVICE_TIER_RESPONSE_FIELDS: Final = {
    "id": "msg_011Cg48jAuekewp9T2EoZqt8",
    "content": [
        {
            "type": "thinking",
            "thinking": "I'll start by reading shop/validation.py first, as the user requested, before looking at the tests.\n\n",
            "signature": CLAUDE_SONNET_5_5_SERVICE_TIER_THINKING_SIGNATURE,
        },
        {
            "type": "text",
            "text": "I'm reading `shop/validation.py` first. `validate_quantity` starts at line 18, so I'll read from just above it to get the imports and helpers, through the end of the function.",
        },
        {
            "type": "tool_use",
            "id": "toolu_017mBCZeT7c9e1duQXRRzS6x",
            "name": "read_file",
            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            "caller": {"type": "direct"},
        },
    ],
    "usage": {
        "input_tokens": 4,
        "cache_creation_input_tokens": 873,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
        "output_tokens": 251,
        "output_tokens_details": {"thinking_tokens": 94},
        "service_tier": "standard",
        "inference_geo": "global",
    },
}

CLAUDE_SONNET_5_5_SERVICE_TIER_TEST_CASE: Final = replace(
    CLAUDE_SONNET_5_5_TEST_CASE,
    scenario="service_tier",
    litellm_request={**CLAUDE_SONNET_5_5_TEST_CASE.litellm_request, "service_tier": "standard_only"},
    expected_provider_request={
        **CLAUDE_SONNET_5_5_TEST_CASE.expected_provider_request,
        "service_tier": "standard_only",
    },
    mock_provider_response={
        **CLAUDE_SONNET_5_5_TEST_CASE.mock_provider_response,
        **CLAUDE_SONNET_5_5_SERVICE_TIER_RESPONSE_FIELDS,
    },
    expected_litellm_response={
        **CLAUDE_SONNET_5_5_TEST_CASE.expected_litellm_response,
        **CLAUDE_SONNET_5_5_SERVICE_TIER_RESPONSE_FIELDS,
    },
)


@pytest.mark.parametrize(
    "case",
    [CLAUDE_SONNET_4_6_SERVICE_TIER_TEST_CASE, CLAUDE_SONNET_5_5_SERVICE_TIER_TEST_CASE],
    ids=lambda case: case.id,
)
def test_messages_service_tier_anthropic(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
