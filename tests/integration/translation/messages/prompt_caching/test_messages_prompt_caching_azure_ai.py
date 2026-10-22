from collections.abc import Mapping
from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.azure_ai import CLAUDE_FABLE_5_1_TEST_CASE, CLAUDE_HAIKU_5_5_TEST_CASE
from integration.translation.runner import assert_translation
from pydantic import JsonValue

CLAUDE_HAIKU_5_5_CACHE_HIT_THINKING_SIGNATURE_0: Final = (
    "CAQS6gUKEAgSGAI4AUIIdGhpbmtpbmcSDLQu922okJauVttVBBoMAgcnYaXPP2pLp7kWIjBjWY/rKcIkRW/cK1zER1RvizqI4Z+33qxyJw9F"
    "e+n1/BHcFPZB/YEHmGpyettyI5QqhwUoNUlt00bsx60ekxYVjn/ocAkKHVywtpoVpn65SrGRBSzMIfVoEtecE/05Gzvy1GgO8L1995thSB+M"
    "E9+Lzc+bvIdirQ/jLWnlnE1L/Vg6/YtM4eUy8a4c8nd//zCPBxgHkZZDrvOKfFE2Md8eKPoB6Uh64DxV0qdcWpPqxdCviDEJzhARb0WMxOWt"
    "irKLLOTF+oc546ftUkF5xsbk+qN/96EVdjNwsYWr8OGv+OQdpEpJsT1ANkkverU2iFQ4oNJnW3ORn0uv6uccIRqFuyTFqXN4JefRGUvc8vtV"
    "iMdrUr3rIu6j/HCu9Lstm2+SVBnV1wWFzFi8HiIupfOebCGYb5LhvYlWR6UJPk8snpbw9/MfiCqLn+wW/pKvaTSQtuKvQUORftkyVeBmy0Mx"
    "UoeSiImk/CqbZ7u7t6wakbFXVZPc5zB1FJe9ymWVCACAFPayHJRLFU93wHjZ2ChPwYhG75HpiA3OordlF5usKP6orO95WU4h8w2FPuX0M2Zz"
    "K78MvESEUn/tcf0qthco+y9A1RCPF+NAdgagx3dDXNatT7/4W33yTdf/ZjFFS8UFKJ/vS83Ehp+OhUzVNB4g1g9134JqvUz040Mrctqnibt1"
    "+7YKm4nLOHuYxETr5JyHW/noo2X/aNVz8dbL64F6grNe/yofT/D2u7YuaiYnYCwmkQLCK5VmmwExUq43pklgYmgv3X2jaF2f4B6Ph84e3rDV"
    "1PQ9AvTgM3WKf0c57NtoMAg4K8NalHtPBELETqI0BnoZt6BW1ltm8ezQoiShdkr8Jy0SpjuaxQmaX9RsoLo1z6LfzaMAf5sXWQFHF8ata5t0"
    "bCzIKmeGV5j0+YHZ+wn4HAWu5XjrLhgB"
)

CLAUDE_HAIKU_5_5_CACHE_HIT_RESPONSE_FIELDS: Final[Mapping[str, JsonValue]] = {
    "id": "msg_011Cg37cRp2MtyQ7pW4fdgJX",
    "content": [
        {
            "type": "thinking",
            "thinking": "I'll read shop/validation.py, starting with a reasonable initial range to see its "
            "contents.\n\n",
            "signature": CLAUDE_HAIKU_5_5_CACHE_HIT_THINKING_SIGNATURE_0,
        },
        {
            "type": "tool_use",
            "id": "toolu_01P1gGqhK77Zpmcf2DrLYAUX",
            "name": "read_file",
            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            "caller": {"type": "direct"},
        },
    ],
    "usage": {
        "input_tokens": 4,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 873,
        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
        "output_tokens": 167,
        "output_tokens_details": {"thinking_tokens": 67},
        "service_tier": "standard",
        "inference_geo": "global",
    },
}

CLAUDE_HAIKU_5_5_CACHE_HIT_TEST_CASE: Final = replace(
    CLAUDE_HAIKU_5_5_TEST_CASE,
    scenario="prompt_cache_hit",
    mock_provider_response={
        **CLAUDE_HAIKU_5_5_TEST_CASE.mock_provider_response,
        **CLAUDE_HAIKU_5_5_CACHE_HIT_RESPONSE_FIELDS,
    },
    expected_litellm_response={
        **CLAUDE_HAIKU_5_5_TEST_CASE.expected_litellm_response,
        **CLAUDE_HAIKU_5_5_CACHE_HIT_RESPONSE_FIELDS,
    },
)

CLAUDE_FABLE_5_1_CACHE_HIT_RESPONSE_FIELDS: Final[Mapping[str, JsonValue]] = {
    "id": "msg_011Cg37VFyjKTzJLau9mXGWM",
    "content": [
        {
            "type": "text",
            "text": "I'll read the validation function along with its surrounding context, since the test "
            "file also matters but doesn't depend on what the validator says, so I'll fetch both "
            "in one go.",
        },
        {
            "type": "tool_use",
            "id": "toolu_01Dr7SroCW5TuKG8UrXY8vad",
            "name": "read_file",
            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 40},
            "caller": {"type": "direct"},
        },
        {
            "type": "tool_use",
            "id": "toolu_0161J4pCkkCDr4kTDeBdrLFE",
            "name": "read_file",
            "input": {"path": "tests/test_validation.py", "start_line": 1, "end_line": 40},
            "caller": {"type": "direct"},
        },
    ],
    "usage": {
        "input_tokens": 4,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 873,
        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
        "output_tokens": 248,
        "output_tokens_details": {"thinking_tokens": 0},
        "service_tier": "standard",
        "inference_geo": "not_available",
    },
}

CLAUDE_FABLE_5_1_CACHE_HIT_TEST_CASE: Final = replace(
    CLAUDE_FABLE_5_1_TEST_CASE,
    scenario="prompt_cache_hit",
    mock_provider_response={
        **CLAUDE_FABLE_5_1_TEST_CASE.mock_provider_response,
        **CLAUDE_FABLE_5_1_CACHE_HIT_RESPONSE_FIELDS,
    },
    expected_litellm_response={
        **CLAUDE_FABLE_5_1_TEST_CASE.expected_litellm_response,
        **CLAUDE_FABLE_5_1_CACHE_HIT_RESPONSE_FIELDS,
    },
)


@pytest.mark.parametrize(
    "case",
    [CLAUDE_HAIKU_5_5_CACHE_HIT_TEST_CASE, CLAUDE_FABLE_5_1_CACHE_HIT_TEST_CASE],
    ids=[CLAUDE_HAIKU_5_5_CACHE_HIT_TEST_CASE.id, CLAUDE_FABLE_5_1_CACHE_HIT_TEST_CASE.id],
)
def test_messages_prompt_cache_hit_azure_ai(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
