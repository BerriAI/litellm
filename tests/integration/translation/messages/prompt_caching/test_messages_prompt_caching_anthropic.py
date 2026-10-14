from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.anthropic import CLAUDE_SONNET_5_5_TEST_CASE
from integration.translation.runner import assert_translation

CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_0: Final = (
    "CAQSiQcKEAgSGAI4AUIIdGhpbmtpbmcSDJmy4nsZ6G9GGfujbxoM2xU6lMsHnAjRbgKPIjDCdWjR2/DWUdgGbUsetewZ+xbrYi08H7fVA3Md"
    "HGr39Zv/cIwTC3Joe9594mXMJ8MqpgbWIL0Gg3BkRvPNxSzamBJMdkjuwxZbY9RPkkXIDDJ3lVI7s/Rw2jZCsw+sFVRvzehlo9ifeweRgUtJ"
    "UQ7Jw5YAXvydQ6afPTlCb5vJuumrJVQtfa2HFER7FOvVmy188kWO/D14rX5YArunnOwN4Tr6WBp36YpACfopAW1H8ysC7qmxlp/C/gXhez+4"
    "dCtWb7wU89gE4VKp3VFLTihst3ZbXvqfyP9FFcocJGswk4P2SXO53DoxTQgrkt6apmWOXffk+Bsc/8POf0D0cfx9WspHgm26HhvbkB+aiN8A"
    "5ow0jT0QKvAOi/C0epWp/uuTII0GiMHy4pH4VQq37n2Abp1yWfHu5c1OvVtpOn7ySN3rX/xRH3+l4KCIVX06RrYaTWC3n9FzUBI/zr5hMfVC"
    "cPxDOFy3rU2Duz/S2yzbhjQIUqTnMEmB93vIfrSWmKXcK37/XTNvHC6HLNpTrIGVmoXvLMJ+k9JcAtvlgtPCEJHULVnruLpMODGuEhKU/CVi"
    "kLkK4dzUbhvGIfhb0JrInjP+rxERma3+1arv+FWw8OVdKskcpZB4w6oY0tyYo38sotyvp9G+ZZp+C0rGLoxnC8ZGF1OXQ3MPexY+krWw1TxT"
    "WRGtWSSC5xxqzBc5CLBy0D0dV2XLFPpMqqo5m6HkBErnSXLbrERDcMnUZv7/+WpvsIqNNwXgtwmnV5FpO95D0yRfs/ru2KOxxYI6RllfGrX4"
    "Rcfd+6umtvFTGIl0YkCoyXAMRplnGSXo5n0xQyXKm1ODptIFml5q/gr31UsdaTvxmbb3hBES+VWskDM+XXNcgS7nNBfyuTT9hnM7SHr1NOqY"
    "osZn2ZzSRAZqz8Nz1ZTC7UQLosjWgG90AxwnW4yXhjm2t8BKv82xKeqQwOeKrxBTzPi5+uPfLb3WFBfk2yHBAYz9obGkN/XPISGiTUsX5h9f"
    "M3MJOPuFjlToForJFiC14XK5bTJnE34jcVC86adk8rRmoYGVVNx1exFhj0Ewadtfo3z/G5Ufl2d1PC0xfzrtKMzJocR0X+ftm0lI39OLeSPo"
    "niP1PuAsUkGk467NXLgH0ulqiRgB"
)

CLAUDE_SONNET_5_5_CACHE_HIT_TEST_CASE: Final = replace(
    CLAUDE_SONNET_5_5_TEST_CASE,
    scenario="prompt_cache_hit",
    mock_provider_response={
        **CLAUDE_SONNET_5_5_TEST_CASE.mock_provider_response,
        "id": "msg_011Cg32qgbJRU3vZX2TB6vPx",
        "content": [
            {
                "type": "thinking",
                "thinking": "I'll start by reading shop/validation.py since the user asked for that "
                "first, then move on to check the related test file after.\n"
                "\n",
                "signature": CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_0,
            },
            {
                "type": "text",
                "text": "I need `shop/validation.py` first, because `validate_quantity` at line 18 holds "
                "the rule. I'll read the function and some of the code around it, in case there "
                "are helpers or constants it depends on.",
            },
            {
                "type": "tool_use",
                "id": "toolu_01U6Ch75cPbVpbpsBEyPP1AN",
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
            "output_tokens": 254,
            "output_tokens_details": {"thinking_tokens": 92},
            "service_tier": "standard",
            "inference_geo": "global",
        },
    },
    expected_litellm_response={
        **CLAUDE_SONNET_5_5_TEST_CASE.expected_litellm_response,
        "id": "msg_011Cg32qgbJRU3vZX2TB6vPx",
        "content": [
            {
                "type": "thinking",
                "thinking": "I'll start by reading shop/validation.py since the user asked for that "
                "first, then move on to check the related test file after.\n"
                "\n",
                "signature": CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_0,
            },
            {
                "type": "text",
                "text": "I need `shop/validation.py` first, because `validate_quantity` at line 18 holds "
                "the rule. I'll read the function and some of the code around it, in case there "
                "are helpers or constants it depends on.",
            },
            {
                "type": "tool_use",
                "id": "toolu_01U6Ch75cPbVpbpsBEyPP1AN",
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
            "output_tokens": 254,
            "output_tokens_details": {"thinking_tokens": 92},
            "service_tier": "standard",
            "inference_geo": "global",
        },
    },
)


@pytest.mark.parametrize(
    "case", [CLAUDE_SONNET_5_5_CACHE_HIT_TEST_CASE], ids=[CLAUDE_SONNET_5_5_CACHE_HIT_TEST_CASE.id]
)
def test_messages_prompt_cache_hit_anthropic(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
