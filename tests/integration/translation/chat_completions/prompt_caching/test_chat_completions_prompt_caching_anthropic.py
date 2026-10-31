from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.bases.anthropic import CLAUDE_SONNET_5_5_TEST_CASE
from integration.translation.runner import assert_translation

CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_0: Final = (
    "CAQSkgkKEAgSGAI4AUIIdGhpbmtpbmcSDF+t4V3r32toxL3AdhoMUGxH3WEiPl1PzrbEIjBXWl/8pilac6QmCoAf2V6ccw1t8goO+vJrZI+Q"
    "P8MYvCDZVMnVW27phzm4id3agp0qrwjlUCjy0oY0HLDs74MWH3D6CY+kO2DNnYDc63CzJfbYmw0R/+pCIuRdqi7uZnep3fjdaHVw27Et0zDr"
    "mohz3SSc9P00Q0lereo8tOVPsiRezXyu5bMXUhNYM14rmH8GI3eMB8dZEZQ2iLomWTqROOfDDCNAiyfBhbJK+cMq2jrPhr19xwwcut3esQnf"
    "5ZhFtfQcHz4GSGWXnQEU53+k7Lixm89JiJrnmls9ZTzFeSkh3FFnX6cnJxzWLgm5jK+a3m1eKjO41QccoioBDI4wgzImAiUTjIptLDB6q9lg"
    "/SW5sc3YUxQPNgVJxdYzbRrKSiBgFli6C5n+yXHai4uA/4K0+uPZ2JbUH1Ofsxs+6x0Jcgawenu8GKAXuaOoqv/DWsWQwm8kNOpYaEgKm0Ld"
    "nX3q8FljhroX0DwHabcKGjh27rMHpSh5JomVChvFOGeUaiO4jKeaD/bY9CmJMTlc5anJV35LxHzvhYHNiG9vFcQjmOopIIq8pe/F9JKyC4tO"
    "v0bCH9jp0dUljq4X55fQZ2xOz2v8VpjH2cVK41N31/ifyDV2fP3BiIP0I9rS5+aos1DzhW22bL5Bsb6qinwk2fRiQXPlFitHITxYY2ORep+E"
    "t7edv5h4wN6iYM9YrVeTJr7gcUdcxeFlR2rHWl+8cyVZrH6WroyWU0DZzsaNCNlv78TBLYRO3ZRJz+mm07GP8zqVVfEHYORcyVSw9uc518uE"
    "t6Xc9cZnRzIc94675OeRflgOPKHcxDjWS7+C91g4mWx8UONWOxNkSj1k7FpP738A+89XCe0X+18JtMKrbZxwnhKZzsLbFQxARsff9dyhjq5D"
    "3ArlPpKF0DquTWuBwf7Wd9BXBNcPxeQbaPIeXFSfhmFPtp5YRQN85yiEaKvOmMxrfMYqEZ/Uu0CP1/j511y9umlnbiEWVqxMHtvqwvdtNiVl"
    "q71QUuNiCVZ4+u7HIgv87ZnuzFBzLzvIlzEPn6Kj+RCifesqVqj9VXw4rHU6ofQHKmePRw3D0t9cXSxRscv6Uks0CY31x0h2X2CTxu5jYG0O"
    "dTyMjaVyoeCtyfHf6DqrCsWJ1U6AzU+kfxIWXRFR+vNmHDlTOSUdtX/HlUB/dpuHAy1l2l9pyF0AuxAIMujraal8Wni899TnkFXSXSy8PqKM"
    "wWjygTVnx2HOvpUveL2xzoepv9kmn90dIax0vNs0fVTDskjls4A1C6xs2NfEH2LspxgcXaMbXBXoz2sm8b8d0NNQZewp/lsFgCbmpb/qjg7c"
    "pyXRKnv/sJARoW/wh27QoFQOM179O2OdGv3gSqRA3XFNuUoJlRxxyQZylTd39nnqXUFL79Hg39eNKWJZ6qF75R+D6nWEK3ZNSsjKXf53HsPw"
    "SF2xdv6nR4Pa58WgGePkzpeInZuZR8ZAgrkXp8zCMSqnKlZfIihm3vkYAQ=="
)

CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_1: Final = (
    "CAQSgAcKEQgSGAI4AUIJbmFycmF0aW9uEgyAZome6lAFZTDHqEAaDPcIxC714yOPmf/uhCIwELGsZh9B7fdio/ac5XWaTOpTTFoN8Viy24I7"
    "n3N4t++qwtl7piCH4qcmDsKRxv0AKpwGVxJEtEWtjrN0np/TwG8KMd8pUAyJBHdVgzT/KEXb3jFIt8Pk7XfQrBfjO/HNUl6l1ETczoLm1Spv"
    "6RAREIVw3VxkMZ5niHMly/eFOkfihJRQc1PvnpbcPB+fTOHknrq8T6srlYML2ZurKAb7W0E6bucp4QLdo3Gt4zKzLS2Xsl3jKEElZL8/U0Ra"
    "s9ymfh/7dO0mIA7QdRwjlJKggy0CzpfWEXj3x8OQSUwMgyYHd/fb5qFUhc6WSIMQt3YI1xrsex8YaI3r2emx/7DaeABpqCYiZAVRrwMSbL/r"
    "6sjBgJ/zGaz+XLIPL3flMvhLeAY7/5rcTQ3ub7OjAxRkmxxL0zkK8IrMLMankXHsniXj+hQA8U+ZUL4ybfXyXb7YZPHr0F/3zf/z4KJ4Q7UU"
    "OJ96n/t1B3c+tREN4c9hRbDIuWlX2WsdSdXFkHZxTO2BCeZ0soempDM+CW5QMIVQ/8fPWwyJi1ZcDq3HmL6NtmhYbswzoOweOcSwrtK+3Td1"
    "GvqOMq1DE/HhXflLVjl5wlaTj7J56GJOCgq+Aw08pJLPBjIoKxGzHrKEVO60JDDj4DLbxi0LoKVtqMzJXdkChdnFYs5ksYHSIEhmYxqSgY6r"
    "ZJ170gU8Kluarhm5GmiB1PPnRCvKEbfGNsZge93iVGMqRsWaScy4Z/pi8a5lwmKATZVVHyan93TXB1lRET9s9DG+8mt2IIm3vrMgDwfVD6pL"
    "fBlxTkxhn+ynRLnEKsuiZ3x2WHTAXIPSXPI8lYRXuJjmIPb0vD6XfPbTDLMfvVIJr7UT/+FtOX5ELKPA3HCVnCQniUNYQDeY0ZdtvmfTg19N"
    "RSbPH552/fk7wxPgFY/HeETIMHA6lTWfcyM7r7EJaA2nqVjoC+7UP8Xid2+uX8yaT02c18EqvhhwbD6GuoIUIOFXPMct1EdxpGYeqg1BRrAM"
    "ciJCkxRTyJ4ySotKH8dwW7r6xk+ZppOoU3suJBLRH+mSqh1JLp2vSoIEezxk4+gviPA79H5I+WfN0D8R8irrnfiUGLpsO1utDIeLE9GqmEEv"
    "2GTYRu5xn9KfzxgB"
)

CLAUDE_SONNET_5_5_CACHE_HIT_TEST_CASE: Final = replace(
    CLAUDE_SONNET_5_5_TEST_CASE,
    scenario="prompt_cache_hit",
    mock_provider_response={
        **CLAUDE_SONNET_5_5_TEST_CASE.mock_provider_response,
        "id": "msg_011Cg32q7o5hyKPksYEKC7na",
        "content": [
            {
                "type": "thinking",
                "thinking": "The user asked to read the validation file first, though reading the test "
                "file too would be independent. I'll stick with their explicit instruction "
                "and read the validation file first.\n"
                "\n",
                "signature": CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": "I'll check `shop/validation.py` to see how `validate_quantity` handles "
                "zero, negative, and non-integer values, and simultaneously review the "
                "zero-quantity test in `tests/test_validation.py`.\n"
                "\n",
                "signature": CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_1,
            },
            {
                "type": "tool_use",
                "id": "toolu_01J1PTHbwMYmFEqmpJuxuFGL",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
                "caller": {"type": "direct"},
            },
            {
                "type": "tool_use",
                "id": "toolu_01JSedGM3Q4DBt1C2aix1QdH",
                "name": "read_file",
                "input": {"path": "tests/test_validation.py", "start_line": 1, "end_line": 50},
                "caller": {"type": "direct"},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 876,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 444,
            "output_tokens_details": {"thinking_tokens": 248},
            "service_tier": "standard",
            "inference_geo": "global",
        },
    },
    expected_litellm_response={
        **CLAUDE_SONNET_5_5_TEST_CASE.expected_litellm_response,
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 2,
                            "caller": {"type": "direct"},
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                                "name": "read_file",
                            },
                            "id": "toolu_01J1PTHbwMYmFEqmpJuxuFGL",
                            "type": "function",
                        },
                        {
                            "index": 3,
                            "caller": {"type": "direct"},
                            "function": {
                                "arguments": '{"path": "tests/test_validation.py", "start_line": 1, "end_line": 50}',
                                "name": "read_file",
                            },
                            "id": "toolu_01JSedGM3Q4DBt1C2aix1QdH",
                            "type": "function",
                        },
                    ],
                    "reasoning_content": "The user asked to read the validation file first, "
                    "though reading the test file too would be independent. "
                    "I'll stick with their explicit instruction and read "
                    "the validation file first.\n"
                    "\n"
                    "I'll check `shop/validation.py` to see how "
                    "`validate_quantity` handles zero, negative, and "
                    "non-integer values, and simultaneously review the "
                    "zero-quantity test in `tests/test_validation.py`.\n"
                    "\n",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "The user asked to read the validation file "
                            "first, though reading the test file too "
                            "would be independent. I'll stick with "
                            "their explicit instruction and read the "
                            "validation file first.\n"
                            "\n",
                            "signature": CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_0,
                        },
                        {
                            "type": "thinking",
                            "thinking": "I'll check `shop/validation.py` to see how "
                            "`validate_quantity` handles zero, "
                            "negative, and non-integer values, and "
                            "simultaneously review the zero-quantity "
                            "test in `tests/test_validation.py`.\n"
                            "\n",
                            "signature": CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_1,
                        },
                    ],
                    "provider_specific_fields": {
                        "citations": None,
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "The user "
                                "asked to read "
                                "the "
                                "validation "
                                "file first, "
                                "though "
                                "reading the "
                                "test file too "
                                "would be "
                                "independent. "
                                "I'll stick "
                                "with their "
                                "explicit "
                                "instruction "
                                "and read the "
                                "validation "
                                "file first.\n"
                                "\n",
                                "signature": CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_0,
                            },
                            {
                                "type": "thinking",
                                "thinking": "I'll check "
                                "`shop/validation.py` "
                                "to see how "
                                "`validate_quantity` "
                                "handles zero, "
                                "negative, and "
                                "non-integer "
                                "values, and "
                                "simultaneously "
                                "review the "
                                "zero-quantity "
                                "test in "
                                "`tests/test_validation.py`.\n"
                                "\n",
                                "signature": CLAUDE_SONNET_5_5_CACHE_HIT_THINKING_SIGNATURE_1,
                            },
                        ],
                    },
                    "content": None,
                },
            }
        ],
        "usage": {
            "completion_tokens": 444,
            "prompt_tokens": 880,
            "total_tokens": 1324,
            "completion_tokens_details": {"reasoning_tokens": 248, "text_tokens": 196},
            "prompt_tokens_details": {
                "cached_tokens": 876,
                "text_tokens": 4,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 876,
            "inference_geo": "global",
            "service_tier": "standard",
        },
    },
)


@pytest.mark.parametrize(
    "case", [CLAUDE_SONNET_5_5_CACHE_HIT_TEST_CASE], ids=[CLAUDE_SONNET_5_5_CACHE_HIT_TEST_CASE.id]
)
def test_chat_completions_prompt_cache_hit_anthropic(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
