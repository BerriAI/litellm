from collections.abc import Mapping
from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.anthropic import (
    CLAUDE_FABLE_5_2_TEST_CASE,
    CLAUDE_SONNET_5_5_TEST_CASE,
)
from integration.translation.runner import assert_translation
from pydantic import JsonValue

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

CLAUDE_SONNET_5_5_CACHE_HIT_RESPONSE_FIELDS: Final[Mapping[str, JsonValue]] = {
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
}

CLAUDE_FABLE_5_2_CACHE_HIT_THINKING_SIGNATURE_0: Final = (
    "CAQSnwcKEAgUGAI4AUIIdGhpbmtpbmcSDO9ot21myDwNtnb8shoM/JOcyPAC8fi4F7K+IjB420J9hGNSC0gtujChLi6js4O8"
    "Q1g4Ya9mNSdw1K2iPBkVDcSvSumqwevWEDmRomEqvAaO8UUGH5dkKXRVtNiY0JxWx1GlyWIbw4bqSHyDMa/G1S3TChvAy6tm"
    "/x4OBSzs697HskYFGgfydz1bvDRturrteMUvxSq3009QrI7smBrE7FHTmK/klOYlTdCArzZackijBFv8p6KO+ZeYkgi45CpS"
    "3pq5c1A6E6pgWe21p9Jc4bYU/bqL9xhUBJ2dGyQ7qAc836LW/RM4kHST91Ehf/qCcDwuovXIo+uJZMmwxy0fdR1xoJGxKf8E"
    "2w3ZphAcyruOwwQ2yuAlk3gAAZ9kcRgB6sTVHEcEZh2XvDUqb8Jy2lcMBqJAwYY7cIGoCc8tNRj5T1GJZ5Uh3/jCO3x2Tl/d"
    "/m1tVL1LklPPlk8dZqfC4PQR1MSjQivvIrtAx/fdvh/imrif6dxaihlmgrXje6q47qqKq7X79ZqO7aX0OqE7wl/tSHYU04wU"
    "4UHBulSC9yqkaYp4I+RHZBB7GhcdMFlHJTySSZLh8qCTaMfJd4Ih+jFefCBivQbDA3Z5iZC/lqgaJ9KLSGmBApZtWO4lseCi"
    "i6hoy7tyvDWNZeKc/iaB84D0VGiHRs4hQA0FLSNKwndfk0TY3fq9FynMBjDnfm4LC87N0o9QV9P+nChlKnTYGLoNEn4n0IZH"
    "CJDeINUbPZHalDDN451eh6qSpkdbwhz1pFfyRtUCw91xOSAq/5wm4w3kkq7tXO2W+TpwOF8sVoO8dLKRaPrvsECYDbALz/fk"
    "Zakw4w+vNUf6FDM9JDaQ+sr52/IU2PF2a2666n3gcRaXb7v4C2iaN0ib9z1MVFugZ+iop8cT40OB28TyEuPBGagpU8sq5pxQ"
    "fu1Nrk6dFpj3dEgpGbnwUsT83TdGJE72ZHUZuZJtfBx9Q9uwGfU61ZhRGkFM/XQeI3YZdFhgeB596bXpztFZ+OB3D+saMXVi"
    "Sfc/zghbCQ3DfORHDLYWsyQrNmr+gx7WHvQT/SH5AhA9YrhXo/8JzL0Cu8ly7ETpKexJpcT3dSMxFxoVVwTX60dhPpXJJpGP"
    "IPL0XBB6xzyqa8SuBWr2LhlYyP9Lk0ltrSvQDRJLJADhKHHGn31Jy9UWnu0gBuxsZsCjOwsM+nYwiS6ZSanvc1aYTtIYAQ=="
)

CLAUDE_FABLE_5_2_CACHE_HIT_THINKING_SIGNATURE_1: Final = (
    "CAQSvAcKEQgUGAI4AUIJbmFycmF0aW9uEgyzPr8z9ml5G1UKW8oaDKMT8f48BH8Nf/gowCIw18OGmlvIwWlJwysb1brecYPf"
    "ldOjcnDHdS8M999+yQUZxELPvNJaLhcnXVoZMswLKtgGkUzT0JLOFe2r+B+6fYA5NBqOWGfB++vQwkmvPw62brwzHk8F3ApH"
    "LGf1zbJdtuKxAEDtwU2lFxGjx0D/IBujr4R3Bfr3qwY7DVDeEOBrwKcaYx7iTwvQ08fn/gHlXluYyMNMb+XBleYN5PPg93BL"
    "H7dA3CrjNo8waajfL8H2J0cQGOqOCRmfDryHV6DIsM3R/JJutlU4N6iMOu+PQF4rtGFXZQhBFL1Hl1UzsR25ChuX6iXOELNq"
    "Ihbj8yiMtV5QwoFy3CXr8aFeynDhH99dx819/EfVVmUMnxdizCYOg7sKh6QGu011dn+Afka4ZHxWDcrUKKB9tIYrju0AOXuL"
    "rFclQxML987OToY4inngLuilDSBVb0L3Z5W2QnNKWhQPaTFpJgrl4IEMLd9macQhY8med7hd7Fn7VqCCoNL+0AnibAleznPD"
    "1ZvymRMeuGR4De8M0VQn4bbjHfvGuztu7cnmr6w3cCZ5xf8mpMkyjrgBUdyTvxMZ44gan3inIdahsLo9rQg+tnOGvFedOD/t"
    "BVA0Anr1plJZcnT+XH2V6tenIbwHgtJXQl9gr2KEWm4YpnNdoQlCFwW2AS8NKRMkJ72//d1aVnFATmv/NY6Y6yq59vR/NzG6"
    "+N5FSrSQSIcRcSqjYifuelCp9vN81isovTcLyxWaWB6sXU/jPraG5Hq+c1ObJ8Sx1HysbXiz7om8dR7cKQsk1LEg6/8WHFx2"
    "1E9u4wYOkuQ7TWoaextf/k2R0cfY81y0E8YXJzTbLMDZkonqOECPZNAiQILmPcxCuyd71q5t1Q+WqtGVHCVVXoyaE6ecuyqR"
    "ovEKkGyM1pCjnZYYcZfG4WXh9bo9yQllLOGip0we2TsNTqES7BlDZcgOAkh1VRELLJjgMj5TsRPnwDZJO+HLMRndZtFqT+wz"
    "tiy4DkERzf0XpLYrFxhz+Kir6KqaTNFjbXCGPm7CvfN/25cryRbWMQPsuwa3+poh6cu4ygdnTI9iwtTriHOJU4crMmm+0X70"
    "oGHj8a/3RqIQTdDDt44Jkb1DleWp5qbBVmF4btCc2bw9uCVNxDF+gdm/uxpjiofsJhOHQPBf3OfN52f1VB6iDf636qTd0Ldu"
    "sjy5c/SaYfRs6SNTw5ApmYtqbnihqlFpbxgB"
)

CLAUDE_FABLE_5_2_CACHE_HIT_TEST_CASE: Final = replace(
    CLAUDE_FABLE_5_2_TEST_CASE,
    scenario="prompt_cache_hit",
    mock_provider_response={
        **CLAUDE_FABLE_5_2_TEST_CASE.mock_provider_response,
        "id": "msg_011ChJEmCiKmGp4ZMwg5CJ2z",
        "content": [
            {
                "type": "thinking",
                "thinking": (
                    "I'll read the validation file first, since that's what the user asked for, and the tests file "
                    "can wait. The function sits at line 18, so I'm pulling roughly lines 1 through 60 with the "
                    "read-only tool.\n\n"
                ),
                "signature": CLAUDE_FABLE_5_2_CACHE_HIT_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": (
                    "Next I'll read `shop/validation.py` from the top, since `validate_quantity` is defined at line "
                    "18 and its imports and helpers affect how zero, negative and non-integer quantities are handled."
                    "\n\n"
                ),
                "signature": CLAUDE_FABLE_5_2_CACHE_HIT_THINKING_SIGNATURE_1,
            },
            {
                "type": "tool_use",
                "id": "toolu_01Rbgt6EnUWcVbUqcA97JUgk",
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
            "output_tokens": 261,
            "output_tokens_details": {"thinking_tokens": 82},
            "service_tier": "standard",
            "inference_geo": "global",
            "speed": "standard",
        },
    },
    expected_litellm_response={
        **CLAUDE_FABLE_5_2_TEST_CASE.expected_litellm_response,
        "id": "msg_011ChJEmCiKmGp4ZMwg5CJ2z",
        "content": [
            {
                "type": "thinking",
                "thinking": (
                    "I'll read the validation file first, since that's what the user asked for, and the tests file "
                    "can wait. The function sits at line 18, so I'm pulling roughly lines 1 through 60 with the "
                    "read-only tool.\n\n"
                ),
                "signature": CLAUDE_FABLE_5_2_CACHE_HIT_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": (
                    "Next I'll read `shop/validation.py` from the top, since `validate_quantity` is defined at line "
                    "18 and its imports and helpers affect how zero, negative and non-integer quantities are handled."
                    "\n\n"
                ),
                "signature": CLAUDE_FABLE_5_2_CACHE_HIT_THINKING_SIGNATURE_1,
            },
            {
                "type": "tool_use",
                "id": "toolu_01Rbgt6EnUWcVbUqcA97JUgk",
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
            "output_tokens": 261,
            "output_tokens_details": {"thinking_tokens": 82},
            "service_tier": "standard",
            "inference_geo": "global",
            "speed": "standard",
        },
    },
)

CLAUDE_SONNET_5_5_CACHE_HIT_TEST_CASE: Final = replace(
    CLAUDE_SONNET_5_5_TEST_CASE,
    scenario="prompt_cache_hit",
    mock_provider_response={
        **CLAUDE_SONNET_5_5_TEST_CASE.mock_provider_response,
        **CLAUDE_SONNET_5_5_CACHE_HIT_RESPONSE_FIELDS,
    },
    expected_litellm_response={
        **CLAUDE_SONNET_5_5_TEST_CASE.expected_litellm_response,
        **CLAUDE_SONNET_5_5_CACHE_HIT_RESPONSE_FIELDS,
    },
)


@pytest.mark.parametrize(
    "case",
    [CLAUDE_FABLE_5_2_CACHE_HIT_TEST_CASE, CLAUDE_SONNET_5_5_CACHE_HIT_TEST_CASE],
    ids=[CLAUDE_FABLE_5_2_CACHE_HIT_TEST_CASE.id, CLAUDE_SONNET_5_5_CACHE_HIT_TEST_CASE.id],
)
def test_messages_prompt_cache_hit_anthropic(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
