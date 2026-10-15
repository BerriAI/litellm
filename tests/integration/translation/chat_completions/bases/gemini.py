from collections.abc import Mapping
from typing import Final
from unittest.mock import ANY

from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.templates import CHAT_COMPLETIONS_REQUEST, CHAT_COMPLETIONS_RESPONSE
from integration.translation.templates.gemini import (
    GEMINI_3_8_FLASH_THOUGHT,
    GEMINI_3_8_FLASH_THOUGHT_SIGNATURE,
    GEMINI_GENERATE_CONTENT_REQUEST,
    GEMINI_GENERATE_CONTENT_RESPONSE,
)
from pydantic import JsonValue

GEMINI_3_8_FLASH_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "gemini/gemini-3.8-flash"},
    expected_provider_endpoint="/models/gemini-3.8-flash:generateContent",
    expected_provider_headers={"content-type": "application/json", "x-goog-api-key": "synthetic-gemini-key"},
    expected_provider_request=GEMINI_GENERATE_CONTENT_REQUEST,
    mock_provider_response=GEMINI_GENERATE_CONTENT_RESPONSE,
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "gemini/gemini-3.8-flash",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": (
                        "To check how zero, negative, and non-integer quantities (such as floats, non-numeric values, or "
                        "values less than or equal to zero) are validated, I need to inspect `shop/validation.py` next."
                    ),
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "provider_specific_fields": {
                                "thought_signature": GEMINI_3_8_FLASH_THOUGHT_SIGNATURE,
                            },
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "end_line": 50, "start_line": 1}',
                                "name": "read_file",
                            },
                            "id": "call_7668__thought__" + GEMINI_3_8_FLASH_THOUGHT_SIGNATURE,
                            "type": "function",
                        },
                    ],
                    "images": [],
                    "reasoning_content": GEMINI_3_8_FLASH_THOUGHT,
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": GEMINI_3_8_FLASH_THOUGHT,
                        },
                    ],
                    "provider_specific_fields": {
                        "thought_signatures": [
                            GEMINI_3_8_FLASH_THOUGHT_SIGNATURE,
                        ],
                    },
                },
                "provider_specific_fields": {
                    "native_finish_reason": "STOP",
                },
            },
        ],
        "usage": {
            "completion_tokens": 655,
            "prompt_tokens": 358,
            "total_tokens": 1013,
            "completion_tokens_details": {
                "reasoning_tokens": 575,
                "text_tokens": 80,
            },
            "prompt_tokens_details": {
                "text_tokens": 358,
            },
        },
        "vertex_ai_citation_metadata": [],
        "vertex_ai_grounding_metadata": [],
        "vertex_ai_safety_results": [],
        "vertex_ai_url_context_metadata": [],
    },
)

GEMINI_3_1_PRO_PREVIEW_THOUGHT: Final = (
    "**Analyzing Quantity Validation in shop/validation.py**\n\n"
    "Okay, so the goal here is pretty clear: the user wants to understand how the system validates th"
    "e `quantity` field, specifically how zero and negative quantities are handled.  My job is to fin"
    "d the relevant code and explain what's happening.\n\n"
    "The search results pointed to `shop/validation.py`, and specifically to a `validate_quantity` fu"
    "nction on line 18. Excellent! That's a direct lead.\n\n"
    "Now, I need to get a look at that code. To do that, I'm going to leverage the `read_file` tool. "
    "I'm thinking I'll read a block of code around that line – let's say from line 10 to 30 – to give"
    " us some context around the function itself.  That should give me enough to validate our underst"
    "anding of the user request.\n\nIn summary, I'm going to read the `shop/validation.py` file to exam"
    "ine the `validate_quantity` function, then I will call the tool.\n\n\n"
)

GEMINI_3_1_PRO_PREVIEW_THOUGHT_SIGNATURE: Final = (
    "ErAFCq0FAWkUfRNlFv/lCrWnTeRSJUQxveeFy7unSAIXuMtxRKgoOGMtvJFe+qjPDFHdUuZP3JdL5ppGcmh8B8rxQ2ixle8Z"
    "4kyuY1ZCCxoXUYj4al2xk6tKOKmB9Nwcyk8EMfl34HkY4wwzPdnHd5LiHDpdOG+LRDtvCTn0sIZ5vp7xEba4z/3OoSJkNZ5Y"
    "dhprkDwNqSEJrCujrSwejQlRBarfIBzKmIMsOuOVuX+jTOZjLjuY2BQ+WaFBze/Nkl3lqBEs1ItCelts/L+6SAFvuzeCdcrt"
    "ahBH6+t8M9ZPOCkWcvrCiuwi8xF3AzAhTdTfEcECHFB4HpiJbuPRGNWVNO+c0Xj1tti+oOM6hamlxxiNiE6qEu6fVGI/f14u"
    "9fPL5XpGla06CtB5YIo94WifNUdpaF1ydl1VNs7N/qJwH0glZmTqiNGuP3SyURtRg/S7TTIHp2dMyTYWc/baAl0p6g5vJuHd"
    "6vN3yBOa8jnUS7di1WKamCFnqKck8vmfXHGwiWsUVHn2SKrI/PKoWOTTw3uc3+Z37pRl/+L9SpNdbfecvOqTrqw12iVi38QY"
    "HEukvJu6FqlRQOvOt3wPbb/RAVStD0fpRTQnTkYrS/lI2inNGrDWWpPpi/0u7dWrzJFOGO/7jV+63BggBPplEVcEkCmZrrZC"
    "2XAsnjzTzAhPzgZ8kR0m0pvjq2lbJgREIeai4ygQ6q0OQkclZmQChISjtw1K/lSuuNlJCrwZFn9Xsp6JaJ6GrGkaJsbgCOOH"
    "bsDyGOba58qNXvvHq33A0km8GthfqiEoEwxLcHwyH3jDDnuYf+/XdqdckdXcMf1T9HcMF1xMhYdkjlYSgVP5iRMIJ/2R1WgK"
    "gXoTgXjLCfHUnUUjFFjpgJwEzqfd8M+7cJfS9tcK07Pu4FSmuZcbCuDX9w=="
)

GEMINI_3_1_PRO_PREVIEW_RESPONSE_FIELDS: Final[Mapping[str, JsonValue]] = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "text": GEMINI_3_1_PRO_PREVIEW_THOUGHT,
                        "thought": True,
                    },
                    {
                        "text": (
                            "I will now read the `shop/validation.py` file to see the implementation of the `validate_quantit"
                            "y` function."
                        ),
                    },
                    {
                        "functionCall": {
                            "name": "read_file",
                            "args": {
                                "path": "shop/validation.py",
                                "start_line": 10,
                                "end_line": 30,
                            },
                            "id": "call_1654638",
                        },
                        "thoughtSignature": GEMINI_3_1_PRO_PREVIEW_THOUGHT_SIGNATURE,
                    },
                ],
                "role": "model",
            },
            "finishReason": "STOP",
            "index": 0,
            "finishMessage": "Model generated function call(s).",
        },
    ],
    "usageMetadata": {
        "promptTokenCount": 358,
        "candidatesTokenCount": 60,
        "totalTokenCount": 568,
        "promptTokensDetails": [
            {
                "modality": "TEXT",
                "tokenCount": 358,
            },
        ],
        "thoughtsTokenCount": 150,
        "serviceTier": "standard",
    },
    "modelVersion": "gemini-3.1-pro-preview",
    "responseId": "2ybQaqy6Gd2i6dkPm82G2AU",
}

GEMINI_3_1_PRO_PREVIEW_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "gemini/gemini-3.1-pro-preview"},
    expected_provider_endpoint="/models/gemini-3.1-pro-preview:generateContent",
    expected_provider_headers={"content-type": "application/json", "x-goog-api-key": "synthetic-gemini-key"},
    expected_provider_request=GEMINI_GENERATE_CONTENT_REQUEST,
    mock_provider_response={**GEMINI_GENERATE_CONTENT_RESPONSE, **GEMINI_3_1_PRO_PREVIEW_RESPONSE_FIELDS},
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "gemini/gemini-3.1-pro-preview",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": (
                        "I will now read the `shop/validation.py` file to see the implementation of the `validate_quantit"
                        "y` function."
                    ),
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "provider_specific_fields": {
                                "thought_signature": GEMINI_3_1_PRO_PREVIEW_THOUGHT_SIGNATURE,
                            },
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 10, "end_line": 30}',
                                "name": "read_file",
                            },
                            "id": "call_1654638__thought__" + GEMINI_3_1_PRO_PREVIEW_THOUGHT_SIGNATURE,
                            "type": "function",
                        },
                    ],
                    "images": [],
                    "reasoning_content": GEMINI_3_1_PRO_PREVIEW_THOUGHT,
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": GEMINI_3_1_PRO_PREVIEW_THOUGHT,
                        },
                    ],
                    "provider_specific_fields": {
                        "thought_signatures": [
                            GEMINI_3_1_PRO_PREVIEW_THOUGHT_SIGNATURE,
                        ],
                    },
                },
                "provider_specific_fields": {
                    "native_finish_reason": "STOP",
                },
            },
        ],
        "usage": {
            "completion_tokens": 210,
            "prompt_tokens": 358,
            "total_tokens": 568,
            "completion_tokens_details": {
                "reasoning_tokens": 150,
                "text_tokens": 60,
            },
            "prompt_tokens_details": {
                "text_tokens": 358,
            },
        },
        "vertex_ai_citation_metadata": [],
        "vertex_ai_grounding_metadata": [],
        "vertex_ai_safety_results": [],
        "vertex_ai_url_context_metadata": [],
    },
)

GEMINI_4_ARGON_THOUGHT: Final = (
    "**My Analysis of Quantity Validation**\n\n"
    "Okay, so the user wants me to start by looking at `shop/validation.py`. Right, that makes perfec"
    "t sense given the instructions and the task. I need to understand how the system handles quantit"
    "ies, especially zero, negative, and potentially non-integer values. It's crucial for me to have "
    "that foundational knowledge before I even *think* about suggesting a fix, and the system wants m"
    "e to ensure this is understood and considered.\n\n"
    "Therefore, I'll state explicitly that I need to examine the `validate_quantity` function within "
    "`shop/validation.py`. This will allow me to properly evaluate how quantities are being handled i"
    "n detail.\n\nNow, I will use the `read_file` tool on `shop/validation.py`.\n\n\n"
)

GEMINI_4_ARGON_THOUGHT_SIGNATURE: Final = (
    "Er0FCroFAWkUfRNrM27jPbX+vFoKJwoTVY0Na19E0vke/4MAfKRQYmXkW61lN+IKyH0lUQrByoh3xOQWzUqZrNCCJXa492UF"
    "RkKVWhi+PVV+fwUBYsmlj7U9Yn379+WUNYt8rkl04KN7jmZJqzrP7T0A58rciLtp4t8vw1I+yi0+OZRBrvg9iU0gyCeWldbp"
    "icQEAiwJBKYIJ9iiy5nFwhwbjANtiqo3WflAuBar1rWfnOq7aU3OOu6x5ZWKVABmTylKPBVCZeVn5ey5t/XJWxPPCF6ObIiF"
    "V99T6S3OI+5wcC1HmHcl3LTCUHsSleVynLZ6Xpg1kpPmCEsFneXBEDqTgTjtYVSyuO/3dqnIM6Z/0OiPp+yoh1cjQguM72xB"
    "2cqE+/OISSZTn85bhcXKXAQYHD4K0rOC7SL2bJVFtMOIrwby5CL4bFWiwFV8TLQzgHzE0Yhs2/tfGezXrofVaI0HApsUEKDz"
    "F4zitTKD5pzw6k0JHOlZ9iETqCnPcQwhw5H2w+QYqn6M7D7HUQvYUF5ZhVWToq55xi/x261Q0K/a3LHUZuBZrYHpYP8NoV29"
    "3v9Vibad/rPZK80wLDSgl+U/HAPuy/nBSoYM4IzQun5AkN6XTCHEM6cKUHLY6GzFTMWiWhBH9EBAUTP34OF4ztaNLxuZvjUn"
    "X/iSiKRZj4cFHN74cfDJBj/SdRo/hlWVXl7D3TXAF9UOsjj3Hq5vjoSyIMJjLAoEU0ZxfrqkHGXE0kKHLqiuZaC9lzK0UvrV"
    "V0MoQ+vf2NkzuN92hjMoFL9yi6DZgogR3B8XCUpghB4Jjx4CPX212Q4kPBxPYAuq7c2+VAm3yX3IDVmDnzENBNBKlqmXkjf6"
    "LMdS+HZdhDwno+M5sq17u75h+FPfOQM8JXh0T5B8zAuQM6jtaKwqANJJYK9/wJM7zBQ5W1PnvaE="
)

GEMINI_4_ARGON_RESPONSE_FIELDS: Final[Mapping[str, JsonValue]] = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "text": GEMINI_4_ARGON_THOUGHT,
                        "thought": True,
                    },
                    {
                        "text": (
                            "I will read `shop/validation.py` next to see how `validate_quantity` handles zero, negative, and"
                            " non-integer quantities."
                        ),
                    },
                    {
                        "functionCall": {
                            "name": "read_file",
                            "args": {
                                "path": "shop/validation.py",
                                "start_line": 1,
                                "end_line": 50,
                            },
                            "id": "call_1426689",
                        },
                        "thoughtSignature": GEMINI_4_ARGON_THOUGHT_SIGNATURE,
                    },
                ],
                "role": "model",
            },
            "finishReason": "STOP",
            "index": 0,
            "finishMessage": "Model generated function call(s).",
        },
    ],
    "usageMetadata": {
        "promptTokenCount": 414,
        "candidatesTokenCount": 69,
        "totalTokenCount": 628,
        "promptTokensDetails": [
            {
                "modality": "TEXT",
                "tokenCount": 414,
            },
        ],
        "candidatesTokensDetails": [
            {
                "modality": "TEXT",
                "tokenCount": 69,
            },
        ],
        "thoughtsTokenCount": 145,
    },
    "modelVersion": "gemini-4-argon",
    "responseId": "4CbQarGpL8SGz7IPxf-zyQ0",
}

GEMINI_4_ARGON_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "gemini/gemini-4-argon"},
    expected_provider_endpoint="/models/gemini-4-argon:generateContent",
    expected_provider_headers={"content-type": "application/json", "x-goog-api-key": "synthetic-gemini-key"},
    expected_provider_request=GEMINI_GENERATE_CONTENT_REQUEST,
    mock_provider_response={**GEMINI_GENERATE_CONTENT_RESPONSE, **GEMINI_4_ARGON_RESPONSE_FIELDS},
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "gemini/gemini-4-argon",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": (
                        "I will read `shop/validation.py` next to see how `validate_quantity` handles zero, negative, and"
                        " non-integer quantities."
                    ),
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "provider_specific_fields": {
                                "thought_signature": GEMINI_4_ARGON_THOUGHT_SIGNATURE,
                            },
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 50}',
                                "name": "read_file",
                            },
                            "id": "call_1426689__thought__" + GEMINI_4_ARGON_THOUGHT_SIGNATURE,
                            "type": "function",
                        },
                    ],
                    "images": [],
                    "reasoning_content": GEMINI_4_ARGON_THOUGHT,
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": GEMINI_4_ARGON_THOUGHT,
                        },
                    ],
                    "provider_specific_fields": {
                        "thought_signatures": [
                            GEMINI_4_ARGON_THOUGHT_SIGNATURE,
                        ],
                    },
                },
                "provider_specific_fields": {
                    "native_finish_reason": "STOP",
                },
            },
        ],
        "usage": {
            "completion_tokens": 214,
            "prompt_tokens": 414,
            "total_tokens": 628,
            "completion_tokens_details": {
                "reasoning_tokens": 145,
                "text_tokens": 69,
            },
            "prompt_tokens_details": {
                "text_tokens": 414,
            },
        },
        "vertex_ai_citation_metadata": [],
        "vertex_ai_grounding_metadata": [],
        "vertex_ai_safety_results": [],
        "vertex_ai_url_context_metadata": [],
    },
)

GEMINI_3_5_FLASH_THOUGHT_SIGNATURE: Final = "ErkECrYEAWkUfRORlQ5j/gkLmyx08FVeIMgg57FJXrjcLwBlt9iuELI6Uc5h+NR/Vm+gEsbdZ5mgFTmKjPF9K/eft8IlXMKm6odJjQOgOYsbt2JHzANo0bpfTmlA6fIi0G2zLbvBVASA6Bdxu1aPQuO4voioQRwm2vomRxH1YbWp8sKXk0DBVbefosldrL0zLJFsi5dFYCtPvw0n9olPVgptHzEdiyXqG+63aPxTooARRQutUH0XWAKR0V+P7qWPt55QKlLaKQqKeBndg9JrEplJihg1sp++y+NAi18fsqXteUS2zIeDtdePGM/GS5oVibwE25zJoziPRdtJhGasFSaA7a3znhW9PF0pBIPAIRPKE4NsQ0FRhpy7ksIXY+0uJ4N+WPPejrtKK6z5x+P0tFkFP0ZNNPM8FZbir1ncVhVxkZS/wWmhc/8TZoRA9ghlTpYhHJ+C4fRVqQqnyRR3SDpVTzB4/sCjBlb434dTH0U3jB4h6V9b/Zx4k4pwUwZTNr2FfgOt2bR7u05DOa+H73OzsNG6zBnMYgBndQdRgk58+l4+UcZdpGKB0lkbHdfD2bminBypEmeJKNRpuc7Smuu0YxcZiY03tzzhHdrUmItqC39OEr2CzRcT9DjFpiWydo3ej9ZkEXeyxoMCckpMmGWh6xbjAnX9gkrPFmUE2rJqblDJWa51i6u/p9Y6ciCq6j4lAy9eBfULIRGQt9pKOXagltOnX0vR0MTDYgWe4dVDSzCI1TV7X/SQYjo="

GEMINI_3_5_FLASH_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "gemini/gemini-3.5-flash",
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/models/gemini-3.5-flash:generateContent",
    expected_provider_headers={"content-type": "application/json", "x-goog-api-key": "synthetic-gemini-key"},
    expected_provider_request={
        "contents": [{"parts": [{"text": "Say hello."}], "role": "user"}],
        "generationConfig": {"max_output_tokens": 1024},
        "system_instruction": {"parts": [{"text": "You are a terse assistant."}]},
    },
    mock_provider_response={
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello.", "thoughtSignature": GEMINI_3_5_FLASH_THOUGHT_SIGNATURE}],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 2,
            "totalTokenCount": 137,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 10}],
            "thoughtsTokenCount": 125,
            "serviceTier": "standard",
        },
        "modelVersion": "gemini-3.5-flash",
        "responseId": "4hjEav32Mt6P6dkPn5iDqAg",
    },
    expected_litellm_response={
        "id": "4hjEav32Mt6P6dkPn5iDqAg",
        "created": ANY,
        "model": "gemini/gemini-3.5-flash",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello.",
                    "role": "assistant",
                    "images": [],
                    "thinking_blocks": [],
                    "provider_specific_fields": {"thought_signatures": [GEMINI_3_5_FLASH_THOUGHT_SIGNATURE]},
                },
                "provider_specific_fields": {"native_finish_reason": "STOP"},
            }
        ],
        "usage": {
            "completion_tokens": 127,
            "prompt_tokens": 10,
            "total_tokens": 137,
            "completion_tokens_details": {"reasoning_tokens": 125, "text_tokens": 2},
            "prompt_tokens_details": {"text_tokens": 10},
        },
        "vertex_ai_grounding_metadata": [],
        "vertex_ai_url_context_metadata": [],
        "vertex_ai_safety_results": [],
        "vertex_ai_citation_metadata": [],
    },
)
