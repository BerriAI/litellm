from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.bases.gemini import GEMINI_3_8_FLASH_TEST_CASE
from integration.translation.chat_completions.bases.vertex_ai import (
    GEMINI_3_8_FLASH_TEST_CASE as VERTEX_AI_GEMINI_3_8_FLASH_TEST_CASE,
)
from integration.translation.chat_completions.templates import CHAT_COMPLETIONS_RESPONSE
from integration.translation.runner import assert_translation
from integration.translation.templates.vertex_ai import (
    VERTEX_AI_GEMINI_GENERATE_CONTENT_RESPONSE,
)

GEMINI_GENERATION_CONFIG: Final = object_value(GEMINI_3_8_FLASH_TEST_CASE.expected_provider_request["generationConfig"])
VERTEX_AI_GENERATION_CONFIG: Final = object_value(
    VERTEX_AI_GEMINI_3_8_FLASH_TEST_CASE.expected_provider_request["generationConfig"]
)

GEMINI_3_8_FLASH_TEMPERATURE_TEST_CASE: Final = replace(
    GEMINI_3_8_FLASH_TEST_CASE,
    scenario="temperature",
    litellm_request={**GEMINI_3_8_FLASH_TEST_CASE.litellm_request, "temperature": 0.4},
    expected_provider_request={
        **GEMINI_3_8_FLASH_TEST_CASE.expected_provider_request,
        "generationConfig": {**GEMINI_GENERATION_CONFIG, "temperature": 0.4},
    },
)

GEMINI_3_8_FLASH_ZERO_TEMPERATURE_TEST_CASE: Final = replace(
    GEMINI_3_8_FLASH_TEST_CASE,
    scenario="zero_temperature",
    litellm_request={**GEMINI_3_8_FLASH_TEST_CASE.litellm_request, "temperature": 0},
    expected_provider_request={
        **GEMINI_3_8_FLASH_TEST_CASE.expected_provider_request,
        "generationConfig": {**GEMINI_GENERATION_CONFIG, "temperature": 0},
    },
)

GEMINI_3_8_FLASH_REASONING_EFFORT_TEST_CASE: Final = replace(
    GEMINI_3_8_FLASH_TEST_CASE,
    scenario="reasoning_effort",
    litellm_request={**GEMINI_3_8_FLASH_TEST_CASE.litellm_request, "reasoning_effort": "medium"},
    expected_provider_request={
        **GEMINI_3_8_FLASH_TEST_CASE.expected_provider_request,
        "generationConfig": {
            **GEMINI_GENERATION_CONFIG,
            "thinkingConfig": {"thinkingLevel": "medium", "includeThoughts": True},
        },
    },
)

VERTEX_AI_GEMINI_3_8_FLASH_TEMPERATURE_THOUGHT_SIGNATURE_0: Final = (
    "AY89a18+3vC8Y6wtLQ8YLNoWj0pZ1qcMTAdhuIp6v+o7VjO8nN81l72KF3fiSKtgKQYvOKc4hz38BNeldTuj8NebwQ7BneePvMzc"
    "hDx/5j5DXPVIIW7bRE2vYMhffaNpyjdltlQRDAxhiqjbSHIIGZQdxxAE7GfdGxlhstemN91VzQR4KnPKk6MXh0at/4JacXcOFRRX"
    "DnYmUqj1a1tSIoSjeTsHMvSj5OpJyxxa8Fafd/yqwL8g4scNZgGzckn4O5quaQmJT9VPtoAsmh8qgXUr/qzNSZoFGzRN6m7OzSuT"
    "yQ246hxKwp0+ZB9P+Nn5fvG96JVXsRq7dF00HBAbhebjUfbNTmIQauZHGSGbF2e+ck0aG5Lf8se2qRu6NCqiF+EnTFMUg7g+1cN1"
    "nIXLw7X6cAPPrpXvVdkfAoHksgSFWY5uUYwW9yDTZYwOWqF6bxDapPFU5HmJj/Bf5YuVH7VhGB7fyO8SVW6AUe0/S/o0/e//wn6A"
    "aFomUST2wXw0hsi/OSZLEhRAi16AeUI0Y4SenblpEYtDAyhzFi77mqsceuF7QIq672m5yWq2qoGhPh7S+5lMEvuD8YLITzrC81Xc"
    "b4HN3rxOvi6skmP4WJqZO8zPIw65WQUhRdhadhxM81HE+3InBToo1IhAF+11qt6hK0EuUf1guaGrUEDZRPAgomcOLBSdK6gQ9pn5"
    "z3SiRPpw9SV4LWMqBjulGRD6K8u7XHXLPAiGZTF1cr/N2sEQuyri57XytugNISVcxh0q/Lvz0v0DyUAZM8H2ymbusxnKdIHanI9Z"
    "US/XIXdia3PwqiOCgXDQaceOmotUnl8i0IXvSKwMjv3wB7UaC4YKyDurZ+gJJOzbjVjZ/mEf9h4JW1ZetNWVtQl2s+oZXCHOtVJD"
    "2qLDyuMPO7EuafP8w7mbpKBTmw4ZAyCfrTL73FnEYhp6M3IEG+c4fdUCHMvT6ecG42DLKhdmWfCqzTdXmXFLj/bePo7YzfYpLOLl"
    "hEzVdO5ojXPd6squqM+YHQS640UkdQ4v20UQ9ghFRkW1xUl7uPQkIAI0KWiDmaoi0YGFBxWpTs1U+wN2K6hfqgtJy80rl5ceUT7f"
    "6B3MnwiIoAjzVc3n3TH6GodPIO08vrhrf4oz+G3OkdCaMHXURfzYkhji50/ThfAxgD7XDuzUYlUWGotbx0FiYEDBbi2b7t7L/w3f"
    "P3nXYoSgaaHumXEnOAtugy6immasBFawtZq+GrOIYHJbH2tZhMpUysSTSnGF3BmqczIctDCwC82AbXU5sKEKFShEGnJSw9B0OlWu"
    "Pa6kC/Yfapk6NzDyNB2ttsLhy0hCkT0BmgDn2P4h5T4PCVgLWUbA7LGQoYO5zbsUFBwtPUsVFK0yLyf/Gg9h4ebosT6NVh7AkzS2"
    "riaNfm//lGUKK/8YDY49PAu0jpHp6JoXYg/z1+BjPvITdIQgf/8a0X36uqO+Yp2HZPzZzl/Ed8sC3ZadKxulNV9kkXN2Gc3v/VyB"
    "ruICqdhHSXJkusQJw+vtClqbP2uT18wQN4SRLjFTC3/ZruElXJFipevCQcC8KNVkXW39qZNWJGmKAR15mprXot6VKXkRHfyQJJz0"
    "lAMTaACA5LJoA1dDn7+UaikqUsfrlLE6ObbCasA5/wIp5zFsGvslVq/vJqGWk2mdXbas2fpWTSUECEB3gGoIFo9rmHTURX4cAALa"
    "oZnjSkaKy1IdC3XDQsqT2vmAX3XjvaLhkCwWg56pst5LvS0AlcRGo49QHATNsY4h0SFkUmMyAGgS7+Pa2VwmCgWUoFFHRbO8ryoy"
    "W57qONicyCxA2/OL28OrKGC/e25UsXwKBAXx+9uamcdxFvOQr9WqEJK1g4Q="
)


VERTEX_AI_GEMINI_3_8_FLASH_TEMPERATURE_TEST_CASE: Final = replace(
    VERTEX_AI_GEMINI_3_8_FLASH_TEST_CASE,
    scenario="temperature",
    litellm_request={**VERTEX_AI_GEMINI_3_8_FLASH_TEST_CASE.litellm_request, "temperature": 0.4},
    expected_provider_request={
        **VERTEX_AI_GEMINI_3_8_FLASH_TEST_CASE.expected_provider_request,
        "generationConfig": {**VERTEX_AI_GENERATION_CONFIG, "temperature": 0.4},
    },
    mock_provider_response={
        **VERTEX_AI_GEMINI_GENERATE_CONTENT_RESPONSE,
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": (
                                "**My Thought Process on Handling Quantity Validation**\n\nAlright, to properly address the handling "
                                "of zero, negative, and potentially non-integer quantities, my immediate next step needs to be "
                                "understanding the existing validation logic. This is crucial before I can even begin to think about "
                                'suggesting any fixes or further code. My protocol dictates that I must "read the relevant files '
                                "before suggesting a fix,\" and that's exactly what I'm prioritizing.\n\nTherefore, I need to examine "
                                "the `shop/validation.py` file. This is where I anticipate finding the rules and checks that govern "
                                "how different types of quantities are processed and validated. My plan is to read a reasonable "
                                "initial segment of this file, say lines 1 through 50, to get a solid grasp of the current "
                                "implementation. After that, I'll be able to determine precisely *which* other files, if any, I'll "
                                "need to investigate next, keeping in mind the specific constraints of zero, negative, and "
                                'non-integer values.\n\nTo do this, I will now use the `read_file` tool:\n\n```json\n{\n  "tool_code": '
                                "\"print(read_file(path='shop/validation.py', start_line=1, end_line=50))\"\n}\n```\n\n"
                            ),
                            "thought": True,
                        },
                        {
                            "text": (
                                "To examine how zero, negative, and non-integer quantities are handled, I need to inspect "
                                "`shop/validation.py` next."
                            ),
                        },
                        {
                            "functionCall": {
                                "name": "read_file",
                                "args": {"start_line": 1, "end_line": 50, "path": "shop/validation.py"},
                                "id": "call_1464376",
                            },
                            "thoughtSignature": VERTEX_AI_GEMINI_3_8_FLASH_TEMPERATURE_THOUGHT_SIGNATURE_0,
                        },
                    ],
                },
                "finishReason": "STOP",
            },
        ],
        "usageMetadata": {
            "promptTokenCount": 246,
            "candidatesTokenCount": 61,
            "totalTokenCount": 646,
            "trafficType": "ON_DEMAND",
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 246}],
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 61}],
            "thoughtsTokenCount": 339,
        },
        "modelVersion": "gemini-3.8-flash",
        "createTime": "2026-10-15T01:22:48.202504Z",
        "responseId": "6CrQaoiuDNKZq8YPjZCokAw",
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "vertex_ai/gemini-3.8-flash",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": (
                        "To examine how zero, negative, and non-integer quantities are handled, I need to inspect "
                        "`shop/validation.py` next."
                    ),
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "provider_specific_fields": {
                                "thought_signature": (
                                    "AY89a18+3vC8Y6wtLQ8YLNoWj0pZ1qcMTAdhuIp6v+o7VjO8nN81l72KF3fiSKtgKQYvOKc4hz38BNeldTuj8NebwQ7BneePvMzc"
                                    "hDx/5j5DXPVIIW7bRE2vYMhffaNpyjdltlQRDAxhiqjbSHIIGZQdxxAE7GfdGxlhstemN91VzQR4KnPKk6MXh0at/4JacXcOFRRX"
                                    "DnYmUqj1a1tSIoSjeTsHMvSj5OpJyxxa8Fafd/yqwL8g4scNZgGzckn4O5quaQmJT9VPtoAsmh8qgXUr/qzNSZoFGzRN6m7OzSuT"
                                    "yQ246hxKwp0+ZB9P+Nn5fvG96JVXsRq7dF00HBAbhebjUfbNTmIQauZHGSGbF2e+ck0aG5Lf8se2qRu6NCqiF+EnTFMUg7g+1cN1"
                                    "nIXLw7X6cAPPrpXvVdkfAoHksgSFWY5uUYwW9yDTZYwOWqF6bxDapPFU5HmJj/Bf5YuVH7VhGB7fyO8SVW6AUe0/S/o0/e//wn6A"
                                    "aFomUST2wXw0hsi/OSZLEhRAi16AeUI0Y4SenblpEYtDAyhzFi77mqsceuF7QIq672m5yWq2qoGhPh7S+5lMEvuD8YLITzrC81Xc"
                                    "b4HN3rxOvi6skmP4WJqZO8zPIw65WQUhRdhadhxM81HE+3InBToo1IhAF+11qt6hK0EuUf1guaGrUEDZRPAgomcOLBSdK6gQ9pn5"
                                    "z3SiRPpw9SV4LWMqBjulGRD6K8u7XHXLPAiGZTF1cr/N2sEQuyri57XytugNISVcxh0q/Lvz0v0DyUAZM8H2ymbusxnKdIHanI9Z"
                                    "US/XIXdia3PwqiOCgXDQaceOmotUnl8i0IXvSKwMjv3wB7UaC4YKyDurZ+gJJOzbjVjZ/mEf9h4JW1ZetNWVtQl2s+oZXCHOtVJD"
                                    "2qLDyuMPO7EuafP8w7mbpKBTmw4ZAyCfrTL73FnEYhp6M3IEG+c4fdUCHMvT6ecG42DLKhdmWfCqzTdXmXFLj/bePo7YzfYpLOLl"
                                    "hEzVdO5ojXPd6squqM+YHQS640UkdQ4v20UQ9ghFRkW1xUl7uPQkIAI0KWiDmaoi0YGFBxWpTs1U+wN2K6hfqgtJy80rl5ceUT7f"
                                    "6B3MnwiIoAjzVc3n3TH6GodPIO08vrhrf4oz+G3OkdCaMHXURfzYkhji50/ThfAxgD7XDuzUYlUWGotbx0FiYEDBbi2b7t7L/w3f"
                                    "P3nXYoSgaaHumXEnOAtugy6immasBFawtZq+GrOIYHJbH2tZhMpUysSTSnGF3BmqczIctDCwC82AbXU5sKEKFShEGnJSw9B0OlWu"
                                    "Pa6kC/Yfapk6NzDyNB2ttsLhy0hCkT0BmgDn2P4h5T4PCVgLWUbA7LGQoYO5zbsUFBwtPUsVFK0yLyf/Gg9h4ebosT6NVh7AkzS2"
                                    "riaNfm//lGUKK/8YDY49PAu0jpHp6JoXYg/z1+BjPvITdIQgf/8a0X36uqO+Yp2HZPzZzl/Ed8sC3ZadKxulNV9kkXN2Gc3v/VyB"
                                    "ruICqdhHSXJkusQJw+vtClqbP2uT18wQN4SRLjFTC3/ZruElXJFipevCQcC8KNVkXW39qZNWJGmKAR15mprXot6VKXkRHfyQJJz0"
                                    "lAMTaACA5LJoA1dDn7+UaikqUsfrlLE6ObbCasA5/wIp5zFsGvslVq/vJqGWk2mdXbas2fpWTSUECEB3gGoIFo9rmHTURX4cAALa"
                                    "oZnjSkaKy1IdC3XDQsqT2vmAX3XjvaLhkCwWg56pst5LvS0AlcRGo49QHATNsY4h0SFkUmMyAGgS7+Pa2VwmCgWUoFFHRbO8ryoy"
                                    "W57qONicyCxA2/OL28OrKGC/e25UsXwKBAXx+9uamcdxFvOQr9WqEJK1g4Q="
                                ),
                            },
                            "function": {
                                "arguments": '{"start_line": 1, "end_line": 50, "path": "shop/validation.py"}',
                                "name": "read_file",
                            },
                            "id": (
                                "call_1464376__thought__AY89a18+3vC8Y6wtLQ8YLNoWj0pZ1qcMTAdhuIp6v+o7VjO8nN81l72KF3fiSKtgKQYvOKc4hz38B"
                                "NeldTuj8NebwQ7BneePvMzchDx/5j5DXPVIIW7bRE2vYMhffaNpyjdltlQRDAxhiqjbSHIIGZQdxxAE7GfdGxlhstemN91VzQR4K"
                                "nPKk6MXh0at/4JacXcOFRRXDnYmUqj1a1tSIoSjeTsHMvSj5OpJyxxa8Fafd/yqwL8g4scNZgGzckn4O5quaQmJT9VPtoAsmh8qg"
                                "XUr/qzNSZoFGzRN6m7OzSuTyQ246hxKwp0+ZB9P+Nn5fvG96JVXsRq7dF00HBAbhebjUfbNTmIQauZHGSGbF2e+ck0aG5Lf8se2q"
                                "Ru6NCqiF+EnTFMUg7g+1cN1nIXLw7X6cAPPrpXvVdkfAoHksgSFWY5uUYwW9yDTZYwOWqF6bxDapPFU5HmJj/Bf5YuVH7VhGB7fy"
                                "O8SVW6AUe0/S/o0/e//wn6AaFomUST2wXw0hsi/OSZLEhRAi16AeUI0Y4SenblpEYtDAyhzFi77mqsceuF7QIq672m5yWq2qoGhP"
                                "h7S+5lMEvuD8YLITzrC81Xcb4HN3rxOvi6skmP4WJqZO8zPIw65WQUhRdhadhxM81HE+3InBToo1IhAF+11qt6hK0EuUf1guaGrU"
                                "EDZRPAgomcOLBSdK6gQ9pn5z3SiRPpw9SV4LWMqBjulGRD6K8u7XHXLPAiGZTF1cr/N2sEQuyri57XytugNISVcxh0q/Lvz0v0Dy"
                                "UAZM8H2ymbusxnKdIHanI9ZUS/XIXdia3PwqiOCgXDQaceOmotUnl8i0IXvSKwMjv3wB7UaC4YKyDurZ+gJJOzbjVjZ/mEf9h4JW"
                                "1ZetNWVtQl2s+oZXCHOtVJD2qLDyuMPO7EuafP8w7mbpKBTmw4ZAyCfrTL73FnEYhp6M3IEG+c4fdUCHMvT6ecG42DLKhdmWfCqz"
                                "TdXmXFLj/bePo7YzfYpLOLlhEzVdO5ojXPd6squqM+YHQS640UkdQ4v20UQ9ghFRkW1xUl7uPQkIAI0KWiDmaoi0YGFBxWpTs1U+"
                                "wN2K6hfqgtJy80rl5ceUT7f6B3MnwiIoAjzVc3n3TH6GodPIO08vrhrf4oz+G3OkdCaMHXURfzYkhji50/ThfAxgD7XDuzUYlUWG"
                                "otbx0FiYEDBbi2b7t7L/w3fP3nXYoSgaaHumXEnOAtugy6immasBFawtZq+GrOIYHJbH2tZhMpUysSTSnGF3BmqczIctDCwC82Ab"
                                "XU5sKEKFShEGnJSw9B0OlWuPa6kC/Yfapk6NzDyNB2ttsLhy0hCkT0BmgDn2P4h5T4PCVgLWUbA7LGQoYO5zbsUFBwtPUsVFK0yL"
                                "yf/Gg9h4ebosT6NVh7AkzS2riaNfm//lGUKK/8YDY49PAu0jpHp6JoXYg/z1+BjPvITdIQgf/8a0X36uqO+Yp2HZPzZzl/Ed8sC3"
                                "ZadKxulNV9kkXN2Gc3v/VyBruICqdhHSXJkusQJw+vtClqbP2uT18wQN4SRLjFTC3/ZruElXJFipevCQcC8KNVkXW39qZNWJGmKA"
                                "R15mprXot6VKXkRHfyQJJz0lAMTaACA5LJoA1dDn7+UaikqUsfrlLE6ObbCasA5/wIp5zFsGvslVq/vJqGWk2mdXbas2fpWTSUEC"
                                "EB3gGoIFo9rmHTURX4cAALaoZnjSkaKy1IdC3XDQsqT2vmAX3XjvaLhkCwWg56pst5LvS0AlcRGo49QHATNsY4h0SFkUmMyAGgS7"
                                "+Pa2VwmCgWUoFFHRbO8ryoyW57qONicyCxA2/OL28OrKGC/e25UsXwKBAXx+9uamcdxFvOQr9WqEJK1g4Q="
                            ),
                            "type": "function",
                        },
                    ],
                    "images": [],
                    "reasoning_content": (
                        "**My Thought Process on Handling Quantity Validation**\n\nAlright, to properly address the handling "
                        "of zero, negative, and potentially non-integer quantities, my immediate next step needs to be "
                        "understanding the existing validation logic. This is crucial before I can even begin to think about "
                        'suggesting any fixes or further code. My protocol dictates that I must "read the relevant files '
                        "before suggesting a fix,\" and that's exactly what I'm prioritizing.\n\nTherefore, I need to examine "
                        "the `shop/validation.py` file. This is where I anticipate finding the rules and checks that govern "
                        "how different types of quantities are processed and validated. My plan is to read a reasonable "
                        "initial segment of this file, say lines 1 through 50, to get a solid grasp of the current "
                        "implementation. After that, I'll be able to determine precisely *which* other files, if any, I'll "
                        "need to investigate next, keeping in mind the specific constraints of zero, negative, and "
                        'non-integer values.\n\nTo do this, I will now use the `read_file` tool:\n\n```json\n{\n  "tool_code": '
                        "\"print(read_file(path='shop/validation.py', start_line=1, end_line=50))\"\n}\n```\n\n"
                    ),
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": (
                                "**My Thought Process on Handling Quantity Validation**\n\nAlright, to properly address the handling "
                                "of zero, negative, and potentially non-integer quantities, my immediate next step needs to be "
                                "understanding the existing validation logic. This is crucial before I can even begin to think about "
                                'suggesting any fixes or further code. My protocol dictates that I must "read the relevant files '
                                "before suggesting a fix,\" and that's exactly what I'm prioritizing.\n\nTherefore, I need to examine "
                                "the `shop/validation.py` file. This is where I anticipate finding the rules and checks that govern "
                                "how different types of quantities are processed and validated. My plan is to read a reasonable "
                                "initial segment of this file, say lines 1 through 50, to get a solid grasp of the current "
                                "implementation. After that, I'll be able to determine precisely *which* other files, if any, I'll "
                                "need to investigate next, keeping in mind the specific constraints of zero, negative, and "
                                'non-integer values.\n\nTo do this, I will now use the `read_file` tool:\n\n```json\n{\n  "tool_code": '
                                "\"print(read_file(path='shop/validation.py', start_line=1, end_line=50))\"\n}\n```\n\n"
                            ),
                        },
                    ],
                    "provider_specific_fields": {
                        "thought_signatures": [
                            (
                                "AY89a18+3vC8Y6wtLQ8YLNoWj0pZ1qcMTAdhuIp6v+o7VjO8nN81l72KF3fiSKtgKQYvOKc4hz38BNeldTuj8NebwQ7BneePvMzc"
                                "hDx/5j5DXPVIIW7bRE2vYMhffaNpyjdltlQRDAxhiqjbSHIIGZQdxxAE7GfdGxlhstemN91VzQR4KnPKk6MXh0at/4JacXcOFRRX"
                                "DnYmUqj1a1tSIoSjeTsHMvSj5OpJyxxa8Fafd/yqwL8g4scNZgGzckn4O5quaQmJT9VPtoAsmh8qgXUr/qzNSZoFGzRN6m7OzSuT"
                                "yQ246hxKwp0+ZB9P+Nn5fvG96JVXsRq7dF00HBAbhebjUfbNTmIQauZHGSGbF2e+ck0aG5Lf8se2qRu6NCqiF+EnTFMUg7g+1cN1"
                                "nIXLw7X6cAPPrpXvVdkfAoHksgSFWY5uUYwW9yDTZYwOWqF6bxDapPFU5HmJj/Bf5YuVH7VhGB7fyO8SVW6AUe0/S/o0/e//wn6A"
                                "aFomUST2wXw0hsi/OSZLEhRAi16AeUI0Y4SenblpEYtDAyhzFi77mqsceuF7QIq672m5yWq2qoGhPh7S+5lMEvuD8YLITzrC81Xc"
                                "b4HN3rxOvi6skmP4WJqZO8zPIw65WQUhRdhadhxM81HE+3InBToo1IhAF+11qt6hK0EuUf1guaGrUEDZRPAgomcOLBSdK6gQ9pn5"
                                "z3SiRPpw9SV4LWMqBjulGRD6K8u7XHXLPAiGZTF1cr/N2sEQuyri57XytugNISVcxh0q/Lvz0v0DyUAZM8H2ymbusxnKdIHanI9Z"
                                "US/XIXdia3PwqiOCgXDQaceOmotUnl8i0IXvSKwMjv3wB7UaC4YKyDurZ+gJJOzbjVjZ/mEf9h4JW1ZetNWVtQl2s+oZXCHOtVJD"
                                "2qLDyuMPO7EuafP8w7mbpKBTmw4ZAyCfrTL73FnEYhp6M3IEG+c4fdUCHMvT6ecG42DLKhdmWfCqzTdXmXFLj/bePo7YzfYpLOLl"
                                "hEzVdO5ojXPd6squqM+YHQS640UkdQ4v20UQ9ghFRkW1xUl7uPQkIAI0KWiDmaoi0YGFBxWpTs1U+wN2K6hfqgtJy80rl5ceUT7f"
                                "6B3MnwiIoAjzVc3n3TH6GodPIO08vrhrf4oz+G3OkdCaMHXURfzYkhji50/ThfAxgD7XDuzUYlUWGotbx0FiYEDBbi2b7t7L/w3f"
                                "P3nXYoSgaaHumXEnOAtugy6immasBFawtZq+GrOIYHJbH2tZhMpUysSTSnGF3BmqczIctDCwC82AbXU5sKEKFShEGnJSw9B0OlWu"
                                "Pa6kC/Yfapk6NzDyNB2ttsLhy0hCkT0BmgDn2P4h5T4PCVgLWUbA7LGQoYO5zbsUFBwtPUsVFK0yLyf/Gg9h4ebosT6NVh7AkzS2"
                                "riaNfm//lGUKK/8YDY49PAu0jpHp6JoXYg/z1+BjPvITdIQgf/8a0X36uqO+Yp2HZPzZzl/Ed8sC3ZadKxulNV9kkXN2Gc3v/VyB"
                                "ruICqdhHSXJkusQJw+vtClqbP2uT18wQN4SRLjFTC3/ZruElXJFipevCQcC8KNVkXW39qZNWJGmKAR15mprXot6VKXkRHfyQJJz0"
                                "lAMTaACA5LJoA1dDn7+UaikqUsfrlLE6ObbCasA5/wIp5zFsGvslVq/vJqGWk2mdXbas2fpWTSUECEB3gGoIFo9rmHTURX4cAALa"
                                "oZnjSkaKy1IdC3XDQsqT2vmAX3XjvaLhkCwWg56pst5LvS0AlcRGo49QHATNsY4h0SFkUmMyAGgS7+Pa2VwmCgWUoFFHRbO8ryoy"
                                "W57qONicyCxA2/OL28OrKGC/e25UsXwKBAXx+9uamcdxFvOQr9WqEJK1g4Q="
                            ),
                        ],
                    },
                },
                "provider_specific_fields": {"native_finish_reason": "STOP"},
            },
        ],
        "usage": {
            "completion_tokens": 400,
            "prompt_tokens": 246,
            "total_tokens": 646,
            "completion_tokens_details": {"reasoning_tokens": 339, "text_tokens": 61},
            "prompt_tokens_details": {"text_tokens": 246},
        },
        "vertex_ai_grounding_metadata": [],
        "vertex_ai_url_context_metadata": [],
        "vertex_ai_safety_results": [],
        "vertex_ai_citation_metadata": [],
    },
)


@pytest.mark.parametrize(
    "case",
    [
        GEMINI_3_8_FLASH_TEMPERATURE_TEST_CASE,
        GEMINI_3_8_FLASH_ZERO_TEMPERATURE_TEST_CASE,
        GEMINI_3_8_FLASH_REASONING_EFFORT_TEST_CASE,
        VERTEX_AI_GEMINI_3_8_FLASH_TEMPERATURE_TEST_CASE,
    ],
    ids=lambda case: case.id,
)
def test_chat_completions_sampling_gemini(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)
