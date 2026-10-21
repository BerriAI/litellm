from collections.abc import Mapping
from typing import Final
from unittest.mock import ANY

from integration.translation.case import TranslationTestCase
from integration.translation.messages.templates import MESSAGES_REQUEST, MESSAGES_RESPONSE
from integration.translation.templates.bedrock_converse import (
    BEDROCK_CONVERSE_REQUEST,
    BEDROCK_CONVERSE_RESPONSE,
)
from pydantic import JsonValue

CLAUDE_HAIKU_4_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-haiku-4-5-20251001-v1%3A0/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
        "inferenceConfig": {"maxTokens": 64},
        "system": [{"text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "metrics": {"latencyMs": 771},
        "output": {"message": {"content": [{"text": "Hello."}], "role": "assistant"}},
        "stopReason": "end_turn",
        "usage": {
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 0,
            "cacheWriteInputTokens": 0,
            "inputTokens": 17,
            "outputTokens": 5,
            "serverToolUsage": {},
            "totalTokens": 22,
        },
    },
    expected_litellm_response={
        "id": ANY,
        "type": "message",
        "role": "assistant",
        "model": "bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "stop_sequence": None,
        "usage": {"input_tokens": 17, "output_tokens": 5},
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)

CLAUDE_SONNET_4_6_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "bedrock/converse/us.anthropic.claude-sonnet-4-6",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-sonnet-4-6/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
        "inferenceConfig": {"maxTokens": 64},
        "system": [{"text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "metrics": {"latencyMs": 997},
        "output": {"message": {"content": [{"text": "Hello!"}], "role": "assistant"}},
        "stopReason": "end_turn",
        "usage": {
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 0,
            "cacheWriteInputTokens": 0,
            "inputTokens": 18,
            "outputTokens": 5,
            "serverToolUsage": {},
            "totalTokens": 23,
        },
    },
    expected_litellm_response={
        "id": ANY,
        "type": "message",
        "role": "assistant",
        "model": "bedrock/converse/us.anthropic.claude-sonnet-4-6",
        "stop_sequence": None,
        "usage": {"input_tokens": 18, "output_tokens": 5},
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)

CLAUDE_SONNET_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "bedrock/converse/us.anthropic.claude-sonnet-5",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-sonnet-5/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
        "inferenceConfig": {"maxTokens": 64},
        "system": [{"text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "metrics": {"latencyMs": 1201},
        "output": {"message": {"content": [{"text": "Hello."}], "role": "assistant"}},
        "stopReason": "end_turn",
        "usage": {
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 0,
            "cacheWriteInputTokens": 0,
            "inputTokens": 21,
            "outputTokens": 6,
            "serverToolUsage": {},
            "totalTokens": 27,
        },
    },
    expected_litellm_response={
        "id": ANY,
        "type": "message",
        "role": "assistant",
        "model": "bedrock/converse/us.anthropic.claude-sonnet-5",
        "stop_sequence": None,
        "usage": {"input_tokens": 21, "output_tokens": 6},
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)

CLAUDE_OPUS_4_8_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "bedrock/converse/us.anthropic.claude-opus-4-8",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-opus-4-8/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
        "inferenceConfig": {"maxTokens": 64},
        "system": [{"text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "metrics": {"latencyMs": 827},
        "output": {"message": {"content": [{"text": "Hello."}], "role": "assistant"}},
        "stopReason": "end_turn",
        "usage": {
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 0,
            "cacheWriteInputTokens": 0,
            "inputTokens": 21,
            "outputTokens": 6,
            "serverToolUsage": {},
            "totalTokens": 27,
        },
    },
    expected_litellm_response={
        "id": ANY,
        "type": "message",
        "role": "assistant",
        "model": "bedrock/converse/us.anthropic.claude-opus-4-8",
        "stop_sequence": None,
        "usage": {"input_tokens": 21, "output_tokens": 6},
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)

BEDROCK_CONVERSE_MESSAGES_RESPONSE: Final[Mapping[str, JsonValue]] = {
    key: value for key, value in MESSAGES_RESPONSE.items() if key not in ("container", "diagnostics")
}

BEDROCK_CONVERSE_MESSAGES_SYSTEM: Final[JsonValue] = [
    {
        "text": "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
        "Explain which file you need next, then call a read-only tool. Never change files. Use the messages "
        "client for this project review. Consider zero, negative and non-integer quantities before choosing "
        "the next file to inspect.",
    },
]

CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSowcKEggSEAEYAjgBQgh0aGlua2luZxIMk7YkOhrvQi9zueMRGgz/wtD/BSP6B5NwRzYiML9G20YP9l1wNfsLU7KSBDRmwZRu6x21pAZl"
    "Wi/ABO1zIPZcSdHWKXKYtMMiben4LCq+BscUTDnI69W7DmSmw2Z/AtQHNqUYtIqAwZ1KP/kuDEmOlF/q/QvdhYOm73hRzbJUjEmVkwU+yxX7"
    "sXM9DB18oM7KbPLbIAHNu5cM0D1uQ7jKNmLv2w2Y61M0iv4Ff+hU2r8zCXHEGsBA3EP6OsfkmhI3CmUWkAorSt1UYPmjSNbW0sQG9qA3fhVN"
    "leKcTQkVpKRV5rY2wcB0Y1OmLdCZCdSj3MXpmjL+Qxh2fW8nFQ1o3XXVIZNTgoJ43J95PtZUJrRMjQGPULeri4oHwXhZowtZmMbnNctjqolX"
    "TqXHAqBZLR7UqtOvaAihr8blpdiU5LrWn8qGuV/j4Me25hjC68jW45xHPW/J3IGg5pNOCGNVCA0ZOFkFkHVjGUGjl0jL0KnjC95I1ngiRhdQ"
    "DiEyfQq/ozP2nwgk7SD4Wdq5T3QtvJpBB7tLYSMiJOPR8s4gyIE0gjG1kLdOHcEROhz5sezE40sCftRKbishKYw3ZE3badsbkBgVVy7sYevj"
    "Hg2E+FycNLGp7VeX3SZgn+zdn2y6ZR427yS7Fhh84PKQMIiX8YQUsg8M4ISBtv0gvqfcrODZeTGKcCprEFC4NNcf9DfunhE9WrVDNaaHAKql"
    "/G3uTl6akwmUyAGWomY7YEGGDymxYYGzhucyYuJiyOC9ncLrXJftPadZvz302hOv/ng15Uh/jQwuWsWCs0BjRWEhkmVje76JbmK4Zau9EEx8"
    "iy7V9ltWvB20iTn1pfgY40+QFv1j75n7yXD0/oJHehfkRqm8feEs0ERnB6C+iF9XAqBoRSlUh728bbxNzK+ogApwtEbBpmnpFz5j7xLJLG8e"
    "a53TCKMyemzqv9bA3xNnenrgO/QYJ1o2heQ6Tyv01cop8EpF5nSqWIorCEbf7yp3f50ABdfZZSnT8Up5hAYHDhTJ/NZBVE8oQaQGbcGRuhl2"
    "5sN156XvCS3GcEP4mHu1L9hoT+bfasIx4aCdwQMB86E5lSAtCjphz3m4sBTT67/7RqVf4ymvPxjEsZroz/pujM7oJN2QEvZKkvKIWpHDXPt9"
    "efVjUpeQcpoSTuazPtnWVWbLvA4FvVXpnDP+Ed7bfkrXw4enrKqw0QYEbJpmGAE="
)

CLAUDE_HAIKU_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "bedrock/converse/us.anthropic.claude-haiku-5-5"},
    expected_provider_endpoint="/model/us.anthropic.claude-haiku-5-5/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        **BEDROCK_CONVERSE_REQUEST,
        "system": BEDROCK_CONVERSE_MESSAGES_SYSTEM,
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_RESPONSE,
        "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 147}}},
        "metrics": {"latencyMs": 4031},
        "output": {
            "message": {
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
                                "text": "I should check shop/validation.py to handle edge cases like zero, "
                                "negative, and non-integer values before deciding on the next file to read. "
                                "I'll read through the file, starting with the first chunk of lines.\n\n",
                            },
                        },
                    },
                    {
                        "text": "I'll read `shop/validation.py` around the `validate_quantity` definition at line "
                        "18, starting from the top of the file so the surrounding helpers are included.",
                    },
                    {
                        "toolUse": {
                            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
                            "name": "read_file",
                            "toolUseId": "tooluse_fjdXCsIbwCrVLDv3Pzvn3B",
                            "type": "tool_use",
                        },
                    },
                ],
                "role": "assistant",
            },
        },
        "usage": {
            "cacheDetails": [{"inputTokens": 853, "ttl": "5m"}],
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 853,
            "cacheWriteInputTokens": 853,
            "inputTokens": 4,
            "outputTokens": 297,
            "serverToolUsage": {},
            "totalTokens": 1154,
        },
    },
    expected_litellm_response={
        **BEDROCK_CONVERSE_MESSAGES_RESPONSE,
        "id": ANY,
        "model": "bedrock/converse/us.anthropic.claude-haiku-5-5",
        "usage": {"input_tokens": 4, "output_tokens": 297, "cache_creation_input_tokens": 853},
        "content": [
            {
                "type": "thinking",
                "thinking": "I should check shop/validation.py to handle edge cases like zero, negative, and "
                "non-integer values before deciding on the next file to read. I'll read through the file, "
                "starting with the first chunk of lines.\n\n",
                "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
            },
            {
                "type": "text",
                "text": "I'll read `shop/validation.py` around the `validate_quantity` definition at line 18, "
                "starting from the top of the file so the surrounding helpers are included.",
            },
            {
                "type": "tool_use",
                "id": "tooluse_fjdXCsIbwCrVLDv3Pzvn3B",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            },
        ],
    },
)

CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQS7wcKEggSEAEYAjgBQgh0aGlua2luZxIMTkG89+rWhZvnNujjGgy9ueQHNGYTXVgc2TsiMOfc/JR8YWRFuzvOfDDqCuTj51nOgpQyrrmL"
    "r5fylFkkoYwi2WWTjeBjxS70RiBErSqKB3OzMtrLHgImTbinps+XcdvrRao0nkPqkXGo1FFfKLf1Tb3/Qytx1scxregT8j4LBtTirSlLJlUX"
    "9zmQOi1DVInxJFYVl6p5KObZBKG8PSiVhdOwMxfnV/euyCWZO3JIZTywuOlsHhpb7k5QW6Ygn8Hciikz+A4dLMdBLdbFVT1y61nbTvGNITkO"
    "0MngJu3DDe3XKO5v/WjMbNJHm7wEa40ZKaU5q6H7gMD/hJIhHuSWQjtNsD9A/uFBw/6o42gaEmndewGx593NobUz9hf5iWfmwezpqGBs+wsr"
    "IMyie0VxW8KNwSR9dpkJfNAToId9EcDHVSlBGdGiQXlaz0Efed6PF57WgV6glU0HVyLKZ1sIDb1G1hhwTe6LKw6mxT/gcakl5S4GB6MJKvEi"
    "vTVAqSWjlvSQPktHDvM4nwPejOt+HFcipmE1Wx8Gopfvg1eE1rwslB+wjUO1EOC31483lw5J14KOz1swzjiZcOTVj85CEFtadz8fPK5SCOsP"
    "VNcdvpV8rh7KCdkDTnD7vTtS18Mu9QoOw0hl66l4O3uEuGa0wiQ1/0H2Mmul/+P+zMxrkYNkVicyDF7UQH9J22lPOaJylisuyZuW6Tamn7PK"
    "sbhnTHbIeZpm37BsL4ZBN33aAtB5FGVjXevjt9/mJAuljybiFyA3WuoCov1/9j50vR+UIBW2XeqFbTqrfymqWzFiNWWM+6qGt6UR1S3o0wlO"
    "2SXXVNhsNkXVRuR/4cJtjjpReVcrYL+LZNtFgHyWAMhRd4NOCLGQ4wQSeHRFyjqJA7xguls6spfcnBZXlgUFsdYWzyDBqbg8EANBfwI0cTQG"
    "X0CZy/beSSxLl9YwDT6cHhQaZU0sJ7yl1AzUuVaf4T5IFDheDn9GKKNCjF6VdY7ycLaH9JfuKvPIM1tzioKcLoNDppFFGtHD+2bwHdw3NAT0"
    "RQjuU7XLoUm9p1roY1m7RltcTl1nl/2K5G7x1//Fl4vN9MoyjiL5OF/vCoyaZuswAxZIQeDWxXvDBQVVZvTy5WA9qJmG8OX0QQM1+XKe84IV"
    "Z/9lqwrTW3ktVSVuNsFHBFypo9KXONnd99ekwaiFyVeCs0W17Gt3qIwk+LEF0yuFEantdcMoJBMXa3V7oxTgu9a42mrpNOtVwQ2RcU2rLsGl"
    "hvj4nDpPAnkNixpyX+SzCJ81FlutT1p5Wpw2GuL2gyZRfNpi8gJOYxgB"
)

CLAUDE_SONNET_5_5_THINKING_SIGNATURE_1: Final = (
    "CAQSigYKEwgSEAEYAjgBQgluYXJyYXRpb24SDAwiB8XjpzUcjkdimhoMwIGz3+l5MeF2P8mjIjAhc1Uec4tzG5fzDJrcwUiHfLl9hpYALDND"
    "+ykJZqmCUizSbZOB45cey71j2vi6fpMqpAWB5p+izlwQyOdW2tEZygJWj2ag/CSfLMDFz4FxatYYUcivtmo/ujCCBekTBt3hDhph2fE64sj3"
    "xsueYPHqSuAdn2n30TPAxBXIR5coXHz6nTZpjouC2A6zXy8grDdO9jSeS1vPprkyCsJw17+jigjWDIkmrjUv8DR1s64GCfuV5dWQ2PdYBI/1"
    "8o0OhR77veuPdOcNGcLfP15XX7vbtxJMP9LWOR5MLTgPjLjSAEL++eiisr52H4rB4Lvhz7n2KMfMClL7hqdxArHMpHh00wPi2NR8CGjKNTIP"
    "HvI/3e13aZDzW8JEDnD22CMmRx7NLgPD8nuncPqoZdVE7jYOECQldD3iHT7Li+m7CAbimbSksM91rzSP2S3CR/FYGtkjqJP5dDlPuRuWVgat"
    "jVM/xUEJl058GNMbTukUAFq2SStLHCCZh6upTBg9DtSNMmNi67602YHuQ+p8g8WRYIJh1PmkU8Y3u1coBp2LKQNuiq7MXT880DxtZ2Fldfi4"
    "xnY4vkYyB8pylwb1MrhUxh/7xHEesoSO18+SB/rHosHX7BQM1zf8KIDP4wqSBJER0EOPu6h+u1wiD6VYsRQXFWGcZ2S77GyNhaIQnvcc8Sp1"
    "sKoXtjNSek706MHLreobNqFfMZBTWtxcmYuzzYIQyPdJMKcdFHSsXLgaoV+5O2ZQmQbhm5Iw/pvLWYobWRXjwHbqR0tka6bzW0nTrX2REUbA"
    "gUV3jAOuZKU2P7DtCDReemU6W0TJpsNIi1/2j2oYwa/Fhgne1WWhadCRN5tHbsAAdrwYbksR6mwIydsu8/on2QBYLRYiLNMN5CtbLMY4zZHB"
    "sKzVE8MUAfiGKwhxBE3usWbXb+RhCo93uKe9RrCnikcdD6z53qQb24EpaAfm+rVvKzCElgZtGAE="
)

CLAUDE_SONNET_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "bedrock/converse/us.anthropic.claude-sonnet-5-5"},
    expected_provider_endpoint="/model/us.anthropic.claude-sonnet-5-5/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        **BEDROCK_CONVERSE_REQUEST,
        "system": BEDROCK_CONVERSE_MESSAGES_SYSTEM,
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_RESPONSE,
        "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 155}}},
        "metrics": {"latencyMs": 6213},
        "output": {
            "message": {
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0,
                                "text": "The user asked to read the validation file first, but since reading the "
                                "test file is independent, I can batch both reads together rather than "
                                "doing them sequentially. I'll read validation.py around line 18 (lines "
                                "1-60) along with the relevant test file section.\n\n",
                            },
                        },
                    },
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_1,
                                "text": "Opening both `shop/validation.py` and `tests/test_validation.py` to check "
                                "how `validate_quantity` handles zero, negative, and non-integer inputs.\n\n",
                            },
                        },
                    },
                    {
                        "toolUse": {
                            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
                            "name": "read_file",
                            "toolUseId": "tooluse_1r54N0OeCEZFcknGLsKsaa",
                            "type": "tool_use",
                        },
                    },
                    {
                        "toolUse": {
                            "input": {"path": "tests/test_validation.py", "start_line": 1, "end_line": 40},
                            "name": "read_file",
                            "toolUseId": "tooluse_VOzhTwpZcyOfeZH2KdVtDj",
                            "type": "tool_use",
                        },
                    },
                ],
                "role": "assistant",
            },
        },
        "usage": {
            "cacheDetails": [{"inputTokens": 853, "ttl": "5m"}],
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 853,
            "cacheWriteInputTokens": 853,
            "inputTokens": 4,
            "outputTokens": 430,
            "serverToolUsage": {},
            "totalTokens": 1287,
        },
    },
    expected_litellm_response={
        **BEDROCK_CONVERSE_MESSAGES_RESPONSE,
        "id": ANY,
        "model": "bedrock/converse/us.anthropic.claude-sonnet-5-5",
        "usage": {"input_tokens": 4, "output_tokens": 430, "cache_creation_input_tokens": 853},
        "content": [
            {
                "type": "thinking",
                "thinking": "The user asked to read the validation file first, but since reading the test file is "
                "independent, I can batch both reads together rather than doing them sequentially. I'll "
                "read validation.py around line 18 (lines 1-60) along with the relevant test file section.\n\n",
                "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": "Opening both `shop/validation.py` and `tests/test_validation.py` to check how "
                "`validate_quantity` handles zero, negative, and non-integer inputs.\n\n",
                "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_1,
            },
            {"type": "text", "text": ""},
            {
                "type": "tool_use",
                "id": "tooluse_1r54N0OeCEZFcknGLsKsaa",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            },
            {
                "type": "tool_use",
                "id": "tooluse_VOzhTwpZcyOfeZH2KdVtDj",
                "name": "read_file",
                "input": {"path": "tests/test_validation.py", "start_line": 1, "end_line": 40},
            },
        ],
    },
)

CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSxgQKEggSEAEYAjgBQgh0aGlua2luZxIMaxDDsYU61QS1+n8RGgz156K29uKC1HWPNQkiMFcfXYkUyc+WiXLtgYu9mq4bS8cW7hn9MRw8"
    "Sjvb5kO5dd7cwmt0bJHftIBtaLIANSrhA3JiSUmAaZwpmq3AgfM0LliKLK/KbQface14HtzKtd/5MF5Nil1AiqDmDblS1U03sFU5L3PyuCc+"
    "hrSaftK7Z1ZBXIB8tOzsko5jeGex0cTsPHFrthM1cH1pay0NMmwU+r1ufrR4EaW7K30oyy3oLol3JpgNF433ra6tRB5EVW2Da1aGlJi6anmp"
    "pp4Zq6VBJdGOFmZqTOQULt6dq5j1tg8pzcuQqkrYRipJREtdyxZSx8t98bdNrAWro5o4UW9cGcIZuWK24E994inMN0H4DAZliIHqpwAqwzAY"
    "pbvild9GOu8lLpIMwyB2mlLADSzwK+J5/gxFSh2WZ1t56d5AlkGd55IywS7EB+AlZelbCCuyEDwJV9d3woohXmGUvEEuaT7dD6K7gHCIPcvI"
    "nVdMnHzb6wlXmtKOUxl8mltIwm4PBBi2uoM8tgPq87fhXxGJgVOSzOdwhaJlbfkHCrVSXAloRx6qwvAZ8JcWBla26lKQo+4vxXo8Y4KgadCR"
    "TMLNfymY5BUsz2eJa0fPm4k5CqR5updBjj8PIcUPU2FF9ZUTWQ3BwEhIJHiSdkSrRKhEy8Ifw8Cdzsbq4WxVlYWJ2H9sdycc43+qO9bgf+xB"
    "gSeNP/G7uwmR2WcZocbodrIG97UYAQ=="
)

CLAUDE_OPUS_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "bedrock/converse/us.anthropic.claude-opus-5-5"},
    expected_provider_endpoint="/model/us.anthropic.claude-opus-5-5/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        **BEDROCK_CONVERSE_REQUEST,
        "system": BEDROCK_CONVERSE_MESSAGES_SYSTEM,
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_RESPONSE,
        "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 35}}},
        "metrics": {"latencyMs": 3730},
        "output": {
            "message": {
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
                                "text": "The user wants the validation file checked first, so I'll look around line "
                                "18 there and hold off on tests for now.\n\n",
                            },
                        },
                    },
                    {
                        "text": "I'll open `shop/validation.py` around `validate_quantity` (line 18) to see how it "
                        "treats zero, negative and non-integer values. After that I'll check the test file.",
                    },
                    {
                        "toolUse": {
                            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 50},
                            "name": "read_file",
                            "toolUseId": "tooluse_g2htBkaHk6egShGW6MsbjH",
                            "type": "tool_use",
                        },
                    },
                ],
                "role": "assistant",
            },
        },
        "usage": {
            "cacheDetails": [{"inputTokens": 853, "ttl": "5m"}],
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 853,
            "cacheWriteInputTokens": 853,
            "inputTokens": 4,
            "outputTokens": 190,
            "serverToolUsage": {},
            "totalTokens": 1047,
        },
    },
    expected_litellm_response={
        **BEDROCK_CONVERSE_MESSAGES_RESPONSE,
        "id": ANY,
        "model": "bedrock/converse/us.anthropic.claude-opus-5-5",
        "usage": {"input_tokens": 4, "output_tokens": 190, "cache_creation_input_tokens": 853},
        "content": [
            {
                "type": "thinking",
                "thinking": "The user wants the validation file checked first, so I'll look around line 18 there and "
                "hold off on tests for now.\n\n",
                "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
            },
            {
                "type": "text",
                "text": "I'll open `shop/validation.py` around `validate_quantity` (line 18) to see how it treats "
                "zero, negative and non-integer values. After that I'll check the test file.",
            },
            {
                "type": "tool_use",
                "id": "tooluse_g2htBkaHk6egShGW6MsbjH",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 50},
            },
        ],
    },
)

CLAUDE_FABLE_5_1_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "bedrock/converse/us.anthropic.claude-fable-5-1"},
    expected_provider_endpoint="/model/us.anthropic.claude-fable-5-1/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        **BEDROCK_CONVERSE_REQUEST,
        "system": BEDROCK_CONVERSE_MESSAGES_SYSTEM,
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "paths": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["query", "paths"],
                                "additionalProperties": False,
                            },
                        },
                        "name": "search_files",
                        "description": "Search source and test files for a literal term.",
                    },
                },
                {
                    "toolSpec": {
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "start_line": {"type": "integer", "minimum": 1},
                                    "end_line": {"type": "integer", "minimum": 1},
                                },
                                "required": ["path", "start_line", "end_line"],
                                "additionalProperties": False,
                            },
                        },
                        "name": "read_file",
                        "description": "Read a range of lines from a project file.",
                    },
                },
            ],
        },
    },
    mock_provider_response={
        **BEDROCK_CONVERSE_RESPONSE,
        "additionalModelResponseFields": {"usage": {"output_tokens_details": {"thinking_tokens": 0}}},
        "metrics": {"latencyMs": 4888},
        "output": {
            "message": {
                "content": [
                    {
                        "text": "I'll read the validation module to see how zero and negative quantities are "
                        "currently treated.",
                    },
                    {
                        "toolUse": {
                            "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 40},
                            "name": "read_file",
                            "toolUseId": "tooluse_ykxX1UplXkgne3EfgDI4Cv",
                            "type": "tool_use",
                        },
                    },
                ],
                "role": "assistant",
            },
        },
        "usage": {
            "cacheDetails": [{"inputTokens": 873, "ttl": "5m"}],
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 873,
            "cacheWriteInputTokens": 873,
            "inputTokens": 4,
            "outputTokens": 125,
            "serverToolUsage": {},
            "totalTokens": 1002,
        },
    },
    expected_litellm_response={
        **BEDROCK_CONVERSE_MESSAGES_RESPONSE,
        "id": ANY,
        "model": "bedrock/converse/us.anthropic.claude-fable-5-1",
        "usage": {"input_tokens": 4, "output_tokens": 125, "cache_creation_input_tokens": 873},
        "content": [
            {
                "type": "text",
                "text": "I'll read the validation module to see how zero and negative quantities are currently "
                "treated.",
            },
            {
                "type": "tool_use",
                "id": "tooluse_ykxX1UplXkgne3EfgDI4Cv",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 40},
            },
        ],
    },
)
