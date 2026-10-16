from typing import Final
from unittest.mock import ANY

from integration._support.client import object_value
from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.templates import (
    CHAT_COMPLETIONS_REQUEST,
    CHAT_COMPLETIONS_RESPONSE,
    CODING_AGENT_REQUEST,
    CODING_AGENT_RESPONSE,
)
from integration.translation.templates.anthropic import (
    ANTHROPIC_AGENT_REQUEST,
    ANTHROPIC_AGENT_RESPONSE,
)
from integration.translation.templates.vertex_ai import (
    VERTEX_AI_ANTHROPIC_MESSAGES_REQUEST,
    VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
    VERTEX_AI_ANTHROPIC_THINKING_SIGNATURE_0,
    VERTEX_AI_ANTHROPIC_THINKING_SIGNATURE_1,
    VERTEX_AI_GEMINI_GENERATE_CONTENT_REQUEST,
    VERTEX_AI_GEMINI_GENERATE_CONTENT_RESPONSE,
)

GEMINI_3_5_FLASH_THOUGHT_SIGNATURE: Final = "AY89a1+kriNs9YsPegJQxgjTWsMlxsqPU/UkHDyz4EK+3FRrE52ygqPIHTvkblgKKiO5Fn8R9D3iu//i+YnuuZ+e7W0yxUoFScas89b5rn3vZHsihnUA7VTe3b7beth1IftteoTGZ3F8OSLrSIP8HKn4xTIx35eeNEt5oiD3K5rTQrdklqO277HvTDhu/LuZhZ0pzaWciGQEc61epIBBpC2ygu8V4L8TekUVZnz6qzSzxipdiQStQ8dZou43334hMnxOR6JjJCGxqt/1K+C0vXX6YP6f+Bph5VhPSnvy2Z1uApi3TUxAcsDDyVewIr+ha2G6DPhN1Oi5uZZNhmbvH01Uq5OOquF1uEICq2m/35LhJIbDNM9N7oI1Z6L94P9xdugeFXM2S7wdJfc8sdg09vEzNeA++GK25zRuB4hU7XyaG4XnWjmu2kswi4sf+EyrO/K0P2tb/ZCJu7kJTtWTiD1wgSIl1eqS8ORKKsXDN2hIbFrtsNR50WIvVzd6"

CLAUDE_HAIKU_4_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "vertex_ai/claude-haiku-4-5@20251001",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-haiku-4-5@20251001:rawPredict",
    expected_provider_headers={
        "authorization": "Bearer scripted-token",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": "scripted-token",
    },
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "anthropic_version": "vertex-2023-10-16",
        "system": [{"type": "text", "text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "model": "claude-haiku-4-5-20251001",
        "id": "msg_vrtx_011CfjuySsPewSSfasuPiaHg",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 17,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 5,
        },
    },
    expected_litellm_status_code=200,
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "vertex_ai/claude-haiku-4-5@20251001",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
        "usage": {
            "completion_tokens": 5,
            "prompt_tokens": 17,
            "total_tokens": 22,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 5},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 17,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)

CLAUDE_SONNET_4_6_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "vertex_ai/claude-sonnet-4-6@default",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-sonnet-4-6@default:rawPredict",
    expected_provider_headers={
        "authorization": "Bearer scripted-token",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": "scripted-token",
    },
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "anthropic_version": "vertex-2023-10-16",
        "system": [{"type": "text", "text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "model": "claude-sonnet-4-6",
        "id": "msg_vrtx_011CfjuxUbJsRvQZCN21qNQq",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 18,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 5,
        },
    },
    expected_litellm_status_code=200,
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "vertex_ai/claude-sonnet-4-6@default",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello!",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
        "usage": {
            "completion_tokens": 5,
            "prompt_tokens": 18,
            "total_tokens": 23,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 5},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 18,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)

CLAUDE_SONNET_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "vertex_ai/claude-sonnet-5@default",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-sonnet-5@default:rawPredict",
    expected_provider_headers={
        "authorization": "Bearer scripted-token",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": "scripted-token",
    },
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "anthropic_version": "vertex-2023-10-16",
        "system": [{"type": "text", "text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "model": "claude-sonnet-5",
        "id": "msg_vrtx_011CfjuygPVDgzuMXdyyW9UP",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 21,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"thinking_tokens": 0},
        },
    },
    expected_litellm_status_code=200,
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "vertex_ai/claude-sonnet-5@default",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
        "usage": {
            "completion_tokens": 6,
            "prompt_tokens": 21,
            "total_tokens": 27,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 6},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 21,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)

CLAUDE_OPUS_4_8_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "vertex_ai/claude-opus-4-8@default",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-opus-4-8@default:rawPredict",
    expected_provider_headers={
        "authorization": "Bearer scripted-token",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": "scripted-token",
    },
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "anthropic_version": "vertex-2023-10-16",
        "system": [{"type": "text", "text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "model": "claude-opus-4-8",
        "id": "msg_vrtx_011CfjuyofoMgSaV8Pbt16N9",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 21,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"thinking_tokens": 0},
        },
    },
    expected_litellm_status_code=200,
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "vertex_ai/claude-opus-4-8@default",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
        "usage": {
            "completion_tokens": 6,
            "prompt_tokens": 21,
            "total_tokens": 27,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 6},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 21,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)

GEMINI_3_5_FLASH_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "vertex_ai/gemini-3.5-flash",
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/google/models/gemini-3.5-flash:generateContent",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer scripted-token"},
    expected_provider_request={
        "contents": [{"role": "user", "parts": [{"text": "Say hello."}]}],
        "system_instruction": {"parts": [{"text": "You are a terse assistant."}]},
        "generationConfig": {"max_output_tokens": 1024},
    },
    mock_provider_response={
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello.", "thoughtSignature": GEMINI_3_5_FLASH_THOUGHT_SIGNATURE}],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 9,
            "candidatesTokenCount": 2,
            "totalTokenCount": 95,
            "trafficType": "ON_DEMAND",
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 9}],
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 2}],
            "thoughtsTokenCount": 84,
        },
        "modelVersion": "gemini-3.5-flash",
        "createTime": "2026-10-05T22:54:50.809002Z",
        "responseId": "uirEaqqwMc-U9LsPjJTYuQU",
    },
    expected_litellm_status_code=200,
    expected_litellm_response={
        "id": "uirEaqqwMc-U9LsPjJTYuQU",
        "created": ANY,
        "model": "vertex_ai/gemini-3.5-flash",
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
            "completion_tokens": 86,
            "prompt_tokens": 9,
            "total_tokens": 95,
            "completion_tokens_details": {"reasoning_tokens": 84, "text_tokens": 2},
            "prompt_tokens_details": {"text_tokens": 9},
        },
        "vertex_ai_grounding_metadata": [],
        "vertex_ai_url_context_metadata": [],
        "vertex_ai_safety_results": [],
        "vertex_ai_citation_metadata": [],
    },
)

CLAUDE_OPUS_5_5_CHAT_COMPLETIONS_THINKING_SIGNATURE_0: Final = (
    "CAQS2AUKEggSEAIYAjgBQgh0aGlua2luZxIMARf7mDY/EfIxNI90GgyxpPEOoCV8iuAHRDwiMEya9+X1kJMbECIvAPWkwThHF5lM"
    "isFNo67kPmn/ek1/pKWvb6bc8EzcFYnNOyBtdCrzBB7ZgNYNQvTVWJrNQAET5gFlr7MeN5oNfe4tnrSysPx7f3rCzYfsUMswfE8O"
    "4sAJk87znpkMO9QRwq+PtgXijy8wWo4ERbvvKh1rPfgDhMSKkiujiaKJgMvMMM8NltTzpaaWlaFcWJNwG/k03dyJHhtvmwhaJyIP"
    "+W8cLd8iVAzHc8KtYlnXOIqpFOeto9p6rRK2NgykT1gZEqCkiFs89qnOv8S0Z3RWwWdMype2EhhFcNI+jWl52OzM/kbZxtnhQ+32"
    "cuntmmgYGQOXUujuAIZwmE4Rvs4U5j5sGjTE5zJZBqZUCtGBUJ9aQv6nVHGqwdy8dmpMexKPhOw/Uccq5nK3/WrufYrSFWuxz7A1"
    "egu/dURaBy1+9T7j0G5VYIeXwsTOh1uJuSk2pgg+dI+9/gk5Lbt1K14ghBxIDvHEfikoAqyGhVBIS1Wn5qIFkwOowwm0S+z7x/s+"
    "/3zy0gQoiEddpn52HkWzKKwrN5qZXJKmsn1pzvdQXyghGSYc5EL4BWZNhvccQgEMcJamK3+QyTGbf7mJM+D4yQNkdRAkurpNqM2O"
    "ofMSaOIXRlhN5zqjNGK5bCSO+2BL0HTaRbsRYDq1jVL+DXLyDymQ1QrXWssmgVo0k3Mzfkj1MenT77LCV2/ar0SYX+6GhBJlHm/B"
    "WmNh0rE8yq8SJY7A+4mXqg+4a0bBoMa7gHBNxTI4x3XBm75ctVkAieHVKORgM23sCQFvngXql0WTZUOC0QHf54MTGOoNBJYIJP7H"
    "Xu2RKL6lL9eqQFNYVEDQK/NyWHX2bw51g0TItCaW3Rl3T6YWXKlcVvAUBH8OQNjQII0j3yF+oGd6mxgB"
)

CLAUDE_OPUS_5_5_CHAT_COMPLETIONS_THINKING_SIGNATURE_1: Final = (
    "CAQSgAYKEwgSEAIYAjgBQgluYXJyYXRpb24SDNLS/zBzyAtsyJk3zBoMp+HCAq92FqxGZNofIjCv16GJ3WIGycSbcFtkGKdAUa4m"
    "GzcV7kH27GOyo/OLT/1GKuFrMp35qn1Jo1EvMBwqmgXZS6p/Jn0PHQj1KEteqdiVwW3Dt6t+vSPrGvq0Ned/d1q62jwgfdWU3xXj"
    "/WkbifL+c3Z1tnsLLtQw5/6FuYlD4NmxypiTWq7MmOrbQ6B5oV5nM9xd86Gqrc9mkSrptbRWyQOUQZFuafCXtzAThnAdaitbS/4I"
    "NvHyTgYwt096mDJB8p/aYQQlfu7bKjmzIwBxibfMRE8EzrmLP9Lnwjo+tU2qFdW/jn96SYW7u8vSmochwPeCLmXWjBwABenlhSsY"
    "ncUrohBqFrfGZ0eoOOltvI7CFIKedDyZEVWgj9k7/fWTg5FZuAYUXTj/NMka1CDe05zAReAqW/0qOVw9XLlqQ/+84A+hQnrdggTf"
    "24KGSmgHJ0gkmLNSnXgzSZd374QaNphAzUJy4MOMoYzu8KIrRev2FE05cDChzCQm3N27gL8oiqZzONNxYsNgbUu/30rMyvRla1uA"
    "e/a9nhXAfvGk0zXoh+cprhGOf9FmVWHg7jrnn7s3ZumAUDKpckvoUyiXTd2raFsOXd/c8fZNaZTyz28M0mfDDbMtWJdNqQLpJJLc"
    "Xr1YTJ7HNZeeA4TZiOp4CX4iEnI+RYylaXGLIodHCCOosriu9EHeceMIHASB+i6lxcL8OUUkTbASFbpe7qHB0wsqwDJGUO35fdJa"
    "mKWY9gM5GEMORkkeVzqT1CsELducpP58h54yC1kEl3z31qko5No2d7Tsb8C0KNAm2sr8O20cOGLxZ9hzAE4e0WQSF0gFqRQ3xAX9"
    "yoUU5tw4Q7PygHCzyWMArMf+GhJfoZZfOrvUl+KYu73xCshFSmi1OUft29RQRLZ9DRNagtwdro4LcBf2jVTzOZcUJ0VWl4J0ebuF"
    "yc4V9fnxZTe4KTHLr2XV5+qytgDzk4MYAQ=="
)

GEMINI_3_1_PRO_PREVIEW_CHAT_COMPLETIONS_THOUGHT_SIGNATURE_0: Final = (
    "AY89a18NmP/rpOzYqPT2gXtTGPJwCXopurRCh2SERSxHAbUcM+kBs62qrKZf6X3eHEdd/leFa2/wBviqJ6WkkEdIsgz94SdkfwWm"
    "53Pd6IC20z+9KTq6PpsXs5CJQzMRyWvSBBolC4JHE99knIqr4nJbUlXHYCknKnExuUNUuWzkcY2zsO/uX81og8IwGDikReqErvvU"
    "+EVyAV0TSDb3Pg0A8eC7RCHBLPfT0ljwzNJDibRUkwoxdKcPkMvdbYhZEAnkOPD7N/CpITooxxwBYVArsbDW1oA4kL78A2gnbsNi"
    "Y2esZSaKgxrftJk1H06rW5b7tAk7y4wMjbj7GD9XQJJ3HzLdUznXNvLRZjkNcr8evFgKPq3YIPqpuCn5klH09/qjGNZRkKoYriBS"
    "f95iWKvGZ7jwEHu+45Qw46o0lID1TFZsrnS4WiwllgqAhvvFZ//DbCtpjMIztpTGYD754oc60k50p6Ny14oPDBlNMOVqwiFGq9QC"
    "FQ7Vzilx+rqAoVdOzvyGSf6+SfXOT0Jv/raw+HIIA2bhOqurVqgdS4CoxdpCcZ5CmOgN4CIA9rQcuxZ/dVaT2w/UL0fUJOAN3kSU"
    "ngcI+FcqIjlbF+IbIEGphpBhsIaTd3qXVCH/57K0tP7lVu9GoAU9zNu/6YPIWZnE9ABUBfQkyTR/DdFZm3Hfsjt7m5YkPlm+kJNH"
    "iHHwdjTo+liy9gBFnQbl+mFgv3e0bcdy+a0pa6mUyluy0IWrKFBVFcfEep4vD7+CRTpkpwhwzL6+v2t3LWGcJGZV6GS/qbbGR3ys"
    "Cyr70DrZtaEnbOnLN3UYP6zDcDfKYTn7ub3hIjaME9LNjg=="
)

GEMINI_4_ARGON_CHAT_COMPLETIONS_THOUGHT_SIGNATURE_0: Final = (
    "AY89a19UcUynGbcdZOu5ucJyuVTEBkP7BWc+WQRD+zdel8qY2zqNh1OLS6DejBKWK9xndkK8coxMqPRuuQytzyB+zVvJz0ouSNh0"
    "nH3Bm0p+S9hFJDzNhlHrk4PRtU0OWxv5SKWtzQURmeBDk5mJV/MU9yJSDs7g+csm6juZi2Lang76vPWyQOHTVPYnJVzLHkP73H0D"
    "WnVPXSdNzBgqPzopqf3GoP+EvALphWyi0Zp/1uv7m4TKbx9dRZjlO15LaqpksnkDvwkF6o8KfaYrC/Skh+xYxlk40yxiiaYv5sr4"
    "PbGutzCGNWCZxjbBCbQBPGMFIL3fDGwLavbGd71tWKcOaSBr3oYbsw5HsfWMFHV8+TQ7rou77MCMWaiJ3mulflSmdBkkTq0m1bPm"
    "iIbwYvVPs6CCtTTZgGYTZhRefefwD1Dx4Y5epmGy1+jST7zikArzdZeexaCTTrIWO4rvqzPdAS65XHnfXjgYEPYK0YUnX6StK/ln"
    "Lrwdb+AK6j+Q2eLuKpx/HJE0QDZ9BgXpw/CQ6my0s2rI3Tjlbun8Bdl7MyFuCbstbTA+uu1RXXv2M1RThJQHiuDGvgzCNg/sgyCb"
    "zluzK/vp2nzki+ESsFeForBBSSCLVAep2tiqKhIOWPWBHYjH04qfgyWkylbRbZeA32+BPYViEcZa+Imi+aYXu241RDOSWRMJvbsJ"
    "oDY15ThjQ+sgLNKJkBoravc7+sQ1Abt+jwCcO9zHBquTWn8G/X0IggqHNuDnbWEW3S9HdxPXw7ZV0ZYXXkZXIYwFIt6Ji7LpXlEY"
    "X0hOwrBdq2ufKAv+EWBQyCU3oyBrIGeVdVx25BiLFsVJzqw3xLTUgfelHyawgwtGvd7J+mQBPp4YH1Rk6fbQMwDEWipSiltG+jG9"
    "ZqXUiF1NtN30tw4a4h2tA03uBV1yXhv6+fL7g2zmN6c4GVX9IzAS60UUnXIMlBgCXjxCU591TuPufe6mJQ=="
)


CLAUDE_SONNET_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "vertex_ai/claude-sonnet-5-5@default"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-sonnet-5-5@default:rawPredict",
    expected_provider_headers={
        "authorization": "Bearer scripted-token",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": "scripted-token",
    },
    expected_provider_request=VERTEX_AI_ANTHROPIC_MESSAGES_REQUEST,
    mock_provider_response={**VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE, "model": "claude-sonnet-5-5"},
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "vertex_ai/claude-sonnet-5-5@default",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 2,
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                                "name": "read_file",
                            },
                            "id": "toolu_vrtx_01SYkCZM7zmQWqcyokJjiAyf",
                            "type": "function",
                        },
                        {
                            "index": 3,
                            "function": {
                                "arguments": '{"path": "tests/test_validation.py", "start_line": 1, "end_line": 50}',
                                "name": "read_file",
                            },
                            "id": "toolu_vrtx_01UhxynEUZYnaNhRkRGC2ZGn",
                            "type": "function",
                        },
                    ],
                    "reasoning_content": (
                        "The user asked to read the validation file first, but since reading the tests is independent, I "
                        "think it's fine to read both files together rather than strictly sequencing them.\n\nI'll check "
                        "`shop/validation.py` for how `validate_quantity` handles zero, negative, and non-integer values, "
                        "and review `tests/test_validation.py` for the expected behavior.\n\n"
                    ),
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": (
                                "The user asked to read the validation file first, but since reading the tests is independent, I "
                                "think it's fine to read both files together rather than strictly sequencing them.\n\n"
                            ),
                            "signature": VERTEX_AI_ANTHROPIC_THINKING_SIGNATURE_0,
                        },
                        {
                            "type": "thinking",
                            "thinking": (
                                "I'll check `shop/validation.py` for how `validate_quantity` handles zero, negative, and non-integer "
                                "values, and review `tests/test_validation.py` for the expected behavior.\n\n"
                            ),
                            "signature": VERTEX_AI_ANTHROPIC_THINKING_SIGNATURE_1,
                        },
                    ],
                    "provider_specific_fields": {
                        "citations": None,
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": (
                                    "The user asked to read the validation file first, but since reading the tests is independent, I "
                                    "think it's fine to read both files together rather than strictly sequencing them.\n\n"
                                ),
                                "signature": VERTEX_AI_ANTHROPIC_THINKING_SIGNATURE_0,
                            },
                            {
                                "type": "thinking",
                                "thinking": (
                                    "I'll check `shop/validation.py` for how `validate_quantity` handles zero, negative, and non-integer "
                                    "values, and review `tests/test_validation.py` for the expected behavior.\n\n"
                                ),
                                "signature": VERTEX_AI_ANTHROPIC_THINKING_SIGNATURE_1,
                            },
                        ],
                    },
                    "content": None,
                },
            },
        ],
        "usage": {
            "completion_tokens": 425,
            "prompt_tokens": 880,
            "total_tokens": 1305,
            "completion_tokens_details": {"reasoning_tokens": 229, "text_tokens": 196},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 4,
                "cache_write_tokens": 876,
                "cache_creation_tokens": 876,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 876, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 876,
            "cache_read_input_tokens": 0,
        },
    },
)


CLAUDE_OPUS_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "vertex_ai/claude-opus-5-5@default"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-opus-5-5@default:rawPredict",
    expected_provider_headers={
        "authorization": "Bearer scripted-token",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": "scripted-token",
    },
    expected_provider_request=VERTEX_AI_ANTHROPIC_MESSAGES_REQUEST,
    mock_provider_response={
        **VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-opus-5-5",
        "id": "msg_vrtx_011Cg38nZLrXXSDtDRwwmsco",
        "content": [
            {
                "type": "thinking",
                "thinking": (
                    "I'll focus on reading validation.py around line 18 first, since the user specifically asked for "
                    "that file before anything else.\n\n"
                ),
                "signature": CLAUDE_OPUS_5_5_CHAT_COMPLETIONS_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": (
                    "I found `validate_quantity` at line 18 in `shop/validation.py`, so I'll check the surrounding code "
                    "to see how it handles zero, negative, and non-integer quantities before looking at the tests.\n\n"
                ),
                "signature": CLAUDE_OPUS_5_5_CHAT_COMPLETIONS_THINKING_SIGNATURE_1,
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_0128Am9iqNFAAAMfU7bBdpQz",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 45},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 876,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 876, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 265,
            "output_tokens_details": {"thinking_tokens": 167},
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "vertex_ai/claude-opus-5-5@default",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 2,
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 45}',
                                "name": "read_file",
                            },
                            "id": "toolu_vrtx_0128Am9iqNFAAAMfU7bBdpQz",
                            "type": "function",
                        },
                    ],
                    "reasoning_content": (
                        "I'll focus on reading validation.py around line 18 first, since the user specifically asked for "
                        "that file before anything else.\n\nI found `validate_quantity` at line 18 in `shop/validation.py`, so "
                        "I'll check the surrounding code to see how it handles zero, negative, and non-integer quantities "
                        "before looking at the tests.\n\n"
                    ),
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": (
                                "I'll focus on reading validation.py around line 18 first, since the user specifically asked for "
                                "that file before anything else.\n\n"
                            ),
                            "signature": CLAUDE_OPUS_5_5_CHAT_COMPLETIONS_THINKING_SIGNATURE_0,
                        },
                        {
                            "type": "thinking",
                            "thinking": (
                                "I found `validate_quantity` at line 18 in `shop/validation.py`, so I'll check the surrounding code "
                                "to see how it handles zero, negative, and non-integer quantities before looking at the tests.\n\n"
                            ),
                            "signature": CLAUDE_OPUS_5_5_CHAT_COMPLETIONS_THINKING_SIGNATURE_1,
                        },
                    ],
                    "provider_specific_fields": {
                        "citations": None,
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": (
                                    "I'll focus on reading validation.py around line 18 first, since the user specifically asked for "
                                    "that file before anything else.\n\n"
                                ),
                                "signature": CLAUDE_OPUS_5_5_CHAT_COMPLETIONS_THINKING_SIGNATURE_0,
                            },
                            {
                                "type": "thinking",
                                "thinking": (
                                    "I found `validate_quantity` at line 18 in `shop/validation.py`, so I'll check the surrounding code "
                                    "to see how it handles zero, negative, and non-integer quantities before looking at the tests.\n\n"
                                ),
                                "signature": CLAUDE_OPUS_5_5_CHAT_COMPLETIONS_THINKING_SIGNATURE_1,
                            },
                        ],
                    },
                    "content": None,
                },
            },
        ],
        "usage": {
            "completion_tokens": 265,
            "prompt_tokens": 880,
            "total_tokens": 1145,
            "completion_tokens_details": {"reasoning_tokens": 167, "text_tokens": 98},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 4,
                "cache_write_tokens": 876,
                "cache_creation_tokens": 876,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 876, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 876,
            "cache_read_input_tokens": 0,
        },
    },
)


CLAUDE_FABLE_5_1_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "vertex_ai/claude-fable-5-1@default"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-fable-5-1@default:rawPredict",
    expected_provider_headers={
        "authorization": "Bearer scripted-token",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": "scripted-token",
    },
    expected_provider_request=VERTEX_AI_ANTHROPIC_MESSAGES_REQUEST,
    mock_provider_response={
        **VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-fable-5-1",
        "id": "msg_vrtx_011Cg38nKaNiEJbdoCePiCCB",
        "content": [
            {
                "type": "text",
                "text": (
                    "I'll read the validation function and the zero-quantity test together, since neither depends on the "
                    "other."
                ),
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01B9TTi6KdxKcFt6W9mhsdWT",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 40},
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01XwRiPmJA1nhRaNgijpqU97",
                "name": "read_file",
                "input": {"path": "tests/test_validation.py", "start_line": 1, "end_line": 30},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 876,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 876, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 226,
            "output_tokens_details": {"thinking_tokens": 0},
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "vertex_ai/claude-fable-5-1@default",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": (
                        "I'll read the validation function and the zero-quantity test together, since neither depends on the "
                        "other."
                    ),
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 1,
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 40}',
                                "name": "read_file",
                            },
                            "id": "toolu_vrtx_01B9TTi6KdxKcFt6W9mhsdWT",
                            "type": "function",
                        },
                        {
                            "index": 2,
                            "function": {
                                "arguments": '{"path": "tests/test_validation.py", "start_line": 1, "end_line": 30}',
                                "name": "read_file",
                            },
                            "id": "toolu_vrtx_01XwRiPmJA1nhRaNgijpqU97",
                            "type": "function",
                        },
                    ],
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            },
        ],
        "usage": {
            "completion_tokens": 226,
            "prompt_tokens": 880,
            "total_tokens": 1106,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 226},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 4,
                "cache_write_tokens": 876,
                "cache_creation_tokens": 876,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 876, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 876,
            "cache_read_input_tokens": 0,
        },
    },
)


GEMINI_3_8_FLASH_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "vertex_ai/gemini-3.8-flash"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/google/models/gemini-3.8-flash:generateContent",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer scripted-token"},
    expected_provider_request=VERTEX_AI_GEMINI_GENERATE_CONTENT_REQUEST,
    mock_provider_response={**VERTEX_AI_GEMINI_GENERATE_CONTENT_RESPONSE, "modelVersion": "gemini-3.8-flash"},
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "vertex_ai/gemini-3.8-flash",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": (
                        "To see how quantities (including zero, negative, and non-integer values) are validated, I need to "
                        "read `shop/validation.py` next."
                    ),
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "provider_specific_fields": {
                                "thought_signature": (
                                    "AY89a1/K8QT9La7l3gDc5jImwnXSF0xUuyg8rAUnAyYS4c2/Yjd/fV3Ve1tHSww5bSooGMdZ0oqljadXXxUqYi+7fkKOWD0Ij80K"
                                    "LFu1O//r13u8cIpIqUxEi3lNiEn8rK6WVW5DPTMTBddZBL+/DcpCEi7U3m5vGYfoJUm9AJwQB8XDRdqLdA1hfiRnCqGpLJsFAQ2g"
                                    "Ln3FpQUPR3+FRoWRlJnNRjk8vWaq3mBKHSHttLMRd3rpvo6VAAZbs2OGoBJXFXA9S+RyN532kDK3fZIGGW2Fk+FSD9EJ1hN1+Qeh"
                                    "v8gzcxyZOin2O/ONFWEQ4zf8lHTzaostuIMLkKFM3v/ssK+Y4VgS0sPFr7VFByhczwoDx73oh141iDD5ch7tqAgnR7YikrWQDuV2"
                                    "8FBSB7+MQ3BrBSt6XUvYgcHsGNbg0giE3D3pFDR7uvLfvgjhFFdIqnCWWmPfBgZ7GOD55unG0IUPA2udP65xg88r8TjfpQOUsOnN"
                                    "KZLOiKovzHrXZX2L0tRGha5nE8ICi427Mc5YF9lm0Ilpo/BrGuuusMG3Zx9RxRyjRa9+zWPANRrvuq7SOVnFvAss4uu9PYf5Orwn"
                                    "nGFJ0xjP1hn22yXOxRYy62ySQQ1ValpWXjPPXvGKZRiUArCuvrBvDEZoUfJhlcVXs26kYnqRikRLev4GZGvBIDnGHyWLvMo0gx4r"
                                    "1kBnAGl5WyfZJu5SFywm/o5fMP70zVUoVdY+zC885uqB5pEV+VsAMyy98NumfmBF7kGtIjbt4l+hAqpzM1ME1z0kRqN7n9JEKn0H"
                                    "DlsQAD9V582O2bBVSTRzF8aFJ7RKohGhrFqyGmYLA+ZAv5Z4kJ1HBZg1LEz2G88MT2vznP6lCEH0m9da48Q5/85pNQ5dlGGEkhpP"
                                    "I+xazh6EZ0cFNGdNY1vAQY5rBQ8LPwizNSKenYvqwMdobWy8xReIQWCsOk2l4QCtlu8kC+iabWrsCPOfjdQmYTlpI+emQwssWXYU"
                                    "UQ+wDGl7QWoaSAnwGu+XQMQ9ndaUycPdBu5+cuw815uiNNtIVfeKejFIbLF66Jq/duKfZ9tgWthPzpoM50u9aie2FZwNOuEtaLIv"
                                    "WcdtBpY9LvnAemSuDUTLLSuiQYwBwRq8HKdFARJ362Wf8gg+ldoeXEV9R6+cscPaeY5j/+E/rsTWyxixg614qiFQyF5xL/7DWjUI"
                                    "LBuIrmJUIVcu8LelTUslepTGF6CBTJ0RABv8pK7S8dUDy+HXtvT8bmz8l9zOsaTagC1jUpvaeGpi44gorN9KU7z56FMxoKSeKF7M"
                                    "B2zuTFBN/GekI9gddLfQRaVK/9LAD4j+uBzdt/XICe2kGqjNu3ds9tnx82/9beezRyTRfFQ+p1V/jfaEt8AMTZ5I2IZQAkgkAnXa"
                                    "KieRKs8qKbkF5yjhtI0krsyg4+BDQZzog+dgkxqYxLDuNyd6hN9HlnMEBAwW8hxWGp4WE8GvL2IU529JVMaBvK/NcrGNnPW/m158"
                                    "BwaIFSfbSxQbqjv6KKqMi30i2bowXBAJit2YQHb3VN1reFBynM9mRm71PGLAbizdVS5VIqNI5zGquZrgpmTqqMF/AjsWpVSTo6Gi"
                                    "WD3O2PJWReehkH3aMJMFeu3jv0/Ru+Ihw1zwFcZr0Yh7rK2NEYQyZjcmf5/vX0rBh89HEVdc78j16arXcFWHNZS2lLAm0NC+Xmak"
                                    "WIP3YUcba6POPYKwqP2huUccyyGg+qFPTcWF488PNm4H86YQRrnUSkPEofjMuaLMrp3vS1TgpjuUZRiZ"
                                ),
                            },
                            "function": {
                                "arguments": '{"start_line": 1, "end_line": 50, "path": "shop/validation.py"}',
                                "name": "read_file",
                            },
                            "id": (
                                "call_267302__thought__AY89a1/K8QT9La7l3gDc5jImwnXSF0xUuyg8rAUnAyYS4c2/Yjd/fV3Ve1tHSww5bSooGMdZ0oqlja"
                                "dXXxUqYi+7fkKOWD0Ij80KLFu1O//r13u8cIpIqUxEi3lNiEn8rK6WVW5DPTMTBddZBL+/DcpCEi7U3m5vGYfoJUm9AJwQB8XDRd"
                                "qLdA1hfiRnCqGpLJsFAQ2gLn3FpQUPR3+FRoWRlJnNRjk8vWaq3mBKHSHttLMRd3rpvo6VAAZbs2OGoBJXFXA9S+RyN532kDK3fZ"
                                "IGGW2Fk+FSD9EJ1hN1+Qehv8gzcxyZOin2O/ONFWEQ4zf8lHTzaostuIMLkKFM3v/ssK+Y4VgS0sPFr7VFByhczwoDx73oh141iD"
                                "D5ch7tqAgnR7YikrWQDuV28FBSB7+MQ3BrBSt6XUvYgcHsGNbg0giE3D3pFDR7uvLfvgjhFFdIqnCWWmPfBgZ7GOD55unG0IUPA2"
                                "udP65xg88r8TjfpQOUsOnNKZLOiKovzHrXZX2L0tRGha5nE8ICi427Mc5YF9lm0Ilpo/BrGuuusMG3Zx9RxRyjRa9+zWPANRrvuq"
                                "7SOVnFvAss4uu9PYf5OrwnnGFJ0xjP1hn22yXOxRYy62ySQQ1ValpWXjPPXvGKZRiUArCuvrBvDEZoUfJhlcVXs26kYnqRikRLev"
                                "4GZGvBIDnGHyWLvMo0gx4r1kBnAGl5WyfZJu5SFywm/o5fMP70zVUoVdY+zC885uqB5pEV+VsAMyy98NumfmBF7kGtIjbt4l+hAq"
                                "pzM1ME1z0kRqN7n9JEKn0HDlsQAD9V582O2bBVSTRzF8aFJ7RKohGhrFqyGmYLA+ZAv5Z4kJ1HBZg1LEz2G88MT2vznP6lCEH0m9"
                                "da48Q5/85pNQ5dlGGEkhpPI+xazh6EZ0cFNGdNY1vAQY5rBQ8LPwizNSKenYvqwMdobWy8xReIQWCsOk2l4QCtlu8kC+iabWrsCP"
                                "OfjdQmYTlpI+emQwssWXYUUQ+wDGl7QWoaSAnwGu+XQMQ9ndaUycPdBu5+cuw815uiNNtIVfeKejFIbLF66Jq/duKfZ9tgWthPzp"
                                "oM50u9aie2FZwNOuEtaLIvWcdtBpY9LvnAemSuDUTLLSuiQYwBwRq8HKdFARJ362Wf8gg+ldoeXEV9R6+cscPaeY5j/+E/rsTWyx"
                                "ixg614qiFQyF5xL/7DWjUILBuIrmJUIVcu8LelTUslepTGF6CBTJ0RABv8pK7S8dUDy+HXtvT8bmz8l9zOsaTagC1jUpvaeGpi44"
                                "gorN9KU7z56FMxoKSeKF7MB2zuTFBN/GekI9gddLfQRaVK/9LAD4j+uBzdt/XICe2kGqjNu3ds9tnx82/9beezRyTRfFQ+p1V/jf"
                                "aEt8AMTZ5I2IZQAkgkAnXaKieRKs8qKbkF5yjhtI0krsyg4+BDQZzog+dgkxqYxLDuNyd6hN9HlnMEBAwW8hxWGp4WE8GvL2IU52"
                                "9JVMaBvK/NcrGNnPW/m158BwaIFSfbSxQbqjv6KKqMi30i2bowXBAJit2YQHb3VN1reFBynM9mRm71PGLAbizdVS5VIqNI5zGquZ"
                                "rgpmTqqMF/AjsWpVSTo6GiWD3O2PJWReehkH3aMJMFeu3jv0/Ru+Ihw1zwFcZr0Yh7rK2NEYQyZjcmf5/vX0rBh89HEVdc78j16a"
                                "rXcFWHNZS2lLAm0NC+XmakWIP3YUcba6POPYKwqP2huUccyyGg+qFPTcWF488PNm4H86YQRrnUSkPEofjMuaLMrp3vS1TgpjuUZR"
                                "iZ"
                            ),
                            "type": "function",
                        },
                    ],
                    "images": [],
                    "reasoning_content": (
                        "**My Thought Process on Analyzing the Request and Planning the Next Action**\n\nRight, so my primary "
                        "objective here is to fulfill the user's request while adhering strictly to the system's operational "
                        'guidelines. The user has specifically asked me to "Read the validation file first so we can '
                        'understand how zero and negative quantities are handled." This is a direct instruction that aligns '
                        'perfectly with the overarching system instruction to "Read the relevant files before suggesting a '
                        'fix."\n\nMy internal checklist, based on the system\'s directives, also reminds me to:\n*   "Explain '
                        'which file you need next, then call a read-only tool."\n*   "Never change files." (This is a '
                        'foundational rule, of course.)\n*   "Use the chat_completions client for this project review." (This '
                        "context is already established.)\n*   Crucially, and directly relevant to the user's query: "
                        '"Consider zero, negative and non-integer quantities before choosing the next file to '
                        'inspect."\n\nNow, looking at the files I have access to, I see `shop/validation.py` and '
                        "`tests/test_validation.py`. The user's request directly points to `shop/validation.py` as the next "
                        "file to examine for validation logic. This makes sense, as the name itself implies it's where the "
                        "core validation rules reside.\n\nBefore I jump straight into invoking a tool, the system requires me "
                        "to articulate *why* I'm selecting this file and *what* I intend to do with it. So, I need to "
                        "explain that I'm going to inspect `shop/validation.py` specifically to understand its handling of "
                        "quantities, particularly concerning zero, negative, and non-integer values, as prompted by the user "
                        "and reinforced by the system's guidance.\n\nFollowing that explanation, I'll then execute the action "
                        "by calling a read-only tool, `read_file`. I'll specify `shop/validation.py` as the path, and to get "
                        "a good initial overview, I'll plan to read a reasonable chunk, say from line 1 to around line 50, "
                        "to capture the initial validation logic. This seems like a good starting point to gather the "
                        "necessary context for the user's specific inquiry.\n\n"
                    ),
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": (
                                "**My Thought Process on Analyzing the Request and Planning the Next Action**\n\nRight, so my primary "
                                "objective here is to fulfill the user's request while adhering strictly to the system's operational "
                                'guidelines. The user has specifically asked me to "Read the validation file first so we can '
                                'understand how zero and negative quantities are handled." This is a direct instruction that aligns '
                                'perfectly with the overarching system instruction to "Read the relevant files before suggesting a '
                                'fix."\n\nMy internal checklist, based on the system\'s directives, also reminds me to:\n*   "Explain '
                                'which file you need next, then call a read-only tool."\n*   "Never change files." (This is a '
                                'foundational rule, of course.)\n*   "Use the chat_completions client for this project review." (This '
                                "context is already established.)\n*   Crucially, and directly relevant to the user's query: "
                                '"Consider zero, negative and non-integer quantities before choosing the next file to '
                                'inspect."\n\nNow, looking at the files I have access to, I see `shop/validation.py` and '
                                "`tests/test_validation.py`. The user's request directly points to `shop/validation.py` as the next "
                                "file to examine for validation logic. This makes sense, as the name itself implies it's where the "
                                "core validation rules reside.\n\nBefore I jump straight into invoking a tool, the system requires me "
                                "to articulate *why* I'm selecting this file and *what* I intend to do with it. So, I need to "
                                "explain that I'm going to inspect `shop/validation.py` specifically to understand its handling of "
                                "quantities, particularly concerning zero, negative, and non-integer values, as prompted by the user "
                                "and reinforced by the system's guidance.\n\nFollowing that explanation, I'll then execute the action "
                                "by calling a read-only tool, `read_file`. I'll specify `shop/validation.py` as the path, and to get "
                                "a good initial overview, I'll plan to read a reasonable chunk, say from line 1 to around line 50, "
                                "to capture the initial validation logic. This seems like a good starting point to gather the "
                                "necessary context for the user's specific inquiry.\n\n"
                            ),
                        },
                    ],
                    "provider_specific_fields": {
                        "thought_signatures": [
                            (
                                "AY89a1/K8QT9La7l3gDc5jImwnXSF0xUuyg8rAUnAyYS4c2/Yjd/fV3Ve1tHSww5bSooGMdZ0oqljadXXxUqYi+7fkKOWD0Ij80K"
                                "LFu1O//r13u8cIpIqUxEi3lNiEn8rK6WVW5DPTMTBddZBL+/DcpCEi7U3m5vGYfoJUm9AJwQB8XDRdqLdA1hfiRnCqGpLJsFAQ2g"
                                "Ln3FpQUPR3+FRoWRlJnNRjk8vWaq3mBKHSHttLMRd3rpvo6VAAZbs2OGoBJXFXA9S+RyN532kDK3fZIGGW2Fk+FSD9EJ1hN1+Qeh"
                                "v8gzcxyZOin2O/ONFWEQ4zf8lHTzaostuIMLkKFM3v/ssK+Y4VgS0sPFr7VFByhczwoDx73oh141iDD5ch7tqAgnR7YikrWQDuV2"
                                "8FBSB7+MQ3BrBSt6XUvYgcHsGNbg0giE3D3pFDR7uvLfvgjhFFdIqnCWWmPfBgZ7GOD55unG0IUPA2udP65xg88r8TjfpQOUsOnN"
                                "KZLOiKovzHrXZX2L0tRGha5nE8ICi427Mc5YF9lm0Ilpo/BrGuuusMG3Zx9RxRyjRa9+zWPANRrvuq7SOVnFvAss4uu9PYf5Orwn"
                                "nGFJ0xjP1hn22yXOxRYy62ySQQ1ValpWXjPPXvGKZRiUArCuvrBvDEZoUfJhlcVXs26kYnqRikRLev4GZGvBIDnGHyWLvMo0gx4r"
                                "1kBnAGl5WyfZJu5SFywm/o5fMP70zVUoVdY+zC885uqB5pEV+VsAMyy98NumfmBF7kGtIjbt4l+hAqpzM1ME1z0kRqN7n9JEKn0H"
                                "DlsQAD9V582O2bBVSTRzF8aFJ7RKohGhrFqyGmYLA+ZAv5Z4kJ1HBZg1LEz2G88MT2vznP6lCEH0m9da48Q5/85pNQ5dlGGEkhpP"
                                "I+xazh6EZ0cFNGdNY1vAQY5rBQ8LPwizNSKenYvqwMdobWy8xReIQWCsOk2l4QCtlu8kC+iabWrsCPOfjdQmYTlpI+emQwssWXYU"
                                "UQ+wDGl7QWoaSAnwGu+XQMQ9ndaUycPdBu5+cuw815uiNNtIVfeKejFIbLF66Jq/duKfZ9tgWthPzpoM50u9aie2FZwNOuEtaLIv"
                                "WcdtBpY9LvnAemSuDUTLLSuiQYwBwRq8HKdFARJ362Wf8gg+ldoeXEV9R6+cscPaeY5j/+E/rsTWyxixg614qiFQyF5xL/7DWjUI"
                                "LBuIrmJUIVcu8LelTUslepTGF6CBTJ0RABv8pK7S8dUDy+HXtvT8bmz8l9zOsaTagC1jUpvaeGpi44gorN9KU7z56FMxoKSeKF7M"
                                "B2zuTFBN/GekI9gddLfQRaVK/9LAD4j+uBzdt/XICe2kGqjNu3ds9tnx82/9beezRyTRfFQ+p1V/jfaEt8AMTZ5I2IZQAkgkAnXa"
                                "KieRKs8qKbkF5yjhtI0krsyg4+BDQZzog+dgkxqYxLDuNyd6hN9HlnMEBAwW8hxWGp4WE8GvL2IU529JVMaBvK/NcrGNnPW/m158"
                                "BwaIFSfbSxQbqjv6KKqMi30i2bowXBAJit2YQHb3VN1reFBynM9mRm71PGLAbizdVS5VIqNI5zGquZrgpmTqqMF/AjsWpVSTo6Gi"
                                "WD3O2PJWReehkH3aMJMFeu3jv0/Ru+Ihw1zwFcZr0Yh7rK2NEYQyZjcmf5/vX0rBh89HEVdc78j16arXcFWHNZS2lLAm0NC+Xmak"
                                "WIP3YUcba6POPYKwqP2huUccyyGg+qFPTcWF488PNm4H86YQRrnUSkPEofjMuaLMrp3vS1TgpjuUZRiZ"
                            ),
                        ],
                    },
                },
                "provider_specific_fields": {"native_finish_reason": "STOP"},
            },
        ],
        "usage": {
            "completion_tokens": 396,
            "prompt_tokens": 246,
            "total_tokens": 642,
            "completion_tokens_details": {"reasoning_tokens": 331, "text_tokens": 65},
            "prompt_tokens_details": {"text_tokens": 246},
        },
        "vertex_ai_grounding_metadata": [],
        "vertex_ai_url_context_metadata": [],
        "vertex_ai_safety_results": [],
        "vertex_ai_citation_metadata": [],
    },
)


GEMINI_3_1_PRO_PREVIEW_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "vertex_ai/gemini-3.1-pro-preview"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/google/models/gemini-3.1-pro-preview:generateContent",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer scripted-token"},
    expected_provider_request=VERTEX_AI_GEMINI_GENERATE_CONTENT_REQUEST,
    mock_provider_response={
        **VERTEX_AI_GEMINI_GENERATE_CONTENT_RESPONSE,
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": (
                                "**My Thought Process**\n\nAlright, first things first, I need to get a handle on how this system is "
                                "dealing with product quantities, specifically those tricky zero and negative values. The best place "
                                "to start is the validation logic itself. I'll bet that `shop/validation.py` file holds the key to "
                                "this.\n\nTo get to the bottom of it, I'm going to use the `read_file` tool. I'll point it directly at "
                                "`shop/validation.py`. I don't need the whole file, just enough to see the relevant validation "
                                "functions. I'll set a reasonable range, say from line 1 to line 50. This should be more than enough "
                                "to capture the `validate_quantity` function, which I anticipate will be at or around line 18, and "
                                "give me the insights I'm looking for into how zero and negative quantities are being treated.\n\n"
                            ),
                            "thought": True,
                        },
                        {
                            "text": (
                                "I'll read the validation file to see the implementation details for the quantity validation, "
                                "specifically focusing on how zero and negative values are processed."
                            ),
                        },
                        {
                            "functionCall": {
                                "name": "read_file",
                                "args": {"end_line": 50, "start_line": 1, "path": "shop/validation.py"},
                                "id": "call_3113284",
                            },
                            "thoughtSignature": GEMINI_3_1_PRO_PREVIEW_CHAT_COMPLETIONS_THOUGHT_SIGNATURE_0,
                        },
                    ],
                },
                "finishReason": "STOP",
            },
        ],
        "usageMetadata": {
            "promptTokenCount": 246,
            "candidatesTokenCount": 61,
            "totalTokenCount": 461,
            "trafficType": "ON_DEMAND",
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 246}],
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 61}],
            "thoughtsTokenCount": 154,
        },
        "modelVersion": "gemini-3.1-pro-preview",
        "createTime": "2026-10-15T01:13:17.173080Z",
        "responseId": "rSjQapjICtCX4_UPjurAmAo",
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "vertex_ai/gemini-3.1-pro-preview",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": (
                        "I'll read the validation file to see the implementation details for the quantity validation, "
                        "specifically focusing on how zero and negative values are processed."
                    ),
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "provider_specific_fields": {
                                "thought_signature": (
                                    "AY89a18NmP/rpOzYqPT2gXtTGPJwCXopurRCh2SERSxHAbUcM+kBs62qrKZf6X3eHEdd/leFa2/wBviqJ6WkkEdIsgz94SdkfwWm"
                                    "53Pd6IC20z+9KTq6PpsXs5CJQzMRyWvSBBolC4JHE99knIqr4nJbUlXHYCknKnExuUNUuWzkcY2zsO/uX81og8IwGDikReqErvvU"
                                    "+EVyAV0TSDb3Pg0A8eC7RCHBLPfT0ljwzNJDibRUkwoxdKcPkMvdbYhZEAnkOPD7N/CpITooxxwBYVArsbDW1oA4kL78A2gnbsNi"
                                    "Y2esZSaKgxrftJk1H06rW5b7tAk7y4wMjbj7GD9XQJJ3HzLdUznXNvLRZjkNcr8evFgKPq3YIPqpuCn5klH09/qjGNZRkKoYriBS"
                                    "f95iWKvGZ7jwEHu+45Qw46o0lID1TFZsrnS4WiwllgqAhvvFZ//DbCtpjMIztpTGYD754oc60k50p6Ny14oPDBlNMOVqwiFGq9QC"
                                    "FQ7Vzilx+rqAoVdOzvyGSf6+SfXOT0Jv/raw+HIIA2bhOqurVqgdS4CoxdpCcZ5CmOgN4CIA9rQcuxZ/dVaT2w/UL0fUJOAN3kSU"
                                    "ngcI+FcqIjlbF+IbIEGphpBhsIaTd3qXVCH/57K0tP7lVu9GoAU9zNu/6YPIWZnE9ABUBfQkyTR/DdFZm3Hfsjt7m5YkPlm+kJNH"
                                    "iHHwdjTo+liy9gBFnQbl+mFgv3e0bcdy+a0pa6mUyluy0IWrKFBVFcfEep4vD7+CRTpkpwhwzL6+v2t3LWGcJGZV6GS/qbbGR3ys"
                                    "Cyr70DrZtaEnbOnLN3UYP6zDcDfKYTn7ub3hIjaME9LNjg=="
                                ),
                            },
                            "function": {
                                "arguments": '{"end_line": 50, "start_line": 1, "path": "shop/validation.py"}',
                                "name": "read_file",
                            },
                            "id": (
                                "call_3113284__thought__AY89a18NmP/rpOzYqPT2gXtTGPJwCXopurRCh2SERSxHAbUcM+kBs62qrKZf6X3eHEdd/leFa2/wB"
                                "viqJ6WkkEdIsgz94SdkfwWm53Pd6IC20z+9KTq6PpsXs5CJQzMRyWvSBBolC4JHE99knIqr4nJbUlXHYCknKnExuUNUuWzkcY2zs"
                                "O/uX81og8IwGDikReqErvvU+EVyAV0TSDb3Pg0A8eC7RCHBLPfT0ljwzNJDibRUkwoxdKcPkMvdbYhZEAnkOPD7N/CpITooxxwBY"
                                "VArsbDW1oA4kL78A2gnbsNiY2esZSaKgxrftJk1H06rW5b7tAk7y4wMjbj7GD9XQJJ3HzLdUznXNvLRZjkNcr8evFgKPq3YIPqpu"
                                "Cn5klH09/qjGNZRkKoYriBSf95iWKvGZ7jwEHu+45Qw46o0lID1TFZsrnS4WiwllgqAhvvFZ//DbCtpjMIztpTGYD754oc60k50p"
                                "6Ny14oPDBlNMOVqwiFGq9QCFQ7Vzilx+rqAoVdOzvyGSf6+SfXOT0Jv/raw+HIIA2bhOqurVqgdS4CoxdpCcZ5CmOgN4CIA9rQcu"
                                "xZ/dVaT2w/UL0fUJOAN3kSUngcI+FcqIjlbF+IbIEGphpBhsIaTd3qXVCH/57K0tP7lVu9GoAU9zNu/6YPIWZnE9ABUBfQkyTR/D"
                                "dFZm3Hfsjt7m5YkPlm+kJNHiHHwdjTo+liy9gBFnQbl+mFgv3e0bcdy+a0pa6mUyluy0IWrKFBVFcfEep4vD7+CRTpkpwhwzL6+v"
                                "2t3LWGcJGZV6GS/qbbGR3ysCyr70DrZtaEnbOnLN3UYP6zDcDfKYTn7ub3hIjaME9LNjg=="
                            ),
                            "type": "function",
                        },
                    ],
                    "images": [],
                    "reasoning_content": (
                        "**My Thought Process**\n\nAlright, first things first, I need to get a handle on how this system is "
                        "dealing with product quantities, specifically those tricky zero and negative values. The best place "
                        "to start is the validation logic itself. I'll bet that `shop/validation.py` file holds the key to "
                        "this.\n\nTo get to the bottom of it, I'm going to use the `read_file` tool. I'll point it directly at "
                        "`shop/validation.py`. I don't need the whole file, just enough to see the relevant validation "
                        "functions. I'll set a reasonable range, say from line 1 to line 50. This should be more than enough "
                        "to capture the `validate_quantity` function, which I anticipate will be at or around line 18, and "
                        "give me the insights I'm looking for into how zero and negative quantities are being treated.\n\n"
                    ),
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": (
                                "**My Thought Process**\n\nAlright, first things first, I need to get a handle on how this system is "
                                "dealing with product quantities, specifically those tricky zero and negative values. The best place "
                                "to start is the validation logic itself. I'll bet that `shop/validation.py` file holds the key to "
                                "this.\n\nTo get to the bottom of it, I'm going to use the `read_file` tool. I'll point it directly at "
                                "`shop/validation.py`. I don't need the whole file, just enough to see the relevant validation "
                                "functions. I'll set a reasonable range, say from line 1 to line 50. This should be more than enough "
                                "to capture the `validate_quantity` function, which I anticipate will be at or around line 18, and "
                                "give me the insights I'm looking for into how zero and negative quantities are being treated.\n\n"
                            ),
                        },
                    ],
                    "provider_specific_fields": {
                        "thought_signatures": [
                            (
                                "AY89a18NmP/rpOzYqPT2gXtTGPJwCXopurRCh2SERSxHAbUcM+kBs62qrKZf6X3eHEdd/leFa2/wBviqJ6WkkEdIsgz94SdkfwWm"
                                "53Pd6IC20z+9KTq6PpsXs5CJQzMRyWvSBBolC4JHE99knIqr4nJbUlXHYCknKnExuUNUuWzkcY2zsO/uX81og8IwGDikReqErvvU"
                                "+EVyAV0TSDb3Pg0A8eC7RCHBLPfT0ljwzNJDibRUkwoxdKcPkMvdbYhZEAnkOPD7N/CpITooxxwBYVArsbDW1oA4kL78A2gnbsNi"
                                "Y2esZSaKgxrftJk1H06rW5b7tAk7y4wMjbj7GD9XQJJ3HzLdUznXNvLRZjkNcr8evFgKPq3YIPqpuCn5klH09/qjGNZRkKoYriBS"
                                "f95iWKvGZ7jwEHu+45Qw46o0lID1TFZsrnS4WiwllgqAhvvFZ//DbCtpjMIztpTGYD754oc60k50p6Ny14oPDBlNMOVqwiFGq9QC"
                                "FQ7Vzilx+rqAoVdOzvyGSf6+SfXOT0Jv/raw+HIIA2bhOqurVqgdS4CoxdpCcZ5CmOgN4CIA9rQcuxZ/dVaT2w/UL0fUJOAN3kSU"
                                "ngcI+FcqIjlbF+IbIEGphpBhsIaTd3qXVCH/57K0tP7lVu9GoAU9zNu/6YPIWZnE9ABUBfQkyTR/DdFZm3Hfsjt7m5YkPlm+kJNH"
                                "iHHwdjTo+liy9gBFnQbl+mFgv3e0bcdy+a0pa6mUyluy0IWrKFBVFcfEep4vD7+CRTpkpwhwzL6+v2t3LWGcJGZV6GS/qbbGR3ys"
                                "Cyr70DrZtaEnbOnLN3UYP6zDcDfKYTn7ub3hIjaME9LNjg=="
                            ),
                        ],
                    },
                },
                "provider_specific_fields": {"native_finish_reason": "STOP"},
            },
        ],
        "usage": {
            "completion_tokens": 215,
            "prompt_tokens": 246,
            "total_tokens": 461,
            "completion_tokens_details": {"reasoning_tokens": 154, "text_tokens": 61},
            "prompt_tokens_details": {"text_tokens": 246},
        },
        "vertex_ai_grounding_metadata": [],
        "vertex_ai_url_context_metadata": [],
        "vertex_ai_safety_results": [],
        "vertex_ai_citation_metadata": [],
    },
)


CLAUDE_FABLE_5_TEST_CASE: Final = TranslationTestCase(
    scenario="agent_tools",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CODING_AGENT_REQUEST, "model": "vertex_ai/claude-fable-5@default"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-fable-5@default:rawPredict",
    expected_provider_headers={
        "content-type": "application/json",
        "x-api-key": "scripted-token",
        "anthropic-version": "2023-06-01",
        "authorization": "Bearer scripted-token",
    },
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
        "anthropic_version": "vertex-2023-10-16",
    },
    mock_provider_response={
        **ANTHROPIC_AGENT_RESPONSE,
        "model": "claude-fable-5",
        "id": "msg_vrtx_011Cg3Ks7bxVSjgjZZkcaRY9",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01HkGRMBmWWCqTp5Ck2nZnJp",
                "name": "search_code",
                "input": {"query": "retries_remaining"},
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01ASaiWVEwF3LUJU8KDzBdrr",
                "name": "run_tests",
                "input": {"path": "tests/test_retry.py"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1184,
            "output_tokens": 105,
            "output_tokens_details": {"thinking_tokens": 0},
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "vertex_ai/claude-fable-5@default",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"query": "retries_remaining"}', "name": "search_code"},
                            "id": "toolu_vrtx_01HkGRMBmWWCqTp5Ck2nZnJp",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "toolu_vrtx_01ASaiWVEwF3LUJU8KDzBdrr",
                            "type": "function",
                        },
                    ],
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                    "content": None,
                },
            }
        ],
        "usage": {
            "completion_tokens": 105,
            "prompt_tokens": 1184,
            "total_tokens": 1289,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 105},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1184,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)


CLAUDE_OPUS_5_TEST_CASE: Final = TranslationTestCase(
    scenario="agent_tools",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CODING_AGENT_REQUEST, "model": "vertex_ai/claude-opus-5@default"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-opus-5@default:rawPredict",
    expected_provider_headers={
        "content-type": "application/json",
        "x-api-key": "scripted-token",
        "anthropic-version": "2023-06-01",
        "authorization": "Bearer scripted-token",
    },
    expected_provider_request={
        **ANTHROPIC_AGENT_REQUEST,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "low"},
        "anthropic_version": "vertex-2023-10-16",
    },
    mock_provider_response={
        **ANTHROPIC_AGENT_RESPONSE,
        "model": "claude-opus-5",
        "id": "msg_vrtx_011Cg3KqBd15Sob6y53PCoBe",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_vrtx_014y2somv8RVnNoX3s7i7RwH",
                "name": "search_code",
                "input": {"query": "retries_remaining"},
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01B8D9djnyoPnNQSccNW99Hi",
                "name": "run_tests",
                "input": {"path": "tests/test_retry.py"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {
            **object_value(ANTHROPIC_AGENT_RESPONSE["usage"]),
            "input_tokens": 1180,
            "output_tokens": 105,
            "output_tokens_details": {"thinking_tokens": 0},
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "model": "vertex_ai/claude-opus-5@default",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"query": "retries_remaining"}', "name": "search_code"},
                            "id": "toolu_vrtx_014y2somv8RVnNoX3s7i7RwH",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path": "tests/test_retry.py"}', "name": "run_tests"},
                            "id": "toolu_vrtx_01B8D9djnyoPnNQSccNW99Hi",
                            "type": "function",
                        },
                    ],
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                    "content": None,
                },
            }
        ],
        "usage": {
            "completion_tokens": 105,
            "prompt_tokens": 1180,
            "total_tokens": 1285,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 105},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 1180,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)


GEMINI_4_ARGON_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "vertex_ai/gemini-4-argon"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/google/models/gemini-4-argon:generateContent",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer scripted-token"},
    expected_provider_request=VERTEX_AI_GEMINI_GENERATE_CONTENT_REQUEST,
    mock_provider_response={
        **VERTEX_AI_GEMINI_GENERATE_CONTENT_RESPONSE,
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": (
                                "**My Thought Process on Reviewing Shop Logic**\n\nAlright, my first order of business is to get a "
                                "firm grasp on how the shop's validation logic handles edge cases, specifically concerning "
                                "quantities. The prompt explicitly mentions zero, negative, and non-integer quantities, and I know "
                                "from my experience that these are often the sources of bugs. To understand this thoroughly, I need "
                                "to dive into the `shop/validation.py` file. This file is the most likely place where these crucial "
                                "validation rules are defined. Once I've reviewed that, I can proceed with a clearer picture of how "
                                "the system is designed to handle these potentially problematic inputs. Therefore, my next action is "
                                "to request a read of `shop/validation.py` so I can examine its contents.\n\n"
                            ),
                            "thought": True,
                        },
                        {
                            "text": (
                                "I need to read `shop/validation.py` next to see how zero, negative, and non-integer quantities are "
                                "currently validated."
                            ),
                        },
                        {
                            "functionCall": {
                                "name": "read_file",
                                "args": {"end_line": 50, "start_line": 1, "path": "shop/validation.py"},
                                "id": "call_290084",
                            },
                            "thoughtSignature": GEMINI_4_ARGON_CHAT_COMPLETIONS_THOUGHT_SIGNATURE_0,
                        },
                    ],
                },
                "finishReason": "STOP",
            },
        ],
        "usageMetadata": {
            "promptTokenCount": 414,
            "candidatesTokenCount": 67,
            "totalTokenCount": 624,
            "trafficType": "ON_DEMAND",
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 414}],
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 67}],
            "thoughtsTokenCount": 143,
        },
        "modelVersion": "gemini-4-argon",
        "createTime": "2026-10-15T01:13:27.878297Z",
        "responseId": "tyjQatnNNZyQ4_UPpcr16QM",
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "vertex_ai/gemini-4-argon",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": (
                        "I need to read `shop/validation.py` next to see how zero, negative, and non-integer quantities are "
                        "currently validated."
                    ),
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "provider_specific_fields": {
                                "thought_signature": (
                                    "AY89a19UcUynGbcdZOu5ucJyuVTEBkP7BWc+WQRD+zdel8qY2zqNh1OLS6DejBKWK9xndkK8coxMqPRuuQytzyB+zVvJz0ouSNh0"
                                    "nH3Bm0p+S9hFJDzNhlHrk4PRtU0OWxv5SKWtzQURmeBDk5mJV/MU9yJSDs7g+csm6juZi2Lang76vPWyQOHTVPYnJVzLHkP73H0D"
                                    "WnVPXSdNzBgqPzopqf3GoP+EvALphWyi0Zp/1uv7m4TKbx9dRZjlO15LaqpksnkDvwkF6o8KfaYrC/Skh+xYxlk40yxiiaYv5sr4"
                                    "PbGutzCGNWCZxjbBCbQBPGMFIL3fDGwLavbGd71tWKcOaSBr3oYbsw5HsfWMFHV8+TQ7rou77MCMWaiJ3mulflSmdBkkTq0m1bPm"
                                    "iIbwYvVPs6CCtTTZgGYTZhRefefwD1Dx4Y5epmGy1+jST7zikArzdZeexaCTTrIWO4rvqzPdAS65XHnfXjgYEPYK0YUnX6StK/ln"
                                    "Lrwdb+AK6j+Q2eLuKpx/HJE0QDZ9BgXpw/CQ6my0s2rI3Tjlbun8Bdl7MyFuCbstbTA+uu1RXXv2M1RThJQHiuDGvgzCNg/sgyCb"
                                    "zluzK/vp2nzki+ESsFeForBBSSCLVAep2tiqKhIOWPWBHYjH04qfgyWkylbRbZeA32+BPYViEcZa+Imi+aYXu241RDOSWRMJvbsJ"
                                    "oDY15ThjQ+sgLNKJkBoravc7+sQ1Abt+jwCcO9zHBquTWn8G/X0IggqHNuDnbWEW3S9HdxPXw7ZV0ZYXXkZXIYwFIt6Ji7LpXlEY"
                                    "X0hOwrBdq2ufKAv+EWBQyCU3oyBrIGeVdVx25BiLFsVJzqw3xLTUgfelHyawgwtGvd7J+mQBPp4YH1Rk6fbQMwDEWipSiltG+jG9"
                                    "ZqXUiF1NtN30tw4a4h2tA03uBV1yXhv6+fL7g2zmN6c4GVX9IzAS60UUnXIMlBgCXjxCU591TuPufe6mJQ=="
                                ),
                            },
                            "function": {
                                "arguments": '{"end_line": 50, "start_line": 1, "path": "shop/validation.py"}',
                                "name": "read_file",
                            },
                            "id": (
                                "call_290084__thought__AY89a19UcUynGbcdZOu5ucJyuVTEBkP7BWc+WQRD+zdel8qY2zqNh1OLS6DejBKWK9xndkK8coxMqP"
                                "RuuQytzyB+zVvJz0ouSNh0nH3Bm0p+S9hFJDzNhlHrk4PRtU0OWxv5SKWtzQURmeBDk5mJV/MU9yJSDs7g+csm6juZi2Lang76vP"
                                "WyQOHTVPYnJVzLHkP73H0DWnVPXSdNzBgqPzopqf3GoP+EvALphWyi0Zp/1uv7m4TKbx9dRZjlO15LaqpksnkDvwkF6o8KfaYrC/"
                                "Skh+xYxlk40yxiiaYv5sr4PbGutzCGNWCZxjbBCbQBPGMFIL3fDGwLavbGd71tWKcOaSBr3oYbsw5HsfWMFHV8+TQ7rou77MCMWa"
                                "iJ3mulflSmdBkkTq0m1bPmiIbwYvVPs6CCtTTZgGYTZhRefefwD1Dx4Y5epmGy1+jST7zikArzdZeexaCTTrIWO4rvqzPdAS65XH"
                                "nfXjgYEPYK0YUnX6StK/lnLrwdb+AK6j+Q2eLuKpx/HJE0QDZ9BgXpw/CQ6my0s2rI3Tjlbun8Bdl7MyFuCbstbTA+uu1RXXv2M1"
                                "RThJQHiuDGvgzCNg/sgyCbzluzK/vp2nzki+ESsFeForBBSSCLVAep2tiqKhIOWPWBHYjH04qfgyWkylbRbZeA32+BPYViEcZa+I"
                                "mi+aYXu241RDOSWRMJvbsJoDY15ThjQ+sgLNKJkBoravc7+sQ1Abt+jwCcO9zHBquTWn8G/X0IggqHNuDnbWEW3S9HdxPXw7ZV0Z"
                                "YXXkZXIYwFIt6Ji7LpXlEYX0hOwrBdq2ufKAv+EWBQyCU3oyBrIGeVdVx25BiLFsVJzqw3xLTUgfelHyawgwtGvd7J+mQBPp4YH1"
                                "Rk6fbQMwDEWipSiltG+jG9ZqXUiF1NtN30tw4a4h2tA03uBV1yXhv6+fL7g2zmN6c4GVX9IzAS60UUnXIMlBgCXjxCU591TuPufe"
                                "6mJQ=="
                            ),
                            "type": "function",
                        },
                    ],
                    "images": [],
                    "reasoning_content": (
                        "**My Thought Process on Reviewing Shop Logic**\n\nAlright, my first order of business is to get a "
                        "firm grasp on how the shop's validation logic handles edge cases, specifically concerning "
                        "quantities. The prompt explicitly mentions zero, negative, and non-integer quantities, and I know "
                        "from my experience that these are often the sources of bugs. To understand this thoroughly, I need "
                        "to dive into the `shop/validation.py` file. This file is the most likely place where these crucial "
                        "validation rules are defined. Once I've reviewed that, I can proceed with a clearer picture of how "
                        "the system is designed to handle these potentially problematic inputs. Therefore, my next action is "
                        "to request a read of `shop/validation.py` so I can examine its contents.\n\n"
                    ),
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": (
                                "**My Thought Process on Reviewing Shop Logic**\n\nAlright, my first order of business is to get a "
                                "firm grasp on how the shop's validation logic handles edge cases, specifically concerning "
                                "quantities. The prompt explicitly mentions zero, negative, and non-integer quantities, and I know "
                                "from my experience that these are often the sources of bugs. To understand this thoroughly, I need "
                                "to dive into the `shop/validation.py` file. This file is the most likely place where these crucial "
                                "validation rules are defined. Once I've reviewed that, I can proceed with a clearer picture of how "
                                "the system is designed to handle these potentially problematic inputs. Therefore, my next action is "
                                "to request a read of `shop/validation.py` so I can examine its contents.\n\n"
                            ),
                        },
                    ],
                    "provider_specific_fields": {
                        "thought_signatures": [
                            (
                                "AY89a19UcUynGbcdZOu5ucJyuVTEBkP7BWc+WQRD+zdel8qY2zqNh1OLS6DejBKWK9xndkK8coxMqPRuuQytzyB+zVvJz0ouSNh0"
                                "nH3Bm0p+S9hFJDzNhlHrk4PRtU0OWxv5SKWtzQURmeBDk5mJV/MU9yJSDs7g+csm6juZi2Lang76vPWyQOHTVPYnJVzLHkP73H0D"
                                "WnVPXSdNzBgqPzopqf3GoP+EvALphWyi0Zp/1uv7m4TKbx9dRZjlO15LaqpksnkDvwkF6o8KfaYrC/Skh+xYxlk40yxiiaYv5sr4"
                                "PbGutzCGNWCZxjbBCbQBPGMFIL3fDGwLavbGd71tWKcOaSBr3oYbsw5HsfWMFHV8+TQ7rou77MCMWaiJ3mulflSmdBkkTq0m1bPm"
                                "iIbwYvVPs6CCtTTZgGYTZhRefefwD1Dx4Y5epmGy1+jST7zikArzdZeexaCTTrIWO4rvqzPdAS65XHnfXjgYEPYK0YUnX6StK/ln"
                                "Lrwdb+AK6j+Q2eLuKpx/HJE0QDZ9BgXpw/CQ6my0s2rI3Tjlbun8Bdl7MyFuCbstbTA+uu1RXXv2M1RThJQHiuDGvgzCNg/sgyCb"
                                "zluzK/vp2nzki+ESsFeForBBSSCLVAep2tiqKhIOWPWBHYjH04qfgyWkylbRbZeA32+BPYViEcZa+Imi+aYXu241RDOSWRMJvbsJ"
                                "oDY15ThjQ+sgLNKJkBoravc7+sQ1Abt+jwCcO9zHBquTWn8G/X0IggqHNuDnbWEW3S9HdxPXw7ZV0ZYXXkZXIYwFIt6Ji7LpXlEY"
                                "X0hOwrBdq2ufKAv+EWBQyCU3oyBrIGeVdVx25BiLFsVJzqw3xLTUgfelHyawgwtGvd7J+mQBPp4YH1Rk6fbQMwDEWipSiltG+jG9"
                                "ZqXUiF1NtN30tw4a4h2tA03uBV1yXhv6+fL7g2zmN6c4GVX9IzAS60UUnXIMlBgCXjxCU591TuPufe6mJQ=="
                            ),
                        ],
                    },
                },
                "provider_specific_fields": {"native_finish_reason": "STOP"},
            },
        ],
        "usage": {
            "completion_tokens": 210,
            "prompt_tokens": 414,
            "total_tokens": 624,
            "completion_tokens_details": {"reasoning_tokens": 143, "text_tokens": 67},
            "prompt_tokens_details": {"text_tokens": 414},
        },
        "vertex_ai_grounding_metadata": [],
        "vertex_ai_url_context_metadata": [],
        "vertex_ai_safety_results": [],
        "vertex_ai_citation_metadata": [],
    },
)
