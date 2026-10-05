from typing import Final
from unittest.mock import ANY

from integration.translation.case import TranslationTestCase

GEMINI_3_5_FLASH_THOUGHT_SIGNATURE: Final = "EtoCCtcCAWkUfRMHqpSzR/FxDYsPJ3NrP1y/mmdCBSDswOdgEoQtZiCJw6qGdrpDDog3bPQS5h72gTN8xFOZ22i9+54pMd+ni6os31/fl2RG7p1s0OtG1D6lnsA9VG6RuaDk5jZHsMQRGBescw3oKaz7YMpGYpe/o4ZemK1zVQ4j7xcGm1MslA4kd7uBDtXWkuGK8S71j8/8DfoBXCZxGTnOBC279dUVKs6IwqEDSW7NwMHIX3koqbhH9VacvtbPuE6/gre0568cSb7RVaMTLC5Gvrpbf1ikskjfmU611G9Ap1H37zU8LBET1+NXb/tBBzbyEYyCoN6PlQn7y20OrtQKulGaxHXk6GBJPKt9D+suW8DQre/X9qKX8DrxmBs8m9UtCdRa+7qGGE4k9c00UujLSxMqper/jshxSSYVPvCubk6vP7BTrqxeD/cJizst1reEKveOFxfozpXptw=="

GEMINI_3_5_FLASH_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "gemini/gemini-3.5-flash",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/models/gemini-3.5-flash:generateContent",
    expected_provider_headers={"content-type": "application/json", "x-goog-api-key": "synthetic-gemini-key"},
    expected_provider_request={
        "contents": [{"parts": [{"text": "Say hello."}], "role": "user"}],
        "generationConfig": {"max_output_tokens": 64, "temperature": 1.0},
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
            "totalTokenCount": 69,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 10}],
            "thoughtsTokenCount": 57,
            "serviceTier": "standard",
        },
        "modelVersion": "gemini-3.5-flash",
        "responseId": "ufDDaqOmFKbVz7IP3oauwAs",
    },
    expected_litellm_response={
        "id": ANY,
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "gemini/gemini-3.5-flash",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello.", "annotations": []}],
                "phase": None,
            }
        ],
        "parallel_tool_calls": False,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "max_output_tokens": 64,
        "previous_response_id": None,
        "reasoning": None,
        "status": "completed",
        "text": {},
        "truncation": None,
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 10,
                "video_tokens": None,
            },
            "output_tokens": 59,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 57, "text_tokens": 2},
            "total_tokens": 69,
            "cost": None,
        },
        "user": None,
        "store": None,
    },
)

GEMINI_3_8_FLASH_THOUGHT_SIGNATURE: Final = "ErYCCrMCAWkUfRN74uA0tYzcgDRg2KSNtSWJy7TymH5/2gchou/ba9aek5JSdNv8pm61HLRFDIM1O1EtldG5hWEET+ngNsd9KqwLbWOzKyT7elyGQ+dznt8qA26HP82TCdVFFyknWVQaQLEnhv7ATTTjJTJoydSd60Nm3PjNELQr+MP1pZ6O9VjUrmJe8b/SaEe5MJy7mzqtVeXqFjiVpaf/Xr4VNWYKupd8ycrbXMZbjhclNwAINZXdfrJfA0aU1Xsg7+pQ9OeV1gg3HdzLTYd6TD+dP+uyQc6yp2IJ9Hd/1EZMOZiC77O6IIUQYff7AUtq6RPAyVMuMcGdp6lvMRVS1545dbXPSsQDN5dJVmMY36jEmouydsfuem39EvvCnmbqV/N/4za3IzTggtZ3hgoN7u3RaF2VAw=="

GEMINI_3_8_FLASH_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "gemini/gemini-3.8-flash",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/models/gemini-3.8-flash:generateContent",
    expected_provider_headers={"content-type": "application/json", "x-goog-api-key": "synthetic-gemini-key"},
    expected_provider_request={
        "contents": [{"parts": [{"text": "Say hello."}], "role": "user"}],
        "generationConfig": {"max_output_tokens": 64, "temperature": 1.0},
        "system_instruction": {"parts": [{"text": "You are a terse assistant."}]},
    },
    mock_provider_response={
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello.", "thoughtSignature": GEMINI_3_8_FLASH_THOUGHT_SIGNATURE}],
                    "role": "model",
                },
                "finishReason": "MAX_TOKENS",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 2,
            "totalTokenCount": 70,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 10}],
            "thoughtsTokenCount": 58,
            "serviceTier": "standard",
        },
        "modelVersion": "gemini-3.8-flash",
        "responseId": "xfDDasyUFMjSjMcPgvOm6Ag",
    },
    expected_litellm_response={
        "id": ANY,
        "created_at": ANY,
        "error": None,
        "incomplete_details": {"reason": "max_output_tokens"},
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "gemini/gemini-3.8-flash",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": ANY,
                "status": "incomplete",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello.", "annotations": []}],
                "phase": None,
            }
        ],
        "parallel_tool_calls": False,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "max_output_tokens": 64,
        "previous_response_id": None,
        "reasoning": None,
        "status": "incomplete",
        "text": {},
        "truncation": None,
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 10,
                "video_tokens": None,
            },
            "output_tokens": 60,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 58, "text_tokens": 2},
            "total_tokens": 70,
            "cost": None,
        },
        "user": None,
        "store": None,
    },
)

GEMINI_3_1_PRO_PREVIEW_THOUGHT_SIGNATURE: Final = "Er0CCroCAWkUfRMMekmx87raUurCnsdbgsVPmIhNFn3Hq6GgkQe+Iq+i1Xj4ST0AU7AawXJvPCqaK0lRTyo1ilHyHJLoUS7a7iOScwHRPoa3L8gnjqJNA+H370nCZq4rOOqa7tIjWo+rsCIy1FUHZKnKjsHvK6nJ9h4mlX8YulO4mK48ox/ZsULbiAZq0uZIssxWAkfzof9Gypmzk/7JIEp9rK3uqWIxTG4BIDKUphNKeti0S68CFrDvm9UEomk+3uCxqvvZmLp9L1OvTcOWb697Yub7pqr7063tzVhLbZilwQovwcJ7vJ+jPBOj2dp5cWBROBtOKKhLKy4I38/9zp03oxg0puNuLUcNtIiNeYkkB8AA9XPQpZ/Xldrbj/cSgVn8A2f55ihglV6Y1yLrM5KVzIVt0nwt5jpw9+qt2pg="

GEMINI_3_1_PRO_PREVIEW_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "gemini/gemini-3.1-pro-preview",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/models/gemini-3.1-pro-preview:generateContent",
    expected_provider_headers={"content-type": "application/json", "x-goog-api-key": "synthetic-gemini-key"},
    expected_provider_request={
        "contents": [{"parts": [{"text": "Say hello."}], "role": "user"}],
        "generationConfig": {"max_output_tokens": 64, "temperature": 1.0},
        "system_instruction": {"parts": [{"text": "You are a terse assistant."}]},
    },
    mock_provider_response={
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello.", "thoughtSignature": GEMINI_3_1_PRO_PREVIEW_THOUGHT_SIGNATURE}],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 2,
            "totalTokenCount": 69,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 10}],
            "thoughtsTokenCount": 57,
            "serviceTier": "standard",
        },
        "modelVersion": "gemini-3.1-pro-preview",
        "responseId": "zPDDar31L57VjMcPpcO-0Ag",
    },
    expected_litellm_response={
        "id": ANY,
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "gemini/gemini-3.1-pro-preview",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello.", "annotations": []}],
                "phase": None,
            }
        ],
        "parallel_tool_calls": False,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "max_output_tokens": 64,
        "previous_response_id": None,
        "reasoning": None,
        "status": "completed",
        "text": {},
        "truncation": None,
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 10,
                "video_tokens": None,
            },
            "output_tokens": 59,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 57, "text_tokens": 2},
            "total_tokens": 69,
            "cost": None,
        },
        "user": None,
        "store": None,
    },
)
