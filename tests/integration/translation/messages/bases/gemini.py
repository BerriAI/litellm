from typing import Final

from integration.translation.case import TranslationTestCase

GEMINI_3_5_FLASH_THOUGHT_SIGNATURE: Final = "EtoCCtcCAWkUfRMHqpSzR/FxDYsPJ3NrP1y/mmdCBSDswOdgEoQtZiCJw6qGdrpDDog3bPQS5h72gTN8xFOZ22i9+54pMd+ni6os31/fl2RG7p1s0OtG1D6lnsA9VG6RuaDk5jZHsMQRGBescw3oKaz7YMpGYpe/o4ZemK1zVQ4j7xcGm1MslA4kd7uBDtXWkuGK8S71j8/8DfoBXCZxGTnOBC279dUVKs6IwqEDSW7NwMHIX3koqbhH9VacvtbPuE6/gre0568cSb7RVaMTLC5Gvrpbf1ikskjfmU611G9Ap1H37zU8LBET1+NXb/tBBzbyEYyCoN6PlQn7y20OrtQKulGaxHXk6GBJPKt9D+suW8DQre/X9qKX8DrxmBs8m9UtCdRa+7qGGE4k9c00UujLSxMqper/jshxSSYVPvCubk6vP7BTrqxeD/cJizst1reEKveOFxfozpXptw=="

GEMINI_3_5_FLASH_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "gemini/gemini-3.5-flash",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
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
        "id": "ufDDaqOmFKbVz7IP3oauwAs",
        "type": "message",
        "role": "assistant",
        "model": "gemini/gemini-3.5-flash",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 59},
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)

GEMINI_3_8_FLASH_THOUGHT_SIGNATURE: Final = "ErYCCrMCAWkUfRN74uA0tYzcgDRg2KSNtSWJy7TymH5/2gchou/ba9aek5JSdNv8pm61HLRFDIM1O1EtldG5hWEET+ngNsd9KqwLbWOzKyT7elyGQ+dznt8qA26HP82TCdVFFyknWVQaQLEnhv7ATTTjJTJoydSd60Nm3PjNELQr+MP1pZ6O9VjUrmJe8b/SaEe5MJy7mzqtVeXqFjiVpaf/Xr4VNWYKupd8ycrbXMZbjhclNwAINZXdfrJfA0aU1Xsg7+pQ9OeV1gg3HdzLTYd6TD+dP+uyQc6yp2IJ9Hd/1EZMOZiC77O6IIUQYff7AUtq6RPAyVMuMcGdp6lvMRVS1545dbXPSsQDN5dJVmMY36jEmouydsfuem39EvvCnmbqV/N/4za3IzTggtZ3hgoN7u3RaF2VAw=="

GEMINI_3_8_FLASH_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "gemini/gemini-3.8-flash",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
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
        "id": "xfDDasyUFMjSjMcPgvOm6Ag",
        "type": "message",
        "role": "assistant",
        "model": "gemini/gemini-3.8-flash",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 60},
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "max_tokens",
        "stop_details": None,
    },
)

GEMINI_3_1_PRO_PREVIEW_THOUGHT_SIGNATURE: Final = "Er0CCroCAWkUfRMMekmx87raUurCnsdbgsVPmIhNFn3Hq6GgkQe+Iq+i1Xj4ST0AU7AawXJvPCqaK0lRTyo1ilHyHJLoUS7a7iOScwHRPoa3L8gnjqJNA+H370nCZq4rOOqa7tIjWo+rsCIy1FUHZKnKjsHvK6nJ9h4mlX8YulO4mK48ox/ZsULbiAZq0uZIssxWAkfzof9Gypmzk/7JIEp9rK3uqWIxTG4BIDKUphNKeti0S68CFrDvm9UEomk+3uCxqvvZmLp9L1OvTcOWb697Yub7pqr7063tzVhLbZilwQovwcJ7vJ+jPBOj2dp5cWBROBtOKKhLKy4I38/9zp03oxg0puNuLUcNtIiNeYkkB8AA9XPQpZ/Xldrbj/cSgVn8A2f55ihglV6Y1yLrM5KVzIVt0nwt5jpw9+qt2pg="

GEMINI_3_1_PRO_PREVIEW_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "gemini/gemini-3.1-pro-preview",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
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
        "id": "zPDDar31L57VjMcPpcO-0Ag",
        "type": "message",
        "role": "assistant",
        "model": "gemini/gemini-3.1-pro-preview",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 59},
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)
