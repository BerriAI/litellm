from pathlib import Path
from typing import Final

from integration._support.client import JSON_OBJECT
from integration.translation.case import TranslationTestCase

CLAUDE_SONNET_4_RESPONSE: Final = JSON_OBJECT.validate_json(
    Path(__file__).with_name("openrouter_claude_reasoning_response.json").read_bytes()
)
CLAUDE_SONNET_4_CHOICE: Final = JSON_OBJECT.validate_python(CLAUDE_SONNET_4_RESPONSE["choices"][0])
CLAUDE_SONNET_4_MESSAGE: Final = JSON_OBJECT.validate_python(CLAUDE_SONNET_4_CHOICE["message"])
CLAUDE_SONNET_4_USAGE: Final = JSON_OBJECT.validate_python(CLAUDE_SONNET_4_RESPONSE["usage"])

GEMINI_3_1_FLASH_IMAGE_RESPONSE: Final = JSON_OBJECT.validate_json(
    Path(__file__).with_name("openrouter_gemini_image_response.json").read_bytes()
)
GEMINI_3_1_FLASH_IMAGE_CHOICE: Final = JSON_OBJECT.validate_python(GEMINI_3_1_FLASH_IMAGE_RESPONSE["choices"][0])
GEMINI_3_1_FLASH_IMAGE_MESSAGE: Final = JSON_OBJECT.validate_python(GEMINI_3_1_FLASH_IMAGE_CHOICE["message"])
GEMINI_3_1_FLASH_IMAGE_USAGE: Final = JSON_OBJECT.validate_python(GEMINI_3_1_FLASH_IMAGE_RESPONSE["usage"])

CLAUDE_SONNET_4_TEST_CASE: Final = TranslationTestCase(
    scenario="reasoning_content",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "openrouter/anthropic/claude-sonnet-4",
        "messages": [{"role": "user", "content": "Hello world"}],
        "reasoning": {"effort": "high"},
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/api/v1/chat/completions",
    expected_provider_headers={
        "authorization": "Bearer synthetic-openrouter-key",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "anthropic/claude-sonnet-4",
        "messages": [{"role": "user", "content": "Hello world"}],
        "reasoning": {"effort": "high"},
        "usage": {"include": True},
    },
    mock_provider_response=CLAUDE_SONNET_4_RESPONSE,
    expected_litellm_response={
        "id": CLAUDE_SONNET_4_RESPONSE["id"],
        "object": "chat.completion",
        "created": CLAUDE_SONNET_4_RESPONSE["created"],
        "model": "openrouter/anthropic/claude-sonnet-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": CLAUDE_SONNET_4_MESSAGE["content"],
                    "reasoning_content": CLAUDE_SONNET_4_MESSAGE["reasoning"],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": CLAUDE_SONNET_4_USAGE["prompt_tokens"],
            "completion_tokens": CLAUDE_SONNET_4_USAGE["completion_tokens"],
            "total_tokens": CLAUDE_SONNET_4_USAGE["total_tokens"],
            "completion_tokens_details": {
                "reasoning_tokens": JSON_OBJECT.validate_python(CLAUDE_SONNET_4_USAGE["completion_tokens_details"])[
                    "reasoning_tokens"
                ]
            },
            "prompt_tokens_details": {
                "cached_tokens": JSON_OBJECT.validate_python(CLAUDE_SONNET_4_USAGE["prompt_tokens_details"])[
                    "cached_tokens"
                ]
            },
        },
    },
)

GEMINI_3_1_FLASH_IMAGE_TEST_CASE: Final = TranslationTestCase(
    scenario="image_generation",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "openrouter/google/gemini-3.1-flash-image",
        "messages": [
            {
                "role": "user",
                "content": "Generate a perfectly flat solid red square with no details, texture, gradients, shadows, or text.",
            }
        ],
        "modalities": ["image", "text"],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/api/v1/chat/completions",
    expected_provider_headers={
        "authorization": "Bearer synthetic-openrouter-key",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "google/gemini-3.1-flash-image",
        "messages": [
            {
                "role": "user",
                "content": "Generate a perfectly flat solid red square with no details, texture, gradients, shadows, or text.",
            }
        ],
        "modalities": ["image", "text"],
        "usage": {"include": True},
    },
    mock_provider_response=GEMINI_3_1_FLASH_IMAGE_RESPONSE,
    expected_litellm_response={
        "id": GEMINI_3_1_FLASH_IMAGE_RESPONSE["id"],
        "object": "chat.completion",
        "created": GEMINI_3_1_FLASH_IMAGE_RESPONSE["created"],
        "model": "openrouter/google/gemini-3.1-flash-image",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "images": GEMINI_3_1_FLASH_IMAGE_MESSAGE["images"],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": GEMINI_3_1_FLASH_IMAGE_USAGE["prompt_tokens"],
            "completion_tokens": GEMINI_3_1_FLASH_IMAGE_USAGE["completion_tokens"],
            "total_tokens": GEMINI_3_1_FLASH_IMAGE_USAGE["total_tokens"],
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    },
)
