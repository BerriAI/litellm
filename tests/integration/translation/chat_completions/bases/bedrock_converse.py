from typing import Final
from unittest.mock import ANY

from integration.translation.case import TranslationTestCase

CLAUDE_HAIKU_4_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
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
        "created": ANY,
        "model": "bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "object": "chat.completion",
        "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "Hello.", "role": "assistant"}}],
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
        "model": "bedrock/converse/us.anthropic.claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
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
        "created": ANY,
        "model": "bedrock/converse/us.anthropic.claude-sonnet-4-6",
        "object": "chat.completion",
        "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "Hello!", "role": "assistant"}}],
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
        "model": "bedrock/converse/us.anthropic.claude-sonnet-5",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
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
        "created": ANY,
        "model": "bedrock/converse/us.anthropic.claude-sonnet-5",
        "object": "chat.completion",
        "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "Hello.", "role": "assistant"}}],
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
        "model": "bedrock/converse/us.anthropic.claude-opus-4-8",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
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
        "created": ANY,
        "model": "bedrock/converse/us.anthropic.claude-opus-4-8",
        "object": "chat.completion",
        "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "Hello.", "role": "assistant"}}],
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
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)

CLAUDE_SONNET_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "bedrock/converse/us.anthropic.claude-sonnet-5-5",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-sonnet-5-5/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
        "inferenceConfig": {"maxTokens": 64},
        "system": [{"text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "metrics": {"latencyMs": 892},
        "output": {"message": {"content": [{"text": "Hello."}], "role": "assistant"}},
        "stopReason": "end_turn",
        "usage": {
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 0,
            "cacheWriteInputTokens": 0,
            "inputTokens": 23,
            "outputTokens": 6,
            "serverToolUsage": {},
            "totalTokens": 29,
        },
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "bedrock/converse/us.anthropic.claude-sonnet-5-5",
        "object": "chat.completion",
        "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "Hello.", "role": "assistant"}}],
        "usage": {
            "completion_tokens": 6,
            "prompt_tokens": 23,
            "total_tokens": 29,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 6},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 23,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)

CLAUDE_OPUS_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "bedrock/converse/us.anthropic.claude-opus-5-5",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/model/us.anthropic.claude-opus-5-5/converse",
    expected_provider_headers={"authorization": "Bearer synthetic-bedrock-key", "content-type": "application/json"},
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
        "inferenceConfig": {"maxTokens": 64},
        "system": [{"text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "metrics": {"latencyMs": 10460},
        "output": {"message": {"content": [{"text": "Hello."}], "role": "assistant"}},
        "stopReason": "end_turn",
        "usage": {
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 0,
            "cacheWriteInputTokens": 0,
            "inputTokens": 23,
            "outputTokens": 6,
            "serverToolUsage": {},
            "totalTokens": 29,
        },
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "bedrock/converse/us.anthropic.claude-opus-5-5",
        "object": "chat.completion",
        "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "Hello.", "role": "assistant"}}],
        "usage": {
            "completion_tokens": 6,
            "prompt_tokens": 23,
            "total_tokens": 29,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 6},
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "text_tokens": 23,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)
