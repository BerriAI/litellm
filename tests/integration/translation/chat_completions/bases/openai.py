from typing import Final

from integration.translation.case import TranslationTestCase


GPT_5_4_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "openai/gpt-5.4",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/chat/completions",
    expected_provider_headers={
        "content-type": "application/json",
        "authorization": "Bearer synthetic-openai-key",
    },
    expected_provider_request={
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "model": "gpt-5.4",
        "max_completion_tokens": 64,
    },
    mock_provider_response={
        "id": "chatcmpl-EVhxnsh4umJNE0UBHCO0G3rCQjLO9",
        "object": "chat.completion",
        "created": 1791225871,
        "model": "gpt-5.4-2026-03-05",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello.", "refusal": None, "annotations": []},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 19,
            "completion_tokens": 5,
            "total_tokens": 24,
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        },
        "service_tier": "default",
        "system_fingerprint": None,
    },
    expected_litellm_response={
        "id": "chatcmpl-EVhxnsh4umJNE0UBHCO0G3rCQjLO9",
        "created": 1791225871,
        "model": "openai/gpt-5.4",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello.",
                    "role": "assistant",
                    "refusal": None,
                    "annotations": [],
                },
                "provider_specific_fields": {},
            }
        ],
        "usage": {
            "completion_tokens": 5,
            "prompt_tokens": 19,
            "total_tokens": 24,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
        },
        "service_tier": "default",
        "system_fingerprint": None,
    },
)


GPT_5_6_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "openai/gpt-5.6-sol",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/chat/completions",
    expected_provider_headers={
        "content-type": "application/json",
        "authorization": "Bearer synthetic-openai-key",
    },
    expected_provider_request={
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "model": "gpt-5.6-sol",
        "max_completion_tokens": 64,
    },
    mock_provider_response={
        "id": "chatcmpl-EVhxr75iujjmwE0F2bgGMQHDX3Bq6",
        "object": "chat.completion",
        "created": 1791225875,
        "model": "gpt-5.6-sol",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!", "refusal": None, "annotations": []},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 19,
            "completion_tokens": 5,
            "total_tokens": 24,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        },
        "service_tier": "default",
        "system_fingerprint": None,
    },
    expected_litellm_response={
        "id": "chatcmpl-EVhxr75iujjmwE0F2bgGMQHDX3Bq6",
        "created": 1791225875,
        "model": "openai/gpt-5.6-sol",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello!",
                    "role": "assistant",
                    "refusal": None,
                    "annotations": [],
                },
                "provider_specific_fields": {},
            }
        ],
        "usage": {
            "completion_tokens": 5,
            "prompt_tokens": 19,
            "total_tokens": 24,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {
                "audio_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
        },
        "service_tier": "default",
        "system_fingerprint": None,
    },
)


GPT_5_6_LUNA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "openai/gpt-5.6-luna",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/chat/completions",
    expected_provider_headers={
        "content-type": "application/json",
        "authorization": "Bearer synthetic-openai-key",
    },
    expected_provider_request={
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "model": "gpt-5.6-luna",
        "max_completion_tokens": 64,
    },
    mock_provider_response={
        "id": "chatcmpl-EVhxvZldBlNE3r8WZ8qJkxC92PmAr",
        "object": "chat.completion",
        "created": 1791225879,
        "model": "gpt-5.6-luna",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!", "refusal": None, "annotations": []},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 19,
            "completion_tokens": 5,
            "total_tokens": 24,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        },
        "service_tier": "default",
        "system_fingerprint": None,
    },
    expected_litellm_response={
        "id": "chatcmpl-EVhxvZldBlNE3r8WZ8qJkxC92PmAr",
        "created": 1791225879,
        "model": "openai/gpt-5.6-luna",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello!",
                    "role": "assistant",
                    "refusal": None,
                    "annotations": [],
                },
                "provider_specific_fields": {},
            }
        ],
        "usage": {
            "completion_tokens": 5,
            "prompt_tokens": 19,
            "total_tokens": 24,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {
                "audio_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
        },
        "service_tier": "default",
        "system_fingerprint": None,
    },
)


GPT_6_LUNA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "openai/gpt-6-luna",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/chat/completions",
    expected_provider_headers={
        "content-type": "application/json",
        "authorization": "Bearer synthetic-openai-key",
    },
    expected_provider_request={
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "model": "gpt-6-luna",
        "max_completion_tokens": 64,
    },
    mock_provider_response={
        "id": "chatcmpl-EVhxzlhJ6vfxv0dDB3QzTZvHJWTno",
        "object": "chat.completion",
        "created": 1791225883,
        "model": "gpt-6-luna",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!", "refusal": None, "annotations": []},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 19,
            "completion_tokens": 5,
            "total_tokens": 24,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        },
        "service_tier": "default",
        "system_fingerprint": None,
    },
    expected_litellm_response={
        "id": "chatcmpl-EVhxzlhJ6vfxv0dDB3QzTZvHJWTno",
        "created": 1791225883,
        "model": "openai/gpt-6-luna",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello!",
                    "role": "assistant",
                    "refusal": None,
                    "annotations": [],
                },
                "provider_specific_fields": {},
            }
        ],
        "usage": {
            "completion_tokens": 5,
            "prompt_tokens": 19,
            "total_tokens": 24,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {
                "audio_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
        },
        "service_tier": "default",
        "system_fingerprint": None,
    },
)


GPT_6_1_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "openai/gpt-6.1-sol",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/chat/completions",
    expected_provider_headers={
        "content-type": "application/json",
        "authorization": "Bearer synthetic-openai-key",
    },
    expected_provider_request={
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "model": "gpt-6.1-sol",
        "max_completion_tokens": 64,
    },
    mock_provider_response={
        "id": "chatcmpl-EVhy2Sc0BN5sjJZK2Wa6l3CtsQ2ot",
        "object": "chat.completion",
        "created": 1791225886,
        "model": "gpt-6.1-sol",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!", "refusal": None, "annotations": []},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 19,
            "completion_tokens": 5,
            "total_tokens": 24,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        },
        "service_tier": "default",
        "system_fingerprint": None,
    },
    expected_litellm_response={
        "id": "chatcmpl-EVhy2Sc0BN5sjJZK2Wa6l3CtsQ2ot",
        "created": 1791225886,
        "model": "openai/gpt-6.1-sol",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello!",
                    "role": "assistant",
                    "refusal": None,
                    "annotations": [],
                },
                "provider_specific_fields": {},
            }
        ],
        "usage": {
            "completion_tokens": 5,
            "prompt_tokens": 19,
            "total_tokens": 24,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {
                "audio_tokens": 0,
                "cached_tokens": 0,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
            },
        },
        "service_tier": "default",
        "system_fingerprint": None,
    },
)
