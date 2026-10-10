from typing import Final

from integration.translation.case import TranslationTestCase

GPT_5_4_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "azure/gpt-5.4",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/openai/deployments/gpt-5.4/chat/completions?api-version=2025-04-01-preview",
    expected_provider_headers={
        "content-type": "application/json",
        "api-key": "synthetic-azure-key",
        "authorization": "Bearer synthetic-azure-key",
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
        "choices": [
            {
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                },
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {"annotations": [], "content": "Hello.", "refusal": None, "role": "assistant"},
            }
        ],
        "created": 1791231864,
        "id": "chatcmpl-EVjWSpHSltjjRtiCWAwJpzcB7dyOQ",
        "model": "gpt-5.4-2026-03-05",
        "object": "chat.completion",
        "prompt_filter_results": [
            {
                "prompt_index": 0,
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                },
            }
        ],
        "routing": {"serving_pipereplica": "d20260919103649-bf00bd58-default-r2-dp0-default"},
        "service_tier": "default",
        "system_fingerprint": None,
        "usage": {
            "completion_tokens": 5,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "latency_checkpoint": {
                "engine_tbt_ms": 11,
                "engine_ttft_ms": 84,
                "engine_ttlt_ms": 152,
                "pre_inference_ms": 63,
                "service_tbt_ms": 12,
                "service_ttft_ms": 516,
                "service_ttlt_ms": 578,
                "user_visible_ttft_ms": 453,
            },
            "prompt_tokens": 19,
            "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
            "total_tokens": 24,
        },
    },
    expected_litellm_response={
        "id": "chatcmpl-EVjWSpHSltjjRtiCWAwJpzcB7dyOQ",
        "type": "message",
        "role": "assistant",
        "model": "azure/gpt-5.4",
        "stop_sequence": None,
        "usage": {"input_tokens": 19, "output_tokens": 5},
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)

GPT_5_6_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "azure/gpt-5.6-sol",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/openai/deployments/gpt-5.6-sol/chat/completions?api-version=2025-04-01-preview",
    expected_provider_headers={
        "content-type": "application/json",
        "api-key": "synthetic-azure-key",
        "authorization": "Bearer synthetic-azure-key",
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
        "choices": [
            {
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                },
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {"annotations": [], "content": "Hello.", "refusal": None, "role": "assistant"},
            }
        ],
        "created": 1791231865,
        "id": "chatcmpl-EVjWTALQyI0QnRBpbPzjE9NvDPIJG",
        "model": "gpt-5.6-sol-2026-07-09",
        "object": "chat.completion",
        "prompt_filter_results": [
            {
                "prompt_index": 0,
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                },
            }
        ],
        "routing": {"serving_pipereplica": "d20260930154032-380c3f99-prefill-r9-dp0-prefill"},
        "service_tier": "default",
        "system_fingerprint": None,
        "usage": {
            "completion_tokens": 5,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "latency_checkpoint": {
                "engine_tbt_ms": 7,
                "engine_ttft_ms": 144,
                "engine_ttlt_ms": 186,
                "pre_inference_ms": 66,
                "service_tbt_ms": 4,
                "service_ttft_ms": 267,
                "service_ttlt_ms": 286,
                "user_visible_ttft_ms": 201,
            },
            "prompt_tokens": 19,
            "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 0},
            "total_tokens": 24,
        },
    },
    expected_litellm_response={
        "id": "chatcmpl-EVjWTALQyI0QnRBpbPzjE9NvDPIJG",
        "type": "message",
        "role": "assistant",
        "model": "azure/gpt-5.6-sol",
        "stop_sequence": None,
        "usage": {"input_tokens": 19, "output_tokens": 5},
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)

GPT_5_6_LUNA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "azure/gpt-5.6-luna",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/openai/deployments/gpt-5.6-luna/chat/completions?api-version=2025-04-01-preview",
    expected_provider_headers={
        "content-type": "application/json",
        "api-key": "synthetic-azure-key",
        "authorization": "Bearer synthetic-azure-key",
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
        "choices": [
            {
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                },
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {"annotations": [], "content": "Hello!", "refusal": None, "role": "assistant"},
            }
        ],
        "created": 1791231867,
        "id": "chatcmpl-EVjWVHRXX2btCuzlg1wSnpqkpZFQZ",
        "model": "gpt-5.6-luna-2026-07-09",
        "object": "chat.completion",
        "prompt_filter_results": [
            {
                "prompt_index": 0,
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                },
            }
        ],
        "routing": {"serving_pipereplica": "d20260925052750-47dc37a0-prefill-r12-dp0-prefill"},
        "service_tier": "default",
        "system_fingerprint": None,
        "usage": {
            "completion_tokens": 5,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "latency_checkpoint": {
                "engine_tbt_ms": 12,
                "engine_ttft_ms": 64,
                "engine_ttlt_ms": 138,
                "pre_inference_ms": 88,
                "service_tbt_ms": 14,
                "service_ttft_ms": 248,
                "service_ttlt_ms": 317,
                "user_visible_ttft_ms": 159,
            },
            "prompt_tokens": 19,
            "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 0},
            "total_tokens": 24,
        },
    },
    expected_litellm_response={
        "id": "chatcmpl-EVjWVHRXX2btCuzlg1wSnpqkpZFQZ",
        "type": "message",
        "role": "assistant",
        "model": "azure/gpt-5.6-luna",
        "stop_sequence": None,
        "usage": {"input_tokens": 19, "output_tokens": 5},
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)

GPT_6_LUNA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "azure/gpt-6-luna",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/openai/deployments/gpt-6-luna/chat/completions?api-version=2025-04-01-preview",
    expected_provider_headers={
        "content-type": "application/json",
        "api-key": "synthetic-azure-key",
        "authorization": "Bearer synthetic-azure-key",
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
        "choices": [
            {
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                },
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {"annotations": [], "content": "Hello!", "refusal": None, "role": "assistant"},
            }
        ],
        "created": 1791231868,
        "id": "chatcmpl-EVjWWeBeTmIBg80iLiSeFNeYYY20z",
        "model": "gpt-6-luna-2026-09-22",
        "object": "chat.completion",
        "prompt_filter_results": [
            {
                "prompt_index": 0,
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                },
            }
        ],
        "routing": {"serving_pipereplica": "d20260930053954-f91e8c7b-prefill-r6-dp0-prefill"},
        "service_tier": "default",
        "system_fingerprint": None,
        "usage": {
            "completion_tokens": 5,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "latency_checkpoint": {
                "engine_tbt_ms": 3,
                "engine_ttft_ms": 63,
                "engine_ttlt_ms": 79,
                "pre_inference_ms": 81,
                "service_tbt_ms": 4,
                "service_ttft_ms": 280,
                "service_ttlt_ms": 301,
                "user_visible_ttft_ms": 199,
            },
            "prompt_tokens": 19,
            "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 0},
            "total_tokens": 24,
        },
    },
    expected_litellm_response={
        "id": "chatcmpl-EVjWWeBeTmIBg80iLiSeFNeYYY20z",
        "type": "message",
        "role": "assistant",
        "model": "azure/gpt-6-luna",
        "stop_sequence": None,
        "usage": {"input_tokens": 19, "output_tokens": 5},
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)

GPT_6_1_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "azure/gpt-6.1-sol",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/openai/deployments/gpt-6.1-sol/chat/completions?api-version=2025-04-01-preview",
    expected_provider_headers={
        "content-type": "application/json",
        "api-key": "synthetic-azure-key",
        "authorization": "Bearer synthetic-azure-key",
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
        "choices": [
            {
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                },
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {"annotations": [], "content": "Hello!", "refusal": None, "role": "assistant"},
            }
        ],
        "created": 1791231869,
        "id": "chatcmpl-EVjWXoW5I8J7jZD6gsZ3ONxzOqZ27",
        "model": "gpt-6.1-sol-2026-09-29",
        "object": "chat.completion",
        "prompt_filter_results": [
            {
                "prompt_index": 0,
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                },
            }
        ],
        "routing": {"serving_pipereplica": "d20260929014647-f3a90410-prefill-r11-dp0-prefill"},
        "service_tier": "default",
        "system_fingerprint": None,
        "usage": {
            "completion_tokens": 5,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "latency_checkpoint": {
                "engine_tbt_ms": 9,
                "engine_ttft_ms": 477,
                "engine_ttlt_ms": 530,
                "pre_inference_ms": 44,
                "service_tbt_ms": 8,
                "service_ttft_ms": 795,
                "service_ttlt_ms": 833,
                "user_visible_ttft_ms": 750,
            },
            "prompt_tokens": 19,
            "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 0},
            "total_tokens": 24,
        },
    },
    expected_litellm_response={
        "id": "chatcmpl-EVjWXoW5I8J7jZD6gsZ3ONxzOqZ27",
        "type": "message",
        "role": "assistant",
        "model": "azure/gpt-6.1-sol",
        "stop_sequence": None,
        "usage": {"input_tokens": 19, "output_tokens": 5},
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)
