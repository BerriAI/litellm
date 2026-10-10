from typing import Final

from integration.translation.case import TranslationTestCase

GPT_5_4_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "azure/gpt-5.4",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
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
        "stream": False,
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
        "created": 1791231844,
        "id": "chatcmpl-EVjW8mtOvQmVYDtEuv7KNgydJESBC",
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
                "engine_tbt_ms": 6,
                "engine_ttft_ms": 134,
                "engine_ttlt_ms": 168,
                "pre_inference_ms": 74,
                "service_tbt_ms": 6,
                "service_ttft_ms": 564,
                "service_ttlt_ms": 592,
                "user_visible_ttft_ms": 490,
            },
            "prompt_tokens": 19,
            "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
            "total_tokens": 24,
        },
    },
    expected_litellm_response={
        "id": "chatcmpl-EVjW8mtOvQmVYDtEuv7KNgydJESBC",
        "created": 1791231844,
        "model": "azure/gpt-5.4",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"content": "Hello.", "role": "assistant", "annotations": [], "refusal": None},
                "provider_specific_fields": {
                    "content_filter_results": {
                        "hate": {"filtered": False, "severity": "safe"},
                        "protected_material_code": {"detected": False, "filtered": False},
                        "protected_material_text": {"detected": False, "filtered": False},
                        "self_harm": {"filtered": False, "severity": "safe"},
                        "sexual": {"filtered": False, "severity": "safe"},
                        "violence": {"filtered": False, "severity": "safe"},
                    }
                },
                "logprobs": None,
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
            "latency_checkpoint": {
                "engine_tbt_ms": 6,
                "engine_ttft_ms": 134,
                "engine_ttlt_ms": 168,
                "pre_inference_ms": 74,
                "service_tbt_ms": 6,
                "service_ttft_ms": 564,
                "service_ttlt_ms": 592,
                "user_visible_ttft_ms": 490,
            },
        },
        "service_tier": "default",
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
        "system_fingerprint": None,
    },
)

GPT_5_6_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "azure/gpt-5.6-sol",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
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
        "stream": False,
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
        "created": 1791231846,
        "id": "chatcmpl-EVjWAtsNezzFIBBSE6qy0TnpRv0Ll",
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
                "engine_ttft_ms": 228,
                "engine_ttlt_ms": 269,
                "pre_inference_ms": 78,
                "service_tbt_ms": 8,
                "service_ttft_ms": 348,
                "service_ttlt_ms": 387,
                "user_visible_ttft_ms": 270,
            },
            "prompt_tokens": 19,
            "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 0},
            "total_tokens": 24,
        },
    },
    expected_litellm_response={
        "id": "chatcmpl-EVjWAtsNezzFIBBSE6qy0TnpRv0Ll",
        "created": 1791231846,
        "model": "azure/gpt-5.6-sol",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"content": "Hello.", "role": "assistant", "annotations": [], "refusal": None},
                "provider_specific_fields": {
                    "content_filter_results": {
                        "hate": {"filtered": False, "severity": "safe"},
                        "protected_material_code": {"detected": False, "filtered": False},
                        "protected_material_text": {"detected": False, "filtered": False},
                        "self_harm": {"filtered": False, "severity": "safe"},
                        "sexual": {"filtered": False, "severity": "safe"},
                        "violence": {"filtered": False, "severity": "safe"},
                    }
                },
                "logprobs": None,
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
            "latency_checkpoint": {
                "engine_tbt_ms": 7,
                "engine_ttft_ms": 228,
                "engine_ttlt_ms": 269,
                "pre_inference_ms": 78,
                "service_tbt_ms": 8,
                "service_ttft_ms": 348,
                "service_ttlt_ms": 387,
                "user_visible_ttft_ms": 270,
            },
        },
        "service_tier": "default",
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
        "system_fingerprint": None,
    },
)

GPT_5_6_LUNA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "azure/gpt-5.6-luna",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
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
        "stream": False,
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
        "created": 1791231848,
        "id": "chatcmpl-EVjWC8K4D1wYOwETMUzs0oFLnsvKg",
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
        "routing": {"serving_pipereplica": "d20260925052750-47dc37a0-prefill-r7-dp0-prefill"},
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
                "engine_ttft_ms": 58,
                "engine_ttlt_ms": 130,
                "pre_inference_ms": 70,
                "service_tbt_ms": 13,
                "service_ttft_ms": 224,
                "service_ttlt_ms": 290,
                "user_visible_ttft_ms": 154,
            },
            "prompt_tokens": 19,
            "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 0},
            "total_tokens": 24,
        },
    },
    expected_litellm_response={
        "id": "chatcmpl-EVjWC8K4D1wYOwETMUzs0oFLnsvKg",
        "created": 1791231848,
        "model": "azure/gpt-5.6-luna",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"content": "Hello!", "role": "assistant", "annotations": [], "refusal": None},
                "provider_specific_fields": {
                    "content_filter_results": {
                        "hate": {"filtered": False, "severity": "safe"},
                        "protected_material_code": {"detected": False, "filtered": False},
                        "protected_material_text": {"detected": False, "filtered": False},
                        "self_harm": {"filtered": False, "severity": "safe"},
                        "sexual": {"filtered": False, "severity": "safe"},
                        "violence": {"filtered": False, "severity": "safe"},
                    }
                },
                "logprobs": None,
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
            "latency_checkpoint": {
                "engine_tbt_ms": 12,
                "engine_ttft_ms": 58,
                "engine_ttlt_ms": 130,
                "pre_inference_ms": 70,
                "service_tbt_ms": 13,
                "service_ttft_ms": 224,
                "service_ttlt_ms": 290,
                "user_visible_ttft_ms": 154,
            },
        },
        "service_tier": "default",
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
        "routing": {"serving_pipereplica": "d20260925052750-47dc37a0-prefill-r7-dp0-prefill"},
        "system_fingerprint": None,
    },
)

GPT_6_LUNA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "azure/gpt-6-luna",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
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
        "stream": False,
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
        "created": 1791231849,
        "id": "chatcmpl-EVjWDVYMA8Y26Lw8Jk3617Q3kWSUS",
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
        "routing": {"serving_pipereplica": "d20260930053954-f91e8c7b-prefill-r7-dp0-prefill"},
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
                "engine_tbt_ms": 2,
                "engine_ttft_ms": 65,
                "engine_ttlt_ms": 80,
                "pre_inference_ms": 83,
                "service_tbt_ms": 5,
                "service_ttft_ms": 282,
                "service_ttlt_ms": 306,
                "user_visible_ttft_ms": 199,
            },
            "prompt_tokens": 19,
            "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 0},
            "total_tokens": 24,
        },
    },
    expected_litellm_response={
        "id": "chatcmpl-EVjWDVYMA8Y26Lw8Jk3617Q3kWSUS",
        "created": 1791231849,
        "model": "azure/gpt-6-luna",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"content": "Hello!", "role": "assistant", "annotations": [], "refusal": None},
                "provider_specific_fields": {
                    "content_filter_results": {
                        "hate": {"filtered": False, "severity": "safe"},
                        "protected_material_code": {"detected": False, "filtered": False},
                        "protected_material_text": {"detected": False, "filtered": False},
                        "self_harm": {"filtered": False, "severity": "safe"},
                        "sexual": {"filtered": False, "severity": "safe"},
                        "violence": {"filtered": False, "severity": "safe"},
                    }
                },
                "logprobs": None,
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
            "latency_checkpoint": {
                "engine_tbt_ms": 2,
                "engine_ttft_ms": 65,
                "engine_ttlt_ms": 80,
                "pre_inference_ms": 83,
                "service_tbt_ms": 5,
                "service_ttft_ms": 282,
                "service_ttlt_ms": 306,
                "user_visible_ttft_ms": 199,
            },
        },
        "service_tier": "default",
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
        "routing": {"serving_pipereplica": "d20260930053954-f91e8c7b-prefill-r7-dp0-prefill"},
        "system_fingerprint": None,
    },
)

GPT_6_1_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "azure/gpt-6.1-sol",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
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
        "stream": False,
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
        "created": 1791231851,
        "id": "chatcmpl-EVjWFBfdrUdRu3rbi6eAXbuO4zfNm",
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
                "engine_tbt_ms": 8,
                "engine_ttft_ms": 440,
                "engine_ttlt_ms": 491,
                "pre_inference_ms": 70,
                "service_tbt_ms": 9,
                "service_ttft_ms": 773,
                "service_ttlt_ms": 819,
                "user_visible_ttft_ms": 703,
            },
            "prompt_tokens": 19,
            "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 0},
            "total_tokens": 24,
        },
    },
    expected_litellm_response={
        "id": "chatcmpl-EVjWFBfdrUdRu3rbi6eAXbuO4zfNm",
        "created": 1791231851,
        "model": "azure/gpt-6.1-sol",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"content": "Hello!", "role": "assistant", "annotations": [], "refusal": None},
                "provider_specific_fields": {
                    "content_filter_results": {
                        "hate": {"filtered": False, "severity": "safe"},
                        "protected_material_code": {"detected": False, "filtered": False},
                        "protected_material_text": {"detected": False, "filtered": False},
                        "self_harm": {"filtered": False, "severity": "safe"},
                        "sexual": {"filtered": False, "severity": "safe"},
                        "violence": {"filtered": False, "severity": "safe"},
                    }
                },
                "logprobs": None,
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
            "latency_checkpoint": {
                "engine_tbt_ms": 8,
                "engine_ttft_ms": 440,
                "engine_ttlt_ms": 491,
                "pre_inference_ms": 70,
                "service_tbt_ms": 9,
                "service_ttft_ms": 773,
                "service_ttlt_ms": 819,
                "user_visible_ttft_ms": 703,
            },
        },
        "service_tier": "default",
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
        "system_fingerprint": None,
    },
)
