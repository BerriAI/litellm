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
from integration.translation.templates.azure import (
    AZURE_RESPONSES_BRIDGE_REQUEST,
    AZURE_RESPONSES_RESPONSE,
)
from integration.translation.templates.openai import (
    OPENAI_RESPONSES_AGENT_REQUEST,
    OPENAI_RESPONSES_AGENT_RESPONSE,
)
from pydantic import JsonValue

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

GPT_6_LUNA_CONTENT_FILTERS: Final[JsonValue] = [
    {
        "blocked": False,
        "source_type": "prompt",
        "content_filter_raw": [],
        "content_filter_results": {
            "hate": {"filtered": False, "severity": "safe"},
            "sexual": {"filtered": False, "severity": "safe"},
            "violence": {"filtered": False, "severity": "safe"},
            "self_harm": {"filtered": False, "severity": "safe"},
            "jailbreak": {"detected": False, "filtered": False},
        },
        "content_filter_offsets": {"start_offset": 0, "end_offset": 575, "check_offset": 0},
    },
    {
        "blocked": False,
        "source_type": "completion",
        "content_filter_raw": [],
        "content_filter_results": {
            "protected_material_code": {"detected": False, "filtered": False},
            "protected_material_text": {"detected": False, "filtered": False},
            "hate": {"filtered": False, "severity": "safe"},
            "sexual": {"filtered": False, "severity": "safe"},
            "violence": {"filtered": False, "severity": "safe"},
            "self_harm": {"filtered": False, "severity": "safe"},
        },
        "content_filter_offsets": {"start_offset": 0, "end_offset": 489, "check_offset": 0},
    },
]

GPT_6_LUNA_ENCRYPTED_REASONING: Final = (
    "gAAAAABq0CO5Co3fPBWyqHQfJ1ms2Y-BIc2sUtT_XTbjdSYTT6QTAaPmp8ylGto8Y3Of8ATTnesNsPFU"
    "7kLy9CJ_Lb2Pl5F7ln4lykRgrtPV-7f-B3FU_zaPLktc2eYem0KqseLyq3kBK90cXXTU0eEKJk458iGU"
    "fjCFXwxXSB2q_f6rHwxaw5XjQ6IJO_-weFQqarHq3kH7oxqfiWR2udVzYFvkmJ3ezaEQBAJFGREpqHVq"
    "NguzyFGuJLNK4urfedbSgOBmOMWZ8jxlPVCpe0FBu4fqvHhKKrBUxBdgvFS9UvhCCAeULfLQrxFBdQk_"
    "ncDkur6n1iT9UuFTkSDEX36MdI9i-Wfx7VGbgmc6sZHJMoHEA5Ut11GmiFdQEgYG5R29Nl1iqAjjBm2g"
    "3ZN2QNYQ7Ez_FLbOgE4dfF5uw5R239QLURkuEYdPAkfHVfewFiNyHstpoRkQ7qBcSfuvJZTbxPQPHpBd"
    "h6tELbKDrZy0upGR-InCK8MGrl9jpSnAtGT5jhS97Je8kay4zEjcFWW7-Km0IchvPLWU__ftBh38tqmn"
    "vprkmBsQpcNDhtx5MLZHEixrcbb-48y92c3_voScqLPDT7vM4sXDh3_8Op6ECc-BBh66af2PKef5wBD2"
    "eAiD9G2dl5UAsQalER0rf5s_Us0CzDfZIc1P4I6WrGK9odBayY1WYpVU57ng5ZeTfnoSYzQt1iaiiEEB"
    "PdVfQHc1Yc44D-Fk1Ipin7dThHshMn4oiYiCQI1ckRcpeLvPCycASykjaPLJyImrGP7CLgKWJPj1wr5R"
    "4XWkGwI5MOI83Y-Zdq3eVUQErpJShokpqITgWVJO-zrClUkwsVbely-9CKjlorLDLTYsAbLNIoJp6xxN"
    "AF2qt3QObhtKr7mxyq2idnVgdsk4Xj02P6XI-NERyqEdWOnUf7hDG8ZB75thRXDfKPiGkRFyCikPCV7Y"
    "4mpeUIE5pytFjYLLOYEq36_GlEcGFosWPyi1Z_V_suyzC0Wy7KGIRMoGdVyyBZrIpCQvVConXD0AFGsY"
    "Mvg4pxnwXK8gk_5N7ALIaGj9KM2X1i8ywKepmDw6Y1OVyfGCPhXyfuQd43ul1XIFczTemXw15PC9q2YX"
    "d3FuAXQQbKYycaPm7beOLi1gGehWqlcbYx0ZbJV4bk4VrxD94TGoii8T-hbckP2qeJInXCGRsT-vXw3P"
    "mGM_IkXTEJvZD_XmfinP7KPyjHyg-Lnx990ODgAOX0UhPCl58lsRlRq0XWldC8soDN-TKQh1E2j3vEGq"
    "YC8goY-zCX-5prx9F__fFBYFMQlRPFtDa3JzBEV2cJ8wavsIZ1r9vow6I4QSguIsp5tToakfVaoou4ct"
    "p94gqzyuymCtslLZ1Ndau4npTMUT1TzhVbQ-KJPRuuXht03dghpmmSe4sBnmpsLJHigktIL7ozhj3hue"
    "Gp6s3t5L8_Cx2vtXKxsOmWsDLM7VxA5Lz_lUGIjgWGyA6LFbssAVkjXLz_okVdFMvDUu2ZkYUnfvwXv1"
    "A1sUGJ_259aAEeIu0jlpMhCS6X54"
)

GPT_6_LUNA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "azure/gpt-6-luna"},
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_headers={"content-type": "application/json", "api-key": "synthetic-azure-key"},
    expected_provider_request={
        **AZURE_RESPONSES_BRIDGE_REQUEST,
        "model": "gpt-6-luna",
        "stream": False,
    },
    mock_provider_response={
        **AZURE_RESPONSES_RESPONSE,
        "id": "resp_0257792c0e8dcbb3006ad023b8112081958c7c4379959d4c6b",
        "created_at": 1792025528,
        "completed_at": 1792025529,
        "content_filters": GPT_6_LUNA_CONTENT_FILTERS,
        "model": "gpt-6-luna",
        "output": [
            {
                "id": "rs_0257792c0e8dcbb3006ad023b8e27481959ae045c67273b7d7",
                "type": "reasoning",
                "content": [],
                "encrypted_content": GPT_6_LUNA_ENCRYPTED_REASONING,
                "summary": [],
            },
            {
                "id": "msg_0257792c0e8dcbb3006ad023b939b88195b37db135f24d9db5",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": (
                            "I need `shop/validation.py` next to see the quantity checks, especially how zero "
                            "and negative values are handled."
                        ),
                    }
                ],
                "phase": "commentary",
                "role": "assistant",
            },
            {
                "id": "fc_0257792c0e8dcbb3006ad023b97338819589076c2f853d85bb",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":160}',
                "call_id": "call_nF1gXonXpNHBt14U2XkvfYZ8",
                "name": "read_file",
            },
        ],
        "usage": {
            "input_tokens": 312,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 129,
            "output_tokens_details": {"reasoning_tokens": 65},
            "total_tokens": 441,
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "id": "resp_0257792c0e8dcbb3006ad023b8112081958c7c4379959d4c6b",
        "model": "azure/gpt-6-luna",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": (
                        "I need `shop/validation.py` next to see the quantity checks, especially how zero "
                        "and negative values are handled."
                    ),
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "arguments": ('{"path":"shop/validation.py","start_line":1,"end_line":160}'),
                                "name": "read_file",
                            },
                            "id": "call_nF1gXonXpNHBt14U2XkvfYZ8",
                            "type": "function",
                        }
                    ],
                    "reasoning_items": [
                        {
                            "type": "reasoning",
                            "id": "rs_0257792c0e8dcbb3006ad023b8e27481959ae045c67273b7d7",
                            "encrypted_content": GPT_6_LUNA_ENCRYPTED_REASONING,
                            "summary": [],
                        }
                    ],
                    "provider_specific_fields": {"phase": "commentary", "openai_responses_replay": ANY},
                },
            }
        ],
        "usage": {
            "completion_tokens": 129,
            "prompt_tokens": 312,
            "total_tokens": 441,
            "completion_tokens_details": {"reasoning_tokens": 65},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "content_filters": GPT_6_LUNA_CONTENT_FILTERS,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
        "tool_usage": AZURE_RESPONSES_RESPONSE["tool_usage"],
    },
)

GPT_6_1_SOL_CONTENT_FILTERS: Final[JsonValue] = [
    {
        "blocked": False,
        "source_type": "prompt",
        "content_filter_raw": [],
        "content_filter_results": {
            "defender_for_ai": {
                "detected": False,
                "filtered": False,
                "details": {
                    "evaluation": "unavailable",
                    "evaluationReason": "missing_or_unknown_provider_evaluation_status",
                },
            },
            "hate": {"filtered": False, "severity": "safe"},
            "sexual": {"filtered": False, "severity": "safe"},
            "violence": {"filtered": False, "severity": "safe"},
            "self_harm": {"filtered": False, "severity": "safe"},
            "jailbreak": {"detected": False, "filtered": False},
        },
        "content_filter_offsets": {"start_offset": 0, "end_offset": 575, "check_offset": 0},
    },
    {
        "blocked": False,
        "source_type": "completion",
        "content_filter_raw": [],
        "content_filter_results": {
            "protected_material_code": {"detected": False, "filtered": False},
            "protected_material_text": {"detected": False, "filtered": False},
            "defender_for_ai": {
                "detected": False,
                "filtered": False,
                "details": {
                    "evaluation": "unavailable",
                    "evaluationReason": "missing_or_unknown_provider_evaluation_status",
                },
            },
            "hate": {"filtered": False, "severity": "safe"},
            "sexual": {"filtered": False, "severity": "safe"},
            "violence": {"filtered": False, "severity": "safe"},
            "self_harm": {"filtered": False, "severity": "safe"},
        },
        "content_filter_offsets": {"start_offset": 0, "end_offset": 524, "check_offset": 0},
    },
]

GPT_6_1_SOL_ENCRYPTED_REASONING: Final = (
    "gAAAAABrBFQR5hOS2Fgn8rkGVtf_7BPwIaLxG7AXRCgT47PkbNx5GQqluW-js-qYYPTuvTnCaFStTYF_sp0ahwRA6JYJl5koERUUv295GRLYHb"
    "qbzPvp9X6fR6Hpg4Q3GKXURzx_Dq8v58TKapLqtSJChzvnGHy7WCRfZuxrtqXjJnZ1ygYyyT6XyeKAhjdH3TyXtlI__vMMAWl0LZGH8MFd7USk"
    "_CYieic0xpOD8J8BeXkhLghQM0XRyjil_caTK6t5hCj8YXi1cdju5Sncbwqfh1DkFRI1uU-6LJNg6accHsCBbooELwM77qNi7K9o5sPo7yKE_q"
    "kICYMWK9XjUC7N3YGeFTDl6Q-ErGP91mK2PSDViD4jt7GuWJ4No9WtEBVtMAERpRS0ajiFiIpaDizCD7V2v9gVRpsWBZ8_8rjri5nsWV5NDZqy"
    "G2ttro7iYKWzDZAWVmfxw1K5dXx510Y8s_bTlN3gxrbUiAYgRGPz8EmPjY99ilsEj1jbMV6sI2zRiFkro_WuQLNBQmhRZjpPKQSbp-rb6cUt9U"
    "6vK54dbX9kV0Yt5N7cwo0D9tdpbFiWX48uBUq1phWEBJiQWfc88bh_3eEOdk38XAAfswOqD_KPn06ec33_5oUHYAwvJd4r_fDYR_z9JW1ogAgK"
    "qLj0-h1iwwjMHpIe8CvATNpv78B3tTEah1WswooBLHaYyAgo7Vr0hV6jZDHQCgGjOCHh4FHUopH8qz2kZyniiFYP7jixE5zlCTq3DqSnnvX1kf"
    "727Lt6jV-xxEVT7NrlI_PbsDmFDK-YIq-J1y12eumpDCbKz_y2U-wWnikD9AQXuMu9DApSG68In9OrpFdsId0mfsNu8qnBJ1EhPBaaPCukmik6"
    "JhVyT5B1NPFCLvDtIJfbgg_q8BuDN0T-D4VetmAT2gG3K3Kva1fxyu6cW2GZ-rIwlod0wPGiEyLMSgk4fd9u-E1GChCL0ZpAIo3b47YLOd2Ydc"
    "SUKQaAQklPg3Wg-mFlkd_QOJ5UWL2fniktWhvar-M0em2e-QadHKlGGm2RlhLnqgznR_9VnZ3JjXlWIZZ4LmtTe9s523vRMcSI143yoxqqXvKu"
    "yopZ6E8rJ0LCh3xOpAsSLQbwFQBIZHb7yOdi1jJSZrESyDQEpgBaoy64uDRkWFHEsGvNqCJf69ESHkbwfpbz_xJMg1vzaSTwJ_Z96MqwwwC_Lp"
    "pxenic__1IFtInFj-hq8VWSBs6bSXjtVNu0tu8hdauvaJL7eJqotSFNcsUaSIpi2QkiWYdHtI_Btq2sxw5HASXzMXE4GBE6v1ZtEU2kInax_7V"
    "-HvFTRXFSX5Atym-2ZtXqysFqVXt0h_emhN5smnHtnrnw4YTMraiESpOtJB0QCkHocZ4G9WkOwSUMNaMd8hzk1fXUcxYY7RNwdVizz9Qv6y8dM"
    "_JkHa2ReIa-D4KPMjLBmWuEdsq_mTjhHtYxhYgH_dgP6ZfdswmtftPtSJeADTEfDm7tsG5O9InA59tUDr05ALPxUz9dmJCKBUaHztHbA5bZ3Af"
    "ttt5tBM7UsZoX-e2d1JcI4uSCetn4gnfACotoCNBvQqD_jPjRhFDEBIyWTN6StpvXb4fQ_Y5242FsuEQAbY7ZUANUaVRvti1byhEW-Ox_qRh6r"
    "XrrOSmNPzohm0JLXkxZwSoI0b_gW5H5N5qxJkSHZ4GsIUT6LpsJ_8JNYExUHfwW-2PHZlH_qIbjPTdJYuWM2x_S77wNyDhugciRSN8XGdE4O9N"
    "KurRgsbnlt6v9t1kgJfh9SO3j-BmcoTH5MbYbQMf229A31Thz-5yY8Wif_xoDVA6s_IIzA8bgFSvmHTIhgmWfGAFXQqmL4DU0Np5bjwqDFq0PJ"
    "bhuwWmN3rWY-k_CmI6aGcQx3tn3cpETh7idbuIYc1rL1uFWHre-dMRWa51_r6dgE7_cmm5ltZ_3mJfbgoVt45YBQWaSKQrROCxD-89j1IvL2LY"
    "wIhUYBpV2Zn4jDq7Z7wdXn5OCq1l94SBNVernpAuEpErGV6lHKTnuXzs1R1YuI0PS8BXjUaXtm7kEfUaVBBeXwJFEPCpo9He4XUGJH9NFBucDj"
    "NLdvJD29bpT_Ggue1stbKAiqa3r6kQazx3T7UwPEMLz8dOeF5BHVVunNJemEgU1JTQ0b8QaT6NcbrnEi41Pa3cZCGgAxQ_0pWjkdOWfOWlCjuU"
    "i-YooGtgZMu3rli27Ly1qWt_aAi3oznad3qowAP7IcMNdJA7SKZHQbwSgXVT0DpRngMB6Kz3FjgRJHZyoj93H74cEXmP3Cmaqq1kSNUqOqveCc"
    "18snaW4g_q5iFYbbSzIFrf3bDtIdtFuIHFPEzzr84Sj9uVAezmMAZoKJ9T75hqAJhn_hZl"
)

GPT_6_1_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "azure/gpt-6.1-sol"},
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_headers={"content-type": "application/json", "api-key": "synthetic-azure-key"},
    expected_provider_request={**AZURE_RESPONSES_BRIDGE_REQUEST, "model": "gpt-6.1-sol", "stream": False},
    mock_provider_response={
        **AZURE_RESPONSES_RESPONSE,
        "id": "resp_01c2d42cd4ede2fb006b04540bf1fc819699733a8c5501bafc",
        "created_at": 1795445772,
        "completed_at": 1795445777,
        "content_filters": GPT_6_1_SOL_CONTENT_FILTERS,
        "instructions": None,
        "output": [
            {
                "id": "rs_01c2d42cd4ede2fb006b04540defc48196958401e7c836a0d5",
                "type": "reasoning",
                "content": [],
                "encrypted_content": GPT_6_1_SOL_ENCRYPTED_REASONING,
                "summary": [],
            },
            {
                "id": "msg_01c2d42cd4ede2fb006b0454103c288196869653b273a7770a",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "I’ll read `shop/validation.py` first to check how it handles zero, negative, and "
                        "non-integer quantities.\n",
                    }
                ],
                "phase": "commentary",
                "role": "assistant",
            },
            {
                "id": "fc_01c2d42cd4ede2fb006b045410ccf48196afd6e2855f15e013",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                "call_id": "call_ZkSvg6xufGGYY0Pp0KdfM3Kw",
                "name": "read_file",
            },
        ],
        "usage": {
            "input_tokens": 312,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 142,
            "output_tokens_details": {"reasoning_tokens": 77},
            "total_tokens": 454,
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "azure/gpt-6.1-sol",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "I’ll read `shop/validation.py` first to check how it handles zero, negative, "
                    "and non-integer quantities.\n",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                                "name": "read_file",
                            },
                            "id": "call_ZkSvg6xufGGYY0Pp0KdfM3Kw",
                            "type": "function",
                        }
                    ],
                    "reasoning_items": [
                        {
                            "type": "reasoning",
                            "id": "rs_01c2d42cd4ede2fb006b04540defc48196958401e7c836a0d5",
                            "encrypted_content": GPT_6_1_SOL_ENCRYPTED_REASONING,
                            "summary": [],
                        }
                    ],
                    "provider_specific_fields": {
                        "phase": "commentary",
                        "openai_responses_replay": ANY,
                    },
                },
            }
        ],
        "usage": {
            "completion_tokens": 142,
            "prompt_tokens": 312,
            "total_tokens": 454,
            "completion_tokens_details": {"reasoning_tokens": 77},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "content_filters": GPT_6_1_SOL_CONTENT_FILTERS,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
        "tool_usage": {
            "image_gen": {
                "input_tokens": 0,
                "input_tokens_details": {"image_tokens": 0, "text_tokens": 0},
                "output_tokens": 0,
                "output_tokens_details": {"image_tokens": 0, "text_tokens": 0},
                "total_tokens": 0,
            },
            "web_search": {"num_requests": 0},
        },
    },
)

GPT_6_ASTRA_CONTENT_FILTERS: Final[JsonValue] = [
    {
        "blocked": False,
        "source_type": "prompt",
        "content_filter_raw": [],
        "content_filter_results": {
            "hate": {"filtered": False, "severity": "safe"},
            "sexual": {"filtered": False, "severity": "safe"},
            "violence": {"filtered": False, "severity": "safe"},
            "self_harm": {"filtered": False, "severity": "safe"},
            "jailbreak": {"detected": False, "filtered": False},
        },
        "content_filter_offsets": {"start_offset": 0, "end_offset": 575, "check_offset": 0},
    },
    {
        "blocked": False,
        "source_type": "completion",
        "content_filter_raw": [],
        "content_filter_results": {
            "protected_material_code": {"detected": False, "filtered": False},
            "protected_material_text": {"detected": False, "filtered": False},
            "hate": {"filtered": False, "severity": "safe"},
            "sexual": {"filtered": False, "severity": "safe"},
            "violence": {"filtered": False, "severity": "safe"},
            "self_harm": {"filtered": False, "severity": "safe"},
        },
        "content_filter_offsets": {"start_offset": 0, "end_offset": 511, "check_offset": 0},
    },
]

GPT_6_ASTRA_ENCRYPTED_REASONING: Final = (
    "gAAAAABq0CPDbn4g-dCLkDZ90pK6ePzjtg1TpduhwuSh1HFNOWpCL6JcLlasgA7lAnoba8Zm8LZe5T23"
    "BMKHJVWyNRygNF8WDBBQvKvUhX2BOjPSf9wUbkXuctD_VtgwabIhmtwFhS6LH3iYbXzztiGqB898M7sF"
    "QPlW8yqde_2hojqasenBXwkhsmeVHZZRxq0TXjc8eZeJ4WEdZOck0TB1eV0xb3UArpyy3amtuONzYCy2"
    "wDVDAAlBVPTCcOC67biyQaEL7lipOMukY-yoGoQ-_SnRfLiVQDtl6U2mRA6svav-EEYm8igRyau6CSbp"
    "CKYAHflNODMH4y27mX2iCRBeFkZ-nDAzEDdxd7UPFFrp-lKbCueG7JYIyBfkQlJMB_YyY2rcF6AWuLMt"
    "K8RauqiMH-lgPZS1p4KhAauH1jSb47s3ZTgF4rowkMLdgAwdxIZD-3aoGXN9l4XOHumUfQNpJMNZbg9c"
    "RayPzWlUEOXWEZ7WBchPzl_yIvZCvtTk3ljpUDmJuEb0Ox_TiWqFEr2__UgFccG49x5Sv9pSXqxrsjMw"
    "y6ysHzZLbveONQ-gGcraErJcnnwSFSq_gsGiDPV-AJAMo-qY9vf5y6wJOSwIgQHpBx2e3WQAlp2R7vLr"
    "QN_-cXtKgTcuVVGGjVLJIazz5Br4Yke70ms5nqPVOQYVfnXTAOzfbOrFfO_U5DoIW6xorPtPd_WFkw7Q"
    "jK-gpm7AfkZL2RTICcwZDXuR6RCnSZMbOkBLCFlvZ3WyliM7j3fcvhxhfhA2Svp5kOxVjzZwY0NNMNA9"
    "9x5td-Ht8ZgFX-UxIxo8H3sxy5cu-1JKd2eQb4QYoMyVXqk8ZOS28HPGgsx_B4pPTl2KOQ8b0KtNoiB1"
    "58YDy9e74nb5u4U9I7Xkf8rS5EGmX1aGFAgW89XEymYCLfBfhtDjZQZAVyPSPxEdQd6CPnoVBy1MGrMK"
    "zbyt8X4xEIJJupb8SLuLmfkxKskE4QHfoX5h9As_Ak9WGEn5TIFtJJWmBk6cwZOoswtBSDbgLf-Xid3n"
    "2qfPX0UwdUnQzMNdc9eG9CU5vtvs-lRCMJbH3qHlK19vF82EoZT_z4lrHNXSyLOZad1kGaCRDFLakFCW"
    "KF7Qil-U69jZCpyaghA1Ebts_es48dSwTLm9Kh323P_y5_o2baqKxT9FUKh5kVztjdiBSLx1LA5mzMUj"
    "wdEWQpKh-RfKRnb9YoRGvCYrfvGtF_TcuYEg1DXQkOGejpaeXd-21Vaq44w1XJ72Kyun9KCJvoZu9O01"
    "vo8uhUnhNksGqx26p8M6bxIePVhfA-_RVbg_meJov3jgH3OHphRLAO6jIECT-GauSo7p3fjmSviQlkZB"
    "520Y5e6n3LPNZ8zpVYvtKooOj1VC0r6F_8vOdN8G5NvjELx2gptVXspfBoM3Ku1q7pFFUKqTG091ZKsS"
    "tiFlIphwqI9D4d5aRaerUp2mN_X_aPJoj02O6svBMsbUHAns0PqVoCDg1HogteEQPSiHJTT_D033QDG6"
    "ybr4oLs="
)

GPT_6_ASTRA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "azure/gpt-6-astra"},
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_headers={"content-type": "application/json", "api-key": "synthetic-azure-key"},
    expected_provider_request={
        **AZURE_RESPONSES_BRIDGE_REQUEST,
        "model": "gpt-6-astra",
        "stream": False,
    },
    mock_provider_response={
        **AZURE_RESPONSES_RESPONSE,
        "id": "resp_03d9860d694cf03b006ad023bfd5f881979bc737bdacd3fdec",
        "created_at": 1792025535,
        "completed_at": 1792025539,
        "content_filters": GPT_6_ASTRA_CONTENT_FILTERS,
        "model": "gpt-6-astra",
        "output": [
            {
                "id": "rs_03d9860d694cf03b006ad023c113c0819784747d4f0860d98a",
                "type": "reasoning",
                "content": [],
                "encrypted_content": GPT_6_ASTRA_ENCRYPTED_REASONING,
                "summary": [],
            },
            {
                "id": "msg_03d9860d694cf03b006ad023c246808197b42e6d5551340e2b",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": (
                            "I’ll read `shop/validation.py` first to see how it handles zero, negative, and "
                            "non-integer quantities.\n"
                        ),
                    }
                ],
                "phase": "commentary",
                "role": "assistant",
            },
            {
                "id": "fc_03d9860d694cf03b006ad023c2b42081978ef9bf4a11041d28",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                "call_id": "call_2kSDJH2w71KGgAbbYNfoELU7",
                "name": "read_file",
            },
        ],
        "usage": {
            "input_tokens": 312,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 135,
            "output_tokens_details": {"reasoning_tokens": 70},
            "total_tokens": 447,
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "id": "resp_03d9860d694cf03b006ad023bfd5f881979bc737bdacd3fdec",
        "model": "azure/gpt-6-astra",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": (
                        "I’ll read `shop/validation.py` first to see how it handles zero, negative, and "
                        "non-integer quantities.\n"
                    ),
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "arguments": ('{"path":"shop/validation.py","start_line":1,"end_line":200}'),
                                "name": "read_file",
                            },
                            "id": "call_2kSDJH2w71KGgAbbYNfoELU7",
                            "type": "function",
                        }
                    ],
                    "reasoning_items": [
                        {
                            "type": "reasoning",
                            "id": "rs_03d9860d694cf03b006ad023c113c0819784747d4f0860d98a",
                            "encrypted_content": GPT_6_ASTRA_ENCRYPTED_REASONING,
                            "summary": [],
                        }
                    ],
                    "provider_specific_fields": {"phase": "commentary", "openai_responses_replay": ANY},
                },
            }
        ],
        "usage": {
            "completion_tokens": 135,
            "prompt_tokens": 312,
            "total_tokens": 447,
            "completion_tokens_details": {"reasoning_tokens": 70},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "content_filters": GPT_6_ASTRA_CONTENT_FILTERS,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
        "tool_usage": AZURE_RESPONSES_RESPONSE["tool_usage"],
    },
)


GPT_5_4_MINI_TEST_CASE: Final = TranslationTestCase(
    scenario="agent_tools",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CODING_AGENT_REQUEST, "model": "azure/gpt-5.4-mini"},
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_headers={"content-type": "application/json", "api-key": "synthetic-agent-key"},
    expected_provider_request={**OPENAI_RESPONSES_AGENT_REQUEST, "model": "gpt-5.4-mini"},
    mock_provider_response={
        **OPENAI_RESPONSES_AGENT_RESPONSE,
        "id": "resp_0f005e55e6e18746006ad04ac8e4548196b106305915e785b5",
        "created_at": 1792035528,
        "status": "completed",
        "access_programs": None,
        "completed_at": 1792035532,
        "content_filters": [
            {
                "blocked": False,
                "source_type": "prompt",
                "content_filter_raw": [],
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 972, "check_offset": 0},
            },
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 253, "check_offset": 0},
            },
        ],
        "model": "gpt-5.4-mini",
        "output": [
            {
                "id": "msg_0f005e55e6e18746006ad04acc3984819690df06110511aa15",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": "I’ll search for callers and run the targeted test before patching.",
                    }
                ],
                "phase": "commentary",
                "role": "assistant",
            },
            {
                "id": "fc_0f005e55e6e18746006ad04acc713c8196b5484f104d908067",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"query":"retries_remaining("}',
                "call_id": "call_d8z67veoQLDjeKe3peLFNYvC",
                "name": "search_code",
            },
            {
                "id": "fc_0f005e55e6e18746006ad04acc72c081969ee09d2b44b50a84",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"tests/test_retry.py"}',
                "call_id": "call_yL8qxv5wZi6Eoesr69xs2UBd",
                "name": "run_tests",
            },
        ],
        "prompt_cache_retention": "in_memory",
        "reasoning": {**object_value(OPENAI_RESPONSES_AGENT_RESPONSE["reasoning"]), "context": "current_turn"},
        "service_tier": "default",
        "usage": {
            "input_tokens": 448,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 75,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 523,
        },
    },
    expected_litellm_response={
        **CODING_AGENT_RESPONSE,
        "id": "resp_0f005e55e6e18746006ad04ac8e4548196b106305915e785b5",
        "model": "azure/gpt-5.4-mini",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "I’ll search for callers and run the targeted test before patching.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"query":"retries_remaining("}', "name": "search_code"},
                            "id": "call_d8z67veoQLDjeKe3peLFNYvC",
                            "type": "function",
                        },
                        {
                            "index": 1,
                            "function": {"arguments": '{"path":"tests/test_retry.py"}', "name": "run_tests"},
                            "id": "call_yL8qxv5wZi6Eoesr69xs2UBd",
                            "type": "function",
                        },
                    ],
                    "provider_specific_fields": {"phase": "commentary", "openai_responses_replay": ANY},
                },
            }
        ],
        "usage": {
            "completion_tokens": 75,
            "prompt_tokens": 448,
            "total_tokens": 523,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "content_filters": [
            {
                "blocked": False,
                "source_type": "prompt",
                "content_filter_raw": [],
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "jailbreak": {"detected": False, "filtered": False},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 972, "check_offset": 0},
            },
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_raw": [],
                "content_filter_results": {
                    "protected_material_code": {"detected": False, "filtered": False},
                    "protected_material_text": {"detected": False, "filtered": False},
                    "hate": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                },
                "content_filter_offsets": {"start_offset": 0, "end_offset": 253, "check_offset": 0},
            },
        ],
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
        "tool_usage": {
            "image_gen": {
                "input_tokens": 0,
                "input_tokens_details": {"image_tokens": 0, "text_tokens": 0},
                "output_tokens": 0,
                "output_tokens_details": {"image_tokens": 0, "text_tokens": 0},
                "total_tokens": 0,
            },
            "web_search": {"num_requests": 0},
        },
    },
)

GPT_5_6_TERRA_CONTENT_FILTERS: Final[JsonValue] = [
    {
        "blocked": False,
        "source_type": "prompt",
        "content_filter_raw": [],
        "content_filter_results": {
            "hate": {"filtered": False, "severity": "safe"},
            "sexual": {"filtered": False, "severity": "safe"},
            "violence": {"filtered": False, "severity": "safe"},
            "self_harm": {"filtered": False, "severity": "safe"},
            "jailbreak": {"detected": False, "filtered": False},
        },
        "content_filter_offsets": {"start_offset": 0, "end_offset": 575, "check_offset": 0},
    },
    {
        "blocked": False,
        "source_type": "completion",
        "content_filter_raw": [],
        "content_filter_results": {
            "protected_material_code": {"detected": False, "filtered": False},
            "protected_material_text": {"detected": False, "filtered": False},
            "hate": {"filtered": False, "severity": "safe"},
            "sexual": {"filtered": False, "severity": "safe"},
            "violence": {"filtered": False, "severity": "safe"},
            "self_harm": {"filtered": False, "severity": "safe"},
        },
        "content_filter_offsets": {"start_offset": 0, "end_offset": 388, "check_offset": 0},
    },
]

GPT_5_6_TERRA_ENCRYPTED_REASONING: Final = (
    "gAAAAABq0CPOOqy44P30Q1NqlPLLgAVH4x-Gd5CYim_MQz5OgD7eB1qP82U2pKslsZhD5378bFPvj194"
    "3FU_ZJKBe0K8qBQ7WC0llLY7GBHxkqG1pk1QoaRjPi5Ns_gwkF3hFSCnGXgjN_6WZVc8mraXVm6kVVCp"
    "YVctOcGPC0FEwPkvFk-UKGyVyEn8--LESX2S0O_3ObP1S4beDMv4L8kAF3LOAlGBKujFdfqk1ch1-yZa"
    "CTA0ePCYXz35wUGwmjpO2NIo-nMPuDUeP-1Y7jzG8gJBdkP9o7-7pR9aCWcurj8wn-ae3Sgtx9WE7I42"
    "eiDBp0CAV0uet_SDAZ5csE0v6Q_amiKBh7TEabBnPMp4CYZxjhVRDAiU3oDciHzq8odZB1Ftwr3_YmEX"
    "YKtuet_lg2wE6s1yEX4itWaYkT6sfh9lR-DafSdF1o9-2jUbgP4FwYrl3LBSadUYlP-mDlsAwrm3caif"
    "GQFATcpLE229i8Yw9aiMmrgWXpJ-vP-f-gM_8EWPZ_HXXTJIrCqYcnD4LjxmLHXesuTdt_QsdeUxSFqX"
    "SzH_Ywl0fNqlH_LUtG1PfO8CEG_TF9QNFsDBY9qOE9NO0LHoTbUsjTCcdglewNQUh_s8W2L_PBMqYpxE"
    "v8xWdvKTsnO1D0aMsE5ovo_T8ROJR46NrrcdAsFPFEceL5G_yjGwLUnB7TRFRvkSm8BSjnaCKqbv9TPy"
    "Uzs3CjMsGmz3_8mHRMJTP8hO0kp97a8oZMKeokVrdSYk4pGv9Oyy9GW1BaUVvCEgUV7Dmrk-mt9znTCf"
    "GlbBkiF6x_LqdQahGjVdSJQk2NpVZQktzKNyQhzHS-ATargehB4TAANyvyRUL4-zyK_QLVCaN2AlIMPx"
    "_6frDw-vmb6bNHXmpc5fyU66Z0JaL_kTpHVTDBdxD5NprC8DEBaaxbnb283hP15dRxLHAb6fOprCkbLd"
    "oJfXyyRmoCD2kkdIBhrzOxFr7h9Ipfvf-Qy-gjRpZVbfOzkUTfqxqe5vI5uAGk8uxz5EQFgdH_NgNaZ7"
    "76ji97ewVoX-RWgnAra0kzzU_ECDOE52lztUcz6yKLwH9Rw2FqxZJucA0Bh6_6lQbsZAlWz9CuYTcCtu"
    "6lnJVfmsVMOJRvnykjYGzaX5tpShxmsu1YiCRmOzXTxB2wdATSHr90OWH3_8w-OkVDBBUO9ByDxlepid"
    "zcX043O9BJhdkXFWBUrwjzBO_NWyKKLm7yRu6Evgg34EvlFhFc1F2p6qqTsml_yNsZyw6I3NErTkw4KH"
    "aYVQH-DCiM7QoqJK6eJxaEk8QBF6Macs8aXVCYC1xXqvApLQELhvuIDLcGDQSQYLeJKdbq6UksNnp-LE"
    "1bt7HPvcSL3heudkey4BtoE5OqnyDV2msEzLtkw="
)

GPT_5_6_TERRA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "azure/gpt-5.6-terra"},
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_headers={"content-type": "application/json", "api-key": "synthetic-azure-key"},
    expected_provider_request={
        **AZURE_RESPONSES_BRIDGE_REQUEST,
        "model": "gpt-5.6-terra",
        "stream": False,
    },
    mock_provider_response={
        **AZURE_RESPONSES_RESPONSE,
        "id": "resp_0146fd70db485046006ad023cbf5b48197a496e74bfbe99f66",
        "created_at": 1792025548,
        "completed_at": 1792025550,
        "content_filters": GPT_5_6_TERRA_CONTENT_FILTERS,
        "model": "gpt-5.6-terra",
        "output": [
            {
                "id": "rs_0146fd70db485046006ad023cd55848197b06f3cbd7f6046e5",
                "type": "reasoning",
                "content": [],
                "encrypted_content": GPT_5_6_TERRA_ENCRYPTED_REASONING,
                "summary": [],
            },
            {
                "id": "msg_0146fd70db485046006ad023cddf1c8197bfc55155a0310a0b",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": (
                            "Next I need `shop/validation.py` to inspect the validation branches for zero, "
                            "negative, and non-integer quantities before reviewing the tests."
                        ),
                    }
                ],
                "phase": "commentary",
                "role": "assistant",
            },
            {
                "id": "fc_0146fd70db485046006ad023ce20d481979de3a49e428ae4e8",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                "call_id": "call_TZsfYd2RtwusZpfe2wrbwnRf",
                "name": "read_file",
            },
        ],
        "usage": {
            "input_tokens": 312,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 110,
            "output_tokens_details": {"reasoning_tokens": 41},
            "total_tokens": 422,
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "id": "resp_0146fd70db485046006ad023cbf5b48197a496e74bfbe99f66",
        "model": "azure/gpt-5.6-terra",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": (
                        "Next I need `shop/validation.py` to inspect the validation branches for zero, "
                        "negative, and non-integer quantities before reviewing the tests."
                    ),
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "arguments": ('{"path":"shop/validation.py","start_line":1,"end_line":200}'),
                                "name": "read_file",
                            },
                            "id": "call_TZsfYd2RtwusZpfe2wrbwnRf",
                            "type": "function",
                        }
                    ],
                    "reasoning_items": [
                        {
                            "type": "reasoning",
                            "id": "rs_0146fd70db485046006ad023cd55848197b06f3cbd7f6046e5",
                            "encrypted_content": GPT_5_6_TERRA_ENCRYPTED_REASONING,
                            "summary": [],
                        }
                    ],
                    "provider_specific_fields": {"phase": "commentary", "openai_responses_replay": ANY},
                },
            }
        ],
        "usage": {
            "completion_tokens": 110,
            "prompt_tokens": 312,
            "total_tokens": 422,
            "completion_tokens_details": {"reasoning_tokens": 41},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "content_filters": GPT_5_6_TERRA_CONTENT_FILTERS,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
        "tool_usage": AZURE_RESPONSES_RESPONSE["tool_usage"],
    },
)
