from typing import Final

from integration.translation.case import TranslationTestCase
from integration.translation.messages.templates import MESSAGES_REQUEST
from integration.translation.templates.azure import AZURE_RESPONSES_BRIDGE_REQUEST, AZURE_RESPONSES_RESPONSE
from pydantic import JsonValue

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
            "protected_material_text": {"detected": False, "filtered": False},
            "protected_material_code": {"detected": False, "filtered": False},
            "hate": {"filtered": False, "severity": "safe"},
            "sexual": {"filtered": False, "severity": "safe"},
            "violence": {"filtered": False, "severity": "safe"},
            "self_harm": {"filtered": False, "severity": "safe"},
        },
        "content_filter_offsets": {"start_offset": 0, "end_offset": 388, "check_offset": 0},
    },
]

GPT_6_LUNA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "azure/gpt-6-luna"},
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_headers={"content-type": "application/json", "api-key": "synthetic-azure-key"},
    expected_provider_request={
        **AZURE_RESPONSES_BRIDGE_REQUEST,
        "model": "gpt-6-luna",
        "instructions": MESSAGES_REQUEST["system"],
    },
    mock_provider_response={
        **AZURE_RESPONSES_RESPONSE,
        "id": "resp_0534b0c3898d6d1b006ad023ba700c8195b63f1819ab327ab3",
        "created_at": 1792025530,
        "completed_at": 1792025531,
        "content_filters": GPT_6_LUNA_CONTENT_FILTERS,
        "instructions": MESSAGES_REQUEST["system"],
        "model": "gpt-6-luna",
        "output": [
            {
                "id": "rs_0534b0c3898d6d1b006ad023bb09b8819590793fe299bffdcf",
                "type": "reasoning",
                "content": [],
                "encrypted_content": (
                    "gAAAAABq0CO7N77UmN8oIxFPRrP2rvE9LuqnrGGHsFzckseHLpNB24KIlg0OKnfemaCgYarrVksxeJBK"
                    "goiQ79xkF1NR9RI8-82w4OGNB0ViGNYNA7nf5kOqOTrMh1_gHecIN0Evxn8I0meiqOds7h1biKfkiZlm"
                    "19b-9r_Y2K--KXONVKFEbe8kP54_ka9KCxQ3_5XnL4IRfRS7lsLfoMjORKnapkVSnMjueRVVOprw_897"
                    "JJypeT61mzzBYM_18T6k68OE70JndMsr4GNRq4c_Lf_rSRoHg7_iFrdHsBdvMfN79dfp_PeLu5MZVRgc"
                    "ag-qM9aKUAvzhTD2-EutMw1C1ZNabU2S5Qa4RXq2OiU_iZljJI6JhpnJ9jT9g_M3W_0mklmFU2HBhQ6O"
                    "csIFmPePVcSaQlmofKgvq2ZThSHyFpLsT15ftBL2ZMZ3OWeodDnkfRr2fY5NDM7v7U1TLcAx7xCnP88W"
                    "03kyb3SuJGSsbPnYTGbNZN7ye3SBJRkCMpRgUfguSYdC46mmMjPdvdkLnTWHZ23GOoDG5A2lffUJZ4WX"
                    "NgPrMdG2FNThtHT0otHeDC75ahpa6i5xtqIsCGiFRVH_iSxhItArxFXoc7e6oBU9YxsgkUkyXW_zz6h5"
                    "h5E6DoJz4QckOAdL8E3SwL1nu6ovpS3_QgeiHa5RewKs6WclFCxeiwiyXO6bfjhl_VtOA8KyWqkO-rt4"
                    "T0RneV_bZHav6KoVanyCX-ZsAYj1WIhn4xEOXh6RdBjZJJB-yDw5cq6zdXkmN1hxs_nmA7hUemHkPclb"
                    "2lGF41N28IwMpcXq1rx2jrw2XleLnqFVKwgCLysinS9erU2vRN1CmmDQrGl7C5lIOrfSVCpB2W_p3BZW"
                    "NDKwAfF4i8uXCxy1_5-EjJ08CJXXYVchceOreHYNYxAXhVTHhJArQqVWQgTwJl9e_HhyA0OFZHojB54B"
                    "jlkMr7QJMm9VE9LIkOJgRgIrtUM2ULIy1tZ-EzuzEy9HYx5NuUKe_12ExKuUVrQmaJ2HRHvj_pPcmSk6"
                    "lOVrN5VvVBMj_KHd3MfkbUWVeVfzIsgWc149TexK6FPtvHGPxf_VQBdyA4Jl77dPp671wxH7puzy184L"
                    "V1IMdxq5eY1JG4QdCOluIDkkqbw_qYEiKGNXImGnp9LsXCcYa6JivYmSAw_vgZ5YhO98L49aLEI6nT3H"
                    "WRWuzY3H39aCkFdA8Y0DttzwXYvhz8UuenFgoGvIk0W7WSOLBxi6pSefdczpRizWLMm10zlrYVFvdMzU"
                    "PEx7OyDUhSXag61gYcsm6L5mcXGb2cojThdHEXcFZg36bQYjQVJzu_ASlvN-Z9AFNajIIyw8qKx3Q8pR"
                    "Vg3HCpQcSyRiDhicvT_Gp8d0X7VhoAhG_-LPg4AOz7U985JqP0wB2dB4sYhRgMZpQ29x7wgvgjNtsyqb"
                    "CA=="
                ),
                "summary": [],
            },
            {
                "id": "msg_0534b0c3898d6d1b006ad023bb52f881958af6ee45425a3b25",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": (
                            "I need `shop/validation.py` next to see how it handles zero, negative, and "
                            "non-integer quantities."
                        ),
                    }
                ],
                "phase": "commentary",
                "role": "assistant",
            },
            {
                "id": "fc_0534b0c3898d6d1b006ad023bb83a881958d95537bd9e16f3e",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":120}',
                "call_id": "call_G1yFCqI4OnAJbTANskb1ROdb",
                "name": "read_file",
            },
        ],
        "usage": {
            "input_tokens": 309,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 114,
            "output_tokens_details": {"reasoning_tokens": 50},
            "total_tokens": 423,
        },
    },
    expected_litellm_response={
        "id": "resp_0534b0c3898d6d1b006ad023ba700c8195b63f1819ab327ab3",
        "type": "message",
        "role": "assistant",
        "model": "azure/gpt-6-luna",
        "stop_sequence": None,
        "usage": {"input_tokens": 309, "output_tokens": 114},
        "content": [
            {
                "type": "text",
                "text": (
                    "I need `shop/validation.py` next to see how it handles zero, negative, and non-integer quantities."
                ),
            },
            {
                "type": "tool_use",
                "id": "call_G1yFCqI4OnAJbTANskb1ROdb",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 120},
            },
        ],
        "stop_reason": "tool_use",
        "stop_details": None,
    },
)

GPT_6_1_SOL_CONTENT_FILTERS: Final[JsonValue] = [
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
            "protected_material_text": {"detected": False, "filtered": False},
            "protected_material_code": {"detected": False, "filtered": False},
            "hate": {"filtered": False, "severity": "safe"},
            "sexual": {"filtered": False, "severity": "safe"},
            "violence": {"filtered": False, "severity": "safe"},
            "self_harm": {"filtered": False, "severity": "safe"},
        },
        "content_filter_offsets": {"start_offset": 0, "end_offset": 475, "check_offset": 0},
    },
]

GPT_6_1_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "azure/gpt-6.1-sol"},
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_headers={"content-type": "application/json", "api-key": "synthetic-azure-key"},
    expected_provider_request={
        **AZURE_RESPONSES_BRIDGE_REQUEST,
        "model": "gpt-6.1-sol",
        "instructions": MESSAGES_REQUEST["system"],
    },
    mock_provider_response={
        **AZURE_RESPONSES_RESPONSE,
        "id": "resp_0a4a896afe94ce65006ad023b047248193999d7c3a0c4fec39",
        "created_at": 1792025520,
        "completed_at": 1792025524,
        "content_filters": GPT_6_1_SOL_CONTENT_FILTERS,
        "instructions": MESSAGES_REQUEST["system"],
        "output": [
            {
                "id": "rs_0a4a896afe94ce65006ad023b19f8481938d51de793f83feed",
                "type": "reasoning",
                "content": [],
                "encrypted_content": (
                    "gAAAAABq0CO0QoINZ4oOVkvDvinFAJAMUBmzqqpXLoppdlyP4fFQsNqPOh-L08mQt8yf4OUtOwzk3E6Q"
                    "7CLHieco0s_tfuYHRlTU0Blr62KF_C77RScOjKYa6p4E2fjdwO44Y2HIiGfIo_GnUpkfwW1EiVZmj6L8"
                    "PO0biPqihvK3KJ-GNKB_R4-lLyKXrrvVN-985glSdHg-_fRrmyVGBvVxPFyERlD-WPm8fPujqMmXnUcA"
                    "v3N-wUoIEccd7PfJGqBs52UzllpH27mNCI71C_qbc2nZ97jZgSabTvAZKF8Q9zSJzWA_Gs72iQqd0X55"
                    "F5vD0-xrMuVdXFTYLV-y-S_1rN8tY1bEbdJWr7Dl84NvNWCYqA5byZgsMxAVyjADWXcVaagrZhNtQ1ol"
                    "II2oePJL2WyrKrOs4d7ZuGFW5BcFCDS5zxk6MwckKbLkSi3HM34z-AhIOjr6SNc5qcR8yur4Phm9pSJQ"
                    "qFscqQwtmEgbAzXqC9U5rwQ_c11N7o6Xc2bnoRXxeXQvKyGCPZWOqQr8AffHBCCOmeKjbPMV0ITek7Hy"
                    "RiOWg_Pjq-_xlVCoUGM7gIq5nq3HG1SqH7Yr6qev5Arq24UALiBbr0NAtmq3p-YGRgia3oDnSrnygIX8"
                    "AbqhnGrHz1kjBbwGr6fXF2XruSKuL96bQghfWGx6nhsbIDH4FI3akERb6Rh1B9XIWsWw4Zzo6-tg6Dnj"
                    "zfaJteZqIY4qS5mHSD9stXEVqu_CfnGh7EdOlZlnkG6kIQ3m6oAo2ZG9E58m32X-KEqZDjD4yUgWss9y"
                    "7cnFbYeEzntQneytoBpaTNXtSEsM1fUoz73Ol7FzqPI0CKIsp3UGje2K7LqAwkV54JrhqCfxkD5HmPJn"
                    "yEOnP0dODvyqD3CK-cbkoPoszI2Ba0dk11re5kF985he-Fjm7dXDNhQX93__nj4awDCtnX8_Ke5ld7O3"
                    "sA1ou7WHhgfGo-Jr4ZcGL5JWflPbI9oPpGOOvLsS-e8CUjTcTRb_eA2jApeGU2nAdW42U-CnXeScaR5e"
                    "JE1RoXTFXIP6MMSpxqpTRWYBfhIDyPEzwTveki-WOWgJtyNJ5W3GJxaEZla2uJ5_cdiD4-HgRy1UwxIJ"
                    "ODUh0F2hKC6niDVIZqoZ6Z4nb4OUvJP00_nGBA5-IL31Ij8I8U740RAZvnOsMOIsKelsJioM8Ii4MdPi"
                    "O2FYSdKjijEJmXLDdbYkQEXeqwfKGC7trXBOyRwO5AF6mLR9Yf8Bu_eiKbuf5DT9sw88P4JtfPi5LLIs"
                    "o2Ki44UnxePraO7vinqg7H7oOi-1gw7BbDaKH7tJs3vUOaxl6mA0BNTg5cDt6TtFZMpFgKNRyanWzSwG"
                    "WQGYlYbfUGVUnM_tjtgoBNFJAQz1s8fd-qOl1NQDggjlryl6TJk3RSiNYKyJ3fmpE10LM3LUtN3AoFMD"
                    "FbEEBstDuVg0zzvwe64zaKosn7PIh_Ym-y7QlMdcvUbsqO-T6pfsq8iqEz2anv3yujHkAXIaa3292lbA"
                    "dFIq80ycQSJbWNd2MtkYQHdWtWmQQoqMX1PdyJGp6N3GG45A4A=="
                ),
                "summary": [],
            },
            {
                "id": "msg_0a4a896afe94ce65006ad023b2cc3081939533c6e664038345",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": (
                            "I’ll read `shop/validation.py` first to check how it handles zero, negative, and "
                            "non-integer quantities.\n"
                        ),
                    }
                ],
                "phase": "commentary",
                "role": "assistant",
            },
            {
                "id": "fc_0a4a896afe94ce65006ad023b3229881938388d9bef2b4a8ef",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                "call_id": "call_5drWfeHZt2S3rfVa9SYefgB1",
                "name": "read_file",
            },
        ],
        "usage": {
            "input_tokens": 309,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 137,
            "output_tokens_details": {"reasoning_tokens": 72},
            "total_tokens": 446,
        },
    },
    expected_litellm_response={
        "id": "resp_0a4a896afe94ce65006ad023b047248193999d7c3a0c4fec39",
        "type": "message",
        "role": "assistant",
        "model": "azure/gpt-6.1-sol",
        "stop_sequence": None,
        "usage": {"input_tokens": 309, "output_tokens": 137},
        "content": [
            {
                "type": "text",
                "text": (
                    "I’ll read `shop/validation.py` first to check how it handles zero, negative, and "
                    "non-integer quantities.\n"
                ),
            },
            {
                "type": "tool_use",
                "id": "call_5drWfeHZt2S3rfVa9SYefgB1",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 200},
            },
        ],
        "stop_reason": "tool_use",
        "stop_details": None,
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
    }
]

GPT_6_ASTRA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "azure/gpt-6-astra"},
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_headers={"content-type": "application/json", "api-key": "synthetic-azure-key"},
    expected_provider_request={
        **AZURE_RESPONSES_BRIDGE_REQUEST,
        "model": "gpt-6-astra",
        "instructions": MESSAGES_REQUEST["system"],
    },
    mock_provider_response={
        **AZURE_RESPONSES_RESPONSE,
        "id": "resp_019af305501e4822006ad023c3a50881958c4cf40623699cf8",
        "created_at": 1792025539,
        "completed_at": 1792025543,
        "content_filters": GPT_6_ASTRA_CONTENT_FILTERS,
        "instructions": MESSAGES_REQUEST["system"],
        "model": "gpt-6-astra",
        "output": [
            {
                "id": "rs_019af305501e4822006ad023c49ce4819590da413b9879e06b",
                "type": "reasoning",
                "content": [],
                "encrypted_content": (
                    "gAAAAABq0CPHdct5jJ3BolLC7QmXutcBsSZ0YzHiI8z2uK07jqwLifVhEROdF12ygdCVRtcrLrHPyHSh"
                    "pVbh41ULxn2L3SyVYLHbbkFMcwbzfky_kqZdUe_1D-PLxsVyI1KFMPLg0Fb9K-N9u_ElvkFcXXsmAhRQ"
                    "CcB4SE9TMe_5tKBHiZbqvOI112wig7cc57rDbvM0mEoKyFn3WgprP_ewHeCeP6DSEvVGDEMjUd_TIXAH"
                    "H7y2Ae3y0WRW7WUag_wAiO7xzAH13mh4S6eKeBTKh62krmYeUH3wETO8BF1WnkuZ_86lS0zWLBzeLJJn"
                    "JiXKXVquKatJ6HvO3m6FkPF0tBXhdc8YYtTF1B_WBMECARKcw4n3mPL3R6J2jpHe05Eo7W7uZkcY5E-q"
                    "mE_cqUK8YQXTXx-mR339iuAvqSe0N2Rgl9Ey1l7z0MBATIV0c6J4wRav6Di5aAxEEXkgDVk1jF9LZSHo"
                    "mcGjh2B2UWFiAZEELRDD7ovpP3c7B05h0QvwazXlZnskJf-xYZEHAl2CNZk0iqagFs6EtNrFguM5W2AH"
                    "6dsuaGjNGmdWM_kt4JqIUeScZLNgMty8qTOHFzd7tiBpuLkQgcnHzOigWb2vY_8ahy_zfzM2lVL-JQxP"
                    "9SB7Gq1m49Ln-byC1O_kPyqEFXELhhQ8peXQY5do2R2T-3I0WLOhDhP5rruLsL66vNHAAMjka6WmjI2g"
                    "oiy-JF_nVCplJd791Wqa3yf7Q7ksfasnSKtjR7-cgTtMcXX7n_5joMnqf4_LsqSq6gHdeKw4VTeT8wGB"
                    "Uo3MHLkgi8PnAIphdb3PKgifSjstE_oH4_ZbmlyCsHaFYq2aoy-S85ndlrR_PAHBtU8II-Ud_42EPzS-"
                    "Etdt5d1Qjjri3KTyPjPO7fAm8aPFr1k3GbjwASPMQ7cUs7yLhPmI-TCm5U-zgK1pIALviiU2o4IM7kdH"
                    "0-Dg1eyyZN8hsFGSHaUB7G_tGB0fj0H0GSztH8ltNjTK_276IOh8Jv5csxjjmlcMOHVBmDCkkE_DlVjv"
                    "NQtWQS9QBt9jd1gFSzaOhXVkKZJK3vTIaq8JxTvYtgLc8WM5pRHbnQr71TfWB-eLEYgN-aX5PueKkXGr"
                    "GauDpKkzVJyNZ6mZ_dVXwl1p5IiNNJ1xzhmZK2GVDFWt0nfOBYyxmCDfIxvQL8YAhXXAi1Hrjal82c68"
                    "lw7UBauyMojDOcbH7EDG9KNbVV-kmkhCW3aH0QCIALzHqO_cukXDqnYX-iVNuxWpW5KUAJdg9cfgwX35"
                    "O246HxDoSpoIYQ1GooqIWSImYrlNQajPHpJcpxnWeSgKOSYIAYCAr70MouYhUg-lsId7n2pr4gspPf1Z"
                    "UrBhAG3h4n9qSguKTlpnAYlln_2MzgVVN1cH5sopvgJJLr20HKS1cfu9lB-CHDF5h74z7q3ydZJljn8d"
                    "QLBqL3csr0ovOZkTjP21rOjPhdsedg4CLKnylgk5nTewbQz99lKxKPhfYK0-XxiDdE6aKvobQUoRupGN"
                    "Hv-gMLqJjehMUynaC6XzBwJ63pPNhH3AlCj3zSte0Mj1FkPQQIY2PjhMB49CmXBV8Z9YJv0="
                ),
                "summary": [],
            },
            {
                "id": "msg_019af305501e4822006ad023c5f73c819597a30ebb95c43472",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": (
                            "I’ll read `shop/validation.py` first to check how it handles zero, negative "
                            "values, and non-integer quantities.\n"
                        ),
                    }
                ],
                "phase": "commentary",
                "role": "assistant",
            },
            {
                "id": "fc_019af305501e4822006ad023c65c408195b9fa8c35e75b1ebd",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                "call_id": "call_zAt0ZCuKW07iCDDK2RlX7lAX",
                "name": "read_file",
            },
        ],
        "usage": {
            "input_tokens": 309,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 138,
            "output_tokens_details": {"reasoning_tokens": 72},
            "total_tokens": 447,
        },
    },
    expected_litellm_response={
        "id": "resp_019af305501e4822006ad023c3a50881958c4cf40623699cf8",
        "type": "message",
        "role": "assistant",
        "model": "azure/gpt-6-astra",
        "stop_sequence": None,
        "usage": {"input_tokens": 309, "output_tokens": 138},
        "content": [
            {
                "type": "text",
                "text": (
                    "I’ll read `shop/validation.py` first to check how it handles zero, negative "
                    "values, and non-integer quantities.\n"
                ),
            },
            {
                "type": "tool_use",
                "id": "call_zAt0ZCuKW07iCDDK2RlX7lAX",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 200},
            },
        ],
        "stop_reason": "tool_use",
        "stop_details": None,
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
        "content_filter_offsets": {"start_offset": 0, "end_offset": 313, "check_offset": 0},
    },
]

GPT_5_6_TERRA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "azure/gpt-5.6-terra"},
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_headers={"content-type": "application/json", "api-key": "synthetic-azure-key"},
    expected_provider_request={
        **AZURE_RESPONSES_BRIDGE_REQUEST,
        "model": "gpt-5.6-terra",
        "instructions": MESSAGES_REQUEST["system"],
    },
    mock_provider_response={
        **AZURE_RESPONSES_RESPONSE,
        "id": "resp_0d178c5d2401a624006ad023cf1c588194ab41d5803ed29401",
        "created_at": 1792025551,
        "completed_at": 1792025552,
        "content_filters": GPT_5_6_TERRA_CONTENT_FILTERS,
        "instructions": MESSAGES_REQUEST["system"],
        "model": "gpt-5.6-terra",
        "output": [
            {
                "id": "rs_0d178c5d2401a624006ad023cfe304819486bdf040ec55eab7",
                "type": "reasoning",
                "content": [],
                "encrypted_content": (
                    "gAAAAABq0CPQDJzIdOV-2rO0aGYCUhKcJifPRv4ZkBaIrKZEARo0rk8CqAe8IV0c2kyBwOFe4M70GMM-"
                    "HrR1L3hIUxan9dcDB5_S7e45yDaxFMiKpbr20izrXNktdwBQfvYOKoMMd07fslVRR-GHOEtz8JvU98v5"
                    "81sxXgVTxNAWGSl_Kb22g2Fds02t1-1g2hT304vsi1sBxbA6P2b-NFNQ8N3NEfX6FoWi7UF0nH69-Dg3"
                    "sCMb4dpZ0gGZaZSHwvzXXHESEuiMv1nJSX18tfrFIt29gmw3Mq3sUxZ3eALkPZ54yDUlfYXl6sm2MHXT"
                    "2xn3l5x6jlG4X8hEqgdfHI76otxmfsXvgtbXCADCes87f00bkWatqWgq27oluRYsY02xDMhtrT4h4C4s"
                    "PIpV7PA9tdhS6Lo3znsQiOEQvMqnSS3dH1NKnZsAiyjNwVZVZa0opIzI5fpNCGjBmMv_4hB3jBRgenG-"
                    "Ku75OcwPp6HXMej9McvpLIKtXb8FMKS5wqiZzYBLp-F25hzOHI0QVl0pGFL5ya9mWU4hkQYeu6HuH6Zj"
                    "Sujzyy-8ZmlBjnRZSQp8mN4t34u2ofk4RtyuZKJkgjJfN2m5MxBTT3AccMalmbiEa7lJv7GV9kLcqR0G"
                    "T2Rzd6UMX6qPG92VdfjIgCj1Rzh1IWrJvMloKkMj5qB3nN2dABqfCtjBleStNoTjO2VDensXf11ceG9-"
                    "KKWeCjfrG0ONgTVXPjTTUPHR0BxtoixnRxIYBZFb8sKdYAPI_QINjby2SVhhQH7UtookBf7N7OqmIJKb"
                    "zR4CLshKnvthH4fJQAZdsfoEPR9I943uBpZi4HnF8mQPF61Aiw9B3bh3USBc9_Oh2rJL044B6qYVD3QK"
                    "KGydzcplRxQWA31m-4Zn9eVxn-xOtx4Tvpdh6BhlFPKX9p2W0mSnb7kjpzX51Ec0J2AsTNUOonaaMFt8"
                    "2MJDELTGcAb3tsAkPzzNK0M_zjQ43CCH5Cu1yV2WhbUzxCu4UM0spAIuzpAZv6JGmW1wF4RtAgLfJjc2"
                    "LxoxcS9MJbLOUPAYdRlwBTj_IL9PIDeNlGVRgVQYL-9xnNKiWD8Iyu2vj8AoslM41wVTEXvOAYyEs25E"
                    "yo9ia2km8909d42DjFi2_ngA5ofZI89KQHtCzGwfCf_VTLbpMmrOB4NDFVTUYQlAMBvuSsXLXJJp0QPl"
                    "n2UBHhZ3erIcQ1D990SFRdx1uY47-E2FnCCPljyhJrWHnFxTlGbuEY4i5vPOweqYzj-uBbLVeDbcfjCL"
                    "jSLFgM4Fa5Zg"
                ),
                "summary": [],
            },
            {
                "id": "msg_0d178c5d2401a624006ad023d011388194a221b5ec1cb6e8bd",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": (
                            "I need `shop/validation.py` next to verify the current behavior for zero, "
                            "negative, and non-integer quantities before reviewing the corresponding tests."
                        ),
                    }
                ],
                "phase": "commentary",
                "role": "assistant",
            },
            {
                "id": "fc_0d178c5d2401a624006ad023d04f708194bb3ccd55f7afb1d0",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                "call_id": "call_xFsX3SKb9MppLeAL2L0aGZ8J",
                "name": "read_file",
            },
        ],
        "usage": {
            "input_tokens": 309,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 92,
            "output_tokens_details": {"reasoning_tokens": 22},
            "total_tokens": 401,
        },
    },
    expected_litellm_response={
        "id": "resp_0d178c5d2401a624006ad023cf1c588194ab41d5803ed29401",
        "type": "message",
        "role": "assistant",
        "model": "azure/gpt-5.6-terra",
        "stop_sequence": None,
        "usage": {"input_tokens": 309, "output_tokens": 92},
        "content": [
            {
                "type": "text",
                "text": (
                    "I need `shop/validation.py` next to verify the current behavior for zero, "
                    "negative, and non-integer quantities before reviewing the corresponding tests."
                ),
            },
            {
                "type": "tool_use",
                "id": "call_xFsX3SKb9MppLeAL2L0aGZ8J",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 200},
            },
        ],
        "stop_reason": "tool_use",
        "stop_details": None,
    },
)
