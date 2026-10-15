from typing import Final
from unittest.mock import ANY

from integration.translation.case import TranslationTestCase
from integration.translation.responses.templates import RESPONSES_NATIVE_RESPONSE, RESPONSES_REQUEST
from integration.translation.templates.azure import AZURE_RESPONSES_REQUEST, AZURE_RESPONSES_RESPONSE
from pydantic import JsonValue

GPT_5_4_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "azure/gpt-5.4",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_headers={"content-type": "application/json", "api-key": "synthetic-azure-key"},
    expected_provider_request={
        "model": "gpt-5.4",
        "input": "Say hello.",
        "instructions": "You are a terse assistant.",
        "max_output_tokens": 64,
    },
    mock_provider_response={
        "id": "resp_0762039bed856005006ac4076cdb5c8194a48664a1b2ef25ae",
        "object": "response",
        "created_at": 1791231852,
        "status": "completed",
        "background": False,
        "completed_at": 1791231853,
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
                "content_filter_offsets": {"start_offset": 0, "end_offset": 842, "check_offset": 0},
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
                "content_filter_offsets": {"start_offset": 0, "end_offset": 6, "check_offset": 0},
            },
        ],
        "error": None,
        "frequency_penalty": 0.0,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "max_output_tokens": 64,
        "max_tool_calls": None,
        "model": "gpt-5.4",
        "moderation": None,
        "output": [
            {
                "id": "msg_0762039bed856005006ac4076d77f08194b260c944a432ba1a",
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": "Hello."}],
                "phase": "final_answer",
                "role": "assistant",
            }
        ],
        "parallel_tool_calls": True,
        "presence_penalty": 0.0,
        "previous_response_id": None,
        "prompt_cache_key": None,
        "prompt_cache_retention": "in_memory",
        "reasoning": {"context": "current_turn", "effort": "none", "mode": "standard", "summary": None},
        "safety_identifier": None,
        "service_tier": "default",
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "auto",
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
        "tools": [],
        "top_logprobs": 0,
        "top_p": 0.98,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 19,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 25,
        },
        "user": None,
        "metadata": {},
    },
    expected_litellm_response={
        "id": ANY,
        "created_at": 1791231852,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "azure/gpt-5.4",
        "object": "response",
        "output": [
            {
                "id": "msg_0762039bed856005006ac4076d77f08194b260c944a432ba1a",
                "content": [{"annotations": [], "text": "Hello.", "type": "output_text", "logprobs": []}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
                "phase": "final_answer",
            }
        ],
        "parallel_tool_calls": True,
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 0.98,
        "max_output_tokens": 64,
        "previous_response_id": None,
        "reasoning": {"context": "current_turn", "effort": "none", "mode": "standard", "summary": None},
        "status": "completed",
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "truncation": "disabled",
        "usage": {
            "input_tokens": 19,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": None,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 6,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": None},
            "total_tokens": 25,
            "cost": None,
        },
        "user": None,
        "store": True,
        "background": False,
        "completed_at": 1791231853,
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
                "content_filter_offsets": {"start_offset": 0, "end_offset": 842, "check_offset": 0},
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
                "content_filter_offsets": {"start_offset": 0, "end_offset": 6, "check_offset": 0},
            },
        ],
        "frequency_penalty": 0.0,
        "max_tool_calls": None,
        "moderation": None,
        "presence_penalty": 0.0,
        "prompt_cache_key": None,
        "prompt_cache_retention": "in_memory",
        "safety_identifier": None,
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
        "top_logprobs": 0,
    },
)

GPT_5_6_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "azure/gpt-5.6-sol",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_headers={"content-type": "application/json", "api-key": "synthetic-azure-key"},
    expected_provider_request={
        "model": "gpt-5.6-sol",
        "input": "Say hello.",
        "instructions": "You are a terse assistant.",
        "max_output_tokens": 64,
    },
    mock_provider_response={
        "id": "resp_08c09f398a5c1990006ac4076e8c5c819498395510175e9c98",
        "object": "response",
        "created_at": 1791231854,
        "status": "completed",
        "background": False,
        "completed_at": 1791231857,
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
                "content_filter_offsets": {"start_offset": 0, "end_offset": 788, "check_offset": 0},
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
                "content_filter_offsets": {"start_offset": 0, "end_offset": 6, "check_offset": 0},
            },
        ],
        "error": None,
        "frequency_penalty": 0.0,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "max_output_tokens": 64,
        "max_tool_calls": None,
        "model": "gpt-5.6-sol",
        "moderation": None,
        "output": [
            {
                "id": "msg_08c09f398a5c1990006ac4077114e48194a52f30a20676db99",
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": "Hello!"}],
                "phase": "final_answer",
                "role": "assistant",
            }
        ],
        "parallel_tool_calls": True,
        "presence_penalty": 0.0,
        "previous_response_id": None,
        "prompt_cache_key": None,
        "prompt_cache_retention": "24h",
        "reasoning": {"context": "all_turns", "effort": "medium", "mode": "standard", "summary": None},
        "safety_identifier": None,
        "service_tier": "default",
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "auto",
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
        "tools": [],
        "top_logprobs": 0,
        "top_p": 0.98,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 19,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 25,
        },
        "user": None,
        "metadata": {},
    },
    expected_litellm_response={
        "id": ANY,
        "created_at": 1791231854,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "azure/gpt-5.6-sol",
        "object": "response",
        "output": [
            {
                "id": "msg_08c09f398a5c1990006ac4077114e48194a52f30a20676db99",
                "content": [{"annotations": [], "text": "Hello!", "type": "output_text", "logprobs": []}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
                "phase": "final_answer",
            }
        ],
        "parallel_tool_calls": True,
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 0.98,
        "max_output_tokens": 64,
        "previous_response_id": None,
        "reasoning": {"context": "all_turns", "effort": "medium", "mode": "standard", "summary": None},
        "status": "completed",
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "truncation": "disabled",
        "usage": {
            "input_tokens": 19,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": None,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 6,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": None},
            "total_tokens": 25,
            "cost": None,
        },
        "user": None,
        "store": True,
        "background": False,
        "completed_at": 1791231857,
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
                "content_filter_offsets": {"start_offset": 0, "end_offset": 788, "check_offset": 0},
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
                "content_filter_offsets": {"start_offset": 0, "end_offset": 6, "check_offset": 0},
            },
        ],
        "frequency_penalty": 0.0,
        "max_tool_calls": None,
        "moderation": None,
        "presence_penalty": 0.0,
        "prompt_cache_key": None,
        "prompt_cache_retention": "24h",
        "safety_identifier": None,
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
        "top_logprobs": 0,
    },
)

GPT_5_6_LUNA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "azure/gpt-5.6-luna",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_headers={"content-type": "application/json", "api-key": "synthetic-azure-key"},
    expected_provider_request={
        "model": "gpt-5.6-luna",
        "input": "Say hello.",
        "instructions": "You are a terse assistant.",
        "max_output_tokens": 64,
    },
    mock_provider_response={
        "id": "resp_00c2ebc008a733e5006ac40772599081938a97e3688d1ff7f9",
        "object": "response",
        "created_at": 1791231858,
        "status": "completed",
        "background": False,
        "completed_at": 1791231859,
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
                "content_filter_offsets": {"start_offset": 0, "end_offset": 788, "check_offset": 0},
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
                "content_filter_offsets": {"start_offset": 0, "end_offset": 6, "check_offset": 0},
            },
        ],
        "error": None,
        "frequency_penalty": 0.0,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "max_output_tokens": 64,
        "max_tool_calls": None,
        "model": "gpt-5.6-luna",
        "moderation": None,
        "output": [
            {
                "id": "msg_00c2ebc008a733e5006ac40772ae908193b01d2c647918c8d4",
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": "Hello!"}],
                "phase": "final_answer",
                "role": "assistant",
            }
        ],
        "parallel_tool_calls": True,
        "presence_penalty": 0.0,
        "previous_response_id": None,
        "prompt_cache_key": None,
        "prompt_cache_retention": "24h",
        "reasoning": {"context": "all_turns", "effort": "medium", "mode": "standard", "summary": None},
        "safety_identifier": None,
        "service_tier": "default",
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "auto",
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
        "tools": [],
        "top_logprobs": 0,
        "top_p": 0.98,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 19,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 25,
        },
        "user": None,
        "metadata": {},
    },
    expected_litellm_response={
        "id": ANY,
        "created_at": 1791231858,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "azure/gpt-5.6-luna",
        "object": "response",
        "output": [
            {
                "id": "msg_00c2ebc008a733e5006ac40772ae908193b01d2c647918c8d4",
                "content": [{"annotations": [], "text": "Hello!", "type": "output_text", "logprobs": []}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
                "phase": "final_answer",
            }
        ],
        "parallel_tool_calls": True,
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 0.98,
        "max_output_tokens": 64,
        "previous_response_id": None,
        "reasoning": {"context": "all_turns", "effort": "medium", "mode": "standard", "summary": None},
        "status": "completed",
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "truncation": "disabled",
        "usage": {
            "input_tokens": 19,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": None,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 6,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": None},
            "total_tokens": 25,
            "cost": None,
        },
        "user": None,
        "store": True,
        "background": False,
        "completed_at": 1791231859,
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
                "content_filter_offsets": {"start_offset": 0, "end_offset": 788, "check_offset": 0},
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
                "content_filter_offsets": {"start_offset": 0, "end_offset": 6, "check_offset": 0},
            },
        ],
        "frequency_penalty": 0.0,
        "max_tool_calls": None,
        "moderation": None,
        "presence_penalty": 0.0,
        "prompt_cache_key": None,
        "prompt_cache_retention": "24h",
        "safety_identifier": None,
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
        "top_logprobs": 0,
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
        "content_filter_offsets": {"start_offset": 0, "end_offset": 384, "check_offset": 0},
    },
]

GPT_6_LUNA_ENCRYPTED_REASONING: Final = (
    "gAAAAABq0CO_lJPFbtrP8DdqMTwl2xlTnCuiBzPQBDrYhcy-xADgVDfRLfYGRE5C1tnchYKxhOh_UwHR"
    "p7D0wxyvh3I3hy6iRIr5phgwGqRcJzzeuDaj63bR6wUNAN1iW4EBIamL6eAo3pg-4bzAkACoCbGT7Uwp"
    "73E83yafb8_vE9QeJ0lSd5faC7-H5zFe9LDgNywIUNp70Ni1d5v6dAF1HdloQhCkqW0cbDzG0WeeT4jT"
    "Yq5yVbiTM_c5PIt3ijbA6OVxbGF5cmtUQb4zOUlR9omxvKRniASAyu9qg201lkwONzoJ7S4rOqROt-1O"
    "o8E0OaC1HTxBPjbqSFT4KLH2DkVN76JG5Mkh8UPHySKMKkTmyCPZMwFFjm4-RGgqPQ82KrpfFd9Kwwwy"
    "G7ZgIIDG-ckYRW1JvZbew_tsi7zEfOUU7qxeyENyfHbRhe98j61WgLoNuct3ywCkDqTMFe-6YY_AAG8Q"
    "FuSc2zfGj6F4UWf1eYiBXm5vzFYgGJLpR1dx8Lthv9gLSKXwyjUDLPev1wkW9QwRGZ0VKUz8r2YC4rT0"
    "PwexJzEXoXmWB6unWPL23bKzUsUDvDal1ZYrB5-CnkEFANPxosqZAjeRYtmUj9dbix7uF6b9aeL41fqC"
    "RXZU1RKyCp-quWtnAB8E-Uz4gLAMOeLT4-C1kwDP8YMhXS9WLrQ3SYk3YRfcMb2LRPIc2-KkMCKaFKX5"
    "_Sov89DzCtGW4VHBqayHzwaKq_qrbnVY5jFOrm-N_A_1Pdrqa4f-oh6TMKlAhFuuUJCKWSMOiydGwtC2"
    "jfPgyJVWWST5JyHOlZcmOq4gHDM57QGHlvqDKz7_bN5SChSAmCnVdgervkHw8HKj_VODkEEwA2U-SGLD"
    "SoJykHAMhsGjx_BoHxddVUi6p8qJBP9XWa52WpFO8eatHoZBjJ9WW7IF2ztljVqlUeKoEsSVUwkFo2wt"
    "i4mhjqjZiaYz6jOz6gYRkJd80UifNe74xisW_IVXTcgT8gXdcf8CVygNjScDDhdzgjUhOhxCjOFNWLc7"
    "01FbRToU2qNg3aQQaMdgOPmczA7HeLghBW7IGZmnYCweoFliXXXo08jSXPlXDt1nhzhjm71or81Hk5PM"
    "iQj4-DH3aY-a1umCDChEGrMNSZAZTqGBiLwW_W8h0OtFkZ_MHqxRl9fyKYORKQAjlbhakN1Dze2106Uy"
    "1WcC4snJwpwxMhd2leREckRD2YpCXDN-WvYwRrDXETUqTJ7-ZLE2gQ8MNK3czzWceHql0pvySKlp9QDo"
    "D_NWTywPag5NlfFE74BivE6oc373ZKOPFcwcoRdFS8boNuIkK8V7HDtW02Wet76ucQ-n4cWhWyz-k_64"
    "8WY1OBfrTyh4Rup4L2BOCGEjC63YyGMjynnEV0IZ1tfZI19jxxz9ctDjeShUf9gPNEu2CxS8ZQ0Q4NPk"
    "8g=="
)

GPT_6_LUNA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "azure/gpt-6-luna"},
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_headers={"content-type": "application/json", "api-key": "synthetic-azure-key"},
    expected_provider_request={**AZURE_RESPONSES_REQUEST, "model": "gpt-6-luna"},
    mock_provider_response={
        **AZURE_RESPONSES_RESPONSE,
        "id": "resp_0c7fa1cbc998fdea006ad023bcd92c81978c48bfd696bb8c5b",
        "created_at": 1792025532,
        "completed_at": 1792025535,
        "content_filters": GPT_6_LUNA_CONTENT_FILTERS,
        "instructions": RESPONSES_REQUEST["instructions"],
        "model": "gpt-6-luna",
        "output": [
            {
                "id": "rs_0c7fa1cbc998fdea006ad023bdb5288197913fbdc89efaee0f",
                "type": "reasoning",
                "content": [],
                "encrypted_content": GPT_6_LUNA_ENCRYPTED_REASONING,
                "summary": [],
            },
            {
                "id": "msg_0c7fa1cbc998fdea006ad023bdf4b08197a483204dce142776",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": (
                            "The next file to inspect is `shop/validation.py`, to see how it handles zero and "
                            "negative quantities."
                        ),
                    }
                ],
                "phase": "commentary",
                "role": "assistant",
            },
            {
                "id": "fc_0c7fa1cbc998fdea006ad023be1b6c8197ac0d31ccff25b0d4",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":120}',
                "call_id": "call_cODuVBuAwDhnb8M0FPGYyS1f",
                "name": "read_file",
            },
        ],
        "tools": [
            {
                "type": "function",
                "description": "Search source and test files for a literal term.",
                "name": "search_files",
                "output_schema": None,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "paths": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["query", "paths"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "description": "Read a range of lines from a project file.",
                "name": "read_file",
                "output_schema": None,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path", "start_line", "end_line"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ],
        "usage": {
            "input_tokens": 309,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 113,
            "output_tokens_details": {"reasoning_tokens": 51},
            "total_tokens": 422,
        },
    },
    expected_litellm_response={
        **RESPONSES_NATIVE_RESPONSE,
        "id": ANY,
        "created_at": 1792025532,
        "model": "azure/gpt-6-luna",
        "output": [
            {
                "id": "rs_0c7fa1cbc998fdea006ad023bdb5288197913fbdc89efaee0f",
                "summary": [],
                "type": "reasoning",
                "content": [],
                "encrypted_content": GPT_6_LUNA_ENCRYPTED_REASONING,
            },
            {
                "id": "msg_0c7fa1cbc998fdea006ad023bdf4b08197a483204dce142776",
                "content": [
                    {
                        "annotations": [],
                        "text": (
                            "The next file to inspect is `shop/validation.py`, to see how it handles zero and "
                            "negative quantities."
                        ),
                        "type": "output_text",
                        "logprobs": [],
                    }
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
                "phase": "commentary",
            },
            {
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":120}',
                "call_id": "call_cODuVBuAwDhnb8M0FPGYyS1f",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_0c7fa1cbc998fdea006ad023be1b6c8197ac0d31ccff25b0d4",
                "namespace": None,
                "status": "completed",
            },
        ],
        "parallel_tool_calls": True,
        "temperature": 1.0,
        "tools": [
            {
                "name": "search_files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "paths": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["query", "paths"],
                    "additionalProperties": False,
                },
                "strict": True,
                "type": "function",
                "defer_loading": None,
                "description": "Search source and test files for a literal term.",
                "output_schema": None,
            },
            {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path", "start_line", "end_line"],
                    "additionalProperties": False,
                },
                "strict": True,
                "type": "function",
                "defer_loading": None,
                "description": "Read a range of lines from a project file.",
                "output_schema": None,
            },
        ],
        "top_p": 0.98,
        "reasoning": {"context": "all_turns", "effort": "high", "mode": "standard", "summary": None},
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "truncation": "disabled",
        "usage": {
            "input_tokens": 309,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": None,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 113,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 51, "text_tokens": None},
            "total_tokens": 422,
            "cost": None,
        },
        "store": True,
        "background": False,
        "completed_at": 1792025535,
        "content_filters": GPT_6_LUNA_CONTENT_FILTERS,
        "frequency_penalty": 0.0,
        "max_tool_calls": None,
        "moderation": None,
        "presence_penalty": 0.0,
        "prompt_cache_key": None,
        "prompt_cache_retention": "24h",
        "safety_identifier": None,
        "service_tier": "default",
        "tool_usage": AZURE_RESPONSES_RESPONSE["tool_usage"],
        "top_logprobs": 0,
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
        "content_filter_offsets": {"start_offset": 0, "end_offset": 519, "check_offset": 0},
    },
]

GPT_6_1_SOL_ENCRYPTED_REASONING: Final = (
    "gAAAAABq0CO37PXMD2-KW8EHtGczorILYppFKOeL-058QC8QCnKldc08rJXqRj2qhEP8qBLUBQLaB-Ku"
    "bhgoZVZsXiobJCVijCX5SDQcINWwFg_SlMsL2pAHEymUIsT99rxULm11jvG8vfVEsBhBMjhkB-8wCua9"
    "q59iXdcK68bVpMhl5BDFsbf1EPIZTY6WFG79yjvXGWqGjsGbCY_EIa8ZsexuLSuZALT1oMI-4KAGDIU9"
    "F29ckEi3rV40L1TUZxG_MtrFGBE3d1DKSe0vz5QjJiMaBzZ8eu4kxfjAX5vwfoRx59zNRfjEjgx1s_eF"
    "4kCBSZn61pC1ME_yY_omKF4baWkKePFysGvv_qbjE_IH5DSVZN0_WNCNx8ISHtr72p2Niy0vZidXoZVg"
    "ByGXlQam3PtAZeRts1wR0VTtb1-xiq8E2hqRXIPmOgBwH35l72tahRBbhtogi_cFxKF-FWYMJOoZQ_dG"
    "_gZEEw1vJrepfOZNQBDq62o2h4BI24w5hf85sETMJwaT_3WiLWd0ukdIxK1bLQpwl4_ZEGmkMZqtwCIO"
    "iZSRfVTsz8GKcAagqEXa1N0Lz-4ejcy1r6aSaNVdXebzgLySS9wWjlF80xV29HgkUQY1kzJA0VwVmf6m"
    "neFdZ9O5yhg3acy_MzALDXbdZ_D4aCO9gtGp06eoO40Cp6fj5CFMrrj4CeC0oV0RrYF3cSb_fSTaKnbH"
    "VEbvlMEfuraBBUWTjjYGu7cXQcj7O43V4x4FFM1j_0Lf2i5w5hrHB9BQy5TiySvlmIeDxfXP-wadoxST"
    "Gdkek8u6Gu-jivpLKgNDQk7Od-E-V2Hx_xdHx3koV-BfQaNpAzSNIJ9gYpbeySEFA29XXIzpk3QAreHa"
    "Z6Ta3yEn717erYIM3QofAOimDyhOUaqFAP4OKD8ljPGa0XAHjJ6iKd3X3oN0ir8072MN5LviK2p3cuwf"
    "y7xd_rChJ5KIyXZMa7ZXis-R1a8tUmjVUITS4lRFRmhGgeq5yAwIXSEzUxY-EWPPm2AvC_qVedS9et_N"
    "2Bb_QIkmPPOvjbvDce01Owhn9O_n0kRhqlvv72qWqr12Up6eugoazsh05Hvf0BBkPboDEhi23zy0FeN5"
    "EaGIJB7D3qe_3awNuq6_2oPVFLT0oQeQegAVyl2Hz5U10mgu_elaX32MV3_54U9o2d2kky17NvAUWrdy"
    "GWkerdLWtI3GFjTt5Uk6jIQOhy00PoaqZtv4_XjH8HADN6NcbGikIowF5KslciZKeFZOURaaOtsrwmgF"
    "eWlwFaV0JVWrA1Ob25Io7rH1v3pCZFf8tCONGDQGlOmoJ_MLegn4HulHcZ5yGpny36P5aPiU8JnJOnpc"
    "xdow0XyjHqS4DFQ9GoQg2IAKWRIk9c8-_PE4nH9OVB_L-3LpZGGWT0gKREmfVmyb2jENTCG6PRFPGgEP"
    "9D4GFJTlOo4cSobLe2rKEWl0F-8WxQWCX1CSxq2-Ik3pe08HcDj3hB4JtGVLW-5RGwtvaxfp22m4-aHi"
    "oVOPhErqa3wvYgv9i4IE4WJ3yjE3rgEHQ3CuqD9bqskSh2WHqJwfX6wb-2etzdFiZPix74VGlaMZR8Hr"
    "Xh8IEHUVj_uL"
)

GPT_6_1_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "azure/gpt-6.1-sol"},
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_headers={"content-type": "application/json", "api-key": "synthetic-azure-key"},
    expected_provider_request={**AZURE_RESPONSES_REQUEST, "model": "gpt-6.1-sol"},
    mock_provider_response={
        **AZURE_RESPONSES_RESPONSE,
        "id": "resp_0895a553a79167f3006ad023b495cc81978a0ed0ada7341347",
        "created_at": 1792025524,
        "completed_at": 1792025527,
        "content_filters": GPT_6_1_SOL_CONTENT_FILTERS,
        "instructions": RESPONSES_REQUEST["instructions"],
        "output": [
            {
                "id": "rs_0895a553a79167f3006ad023b5d6348197ac3ab4cda17f113c",
                "type": "reasoning",
                "content": [],
                "encrypted_content": GPT_6_1_SOL_ENCRYPTED_REASONING,
                "summary": [],
            },
            {
                "id": "msg_0895a553a79167f3006ad023b6ca3c81978c47c3a9216638f6",
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
                "id": "fc_0895a553a79167f3006ad023b71d648197afe5314dfd20909f",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                "call_id": "call_h9vdLDpm7EuHrkHXzdKY0Yb2",
                "name": "read_file",
            },
        ],
        "tools": [
            {
                "type": "function",
                "description": "Search source and test files for a literal term.",
                "name": "search_files",
                "output_schema": None,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "paths": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["query", "paths"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "description": "Read a range of lines from a project file.",
                "name": "read_file",
                "output_schema": None,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path", "start_line", "end_line"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ],
        "usage": {
            "input_tokens": 309,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 140,
            "output_tokens_details": {"reasoning_tokens": 75},
            "total_tokens": 449,
        },
    },
    expected_litellm_response={
        **RESPONSES_NATIVE_RESPONSE,
        "id": ANY,
        "created_at": 1792025524,
        "model": "azure/gpt-6.1-sol",
        "output": [
            {
                "id": "rs_0895a553a79167f3006ad023b5d6348197ac3ab4cda17f113c",
                "summary": [],
                "type": "reasoning",
                "content": [],
                "encrypted_content": GPT_6_1_SOL_ENCRYPTED_REASONING,
            },
            {
                "id": "msg_0895a553a79167f3006ad023b6ca3c81978c47c3a9216638f6",
                "content": [
                    {
                        "annotations": [],
                        "text": (
                            "I’ll read `shop/validation.py` first to check how it handles zero, negative, and "
                            "non-integer quantities.\n"
                        ),
                        "type": "output_text",
                        "logprobs": [],
                    }
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
                "phase": "commentary",
            },
            {
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                "call_id": "call_h9vdLDpm7EuHrkHXzdKY0Yb2",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_0895a553a79167f3006ad023b71d648197afe5314dfd20909f",
                "namespace": None,
                "status": "completed",
            },
        ],
        "parallel_tool_calls": True,
        "temperature": 1.0,
        "tools": [
            {
                "name": "search_files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "paths": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["query", "paths"],
                    "additionalProperties": False,
                },
                "strict": True,
                "type": "function",
                "defer_loading": None,
                "description": "Search source and test files for a literal term.",
                "output_schema": None,
            },
            {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path", "start_line", "end_line"],
                    "additionalProperties": False,
                },
                "strict": True,
                "type": "function",
                "defer_loading": None,
                "description": "Read a range of lines from a project file.",
                "output_schema": None,
            },
        ],
        "top_p": 0.98,
        "reasoning": {"context": "all_turns", "effort": "high", "mode": "standard", "summary": None},
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "truncation": "disabled",
        "usage": {
            "input_tokens": 309,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": None,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 140,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 75, "text_tokens": None},
            "total_tokens": 449,
            "cost": None,
        },
        "store": True,
        "background": False,
        "completed_at": 1792025527,
        "content_filters": GPT_6_1_SOL_CONTENT_FILTERS,
        "frequency_penalty": 0.0,
        "max_tool_calls": None,
        "moderation": None,
        "presence_penalty": 0.0,
        "prompt_cache_key": None,
        "prompt_cache_retention": "24h",
        "safety_identifier": None,
        "service_tier": "default",
        "tool_usage": AZURE_RESPONSES_RESPONSE["tool_usage"],
        "top_logprobs": 0,
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
        "content_filter_offsets": {"start_offset": 0, "end_offset": 567, "check_offset": 0},
    },
]

GPT_6_ASTRA_ENCRYPTED_REASONING: Final = (
    "gAAAAABq0CPLsQ7EOhGsCUWnH7tYlsJHraBIt-auus4qIwbzqaCzNqSCmiefHbKwVcOnreZc_AwoO-HK"
    "M_0Ataa_BhrACZn7EkzTtUPaypdS0R9INoJ6hsR5vQ8YWibtIffS3BR5XaxsHpCcqFiNrIXvd0QvefYv"
    "vJ7nyPfJvPxCAw0UoWU-XTXeexjM8twO6AvDlTqzP5yCG0hwmYco7zeiwWGxoleRMRoEiYIiiHwWxPXZ"
    "nr9DuZeyqKFgByJsZ19pUAQSlyCKQkKEvMw8KMyPn8iYqxXaQ1vgUBIKopAk9UBkdwDZ9Zd03ZjLodo-"
    "UQx6H0gwQQHVBTej7HJ-52cT0pOJ3vsPjKEx6m_RSw6-s648hLZNWzDYqgfVI-UGID2kw_qtcYQtVhDG"
    "GikbbkwpgJPYxnEmVw3ywlrXxGg9Crlgy0K8S0F1jNkeTuPAMkMRkZOCjiii2DyYrwYAAQpb6ARZEcUv"
    "mMKlpPyhW70JZ3qBHLsSxCGAMaTNsiKPxLHai_nCgoGwrbNTspoCNy55RSDqu2jlhaGfUiF6u3j1Tw_k"
    "zkPY45URBKIaQH055M1D48QxOmci903Rnai715k4RGjHyG30kqCHeZ9F-C7CnRnEENZPi69-U6pIvNF3"
    "s9PkIgXxnOQgBTVO8s9Q1cLoooM5N4s_KFIrESrJCivCebrrOGUaYUFpZ02FS_UGu_CS6Oms8HroLSsL"
    "bKoTD3AO_n5xM5lFu3eTtFCXWGK0_IaR_dMQ_s-bdyZ8KtcrHlG_UKR9TaluS17kGtokuTkaWlQMCvf4"
    "zIUY3mcZU44dz7EuMXupipNLkxrAKZFL6ypcaqjGgSN2eTQFJCO6oCbkI34vbg7-2524ttNkuTXNRCTt"
    "qrbcNOXoDS8asvpa7JbYsKmtLD5bCZGyR_tOgbjSBVmfnT6eQsh2Y6dIf0UJrvCmtOt316p1mydz7-Ek"
    "DEFuAJfTTk7H2zhFncIa2rej33J52FkOIHarQYr9gFG95gTXtqswWuOQ10_C9EG4pd3xJiG9BfC7sefP"
    "XFORAIQ6DJ2rwJWTLixquGrdEN9oe1agij-jRGi5kXfcu12s8bw8d77wW6T9hDDTepEJ8qpNC-DXCvZw"
    "Kt8CxMQvPoTH8nJx9zeeu8mbahkTBLQ_DS18bbFvExbyzDWASZ6oOZD0Ni8mxIxlh18ZzhzHOq82XvH5"
    "YdYkuivNezw-SvImaCXKCDYvV8kPxNnT6D7SCYwJWC4FEnc9ZHw1JthXMVt5NZJLGs4kQS2p2HsfMLMk"
    "rFgIY7Ra4WXvLduVcxuqa2lV3y3Rp5b5Vx8pkIRtHdI4bNfcV5nVsXskjIlRdLZ_nAzEZKbPFxuVnS3r"
    "KXZi8LNgl138jVxid4ETBk5-YmnogzRYZZ3wb-ULgDP7Y3VlXbkqniAbDY1BsZK9I7V5IgEgbEl1J0Ck"
    "fjH9Q5m1aeLBA1vqLtEMtwVEvM78mNctefN9LOCIXGzWKvVPs72ez-DQ71HZ_nVexKpvGKwoPTVCn92M"
    "xs5l9ZT5YSXu0ifdAEKo3334mU_DqyvFqpUeVkaRnSg6rHTtnkraG6ieExe9cGguMdUHAb47VEVCgMHg"
    "9ItzhlhOLuX82fHONdmUeWP04AQrZTI3U5OaN8C15GdirZAc87uQzEJ0Eh54uOLDcS1wqsSk6Hu6"
)

GPT_6_ASTRA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "azure/gpt-6-astra"},
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_headers={"content-type": "application/json", "api-key": "synthetic-azure-key"},
    expected_provider_request={**AZURE_RESPONSES_REQUEST, "model": "gpt-6-astra"},
    mock_provider_response={
        **AZURE_RESPONSES_RESPONSE,
        "id": "resp_09fa191d4c7fb192006ad023c7952881958572a6686bbffea3",
        "created_at": 1792025543,
        "completed_at": 1792025547,
        "content_filters": GPT_6_ASTRA_CONTENT_FILTERS,
        "instructions": RESPONSES_REQUEST["instructions"],
        "model": "gpt-6-astra",
        "output": [
            {
                "id": "rs_09fa191d4c7fb192006ad023c898e88195a30e3b694f965fb6",
                "type": "reasoning",
                "content": [],
                "encrypted_content": GPT_6_ASTRA_ENCRYPTED_REASONING,
                "summary": [],
            },
            {
                "id": "msg_09fa191d4c7fb192006ad023ca32f0819596d7f46dd3c29f03",
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
                "id": "fc_09fa191d4c7fb192006ad023ca9800819586b8ea16baef56cb",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                "call_id": "call_PuG8zuJ8TpdNdpQFRquUj6ZL",
                "name": "read_file",
            },
        ],
        "tools": [
            {
                "type": "function",
                "description": "Search source and test files for a literal term.",
                "name": "search_files",
                "output_schema": None,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "paths": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["query", "paths"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "description": "Read a range of lines from a project file.",
                "name": "read_file",
                "output_schema": None,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path", "start_line", "end_line"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ],
        "usage": {
            "input_tokens": 309,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 152,
            "output_tokens_details": {"reasoning_tokens": 87},
            "total_tokens": 461,
        },
    },
    expected_litellm_response={
        **RESPONSES_NATIVE_RESPONSE,
        "id": ANY,
        "created_at": 1792025543,
        "model": "azure/gpt-6-astra",
        "output": [
            {
                "id": "rs_09fa191d4c7fb192006ad023c898e88195a30e3b694f965fb6",
                "summary": [],
                "type": "reasoning",
                "content": [],
                "encrypted_content": GPT_6_ASTRA_ENCRYPTED_REASONING,
            },
            {
                "id": "msg_09fa191d4c7fb192006ad023ca32f0819596d7f46dd3c29f03",
                "content": [
                    {
                        "annotations": [],
                        "text": (
                            "I’ll read `shop/validation.py` first to check how it handles zero, negative, and "
                            "non-integer quantities.\n"
                        ),
                        "type": "output_text",
                        "logprobs": [],
                    }
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
                "phase": "commentary",
            },
            {
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                "call_id": "call_PuG8zuJ8TpdNdpQFRquUj6ZL",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_09fa191d4c7fb192006ad023ca9800819586b8ea16baef56cb",
                "namespace": None,
                "status": "completed",
            },
        ],
        "parallel_tool_calls": True,
        "temperature": 1.0,
        "tools": [
            {
                "name": "search_files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "paths": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["query", "paths"],
                    "additionalProperties": False,
                },
                "strict": True,
                "type": "function",
                "defer_loading": None,
                "description": "Search source and test files for a literal term.",
                "output_schema": None,
            },
            {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path", "start_line", "end_line"],
                    "additionalProperties": False,
                },
                "strict": True,
                "type": "function",
                "defer_loading": None,
                "description": "Read a range of lines from a project file.",
                "output_schema": None,
            },
        ],
        "top_p": 0.98,
        "reasoning": {"context": "all_turns", "effort": "high", "mode": "standard", "summary": None},
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "truncation": "disabled",
        "usage": {
            "input_tokens": 309,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": None,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 152,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 87, "text_tokens": None},
            "total_tokens": 461,
            "cost": None,
        },
        "store": True,
        "background": False,
        "completed_at": 1792025547,
        "content_filters": GPT_6_ASTRA_CONTENT_FILTERS,
        "frequency_penalty": 0.0,
        "max_tool_calls": None,
        "moderation": None,
        "presence_penalty": 0.0,
        "prompt_cache_key": None,
        "prompt_cache_retention": "24h",
        "safety_identifier": None,
        "service_tier": "default",
        "tool_usage": AZURE_RESPONSES_RESPONSE["tool_usage"],
        "top_logprobs": 0,
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
        "content_filter_offsets": {"start_offset": 0, "end_offset": 297, "check_offset": 0},
    },
]

GPT_5_6_TERRA_ENCRYPTED_REASONING: Final = (
    "gAAAAABq0CPSc68PgJKpbMkIorqdkFkGusPg2cNuR1B1t7kGmuehEWf0bpLWzdkOUkQgxYbzawyuBLR5"
    "sb-Wflsld_-Qj5WlHfYZmRJU7n21c2JKaWMtvoEf2WwktXCuHJCwg9hwBNUhIGuLjIJFqx4gsHMcoNSZ"
    "reXGm51nnzkzX48zanXTWOgfELEUNPazQlPkhMWsO0ZvjomVM-OaWVWCzvD0MIoAe42ZPAeA_448XG1k"
    "noYd5r4kxIn2ggODmjP2U-ijPUkLWt1ulClD2k8B_tGPUIAfMfN_jsx1KLLbLw0cppoedYXYl_PNWJn_"
    "N3L0Us5Vtj0QLwO1z4MfNus_oTJ4PLa3K43al71izDw9gJlWZmT_Tme34cOuTb5_3O3rPSMSMVvNoXjG"
    "MjNzlL0lNQwu5O-scO8U-6nwfSZt8gSkQ5h5GflqHLgFQkOFWFu9tAcifvaTjznOFlinK43vpGIf796d"
    "ZFq_MSIBgGdnNGEUsDsFvMUXomn4Zlw2_v9icKiaFr505BJehAgj4f7WRI4vnIJCMI5WByUupPB9Rawi"
    "rB_b5OhDLfrI9jEDE8DtkVKCGf1nwrI99Bllt9-EtyOTSWspKOzDinbIkS7bhOXy0oZh_8_6vFJDdzzV"
    "C2yRRfHPVUGndylD--H5P-lkgDN_qwWTDIOhhTd-9TnewshMmrah0EfNimFLaxYWQ7vqBA1C5_Dh9gAd"
    "5SZ3Pb0C4iOjsHepbUl4l-9DtrGIGoJ4pz0XukOZRa-SXLxHOAI2eil9Rnj08MR37yYy_lrIjq75F_0R"
    "d-11MzV0_ej6UDABd-h9Myzn_0WR-qpBfnCyH-z9w_SB8X7ErzXOImlzSvIo4B-Oy_xSLVwSjNRW3lbW"
    "Z8uoNHPXFj3miv5w4lb2TZTd_6ltOeH-I746jdKa0jxgZOY2oHEKH2eFEqYjQvNd06TGMvJYByM1urkI"
    "h1GnTRH9VVoHa7QbyLAxL9hdV0pGc5Rrh9eIvdz830irEwVO6nS8t6tXQ2DgKy555nmlVQ6_inLzvw1t"
    "uU0di7-JZK28JCgo7zMw_UwIDok7oeLXFAbUtlVNvyp2mJ5kz9cm-xK7veFgiKNrRi3NqospaOfEt3NF"
    "_NV62O8WKwudIXpg2aTEnxI66GNphwllNQSoZjYrsvMIsTk6GNdqVE7OXpKTRhmVEWh2vb8u4GSMZ7sa"
    "Kqf6ju6RbjQF1FyazaFCuBGIVANnTPLPcvt7HIv6HzyUb75S1sjoFimhyRByRn906fCM218="
)

GPT_5_6_TERRA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "azure/gpt-5.6-terra"},
    expected_provider_endpoint="/openai/responses?api-version=2025-04-01-preview",
    expected_provider_headers={"content-type": "application/json", "api-key": "synthetic-azure-key"},
    expected_provider_request={**AZURE_RESPONSES_REQUEST, "model": "gpt-5.6-terra"},
    mock_provider_response={
        **AZURE_RESPONSES_RESPONSE,
        "id": "resp_0e1796d6e4004e2f006ad023d108548193a6493b2aac71acd0",
        "created_at": 1792025553,
        "completed_at": 1792025554,
        "content_filters": GPT_5_6_TERRA_CONTENT_FILTERS,
        "instructions": RESPONSES_REQUEST["instructions"],
        "model": "gpt-5.6-terra",
        "output": [
            {
                "id": "rs_0e1796d6e4004e2f006ad023d1f5ac81939dc5121afcbd9967",
                "type": "reasoning",
                "content": [],
                "encrypted_content": GPT_5_6_TERRA_ENCRYPTED_REASONING,
                "summary": [],
            },
            {
                "id": "msg_0e1796d6e4004e2f006ad023d2347c8193a86acc05aee89b6f",
                "type": "message",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": (
                            "The next file is `shop/validation.py`, which contains the validator and will "
                            "show how zero, negative, and non-integer quantities are currently handled."
                        ),
                    }
                ],
                "phase": "commentary",
                "role": "assistant",
            },
            {
                "id": "fc_0e1796d6e4004e2f006ad023d26b5081939200dbb9bfa0382d",
                "type": "function_call",
                "status": "completed",
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                "call_id": "call_IuSjfDiujWaxiYl7MptZoMXp",
                "name": "read_file",
            },
        ],
        "tools": [
            {
                "type": "function",
                "description": "Search source and test files for a literal term.",
                "name": "search_files",
                "output_schema": None,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "paths": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["query", "paths"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "description": "Read a range of lines from a project file.",
                "name": "read_file",
                "output_schema": None,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path", "start_line", "end_line"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ],
        "usage": {
            "input_tokens": 309,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 95,
            "output_tokens_details": {"reasoning_tokens": 24},
            "total_tokens": 404,
        },
    },
    expected_litellm_response={
        **RESPONSES_NATIVE_RESPONSE,
        "id": ANY,
        "created_at": 1792025553,
        "model": "azure/gpt-5.6-terra",
        "output": [
            {
                "id": "rs_0e1796d6e4004e2f006ad023d1f5ac81939dc5121afcbd9967",
                "summary": [],
                "type": "reasoning",
                "content": [],
                "encrypted_content": GPT_5_6_TERRA_ENCRYPTED_REASONING,
            },
            {
                "id": "msg_0e1796d6e4004e2f006ad023d2347c8193a86acc05aee89b6f",
                "content": [
                    {
                        "annotations": [],
                        "text": (
                            "The next file is `shop/validation.py`, which contains the validator and will "
                            "show how zero, negative, and non-integer quantities are currently handled."
                        ),
                        "type": "output_text",
                        "logprobs": [],
                    }
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
                "phase": "commentary",
            },
            {
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                "call_id": "call_IuSjfDiujWaxiYl7MptZoMXp",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_0e1796d6e4004e2f006ad023d26b5081939200dbb9bfa0382d",
                "namespace": None,
                "status": "completed",
            },
        ],
        "parallel_tool_calls": True,
        "temperature": 1.0,
        "tools": [
            {
                "name": "search_files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "paths": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["query", "paths"],
                    "additionalProperties": False,
                },
                "strict": True,
                "type": "function",
                "defer_loading": None,
                "description": "Search source and test files for a literal term.",
                "output_schema": None,
            },
            {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path", "start_line", "end_line"],
                    "additionalProperties": False,
                },
                "strict": True,
                "type": "function",
                "defer_loading": None,
                "description": "Read a range of lines from a project file.",
                "output_schema": None,
            },
        ],
        "top_p": 0.98,
        "reasoning": {"context": "all_turns", "effort": "high", "mode": "standard", "summary": None},
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "truncation": "disabled",
        "usage": {
            "input_tokens": 309,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": None,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 95,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 24, "text_tokens": None},
            "total_tokens": 404,
            "cost": None,
        },
        "store": True,
        "background": False,
        "completed_at": 1792025554,
        "content_filters": GPT_5_6_TERRA_CONTENT_FILTERS,
        "frequency_penalty": 0.0,
        "max_tool_calls": None,
        "moderation": None,
        "presence_penalty": 0.0,
        "prompt_cache_key": None,
        "prompt_cache_retention": "24h",
        "safety_identifier": None,
        "service_tier": "default",
        "tool_usage": AZURE_RESPONSES_RESPONSE["tool_usage"],
        "top_logprobs": 0,
    },
)
