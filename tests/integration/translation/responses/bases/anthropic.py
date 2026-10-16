import json
from typing import Final
from unittest.mock import ANY

from integration.translation.case import TranslationTestCase
from integration.translation.responses.templates import (
    CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0,
    CLAUDE_SONNET_5_5_THINKING_SIGNATURE_1,
    RESPONSES_REQUEST,
    RESPONSES_RESPONSE,
)
from integration.translation.templates.anthropic import ANTHROPIC_MESSAGES_REQUEST, ANTHROPIC_MESSAGES_RESPONSE

CLAUDE_HAIKU_4_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "anthropic/claude-haiku-4-5",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "claude-haiku-4-5",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "model": "claude-haiku-4-5-20251001",
        "id": "msg_011CfgDzBnuPYc7TdysZFRpo",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 17,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 5,
            "service_tier": "standard",
            "inference_geo": "not_available",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "id": ANY,
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "anthropic/claude-haiku-4-5",
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
            "input_tokens": 17,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 17,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 5,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": 5},
            "total_tokens": 22,
            "cost": None,
        },
        "user": None,
        "store": None,
        "provider_specific_fields": {"citations": None, "thinking_blocks": None},
    },
)

CLAUDE_SONNET_4_6_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "anthropic/claude-sonnet-4-6",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "model": "claude-sonnet-4-6",
        "id": "msg_011CfgDzH3AcaSNXtvurLgAG",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 18,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 5,
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "id": ANY,
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "anthropic/claude-sonnet-4-6",
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello!", "annotations": []}],
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
            "input_tokens": 18,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 18,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 5,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": 5},
            "total_tokens": 23,
            "cost": None,
        },
        "user": None,
        "store": None,
        "provider_specific_fields": {"citations": None, "thinking_blocks": None},
    },
)

CLAUDE_SONNET_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "anthropic/claude-sonnet-5",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "claude-sonnet-5",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "model": "claude-sonnet-5",
        "id": "msg_011CfgDzPkj2HxBq8qGJsGSi",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 21,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "id": ANY,
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "anthropic/claude-sonnet-5",
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
            "input_tokens": 21,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 21,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 6,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": 6},
            "total_tokens": 27,
            "cost": None,
        },
        "user": None,
        "store": None,
        "provider_specific_fields": {"citations": None, "thinking_blocks": None},
    },
)

CLAUDE_OPUS_4_8_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "anthropic/claude-opus-4-8",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "claude-opus-4-8",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "model": "claude-opus-4-8",
        "id": "msg_011CfgDzV9vZCkuQJt15n3Z1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 21,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "global",
        },
        "diagnostics": None,
    },
    expected_litellm_response={
        "id": ANY,
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "anthropic/claude-opus-4-8",
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
            "input_tokens": 21,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 21,
                "video_tokens": None,
                "cache_write_tokens": 0,
            },
            "output_tokens": 6,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": 6},
            "total_tokens": 27,
            "cost": None,
        },
        "user": None,
        "store": None,
        "provider_specific_fields": {"citations": None, "thinking_blocks": None},
    },
)

CLAUDE_SONNET_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "anthropic/claude-sonnet-5-5"},
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        **ANTHROPIC_MESSAGES_REQUEST,
        "model": "claude-sonnet-5-5",
        "system": [
            {
                "type": "text",
                "text": "You help maintain a synthetic Python project. Read the relevant files before "
                "suggesting a fix. Explain which file you need next, then call a read-only tool. "
                "Never change files. Use the responses client for this project review. Consider "
                "zero, negative and non-integer quantities before choosing the next file to "
                "inspect.",
            }
        ],
    },
    mock_provider_response={
        **ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-sonnet-5-5",
        "id": "msg_011Cg32qrmyWNAPnHJhxBL3M",
        "content": [
            {
                "type": "thinking",
                "thinking": "The user wants the validation file read first, but since the test reads are "
                "independent, I can just include both in the same batch since it's "
                "effectively still prioritizing validation.\n"
                "\n",
                "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": "I'll check `shop/validation.py` to see how `validate_quantity` handles "
                "zero, negative, and non-integer values, and read `tests/test_validation.py` "
                "alongside it to confirm the expected behavior.\n"
                "\n",
                "signature": CLAUDE_SONNET_5_5_THINKING_SIGNATURE_1,
            },
            {
                "type": "tool_use",
                "id": "toolu_01GYseiK8hh9QxfWJ6HbiAt3",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
                "caller": {"type": "direct"},
            },
            {
                "type": "tool_use",
                "id": "toolu_01U3TuUTifkSBLVzyTFnbMN1",
                "name": "read_file",
                "input": {"path": "tests/test_validation.py", "start_line": 1, "end_line": 40},
                "caller": {"type": "direct"},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 509,
            "output_tokens_details": {"thinking_tokens": 313},
            "service_tier": "standard",
            "inference_geo": "global",
        },
    },
    expected_litellm_response={**RESPONSES_RESPONSE, "model": "anthropic/claude-sonnet-5-5"},
)


CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSkgYKEAgSGAI4AUIIdGhpbmtpbmcSDDNkbcUmfOcV76xw1BoM1d74UojfhlwUKXJRIjDruKpmF7Xkn4wU+s/jLhdQL1oTKBrwFivaGL2z"
    "2qnyjXgtCgYeYxa4syAiIlaJMkwqrwXgBL0q4FtZvicmkVAfs3GMC/MvUk3zJ5psHQWMfE+EYU436rT6+Xcoj47t9XgRI+Qs5vio51d6AvNR"
    "snONcWIVexutCBNKfDOcnBZwzcgr6xlR8+bvZI0a4kgGEkJzqUm5nkST8scZwMkJPsgk0Zrr8aDDcGGmtc161vfYqDIMq6+IpnOlib1Q/FA0"
    "gO2Trdo93xLjKvHUXbmSzXokO5x7s60ymFjVd6RzmoLStGgBwiMXElWT40WPh/T2tx2GtuYJ6EPuMOpIqzmI3pLbYplVdVBGdEhUL+wPD8gT"
    "mHbBq1Oz8cwSdUyRWBTMCgyZdMkuIw6lPFRAec/19JbDZQWQbnKCOmZK9YwicptzrbllIKqNjJTX4sww8MvMp69lMRxeFqM9tBnC7oIYevEg"
    "witiRyKsVfbQTfmexva4g1WenMsqKDub77WJPsbhN0ihuQvlLssmGtRLKh/C4waaaCTzqsj4mDO1UEnflUczRoQHF2fxlsQBVjMc0QcuICYL"
    "nwcc2ZL8zvL14sELhiSx2PvxjOIwNuEezF3w3uhS8o/9CSbfJnJL6lQ3lldTA8+WL+cvzN+73+9rTN89GRCJPGni52cMkuGV5ObB2dAyNsBi"
    "/WsPwiUetMhWeds/kP+Ct1cqagNuCTenGF+oKf0qNsvIPnWoc8a8YW9bxlEgQ9qBF0NXT26RkEE90Qsp4aseb/qmEDKxE2DMcdMwOru3CEvD"
    "xQNy07Dak3+tiSS63kRZwsRFbMTb42W98NrDXqq+sfyvMQarPJj/OmWExt9rna3Y+BlWNq0j82gUyCY4gzv4VHM/OyuhNcmNKDdDsgvSwub9"
    "wRSKmuC4LDAhrRMuTQ/cMZq1+ZyQHiuFvnsArZxCusQ5QxDjxRGsI+uMeWmkPY3TN1ofMSIbeWpyiwnsR8EYAQ=="
)

CLAUDE_HAIKU_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "anthropic/claude-haiku-5-5"},
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        **ANTHROPIC_MESSAGES_REQUEST,
        "model": "claude-haiku-5-5",
        "system": [
            {
                "type": "text",
                "text": "You help maintain a synthetic Python project. Read the relevant files before "
                "suggesting a fix. Explain which file you need next, then call a read-only tool. "
                "Never change files. Use the responses client for this project review. Consider "
                "zero, negative and non-integer quantities before choosing the next file to "
                "inspect.",
            }
        ],
    },
    mock_provider_response={
        **ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-haiku-5-5",
        "id": "msg_011Cg3716ZuhZgb2YrqgDQVR",
        "content": [
            {
                "type": "thinking",
                "thinking": "I'll just read the validation.py file directly, starting with a reasonable range "
                "near the top of the file since no specific lines were requested.\n"
                "\n",
                "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
            },
            {
                "type": "tool_use",
                "id": "toolu_01H53znxFvBbuHPXNePjEhW5",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
                "caller": {"type": "direct"},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 167,
            "output_tokens_details": {"thinking_tokens": 67},
            "service_tier": "standard",
            "inference_geo": "global",
        },
    },
    expected_litellm_response={
        **RESPONSES_RESPONSE,
        "model": "anthropic/claude-haiku-5-5",
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I'll just read the validation.py file directly, starting with a "
                        "reasonable range near the top of the file since no specific lines were "
                        "requested.\n"
                        "\n",
                        "annotations": [],
                    },
                ],
                "phase": None,
                "encrypted_content": json.dumps(
                    [
                        {
                            "type": "thinking",
                            "thinking": "I'll just read the validation.py file directly, starting with a "
                            "reasonable range near the top of the file since no specific lines were "
                            "requested.\n"
                            "\n",
                            "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
                        },
                    ],
                    separators=(",", ":"),
                ),
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                "call_id": "toolu_01H53znxFvBbuHPXNePjEhW5",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_01H53znxFvBbuHPXNePjEhW5",
                "namespace": None,
                "status": "completed",
            },
        ],
        "usage": {
            "input_tokens": 877,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 4,
                "video_tokens": None,
                "cache_write_tokens": 873,
            },
            "output_tokens": 167,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 67, "text_tokens": 100},
            "total_tokens": 1044,
            "cost": None,
        },
        "provider_specific_fields": {
            "citations": None,
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "I'll just read the validation.py file directly, starting with a reasonable "
                    "range near the top of the file since no specific lines were requested.\n"
                    "\n",
                    "signature": CLAUDE_HAIKU_5_5_THINKING_SIGNATURE_0,
                },
            ],
        },
    },
)

CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0: Final = (
    "CAQSmwYKEAgSGAI4AUIIdGhpbmtpbmcSDAp614h0T9spOzVRohoMRXG5Q7QnUaybHmMQIjDVSvcKksTqmvoyviUgrnx25thBdpJEPfO61S1d"
    "hV+zrTe/1FP4+d14zYjlYEZ8Id0quAW5bhrtPXA2ESV73lWwlm/w+SYE+QsiZA9rbSOkSdKLTg37S3i/w+ad4VIFTr1iNoV8RfYZog2D3q69"
    "miBOF29U8ILsi+tUM66QNtGdGWDahlsimm+d/IMs+Ie4uP8OHr5X/bJ7NS2MJ407WhS1E+rYrmOce5lHOmhVPGzcL2EM+ToptZ84DR8N7iIq"
    "k9pP8CoQL7Z10aCTkG4r2tbdIup7RE3WXrsBHNbs+RAqF5LXbEpsQZaXLx71U1DpDffx6OZ3OSCw1tEcA4o2FDVFNv8CQO3wBnWqkZyjehqb"
    "sMarphzyvDxDO7T//sTTVM1593gEbLoRNeUne3bwtgdwoDPMXminh541bLo8vjo25kERKJedEUG/CDZqT16YziXqlSvzV/TgQeddzb2TLnJT"
    "LoYoXsmn8789+6nBPdIe3SQu1nhVP5XUgC0+dI/iKqbiuswuslOSx3ZuIEan3i/HZ+XflqXy4ewB0+5sLHbiwzmGEbrZcWl0wbebc2Da0LnY"
    "rgAQHqKH0vl4o3fGmaPMxcPml9RP4zG0KblC24o6acb0DYAFTSfblE3dKoMgZwPfiMijqiXriZQSuF5Y3VGY6CcWGQDi5kLzbFe5D2HA/nM4"
    "66bjH9VKvVsVglODnnDPQD84e9YM86EelbaxCQcNo9MrCpp7g0tWRvMEiAlfKo0W4hvsYaa2op3XMbHVlJeWJcoU/WPodcEnSEPl6/pz522X"
    "I/ItEKXeKIqs0VG9giw0pxuHB8+ig+8bn74Lezmw4LLHk67D2/PCVKLSqrIkHfE41ASdJSFAopc7YY8iovyAVdvUfm+7Vefmmphkx7C162Ze"
    "gVLS7mSAxuQCWsIlFyui50D75X8xEa2CdijsbrTvCqKIQHcywq8oLLGmDilMRs1DlnQiAWETjHvpzp3WEs9eZBi4tHPz5CMYAQ=="
)

CLAUDE_OPUS_5_5_THINKING_SIGNATURE_1: Final = (
    "CAQS0QYKEQgSGAI4AUIJbmFycmF0aW9uEgwL0nQCUkLBdQqyQ48aDBNdOXSFQIfQqDyzeyIwgBeYVWjfsySdPznFBTS6nfeLZiDt+L2EHka6"
    "Wp6pH2lO/ZVHd45V/Yy8qr0uX3xrKu0F3QlrjBy505vPXf94O0o7bPvrihxLhHnG9Wf14JgZfAXTqiM7kZxqRCJF2DlP2S4BzspQGIILMu+e"
    "gnizwTdKpM7I2XgSqV4ZnQRhjPgRVgQkQrbRfSRtfFWDTcJtz/Lp/2KQw6ijdj1tWc66QB8uNSkTStOlz1sseMaBWNOps0WK+vkJw6J+ZbZK"
    "dFdZSgNQ2Iwxl3Pys57YHUrqHaVi3NJVFK760EZeavxciDafSs+ZlTuzeghIy/iLQdSSeUhCl0pZE6SCkEWca+A02GewaHguZ9eRoGnCVsSw"
    "soQT3aeK7sUbas1sFDyWSvieJ0naBRUN6ae1QSBAWqYnsvEEx11wWcN/DB3SzjM5j48po9ov+wbeoBl2nex/iUVSfBBixe36nZ8Vt37dC6f+"
    "0wgw5DHVtiqFKYSH/b3owUf2RJuYUut7LPkT5qFm0mPlkCqSfZqpRwijVAmV0obs61IgrTVlmFq2igWs0rt9v5iuM7+zjI/DtLL6NgGaFgXW"
    "Oc8VjDsrHnUMCK0/QtHuC51aE8mqzw47i4wTQ3O5+/8F8ZJn0PkmxB2lZixCSZ42XhU5pqPiTAWjuXciwXzkCcJKz+CR2cl2gFWw3oQR1N6B"
    "f9Wlk3f6DrtPywDFDlY7piBz25k958iZDRvTYkleo8daA+DAWMXS+3k0doIpNBSQMoE67GCgqk3A22huvgYdCqlD142RbrdB7xlUviysajfB"
    "lbOFXNjs5Lp+v7UVPiNCDJwha/ZIZfmgp86OSKryKTRrBiMNZRv7v9xF5NyQhB+MzGG3pDzzhqYUlzpIE2vmAd4Hk3/T4X5WvIGW+Juv2U+r"
    "pHW59B5K/+2tF3VczG//CpHoB6R+8fNWTDp4Bx+l7v+Pep1lBvVG+XAZZHwaPxeXgaOZcUYs/SMdxYDBflXfvxusttZoXNiXJROZle/lIIeo"
    "fDbMvuc14FSkdLJ+N41d4et8zwXI5BcOxPIB7ryDVejwgF/F5kAMtnVdzQEYAQ=="
)

CLAUDE_OPUS_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "anthropic/claude-opus-5-5"},
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        **ANTHROPIC_MESSAGES_REQUEST,
        "model": "claude-opus-5-5",
        "system": [
            {
                "type": "text",
                "text": "You help maintain a synthetic Python project. Read the relevant files before "
                "suggesting a fix. Explain which file you need next, then call a read-only tool. "
                "Never change files. Use the responses client for this project review. Consider "
                "zero, negative and non-integer quantities before choosing the next file to "
                "inspect.",
            }
        ],
    },
    mock_provider_response={
        **ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-opus-5-5",
        "id": "msg_011Cg372KDhhjt1fiWYa1qCb",
        "content": [
            {
                "type": "thinking",
                "thinking": "Since the user asked for the validation file first, I'll start by reading "
                "shop/validation.py around line 18 before looking at anything else.\n"
                "\n",
                "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": "I'll check validate_quantity in shop/validation.py to see how it handles zero, "
                "negative, and non-integer inputs before moving to the test file.\n"
                "\n",
                "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_1,
            },
            {
                "type": "tool_use",
                "id": "toolu_0186wos82AWCLwkPWVeVegg8",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 50},
                "caller": {"type": "direct"},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 228,
            "output_tokens_details": {"thinking_tokens": 130},
            "service_tier": "standard",
            "inference_geo": "global",
        },
    },
    expected_litellm_response={
        **RESPONSES_RESPONSE,
        "model": "anthropic/claude-opus-5-5",
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Since the user asked for the validation file first, I'll start by reading "
                        "shop/validation.py around line 18 before looking at anything else.\n"
                        "\n"
                        "I'll check validate_quantity in shop/validation.py to see how it handles "
                        "zero, negative, and non-integer inputs before moving to the test file.\n"
                        "\n",
                        "annotations": [],
                    },
                ],
                "phase": None,
                "encrypted_content": json.dumps(
                    [
                        {
                            "type": "thinking",
                            "thinking": "Since the user asked for the validation file first, I'll start by reading "
                            "shop/validation.py around line 18 before looking at anything else.\n"
                            "\n",
                            "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
                        },
                        {
                            "type": "thinking",
                            "thinking": "I'll check validate_quantity in shop/validation.py to see how it handles "
                            "zero, negative, and non-integer inputs before moving to the test file.\n"
                            "\n",
                            "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_1,
                        },
                    ],
                    separators=(",", ":"),
                ),
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 50}',
                "call_id": "toolu_0186wos82AWCLwkPWVeVegg8",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_0186wos82AWCLwkPWVeVegg8",
                "namespace": None,
                "status": "completed",
            },
        ],
        "usage": {
            "input_tokens": 877,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 4,
                "video_tokens": None,
                "cache_write_tokens": 873,
            },
            "output_tokens": 228,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 130, "text_tokens": 98},
            "total_tokens": 1105,
            "cost": None,
        },
        "provider_specific_fields": {
            "citations": None,
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "Since the user asked for the validation file first, I'll start by reading "
                    "shop/validation.py around line 18 before looking at anything else.\n"
                    "\n",
                    "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
                },
                {
                    "type": "thinking",
                    "thinking": "I'll check validate_quantity in shop/validation.py to see how it handles "
                    "zero, negative, and non-integer inputs before moving to the test file.\n"
                    "\n",
                    "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_1,
                },
            ],
        },
    },
)

CLAUDE_FABLE_5_1_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "anthropic/claude-fable-5-1"},
    expected_provider_endpoint="/v1/messages",
    expected_provider_headers={
        "x-api-key": "synthetic-anthropic-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    expected_provider_request={
        **ANTHROPIC_MESSAGES_REQUEST,
        "model": "claude-fable-5-1",
        "system": [
            {
                "type": "text",
                "text": "You help maintain a synthetic Python project. Read the relevant files before "
                "suggesting a fix. Explain which file you need next, then call a read-only tool. "
                "Never change files. Use the responses client for this project review. Consider "
                "zero, negative and non-integer quantities before choosing the next file to "
                "inspect.",
            }
        ],
    },
    mock_provider_response={
        **ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-fable-5-1",
        "id": "msg_011Cg3734Dqb6rpmJnPKii79",
        "content": [
            {
                "type": "text",
                "text": "Reading `shop/validation.py` first to see the rule behind the zero rejection, "
                "then I'll read the test file in a separate step since there's nothing to derive "
                "from it for the read.",
            },
            {
                "type": "tool_use",
                "id": "toolu_01QKMW1txMKCm8oSZyEwo7wX",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 40},
                "caller": {"type": "direct"},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 155,
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
            "inference_geo": "global",
        },
    },
    expected_litellm_response={
        **RESPONSES_RESPONSE,
        "model": "anthropic/claude-fable-5-1",
        "output": [
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Reading `shop/validation.py` first to see the rule behind the zero "
                        "rejection, then I'll read the test file in a separate step since there's "
                        "nothing to derive from it for the read.",
                        "annotations": [],
                    },
                ],
                "phase": None,
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 40}',
                "call_id": "toolu_01QKMW1txMKCm8oSZyEwo7wX",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_01QKMW1txMKCm8oSZyEwo7wX",
                "namespace": None,
                "status": "completed",
            },
        ],
        "usage": {
            "input_tokens": 877,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 4,
                "video_tokens": None,
                "cache_write_tokens": 873,
            },
            "output_tokens": 155,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0, "text_tokens": 155},
            "total_tokens": 1032,
            "cost": None,
        },
        "provider_specific_fields": {"citations": None, "thinking_blocks": None},
    },
)
