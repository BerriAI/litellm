import json
from typing import Final
from unittest.mock import ANY

from integration.translation.case import TranslationTestCase
from integration.translation.responses.templates import RESPONSES_REQUEST, RESPONSES_RESPONSE
from integration.translation.templates.vertex_ai import (
    VERTEX_AI_ANTHROPIC_MESSAGES_REQUEST,
    VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
    VERTEX_AI_GEMINI_GENERATE_CONTENT_REQUEST,
    VERTEX_AI_GEMINI_GENERATE_CONTENT_RESPONSE,
)

GEMINI_3_5_FLASH_THOUGHT_SIGNATURE: Final = "AY89a1+kriNs9YsPegJQxgjTWsMlxsqPU/UkHDyz4EK+3FRrE52ygqPIHTvkblgKKiO5Fn8R9D3iu//i+YnuuZ+e7W0yxUoFScas89b5rn3vZHsihnUA7VTe3b7beth1IftteoTGZ3F8OSLrSIP8HKn4xTIx35eeNEt5oiD3K5rTQrdklqO277HvTDhu/LuZhZ0pzaWciGQEc61epIBBpC2ygu8V4L8TekUVZnz6qzSzxipdiQStQ8dZou43334hMnxOR6JjJCGxqt/1K+C0vXX6YP6f+Bph5VhPSnvy2Z1uApi3TUxAcsDDyVewIr+ha2G6DPhN1Oi5uZZNhmbvH01Uq5OOquF1uEICq2m/35LhJIbDNM9N7oI1Z6L94P9xdugeFXM2S7wdJfc8sdg09vEzNeA++GK25zRuB4hU7XyaG4XnWjmu2kswi4sf+EyrO/K0P2tb/ZCJu7kJTtWTiD1wgSIl1eqS8ORKKsXDN2hIbFrtsNR50WIvVzd6"
CLAUDE_HAIKU_5_5_RESPONSES_THINKING_SIGNATURE: Final = (
    "CAQSnQYKEggUEAIYAjgBQgh0aGlua2luZxIMkVDSVxiTnd+9SfQfGgzwTUI2p4UsBx3DEsoiMNjpaFYOWnhwVaL3/Eryb0c8fnMn"
    "ti6UJ7fYeYlGswrt9vI0qWxkxp6RPGQWSeGc1yq4BWRvXHaCDMKPCCPdKE8d+PYABUgl95dkyR4icTz4lL576OVydhVIZ5AZCcdb"
    "3rH6aGBCT6KKgzqecwexnMan1ksmnDcZaQ4Od+EgeWSfoomtXP2tQl4yNk3JHvD1aANRm38lTTcyRH+i7gTfxNdlUAwa82Jmfmks"
    "dIpAhOfy48wk/M06bCFR8Cj42CPnXPQXQONmlickYZx4vmFUx5knpJhN63e3geWQ9yQ/y1NgXwbmPieK8cf6hgmNDUJ5d+6ZQusM"
    "vnzDD3oPLbjcz70UrgaB6e602cGpjjFL6foHDEPiu9/jdoWe4vzXT5xLwZCrx66RCc6RT+MTjDmf0SbMovTK2ZCTa8gcXGwnKGjH"
    "Tblya/EOkSLO9s12UxeYy0mrpRp5mjhVynkLkftR4hIRvZtcp3CYPScINhrM0svXE1RZ4VdSNL9TA5ZW9ba0rr8886iuCgGzEmKp"
    "v1KMaqp5v91VUNZ8v26P9n4+qawMkfcE50n04mrZHaMlW8pClEBVfMj5MlpGyI4illQH7KlEtPluo2PpWjTaB+tAjKgO/bvNdyGS"
    "fr4vnjWgWfzMndOSY1UcvKruVgkizBrr53sSj+YshGjECCUt5/S46irNPDgGpMSgbmo9O33C8vt1wOi6yhJrGAYfKIlj+Uhf664S"
    "oiDj6si+rCcpLkqrZEEarCLhMKNT6CsMjmB5LaUGtMG60h7s+3Yzvf0dpj0JNMcVksYQ/n+4hnLLVSz78DdNYujHMV8cEYd3UCRA"
    "PBDqOpl1qocZVHmWDAMEfx18+JVFUdssDiIx1cc2Ja1LQPm2qUaUr8cTHOy9zYCZEZCZ3eSzFvii8fiRiELdX+QTWHNqXFTdCNKO"
    "VGlegzhhbAynXLzC3YSaRwoiTT2IBRR7tgfeh4sRXZPyRdyUphGTrWE29aXtbtX2O5N4yhgB"
)

CLAUDE_HAIKU_4_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "vertex_ai/claude-haiku-4-5@20251001",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-haiku-4-5@20251001:rawPredict",
    expected_provider_headers={
        "authorization": "Bearer scripted-token",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": "scripted-token",
    },
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "anthropic_version": "vertex-2023-10-16",
        "system": [{"type": "text", "text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "model": "claude-haiku-4-5-20251001",
        "id": "msg_vrtx_011ChJAfDJfPTELa9d6zjHZQ",
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
            "speed": "standard",
        },
    },
    expected_litellm_status_code=200,
    expected_litellm_response={
        "id": ANY,
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "vertex_ai/claude-haiku-4-5@20251001",
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
        "model": "vertex_ai/claude-sonnet-4-6@default",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-sonnet-4-6@default:rawPredict",
    expected_provider_headers={
        "authorization": "Bearer scripted-token",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": "scripted-token",
    },
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "anthropic_version": "vertex-2023-10-16",
        "system": [{"type": "text", "text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "model": "claude-sonnet-4-6",
        "id": "msg_vrtx_011ChJAfExswMfbMU9g7MHdR",
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
            "speed": "standard",
        },
    },
    expected_litellm_status_code=200,
    expected_litellm_response={
        "id": ANY,
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "vertex_ai/claude-sonnet-4-6@default",
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
        "model": "vertex_ai/claude-sonnet-5@default",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-sonnet-5@default:rawPredict",
    expected_provider_headers={
        "authorization": "Bearer scripted-token",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": "scripted-token",
    },
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "anthropic_version": "vertex-2023-10-16",
        "system": [{"type": "text", "text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "model": "claude-sonnet-5",
        "id": "msg_vrtx_011ChJAfFEjdYucFpDiyMtgB",
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
            "speed": "standard",
        },
    },
    expected_litellm_status_code=200,
    expected_litellm_response={
        "id": ANY,
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "vertex_ai/claude-sonnet-5@default",
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
        "model": "vertex_ai/claude-opus-4-8@default",
        "max_output_tokens": 64,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-opus-4-8@default:rawPredict",
    expected_provider_headers={
        "authorization": "Bearer scripted-token",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": "scripted-token",
    },
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "anthropic_version": "vertex-2023-10-16",
        "system": [{"type": "text", "text": "You are a terse assistant."}],
    },
    mock_provider_response={
        "model": "claude-opus-4-8",
        "id": "msg_vrtx_011ChJAfK6urZiDs3aHUbRCF",
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
            "speed": "standard",
        },
    },
    expected_litellm_status_code=200,
    expected_litellm_response={
        "id": ANY,
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "vertex_ai/claude-opus-4-8@default",
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

GEMINI_3_5_FLASH_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={
        "model": "vertex_ai/gemini-3.5-flash",
        "max_output_tokens": 1024,
        "instructions": "You are a terse assistant.",
        "input": "Say hello.",
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/google/models/gemini-3.5-flash:generateContent",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer scripted-token"},
    expected_provider_request={
        "contents": [{"role": "user", "parts": [{"text": "Say hello."}]}],
        "system_instruction": {"parts": [{"text": "You are a terse assistant."}]},
        "generationConfig": {"max_output_tokens": 1024},
    },
    mock_provider_response={
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello.", "thoughtSignature": GEMINI_3_5_FLASH_THOUGHT_SIGNATURE}],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 9,
            "candidatesTokenCount": 2,
            "totalTokenCount": 95,
            "trafficType": "ON_DEMAND",
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 9}],
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 2}],
            "thoughtsTokenCount": 84,
        },
        "modelVersion": "gemini-3.5-flash",
        "createTime": "2026-10-05T22:54:50.809002Z",
        "responseId": "uirEaqqwMc-U9LsPjJTYuQU",
    },
    expected_litellm_status_code=200,
    expected_litellm_response={
        "id": ANY,
        "created_at": ANY,
        "error": None,
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "metadata": {},
        "model": "vertex_ai/gemini-3.5-flash",
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
        "max_output_tokens": 1024,
        "previous_response_id": None,
        "reasoning": None,
        "status": "completed",
        "text": {},
        "truncation": None,
        "usage": {
            "input_tokens": 9,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 9,
                "video_tokens": None,
            },
            "output_tokens": 86,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 84, "text_tokens": 2},
            "total_tokens": 95,
            "cost": None,
        },
        "user": None,
        "store": None,
        "provider_specific_fields": {"traffic_type": "ON_DEMAND"},
    },
)

CLAUDE_SONNET_5_5_RESPONSES_THINKING_SIGNATURE_0: Final = (
    "CAQS0wkKEggUEAIYAjgBQgh0aGlua2luZxIMvc2gVcf3TKUGh5LgGgzri2+movs6Xp1fNUQiMBi3tJV3peQw7YOlZsgUlavAeb1L"
    "dUC8Ze7GGNTeHG1PMv/N9E8CbdkK1Ggzz+A0DiruCHvT2JjNp0WYfE0db9FZ8Uv5NG4G4R4NI4BhYvcHrp6K5+zTl6M0p/IKfBB+"
    "74445dDQQR2sTgSE/xtn2s2I1fu5crfmte+xtCDkUfU5VGso5Ewg38YCJHgE8jwnw9ncqbwlcmOnC1W3Lf0pZ3Y9tmf0U5Isou+f"
    "vVmJCKJg5gOWz0bQXtefTI24QcKMfSlKgFWEy07gMtCPwcZj/0Ip1EdVw2wexWl0ad9bgUwI/oVbpx8LOEMkhlOWbAwXDthVe83t"
    "D9wqOkmDcmb3tp4hpGDaXtKDbkVZujql3ATDYDAVZleDnsWpisLhqNJQyQpHfkhtV1udbDlUH1ejsERNYFLMkThYa8lN6V3UUNWO"
    "4oEm4oQ5F3qgL6BP7VpYdyQMkOTamRt1WmJ6Tat7D/061Kw/u/9w5VYA+EbUjPFEwlveY3/P6/F+Z6W5fuX1oEQJoC/sEvRn7nHI"
    "8bTngKOTZ6WzbrJDZANlpEW+eQ6Bu8b5bGRAQ6XtHTE1V9cKw9X8CsMS+saZDZ25w38M7+idANyZljWu0MrbHNeXtlnkSbkinJfR"
    "iNe/7wkROW/OP3rYYQRhy+1adRWH6xVLhXCmYS8VDjNliyQpDkDLryl5wQX1oP6t88xasAi7OQikVyo1HCw+Ygrz9qmVr7qk+b4R"
    "3s+SniNLmJ8v1KVCDEb5Sl8vmive3fN1WQJpO6tAtJiaHy/SzGxg6fvaq+ZCvTw3Icb+ujUzqpJKECUkF0J/GWKRyoXCAiye4mpI"
    "3Qf1Hoi3sf88fyB8vrq0Tb+G0EL3CLo6NbZrHYRjKWmYCqykWtET09Q+roEuoSU7DoGjC8MzprqUB6lfymL9DnZRD+1vklSbZa0w"
    "aELAh/G7c0vuOgp1NGYzcmtUuFmVB1SE/J85ynVQSloWGxjVcA2DFlDpgWcI6TVmECH7EiNveiSRMeX7N/Z6hQvioYCAsMuD6lmJ"
    "gHAj7hBrSsxVrDEL5KjnSWV9F7IznxLhXeT4EJExcxeUd0J5fT8pW2tOUwED5ZkHlTCuyE6CrqZ+QFP0PceF3+is2rgnd33zlXEM"
    "KmVbbwOSsrC2R3fuUBU7/8eVY7ZMtmwFEGECrMsJhdxXwrcHclSEgUwx/Pp88scH2jw3wfu0THXWdV8osZvq0gic7uiN54jE1LtF"
    "5KsjqsGqsywQjldnr3f2WxDcIGJni1/CqpuJRxo5ooOJLKxMRbyo98oZRaWWOfCP3CGfEWkD1i0WPevoSpsolou3HhMaxFUfRO5R"
    "bUasHeivOgfK5IJbDRC+B6msD7JxtiML8nz89qwCVU3oxje67ZMe459RJtcA9VSCVf0yHqW6PGLcfN4kzxqRS+wSqdMZcHGbwks8"
    "lqbNZYEdRwwelbyeVrsLwGbdDIV/bTIBQ5/z6OP/fF6d8GS6D7EnT9l/vhFf9MiuJEfh940efe8nKcJNKFTT0afHkWjaeFsyigKp"
    "wZDSewaADhI2sfwQUozVCSqWpBjHf+CPhTmpENS3dgcjTrLJkQ3n6BgB"
)

CLAUDE_SONNET_5_5_RESPONSES_THINKING_SIGNATURE_1: Final = (
    "CAQS/QUKEwgUEAIYAjgBQgluYXJyYXRpb24SDMX15vS87/bkxpLprRoMWIq8keWZ0GdK0scvIjD0rSXKmQ4b5BSq8TEuvmfUVExm"
    "yUPbx8pdAVnI7KcR3ObUxEGFkddoDhdAq/EVe/cqlwXzdbPCiuM+6OACa4QamlUB3nkwOtx61dioigK3l7w5PAwu1yiR0DgOyH1p"
    "IegzDYjYqtQJqX2iLiZWlOkDQUba8zA5EQi3IWC7NMy3P1uH39VA9gs+o2ueKlCjm8vwgmfaie8+aitmM5q/xV88/ajAs1zvvVCx"
    "brUwxghm1pRYTwP9DS6Y0Rka7bxHRPFZd8GsaIMhkORqNurnkxhIQVUHsbGeve3BndXWxg+isKGHndOcMfinS+sqQdXxCZ6IjBXO"
    "rRJ31lJpTaGuMeHInlZSPve8HgxstOtqBjRo0ip3MBbtMnOAu18NzKfTAQe73gDwRtO92bs0Br76dMPGJ7Sin0z64WX52JQKZntp"
    "bGjqPbFjrcM1bT6MNc05x1J5Qg+cgyZ8giUyHl5HYRgq0mArcLKuy7OX3w4q5mAleYheovXky7GptD25WZKVd+q3ulmpRMhviSbV"
    "l/rmmyV8doffCz4McLiMvjcoUoGbvbmvVc4Tm5M+oKyV7CN7u0zWhHynZ6jck316nZDK/7aSZGEkjQuMJF1pBvZFOi+IBmd9iDij"
    "qH/w75Uy4Elgf5k77lG+ukbsnpyDMNmeJmShGbTaeZXKkvyNw8ZAkU7/O3QJXlzXw3h20misuJwRN6y+fo3gi+kKD3Hs1PTXWLfI"
    "DFVB0BB7sFcG2KXAXNjVpsccnOhDxiYPPQ5A9Aoi5uzIEJfbs1o8o4Zvw1ltl0TSSU5eeqbG7b7gwQiYytECg1V5JxC8JGJagqeq"
    "9hFHsx9xz0JQ7gvlsTvEJkm+mYgFnmuVUdjgKUDBznjPBt/9XEl7jiQLdFiCkTJqfEiwzwGu+nJgQSiCj5BAW0odYa2+rc4G1jtT"
    "6B3RfpMBcWTkABgVS4p66Goq6B0YAQ=="
)

CLAUDE_OPUS_5_5_RESPONSES_THINKING_SIGNATURE_0: Final = (
    "CAQSrQUKEggUEAIYAjgBQgh0aGlua2luZxIMdji7Q6XlLNBgAS1vGgxGN7FjuexLN6BcfpkiME3pKyBqlvuw4KCfaiQBCLeGugSh"
    "bOVBc/RrPbq1Ta2C74JIUDhSO9Q0j4KDog+yNSrIBIW4fE5E566Kgzmtg5DUjKfqJNTAYvUyaHrXLkw3hgZOjV2+PvoUh8zlZcfq"
    "iqkP/rxTYVN9/oaS6kpoZcllOQkIf6pyUzZSHISbqSo+9ZnxgZ6wvK8LVsRHdDsvQiExtPtd1DR45LjrqKzSHXPY1p+gpOntaAcU"
    "N69R29CeVy1J+Q0tWOY1++hSd9jloPli/6ORx8QmfttLtRfudpLynccDjluHgWjlnY4ImDX/22R/yP72NxjHNMcigSH5dAa0VCy2"
    "JkNMnwggojgqRy8EU6yBeYOrnnnMkJ8Gj7khVKJ5akAOfh0m0pAbwTam3pksx2wp/fDr54Nspi9w7tGL6PUDnoQ/Px9Xq2Xg9z5k"
    "rEPuzejpkOPnZxEXzgf5evlHXdMY6YcsRoVfF4o92x4Ajk4jdV2MDmcopPz+GD3JnfaiSn/ge5HjYhIXw5Kyl5yEeyBRA0y+wyTK"
    "loxLE0/cxbiKTc2wB/+tIGVGY3zj2pvSZUSgdYslOpuGfDk7HxCN16CecnKl0jiQHstGcR6DILwCz+Fwvw55DDt3RfFMSC9t/qB8"
    "X0VqRNysiemdKgdAgLsN4V0GYxgXaI4owIft8PfJRv+1HSP6b/6i21F1WzlPkRy3uzIUxF6iC+v1/+dnLpIwdSx4Yx+VpbayTARq"
    "jLJB1tKTv5zHxXFk8Cbs118jL8sIthyjR04qaUgQ3y1bL1vcKLTvMASz7yRt7Mo3JtbyIkk+F1aKmFje0BTFeu3NLwfv6+AUztoK"
    "vxdLjJtGh0G+eH7PGXy9GAE="
)

CLAUDE_OPUS_5_5_RESPONSES_THINKING_SIGNATURE_1: Final = (
    "CAQSkwYKEwgUEAIYAjgBQgluYXJyYXRpb24SDPpokyy3U6eVehXBEBoMTlKlTqWU0XOMv4f2IjCaNLfibgPtw4FasYBQ0gkm0aK5"
    "nF/7zONOubIzCpO3Edkwz7Ygvw7zVCMrMAhisokqrQVXkuAhA+9AOtxDUc4Jez1QVqFffKYdn3/C8ZkPx7+XL+AXdPHFDj2A5PDj"
    "q2tfuhzuUAqE6GKpCa6G8wZnaG0bt9j9ts1hQV0tKv7dap5xoaapNkD147XzmBGrNQrDIBnMujJywqnkVPGKosDpq67ooHsILNss"
    "pPg3vptPnEGur+o06cDyjDkjTcCstjxiv48AtEwtg7LY3gnVE5t3V4gPLCmMIV8AIfnPcl/OpLdBGPgwQdASo+RzmBkACppWpPlP"
    "NVxE7s6ZCv0DjmYAT6lUcXLLlda2RKkmAGxDD2ktu7omdkqyV23QPW6EQDWOT3q4qsX6ii1vIMWSrgTAGrvV0c/X/5r+zjuaDgz3"
    "hGcWbmC/5yiu7eO2mvOYICvClNbH0pCopLRrMTztzQuiMcHSrLSrCSAaPDj+6okVkPiIuPTD/BFvNXxb7R0bnxEbdJMvUXYUQm58"
    "l0qDw7AZuOpvtuxRX9vOVsvueJHtAo/csBmo/ot9gr/cbXNYEvyUv+OhrVvJNlDTD5FWjzuD6ihjYuVHC2Tn9CgfSGGgejJ+JMPf"
    "ITUOnTYY5p3ElN2uhpt93fWNZ84E3xPq3GSCUQrgZNSRJrtaa+jwYqMup5W4y/1/SOPcHPf32mdzDEdFXzpiHY8IXS0qQgP9cLSo"
    "2tenHpjOOceON6AaO+RewoHq92urtU5BPunIsEO5/+qDCZr5SxOb8VrY0SxHNVAJKRhFj+V3hAu0k2zDIX/IIerRuvCML/qsD8wZ"
    "7NXUYQDXF5VPIWNi8z/1ke/TXZmTD+lKTC9TavTc8EhGXNUv53ly7oL4JwhT7TF9nFOJVCNk92WaHy6Cf3zZ+GuMqO/P4s/Vybvn"
    "zn0o04dTHB1Q6Ilr9RZhRFwue4nadUKHvNF70A/ZeaGp9R+R8dtCMEv3GAE="
)

CLAUDE_FABLE_5_1_RESPONSES_THINKING_SIGNATURE_0: Final = (
    "CAQS/AgKEggSEAIYAjgBQgh0aGlua2luZxIM2adaKStJUdqWjb0JGgzx6a0ntxWBsWf5SPoiMJUxhiE/xFsZl3STVS3np3WJ/BKT"
    "beLsjANHlerzAYMPNBVAhHn1udBy1TF2WKg5HCqXCPpPMw5IgVP28vObMyGSjS/x+oIuZ28MO+NKO2I9UT9jjmIpo09b4xA1gVML"
    "V74CHnfCfqAiSyVxuu5tltOyp1Jn8ctKef4JPBV4KRYl7wRV5MUKjicadiTPfsJrSDigGXTlNBnP4kxP7FHx3a39sBMOZaeqcM+6"
    "PeiXYCkNQdFWMIlRkELFdSTwxiBJLzzv1cCOQWKic4RxLuL/9vys/Yi+lNFHweovcRg7YbQQ5SisK4wSUoaFvTfOQbvHeN+THhfh"
    "8YCJFz0eckG2tAJ641KFkuanRvvvTrAW8d+ObpoHjtFpqF8t2NGqmKLnQoWxoYixWS+KPOnOKNs2qM1LGI2Ng37JlQGRKhHEmdHb"
    "uTv8MQCZYAeyp+6VQM5cP2CS7qsae8aLJ4uLAkwack/chb/QxLtpmkfwePDcPhmYC+mIpaWojs/M2XfbJmPTa4w2NklBELbpvTue"
    "QRgGhOxBkY4vhWu73zeJ6/NM2CoiCpoLJnty5eM//6/IA95o4cIzNY/3WPblFKh+fVV+Ilsdav3KvLyBGnmt7EcM6VTmLmi2Idvr"
    "VuHaaFNB0PRSNPP6TXlEd0AKOBkCLZpKd0wW5vRCo/tgbj7Hat3ZC8ZHGMJhnGGr714wWH9Wog7qZkj+F9C5LlythT24fMt+C4Fl"
    "lK76u5XMxI9tbfirIq0sMkzqEaJk71QqkCvr6gs8lSJyXhaWC17X8enFEIxcMFlX6NDBsLYs5YVUUqlNq8Wh1JuX/irKmm+T2OIh"
    "+7WHUnmTKAgQpEOIuzA5ia95dLxRWP/jCUvwDRbnvug+qo1iQBz3cOkpKUjtbIEx1ory/wm3wEf61zaszTOJDy9wWe/BqpyS5nJO"
    "g3ZEKgy7Yjle2J411sQc1iyZ3dVuCrGLsZoa4ANoBF6SFcm8evZcq9AWWExIG2mQtX3y2SzWyIIw18/w2iTYTniO8LFkzNZlokDj"
    "e6SV/6d9Zpcvm75FtQurYETKQd2WPJ6jnYZKPR+DYVEro/urBAYtL/2fII5pFNi8RMJaXOsYqmHllDWY2a6fX6KzlIjod7XDBatD"
    "IwkUkQNUsz8Tmdo/OLY1kcC+bWjPcXzod/Spxc4MUeung8UpF6WUm6yLyTAbkdieIaTDlXg3SlEFWz55qHhskzHmi6i0vOCly7Dc"
    "nngc1swUh337xmnQIKZnrKYom9X4UiItcex4cQ+C8oKjmHdZT3l3zA6b7m8QZZpd3V6JQQC0IDOHxbvtuO/OC14OCjzeDzBKnJmv"
    "qJIgA7JG3hjb2eDPDGp/THA6XDpKFOHjtRX/dajRL5NpSw4NFxrCOCN9UTXmzFqC7rwM7IiK1Eu1VYPh6gXKlDwqCP6hDu+zEajU"
    "HOu0HiQvlm3OECugNo2mtm5gbrtrYq+LYZ8MjxgB"
)

CLAUDE_FABLE_5_1_RESPONSES_THINKING_SIGNATURE_1: Final = (
    "CAQS5gUKEwgSEAIYAjgBQgluYXJyYXRpb24SDKtQ5HPiP+fYCKzUGxoMyrwawBjxXUkBTvgEIjCDIS/QuqY9Vm/JNNkKKFxwTX58"
    "kwQICd+wgim/pJgr2sDZRzdBVklH+aNI+pFM37UqgAXXpvcziXDsrtYuOqyrTaXNs0t7+LUPjZMRiY4pfcVvJbdJTq0R4HF1VfsY"
    "GIOJuJLKw2uu8JM3t3+wILwrnaLBjs/nHojjdOuCCLEzatKp430zN7+slVpY9GmEUSnJunVI38Xye9f3YA3jAl/yEtrXab4nFn+w"
    "KOx9uTlHg3s7TZC5W6h8H4dfb71fnWFU5q0hBGw67aKa16tr3xU6/J66gOlFnah7LygzYwn2Dv9O34Rf1WMzIXZgf0vv+BWtCNP+"
    "u3hOtSlAprySuawJx6T0hSXqDIcRjC3VQqnDxw1M0xEQSegpFiBEZsrxcS6xg2AEpIJZaortESXGlKDv4WSZ/WvwgcKlZY1IttNd"
    "TDlYvRwlKiHv7R2KVGzpyCx5avIVV/I+QLAsiWfoxwhauiFSnviTngaoOujAYrKe/KTZi3Orx0PIKDga5829zgwSwRkJn59iLcpr"
    "D4clFG2xBRSnjxa/g806JHH0uK/Ko/wJ190f4oV0CSBlv2A/4Gd+7OwwTDmXpc+Q9uhhcam/isnJOo93Y2qNCYYBLkv0nHaXry9G"
    "UD0yqRT3swkDdxrX/08DprwAtht01SmPl/yRVG26stOuNRXhPFqat5AmoiWQbMG8fYll3/lcqn2LBxnQZFTN8LVZ/Hu3NiEi11Ui"
    "Rq7O4owEeyvdfh5YAjc50bhzTWhi37W9Y3YcbuAcOsMlOSLV9dwnTz2WGfSPDBXuFV5ltbbN4tHfWG1l2zs8+MQ7FGBHQgO6aLP8"
    "ZbNJW02ePpv0OOQnbw3DOw83MZahNfOABCQvwSsW6//YPhbSu+fZFcIwhFf13ydB1ZRu2xY/9wpfQTRKydDUEw0DyGOX8aC/GAE="
)

GEMINI_3_8_FLASH_RESPONSES_THOUGHT_SIGNATURE_0: Final = (
    "AY89a18zqWYsP1fCe9VRYni0L9f03IJNDHvm1boNA4Q2dVNJPzZZvrxcHGuC7kCwx5e9kBdKsGEQ9/Nr4AjO5c6ukC2h2NEeTinm"
    "hak8IRJNBkDQ7OZ7at4q5KYoqYfDZ8aNohtXWBxVdoSK9YZHp+nPGYE7zmXk1icksF+WJG0ga0jBaivzjYeChB7DEp/pEfxa5e/5"
    "etULNzsfurb4+l7Aht/DYaH/Z+EEttmrxrDlKI9SHGOfDy8YRn7XcDJyjYjIJ/Hgcoa3/STN5kaEJs+7FzyrN64sSOyhzsP1H5dS"
    "M27anFBLUjs5KHg15CoDirSbIlRLlOHJWQ2k9Ql3ljBfNePDAds4/zH2VchSqiZfsIS5XwUGm8D7L565vzTzknZtb+zSZivDPPs/"
    "fCe3iCNYY4YEeSt2Zz/hy7baM06u6Vb1cljWtmckLoQgpiHJhY/yA5IhrHLqTx+PnBTgud6+yx7r/9Wiu8ttxeU8vupqAERoBj9y"
    "OmysvjloE6N7g/9qwsRvvoxICo8luVQmA1xWqkDMTLT3LNfEwIpWmEr0Tx+y4chAZitmuKUBZdtPfrayO96GB7iPjRUarEKs/px3"
    "hHyaSs7QzZZM/ymeeihqYyWzr81GN14/MsHs72eX3sl+ooYhxM/qhpOSjFiMgffLrD7NlAwcBtXfNI7fQWjbkLYf+mu6hCMSY8gs"
    "hIkSLFcRv8ijH3eK2RBQWzvDsjkNgiuTlXetgYFVe5/Q3YJvCCdPsbn/Fd/dc3etq5XTGJ9B1ecsI7KFdWOX8Q5HrrfWFOuo7BuW"
    "XNnFBN7pd1RwzWI3cveysx77EgQ/fMS0Iqb9FsM4i8SIY6tW5KIj+Q0SdnGR2Fkh214QxDUwT53ThPDes9mapBpR4lxIqvROoMcM"
    "PoMFlcvENF9S8oeE17cifQTVFrQT6xGeWL3vV7AVGVhuP7RjIX9cU/NK2qBof50qfI2BC+aqiu6WtzWQ2IdfrhCfCChccr0Jh4Il"
    "dyF/0mW45PijT0mguv0vwceuVrtwLkYSK7gsnb6REGUmQ4WXWy9VwrCmcYzmX5igAN/ml0sML0+k1pDHtYVcflnjAT1s/wiQFJ4P"
    "g1vUXmpjGSOUb7KiyKeiUvgTTdqD1EXHAWxnFJtgwwz/6p+VhkiZvvwOJY/GSciX14pMtA57VsGd+2eUTRdpuEbffVWjuuyeyZd6"
    "Xgm2PNLnCIvP8sS0FLJAKhOYau1OmGwkl5gsteZqSgjg2l7hXJf3lHhXv4UrRRj/YwX8SEg/YPSzhurb3bBdLKBUPfQs3h/DlOnG"
    "bQqi7efhq5seYJBldPpiEtwChm7o6mJ6J4cr1AqLTtDmQR6XF5ZN/oE6XGHpa2XY9N4tJmSrnIq6kuunngTfpD7yejAg9bHVy8r4"
    "FWjU9wU2AgYHBZneDXTiVKDaJDzYPdoufiADokGyWntWksp+B3jH+El+kI7c3uu/tL28HVep3u8Jz/XlLqEgh4oViCrcsyqqjLsU"
    "rfQ8oDmiDJZfbxECR7GHDxJ9yA=="
)

GEMINI_3_1_PRO_PREVIEW_RESPONSES_THOUGHT_SIGNATURE_0: Final = (
    "AY89a18mH4gPpBuF8sZq3BxEt7fVfeky9tDk6WrFiAYqUNfy1TwKL5UEuc60FQFzZzoY6BhtmNYM/YnvWntgJG06tpD+R69L+px8"
    "wJWN2jf96nYOxDNe5xM3Z6yz7QO1XikWun/bqStP3GVvN2xCTRBeQxkN0n5+NPDWYXC94zo2uikRYf1XZ5IsCrbwhDgFhqzyv+3C"
    "pEmBpN7ghzonZp6GL23N0ELNi95QEzZMcGrD31Z3xx6Db+KblQL9yVKVj7STHJ4Bno3sU6heJx9NMJbM55wI5SNcZqi8YqkS3Kli"
    "WRxcPdWbkekAOCWUFYYFG0JEB0iOibkqcCV6j7bZaJD3v9qdqpaSbG1Q3yl5Ncb1KMMGLwbMyjN9t3XrFQuqtM1x3OE8hofM3bNP"
    "MYGna3DDEmlporSsQlE6W8Cxa70IsBaFY5h7eQjKCkxLSOhZmGfmuukLLpxXWG8ZHzdtOB7vmpIcBolW2TcQCamojGpvWdxKz/Jh"
    "U9WB9xac6A5luWXnIA+YC/Pu793XzkDrST8jT4bei8Wi37sfAhRUkVeUw6VAAYbZKONNM43U/ih3BwJvpJoY5voZVyhm4JTT66q9"
    "vSTUSiR4oQz/ipXDUKVfWFhqEWoh3Nh5elhFe1oZyx8q+lGwXm4VcrGNtb5xwWGnAsQN9GV30egFGNcEKFq3rnODehteSTTaHF6Y"
    "aqklMypgS/K09SfshF0waKvX0GDouVMbnO3bFfMxGU1CofWB3b3q6RbIdv+O37WekKo0O8S6OwQRdpfOCNXcwpwsRczX4s9IoGRK"
    "5IIKr1/6cQze7LA2Gi00gBIe9RpbSMJHo/XJyDsP6KT1IAAtoEkyGBzsVjBEU4ggVsG6n8ka5y4myFgwMQeNrN5DNqFL+nVzxqUF"
    "97pRU7Xw37SlNNi9DgGnqhyWY/T6Vvr4yhCsYR2BElGCq0C1YUF9OQ8+gp2CBrtd+bRLHz5iNyUMoPumSAn6B1+GyU5JFPQwC9Yv"
    "NK6ug+9s2akQgYYgNAnDS7aXj2djQ57nwPqnsg=="
)

GEMINI_4_ARGON_RESPONSES_THOUGHT_SIGNATURE_0: Final = (
    "AY89a180ZmYN6Zq/ICj0BCu+eBS75TWGlATeX2fsIhoKaFuGLfTjmizARO/YOhSzt4TK8d1jl7skKbDq23TdvRVOJfjPsI85UiVC"
    "F8DZC4gSdHQbTXiDtBUsue7BxMBnoNfPJe34Y1p6WqjbaK44zsfJybZVsWJSmQOW5Ky8qnBZXfEL9dyz2j0rQbN5IMGUV36wrA4v"
    "KLfUj6vS1dwzz2LETuK1rT/EFKXDG2Pwgo11yMzoZuPvC8VvNrQXqJ/Tizd/cuuc+Fiht7V4EvrEDnDH0ZFMW4gyS52mfjE8/hnI"
    "FIgzaFkhP7mWHBOwEt3ZSPZPQ2d6ZEOXD+1clQXSOWMv9Ks4Iwvy8hrcri60cdqbFdbHxF48OgfWUy2dUycP6kuKwWUJ+sDNT1pN"
    "lb6pfhKdmP7wd6QvjudKrrunGZbWr9CjkiAGVOD8RfcQK8Z7LUC11RyHqGDXLxJA1kzR4W4wct1Y6uWF3TRN8QsSLpuXLS+sITYs"
    "vEaRtT/wcCBVYu+dHephEAo2UXmaMfkZ8QxrWtf/VOwDZegQiVpBkxOZyAoByiDJrCwFa/osvXXcj0BtweIKpKKUdLy77BVE7IHa"
    "JikXSIRmpwYZRnA="
)


CLAUDE_SONNET_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "vertex_ai/claude-sonnet-5-5@default"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-sonnet-5-5@default:rawPredict",
    expected_provider_headers={
        "authorization": "Bearer scripted-token",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": "scripted-token",
    },
    expected_provider_request={
        **VERTEX_AI_ANTHROPIC_MESSAGES_REQUEST,
        "system": [
            {
                "type": "text",
                "text": (
                    "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
                    "Explain which file you need next, then call a read-only tool. Never change files. Use the responses "
                    "client for this project review. Consider zero, negative and non-integer quantities before choosing "
                    "the next file to inspect."
                ),
            }
        ],
    },
    mock_provider_response={
        **VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-sonnet-5-5",
        "id": "msg_vrtx_011ChJAfXeibANtUZMM6eNQx",
        "content": [
            {
                "type": "thinking",
                "thinking": "I need to read validation.py before anything else, since the user asked for it first. I'm also weighing whether the test file could be read in the same batch, since those reads don't depend on each other, though the user's ordering makes me lean toward validation alone.\n\nI'll check whether reading the tests alongside validation.py is fine, since they don't depend on each other, but the user asked for validation first, so I'll go ahead and read both to move things along.\n\n",
                "signature": CLAUDE_SONNET_5_5_RESPONSES_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": "Next I'm reading `shop/validation.py` to see how `validate_quantity` handles zero, negative and non-integer values, and `tests/test_validation.py` alongside it.\n\n",
                "signature": CLAUDE_SONNET_5_5_RESPONSES_THINKING_SIGNATURE_1,
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01MgSj33NjzUQH1GHcBapZdP",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01PKoiHzUkfqVMiybYwkpGQy",
                "name": "read_file",
                "input": {"path": "tests/test_validation.py", "start_line": 1, "end_line": 40},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 506,
            "output_tokens_details": {"thinking_tokens": 228},
            "speed": "standard",
        },
    },
    expected_litellm_response={
        **RESPONSES_RESPONSE,
        "model": "vertex_ai/claude-sonnet-5-5@default",
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            "I need to read validation.py before anything else, since the user asked for it first. I'm also w"
                            "eighing whether the test file could be read in the same batch, since those reads don't depend on"
                            " each other, though the user's ordering makes me lean toward validation alone.\n\nI'll check wheth"
                            "er reading the tests alongside validation.py is fine, since they don't depend on each other, but"
                            " the user asked for validation first, so I'll go ahead and read both to move things along.\n\nNext"
                            " I'm reading `shop/validation.py` to see how `validate_quantity` handles zero, negative and non-"
                            "integer values, and `tests/test_validation.py` alongside it.\n\n"
                        ),
                        "annotations": [],
                    }
                ],
                "phase": None,
                "encrypted_content": json.dumps(
                    [
                        {
                            "type": "thinking",
                            "thinking": (
                                "I need to read validation.py before anything else, since the user asked for it first. I'm also w"
                                "eighing whether the test file could be read in the same batch, since those reads don't depend on"
                                " each other, though the user's ordering makes me lean toward validation alone.\n\nI'll check wheth"
                                "er reading the tests alongside validation.py is fine, since they don't depend on each other, but"
                                " the user asked for validation first, so I'll go ahead and read both to move things along.\n\n"
                            ),
                            "signature": CLAUDE_SONNET_5_5_RESPONSES_THINKING_SIGNATURE_0,
                        },
                        {
                            "type": "thinking",
                            "thinking": (
                                "Next I'm reading `shop/validation.py` to see how `validate_quantity` handles zero, negative and "
                                "non-integer values, and `tests/test_validation.py` alongside it.\n\n"
                            ),
                            "signature": CLAUDE_SONNET_5_5_RESPONSES_THINKING_SIGNATURE_1,
                        },
                    ],
                    separators=(",", ":"),
                ),
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                "call_id": "toolu_vrtx_01MgSj33NjzUQH1GHcBapZdP",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_vrtx_01MgSj33NjzUQH1GHcBapZdP",
                "namespace": None,
                "status": "completed",
            },
            {
                "arguments": '{"path": "tests/test_validation.py", "start_line": 1, "end_line": 40}',
                "call_id": "toolu_vrtx_01PKoiHzUkfqVMiybYwkpGQy",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_vrtx_01PKoiHzUkfqVMiybYwkpGQy",
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
                "cache_write_tokens": 873,
                "image_tokens": None,
                "text_tokens": 4,
                "video_tokens": None,
            },
            "output_tokens": 506,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 228, "text_tokens": 278},
            "total_tokens": 1383,
            "cost": None,
        },
        "provider_specific_fields": {
            "citations": None,
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": (
                        "I need to read validation.py before anything else, since the user asked for it first. I'm also w"
                        "eighing whether the test file could be read in the same batch, since those reads don't depend on"
                        " each other, though the user's ordering makes me lean toward validation alone.\n\nI'll check wheth"
                        "er reading the tests alongside validation.py is fine, since they don't depend on each other, but"
                        " the user asked for validation first, so I'll go ahead and read both to move things along.\n\n"
                    ),
                    "signature": CLAUDE_SONNET_5_5_RESPONSES_THINKING_SIGNATURE_0,
                },
                {
                    "type": "thinking",
                    "thinking": (
                        "Next I'm reading `shop/validation.py` to see how `validate_quantity` handles zero, negative and "
                        "non-integer values, and `tests/test_validation.py` alongside it.\n\n"
                    ),
                    "signature": CLAUDE_SONNET_5_5_RESPONSES_THINKING_SIGNATURE_1,
                },
            ],
        },
    },
)


CLAUDE_HAIKU_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "vertex_ai/claude-haiku-5-5@default"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-haiku-5-5@default:rawPredict",
    expected_provider_headers={
        "authorization": "Bearer scripted-token",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": "scripted-token",
    },
    expected_provider_request={
        **VERTEX_AI_ANTHROPIC_MESSAGES_REQUEST,
        "system": [
            {
                "type": "text",
                "text": (
                    "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
                    "Explain which file you need next, then call a read-only tool. Never change files. Use the responses "
                    "client for this project review. Consider zero, negative and non-integer quantities before choosing "
                    "the next file to inspect."
                ),
            }
        ],
    },
    mock_provider_response={
        **VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-haiku-5-5",
        "id": "msg_vrtx_011ChJAfYKtmiiNeYQHNwyEv",
        "content": [
            {
                "type": "thinking",
                "thinking": "I'm going to read shop/validation.py, starting with lines 1-60 since I don't know its full length yet. I'll make a single read call, and I should first say which file I need next. I also want to check how it handles zero, negative, and non-integer quantities. The tests could be pulled in alongside, but the user asked for validation first, so I'll stick to that.\n\n",
                "signature": CLAUDE_HAIKU_5_5_RESPONSES_THINKING_SIGNATURE,
            },
            {
                "type": "text",
                "text": "I'll read `shop/validation.py` first, since `validate_quantity` is where zero and negative quantities should be handled.",
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01HivpBXDNZ4qTbRKPHwxZzX",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 249,
            "output_tokens_details": {"thinking_tokens": 110},
            "speed": "standard",
        },
    },
    expected_litellm_response={
        **RESPONSES_RESPONSE,
        "model": "vertex_ai/claude-haiku-5-5@default",
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            "I'm going to read shop/validation.py, starting with lines 1-60 since I don't know its full lengt"
                            "h yet. I'll make a single read call, and I should first say which file I need next. I also want "
                            "to check how it handles zero, negative, and non-integer quantities. The tests could be pulled in"
                            " alongside, but the user asked for validation first, so I'll stick to that.\n\n"
                        ),
                        "annotations": [],
                    }
                ],
                "phase": None,
                "encrypted_content": json.dumps(
                    [
                        {
                            "type": "thinking",
                            "thinking": (
                                "I'm going to read shop/validation.py, starting with lines 1-60 since I don't know its full lengt"
                                "h yet. I'll make a single read call, and I should first say which file I need next. I also want "
                                "to check how it handles zero, negative, and non-integer quantities. The tests could be pulled in"
                                " alongside, but the user asked for validation first, so I'll stick to that.\n\n"
                            ),
                            "signature": CLAUDE_HAIKU_5_5_RESPONSES_THINKING_SIGNATURE,
                        }
                    ],
                    separators=(",", ":"),
                ),
            },
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            "I'll read `shop/validation.py` first, since `validate_quantity` is where zero and negative quant"
                            "ities should be handled."
                        ),
                        "annotations": [],
                    }
                ],
                "phase": None,
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                "call_id": "toolu_vrtx_01HivpBXDNZ4qTbRKPHwxZzX",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_vrtx_01HivpBXDNZ4qTbRKPHwxZzX",
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
                "cache_write_tokens": 873,
                "image_tokens": None,
                "text_tokens": 4,
                "video_tokens": None,
            },
            "output_tokens": 249,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 110, "text_tokens": 139},
            "total_tokens": 1126,
            "cost": None,
        },
        "provider_specific_fields": {
            "citations": None,
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": (
                        "I'm going to read shop/validation.py, starting with lines 1-60 since I don't know its full lengt"
                        "h yet. I'll make a single read call, and I should first say which file I need next. I also want "
                        "to check how it handles zero, negative, and non-integer quantities. The tests could be pulled in"
                        " alongside, but the user asked for validation first, so I'll stick to that.\n\n"
                    ),
                    "signature": CLAUDE_HAIKU_5_5_RESPONSES_THINKING_SIGNATURE,
                }
            ],
        },
    },
)


CLAUDE_OPUS_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "vertex_ai/claude-opus-5-5@default"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-opus-5-5@default:rawPredict",
    expected_provider_headers={
        "authorization": "Bearer scripted-token",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": "scripted-token",
    },
    expected_provider_request={
        **VERTEX_AI_ANTHROPIC_MESSAGES_REQUEST,
        "system": [
            {
                "type": "text",
                "text": (
                    "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
                    "Explain which file you need next, then call a read-only tool. Never change files. Use the responses "
                    "client for this project review. Consider zero, negative and non-integer quantities before choosing "
                    "the next file to inspect."
                ),
            }
        ],
    },
    mock_provider_response={
        **VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-opus-5-5",
        "id": "msg_vrtx_011ChJAfbF1xaabUQHbX9KTY",
        "content": [
            {
                "type": "thinking",
                "thinking": "I'm going to read the validation file first, as the user asked, and I'm leaving the test file for later. I'll call read_file on the validation file to start.\n\n",
                "signature": CLAUDE_OPUS_5_5_RESPONSES_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": "I'm opening shop/validation.py around line 18 to check how validate_quantity handles zero, negative, non-integer and non-numeric inputs.\n\n",
                "signature": CLAUDE_OPUS_5_5_RESPONSES_THINKING_SIGNATURE_1,
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_016XqZQZ1bkRjw6xdLRvC7Rq",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 50},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 261,
            "output_tokens_details": {"thinking_tokens": 65},
            "speed": "standard",
        },
    },
    expected_litellm_response={
        **RESPONSES_RESPONSE,
        "model": "vertex_ai/claude-opus-5-5@default",
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            "I'm going to read the validation file first, as the user asked, and I'm leaving the test file fo"
                            "r later. I'll call read_file on the validation file to start.\n\nI'm opening shop/validation.py ar"
                            "ound line 18 to check how validate_quantity handles zero, negative, non-integer and non-numeric "
                            "inputs.\n\n"
                        ),
                        "annotations": [],
                    }
                ],
                "phase": None,
                "encrypted_content": json.dumps(
                    [
                        {
                            "type": "thinking",
                            "thinking": (
                                "I'm going to read the validation file first, as the user asked, and I'm leaving the test file fo"
                                "r later. I'll call read_file on the validation file to start.\n\n"
                            ),
                            "signature": CLAUDE_OPUS_5_5_RESPONSES_THINKING_SIGNATURE_0,
                        },
                        {
                            "type": "thinking",
                            "thinking": (
                                "I'm opening shop/validation.py around line 18 to check how validate_quantity handles zero, negat"
                                "ive, non-integer and non-numeric inputs.\n\n"
                            ),
                            "signature": CLAUDE_OPUS_5_5_RESPONSES_THINKING_SIGNATURE_1,
                        },
                    ],
                    separators=(",", ":"),
                ),
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 50}',
                "call_id": "toolu_vrtx_016XqZQZ1bkRjw6xdLRvC7Rq",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_vrtx_016XqZQZ1bkRjw6xdLRvC7Rq",
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
                "cache_write_tokens": 873,
                "image_tokens": None,
                "text_tokens": 4,
                "video_tokens": None,
            },
            "output_tokens": 261,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 65, "text_tokens": 196},
            "total_tokens": 1138,
            "cost": None,
        },
        "provider_specific_fields": {
            "citations": None,
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": (
                        "I'm going to read the validation file first, as the user asked, and I'm leaving the test file fo"
                        "r later. I'll call read_file on the validation file to start.\n\n"
                    ),
                    "signature": CLAUDE_OPUS_5_5_RESPONSES_THINKING_SIGNATURE_0,
                },
                {
                    "type": "thinking",
                    "thinking": (
                        "I'm opening shop/validation.py around line 18 to check how validate_quantity handles zero, negat"
                        "ive, non-integer and non-numeric inputs.\n\n"
                    ),
                    "signature": CLAUDE_OPUS_5_5_RESPONSES_THINKING_SIGNATURE_1,
                },
            ],
        },
    },
)


CLAUDE_FABLE_5_1_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "vertex_ai/claude-fable-5-1@default"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-fable-5-1@default:rawPredict",
    expected_provider_headers={
        "authorization": "Bearer scripted-token",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": "scripted-token",
    },
    expected_provider_request={
        **VERTEX_AI_ANTHROPIC_MESSAGES_REQUEST,
        "system": [
            {
                "type": "text",
                "text": (
                    "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
                    "Explain which file you need next, then call a read-only tool. Never change files. Use the responses "
                    "client for this project review. Consider zero, negative and non-integer quantities before choosing "
                    "the next file to inspect."
                ),
            },
        ],
    },
    mock_provider_response={
        **VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-fable-5-1",
        "id": "msg_vrtx_011Cg38r5dMDQufLUFGVTBXP",
        "content": [
            {
                "type": "thinking",
                "thinking": (
                    "The user asked to read the validation file first, so even though the files are independent, I'll "
                    "respect that ordering and read validation.py before anything else.\n\n"
                ),
                "signature": CLAUDE_FABLE_5_1_RESPONSES_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": (
                    "I'll check `validate_quantity` in `shop/validation.py` for how it handles zero/negative values, "
                    "along with the corresponding test in `tests/test_validation.py`.\n\n"
                ),
                "signature": CLAUDE_FABLE_5_1_RESPONSES_THINKING_SIGNATURE_1,
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01WCXdQndfe72gQeCatYYmiP",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 40},
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01MDyzK8nTHNuCpY7jS6YJ2Y",
                "name": "read_file",
                "input": {"path": "tests/test_validation.py", "start_line": 1, "end_line": 30},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 461,
            "output_tokens_details": {"thinking_tokens": 265},
        },
    },
    expected_litellm_response={
        **RESPONSES_RESPONSE,
        "model": "vertex_ai/claude-fable-5-1@default",
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            "The user asked to read the validation file first, so even though the files are independent, I'll "
                            "respect that ordering and read validation.py before anything else.\n\nI'll check `validate_quantity` "
                            "in `shop/validation.py` for how it handles zero/negative values, along with the corresponding test "
                            "in `tests/test_validation.py`.\n\n"
                        ),
                        "annotations": [],
                    },
                ],
                "phase": None,
                "encrypted_content": json.dumps(
                    [
                        {
                            "type": "thinking",
                            "thinking": (
                                "The user asked to read the validation file first, so even though the files are independent, I'll "
                                "respect that ordering and read validation.py before anything else.\n\n"
                            ),
                            "signature": CLAUDE_FABLE_5_1_RESPONSES_THINKING_SIGNATURE_0,
                        },
                        {
                            "type": "thinking",
                            "thinking": (
                                "I'll check `validate_quantity` in `shop/validation.py` for how it handles zero/negative values, "
                                "along with the corresponding test in `tests/test_validation.py`.\n\n"
                            ),
                            "signature": CLAUDE_FABLE_5_1_RESPONSES_THINKING_SIGNATURE_1,
                        },
                    ],
                    separators=(",", ":"),
                ),
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 40}',
                "call_id": "toolu_vrtx_01WCXdQndfe72gQeCatYYmiP",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_vrtx_01WCXdQndfe72gQeCatYYmiP",
                "namespace": None,
                "status": "completed",
            },
            {
                "arguments": '{"path": "tests/test_validation.py", "start_line": 1, "end_line": 30}',
                "call_id": "toolu_vrtx_01MDyzK8nTHNuCpY7jS6YJ2Y",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_vrtx_01MDyzK8nTHNuCpY7jS6YJ2Y",
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
            "output_tokens": 461,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 265, "text_tokens": 196},
            "total_tokens": 1338,
            "cost": None,
        },
        "provider_specific_fields": {
            "citations": None,
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": (
                        "The user asked to read the validation file first, so even though the files are independent, I'll "
                        "respect that ordering and read validation.py before anything else.\n\n"
                    ),
                    "signature": CLAUDE_FABLE_5_1_RESPONSES_THINKING_SIGNATURE_0,
                },
                {
                    "type": "thinking",
                    "thinking": (
                        "I'll check `validate_quantity` in `shop/validation.py` for how it handles zero/negative values, "
                        "along with the corresponding test in `tests/test_validation.py`.\n\n"
                    ),
                    "signature": CLAUDE_FABLE_5_1_RESPONSES_THINKING_SIGNATURE_1,
                },
            ],
        },
    },
)


GEMINI_3_8_FLASH_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "vertex_ai/gemini-3.8-flash"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/google/models/gemini-3.8-flash:generateContent",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer scripted-token"},
    expected_provider_request={
        **VERTEX_AI_GEMINI_GENERATE_CONTENT_REQUEST,
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
                        "Explain which file you need next, then call a read-only tool. Never change files. Use the responses "
                        "client for this project review. Consider zero, negative and non-integer quantities before choosing "
                        "the next file to inspect."
                    ),
                },
            ],
        },
    },
    mock_provider_response={
        **VERTEX_AI_GEMINI_GENERATE_CONTENT_RESPONSE,
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": (
                                "**My Thought Process on Inspecting `shop/validation.py`**\n\nAlright, my primary objective right now "
                                "is to understand how the system handles invalid quantity inputs, specifically zero, negative, and "
                                "potentially non-integer values. This is crucial for robust error handling and ensuring data "
                                "integrity within the `shop` module.\n\nTo achieve this, I need to examine the validation logic "
                                "directly. My instinct is that the `shop/validation.py` file will contain the relevant functions for "
                                "this purpose. I'm looking for something like a `validate_quantity` function, which I anticipate "
                                "will be located around line 18, as per my internal knowledge base or a quick scan.\n\nTherefore, the "
                                "next logical step is to access the content of `shop/validation.py`. I need to read this file to see "
                                "the exact implementation of the quantity validation. I'll be focusing on the code that dictates "
                                "whether zero or negative quantities are permitted, or how they are processed if they are not. I'll "
                                "also be keeping an eye out for any checks related to non-integer values, though the initial prompt "
                                "emphasized zero and negatives.\n\nI will then use the `read_file` tool to retrieve the content of "
                                "`shop/validation.py`. I'll specify a reasonable range for the lines to read, perhaps from the "
                                "beginning of the file up to line 50, to ensure I capture the relevant validation function and its "
                                "surrounding context. This will allow me to analyze the validation rules in detail before proceeding "
                                "to any other files.\n\n"
                            ),
                            "thought": True,
                        },
                        {
                            "text": (
                                "I need to inspect `shop/validation.py` next to see how `validate_quantity` handles zero, negative, "
                                "and non-integer quantities."
                            ),
                        },
                        {
                            "functionCall": {
                                "name": "read_file",
                                "args": {"path": "shop/validation.py", "start_line": 1, "end_line": 40},
                                "id": "call_316157",
                            },
                            "thoughtSignature": GEMINI_3_8_FLASH_RESPONSES_THOUGHT_SIGNATURE_0,
                        },
                    ],
                },
                "finishReason": "STOP",
            },
        ],
        "usageMetadata": {
            "promptTokenCount": 242,
            "candidatesTokenCount": 64,
            "totalTokenCount": 571,
            "trafficType": "ON_DEMAND",
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 242}],
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 64}],
            "thoughtsTokenCount": 265,
        },
        "modelVersion": "gemini-3.8-flash",
        "createTime": "2026-10-15T01:14:17.560771Z",
        "responseId": "6SjQaoOdIvCfq8YP89zFuQk",
    },
    expected_litellm_response={
        **RESPONSES_RESPONSE,
        "model": "vertex_ai/gemini-3.8-flash",
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            "**My Thought Process on Inspecting `shop/validation.py`**\n\nAlright, my primary objective right now "
                            "is to understand how the system handles invalid quantity inputs, specifically zero, negative, and "
                            "potentially non-integer values. This is crucial for robust error handling and ensuring data "
                            "integrity within the `shop` module.\n\nTo achieve this, I need to examine the validation logic "
                            "directly. My instinct is that the `shop/validation.py` file will contain the relevant functions for "
                            "this purpose. I'm looking for something like a `validate_quantity` function, which I anticipate "
                            "will be located around line 18, as per my internal knowledge base or a quick scan.\n\nTherefore, the "
                            "next logical step is to access the content of `shop/validation.py`. I need to read this file to see "
                            "the exact implementation of the quantity validation. I'll be focusing on the code that dictates "
                            "whether zero or negative quantities are permitted, or how they are processed if they are not. I'll "
                            "also be keeping an eye out for any checks related to non-integer values, though the initial prompt "
                            "emphasized zero and negatives.\n\nI will then use the `read_file` tool to retrieve the content of "
                            "`shop/validation.py`. I'll specify a reasonable range for the lines to read, perhaps from the "
                            "beginning of the file up to line 50, to ensure I capture the relevant validation function and its "
                            "surrounding context. This will allow me to analyze the validation rules in detail before proceeding "
                            "to any other files.\n\n"
                        ),
                        "annotations": [],
                    },
                ],
                "phase": None,
            },
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            "I need to inspect `shop/validation.py` next to see how `validate_quantity` handles zero, negative, "
                            "and non-integer quantities."
                        ),
                        "annotations": [],
                    },
                ],
                "phase": None,
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 40}',
                "call_id": (
                    "call_316157__thought__AY89a18zqWYsP1fCe9VRYni0L9f03IJNDHvm1boNA4Q2dVNJPzZZvrxcHGuC7kCwx5e9kBdKsGEQ9/"
                    "Nr4AjO5c6ukC2h2NEeTinmhak8IRJNBkDQ7OZ7at4q5KYoqYfDZ8aNohtXWBxVdoSK9YZHp+nPGYE7zmXk1icksF+WJG0ga0jBai"
                    "vzjYeChB7DEp/pEfxa5e/5etULNzsfurb4+l7Aht/DYaH/Z+EEttmrxrDlKI9SHGOfDy8YRn7XcDJyjYjIJ/Hgcoa3/STN5kaEJs"
                    "+7FzyrN64sSOyhzsP1H5dSM27anFBLUjs5KHg15CoDirSbIlRLlOHJWQ2k9Ql3ljBfNePDAds4/zH2VchSqiZfsIS5XwUGm8D7L5"
                    "65vzTzknZtb+zSZivDPPs/fCe3iCNYY4YEeSt2Zz/hy7baM06u6Vb1cljWtmckLoQgpiHJhY/yA5IhrHLqTx+PnBTgud6+yx7r/9"
                    "Wiu8ttxeU8vupqAERoBj9yOmysvjloE6N7g/9qwsRvvoxICo8luVQmA1xWqkDMTLT3LNfEwIpWmEr0Tx+y4chAZitmuKUBZdtPfr"
                    "ayO96GB7iPjRUarEKs/px3hHyaSs7QzZZM/ymeeihqYyWzr81GN14/MsHs72eX3sl+ooYhxM/qhpOSjFiMgffLrD7NlAwcBtXfNI"
                    "7fQWjbkLYf+mu6hCMSY8gshIkSLFcRv8ijH3eK2RBQWzvDsjkNgiuTlXetgYFVe5/Q3YJvCCdPsbn/Fd/dc3etq5XTGJ9B1ecsI7"
                    "KFdWOX8Q5HrrfWFOuo7BuWXNnFBN7pd1RwzWI3cveysx77EgQ/fMS0Iqb9FsM4i8SIY6tW5KIj+Q0SdnGR2Fkh214QxDUwT53ThP"
                    "Des9mapBpR4lxIqvROoMcMPoMFlcvENF9S8oeE17cifQTVFrQT6xGeWL3vV7AVGVhuP7RjIX9cU/NK2qBof50qfI2BC+aqiu6Wtz"
                    "WQ2IdfrhCfCChccr0Jh4IldyF/0mW45PijT0mguv0vwceuVrtwLkYSK7gsnb6REGUmQ4WXWy9VwrCmcYzmX5igAN/ml0sML0+k1p"
                    "DHtYVcflnjAT1s/wiQFJ4Pg1vUXmpjGSOUb7KiyKeiUvgTTdqD1EXHAWxnFJtgwwz/6p+VhkiZvvwOJY/GSciX14pMtA57VsGd+2"
                    "eUTRdpuEbffVWjuuyeyZd6Xgm2PNLnCIvP8sS0FLJAKhOYau1OmGwkl5gsteZqSgjg2l7hXJf3lHhXv4UrRRj/YwX8SEg/YPSzhu"
                    "rb3bBdLKBUPfQs3h/DlOnGbQqi7efhq5seYJBldPpiEtwChm7o6mJ6J4cr1AqLTtDmQR6XF5ZN/oE6XGHpa2XY9N4tJmSrnIq6ku"
                    "unngTfpD7yejAg9bHVy8r4FWjU9wU2AgYHBZneDXTiVKDaJDzYPdoufiADokGyWntWksp+B3jH+El+kI7c3uu/tL28HVep3u8Jz/"
                    "XlLqEgh4oViCrcsyqqjLsUrfQ8oDmiDJZfbxECR7GHDxJ9yA=="
                ),
                "name": "read_file",
                "type": "function_call",
                "id": (
                    "fc_call_316157__thought__AY89a18zqWYsP1fCe9VRYni0L9f03IJNDHvm1boNA4Q2dVNJPzZZvrxcHGuC7kCwx5e9kBdKsGE"
                    "Q9/Nr4AjO5c6ukC2h2NEeTinmhak8IRJNBkDQ7OZ7at4q5KYoqYfDZ8aNohtXWBxVdoSK9YZHp+nPGYE7zmXk1icksF+WJG0ga0j"
                    "BaivzjYeChB7DEp/pEfxa5e/5etULNzsfurb4+l7Aht/DYaH/Z+EEttmrxrDlKI9SHGOfDy8YRn7XcDJyjYjIJ/Hgcoa3/STN5ka"
                    "EJs+7FzyrN64sSOyhzsP1H5dSM27anFBLUjs5KHg15CoDirSbIlRLlOHJWQ2k9Ql3ljBfNePDAds4/zH2VchSqiZfsIS5XwUGm8D"
                    "7L565vzTzknZtb+zSZivDPPs/fCe3iCNYY4YEeSt2Zz/hy7baM06u6Vb1cljWtmckLoQgpiHJhY/yA5IhrHLqTx+PnBTgud6+yx7"
                    "r/9Wiu8ttxeU8vupqAERoBj9yOmysvjloE6N7g/9qwsRvvoxICo8luVQmA1xWqkDMTLT3LNfEwIpWmEr0Tx+y4chAZitmuKUBZdt"
                    "PfrayO96GB7iPjRUarEKs/px3hHyaSs7QzZZM/ymeeihqYyWzr81GN14/MsHs72eX3sl+ooYhxM/qhpOSjFiMgffLrD7NlAwcBtX"
                    "fNI7fQWjbkLYf+mu6hCMSY8gshIkSLFcRv8ijH3eK2RBQWzvDsjkNgiuTlXetgYFVe5/Q3YJvCCdPsbn/Fd/dc3etq5XTGJ9B1ec"
                    "sI7KFdWOX8Q5HrrfWFOuo7BuWXNnFBN7pd1RwzWI3cveysx77EgQ/fMS0Iqb9FsM4i8SIY6tW5KIj+Q0SdnGR2Fkh214QxDUwT53"
                    "ThPDes9mapBpR4lxIqvROoMcMPoMFlcvENF9S8oeE17cifQTVFrQT6xGeWL3vV7AVGVhuP7RjIX9cU/NK2qBof50qfI2BC+aqiu6"
                    "WtzWQ2IdfrhCfCChccr0Jh4IldyF/0mW45PijT0mguv0vwceuVrtwLkYSK7gsnb6REGUmQ4WXWy9VwrCmcYzmX5igAN/ml0sML0+"
                    "k1pDHtYVcflnjAT1s/wiQFJ4Pg1vUXmpjGSOUb7KiyKeiUvgTTdqD1EXHAWxnFJtgwwz/6p+VhkiZvvwOJY/GSciX14pMtA57VsG"
                    "d+2eUTRdpuEbffVWjuuyeyZd6Xgm2PNLnCIvP8sS0FLJAKhOYau1OmGwkl5gsteZqSgjg2l7hXJf3lHhXv4UrRRj/YwX8SEg/YPS"
                    "zhurb3bBdLKBUPfQs3h/DlOnGbQqi7efhq5seYJBldPpiEtwChm7o6mJ6J4cr1AqLTtDmQR6XF5ZN/oE6XGHpa2XY9N4tJmSrnIq"
                    "6kuunngTfpD7yejAg9bHVy8r4FWjU9wU2AgYHBZneDXTiVKDaJDzYPdoufiADokGyWntWksp+B3jH+El+kI7c3uu/tL28HVep3u8"
                    "Jz/XlLqEgh4oViCrcsyqqjLsUrfQ8oDmiDJZfbxECR7GHDxJ9yA=="
                ),
                "namespace": None,
                "status": "completed",
                "provider_specific_fields": {
                    "thought_signature": (
                        "AY89a18zqWYsP1fCe9VRYni0L9f03IJNDHvm1boNA4Q2dVNJPzZZvrxcHGuC7kCwx5e9kBdKsGEQ9/Nr4AjO5c6ukC2h2NEeTinm"
                        "hak8IRJNBkDQ7OZ7at4q5KYoqYfDZ8aNohtXWBxVdoSK9YZHp+nPGYE7zmXk1icksF+WJG0ga0jBaivzjYeChB7DEp/pEfxa5e/5"
                        "etULNzsfurb4+l7Aht/DYaH/Z+EEttmrxrDlKI9SHGOfDy8YRn7XcDJyjYjIJ/Hgcoa3/STN5kaEJs+7FzyrN64sSOyhzsP1H5dS"
                        "M27anFBLUjs5KHg15CoDirSbIlRLlOHJWQ2k9Ql3ljBfNePDAds4/zH2VchSqiZfsIS5XwUGm8D7L565vzTzknZtb+zSZivDPPs/"
                        "fCe3iCNYY4YEeSt2Zz/hy7baM06u6Vb1cljWtmckLoQgpiHJhY/yA5IhrHLqTx+PnBTgud6+yx7r/9Wiu8ttxeU8vupqAERoBj9y"
                        "OmysvjloE6N7g/9qwsRvvoxICo8luVQmA1xWqkDMTLT3LNfEwIpWmEr0Tx+y4chAZitmuKUBZdtPfrayO96GB7iPjRUarEKs/px3"
                        "hHyaSs7QzZZM/ymeeihqYyWzr81GN14/MsHs72eX3sl+ooYhxM/qhpOSjFiMgffLrD7NlAwcBtXfNI7fQWjbkLYf+mu6hCMSY8gs"
                        "hIkSLFcRv8ijH3eK2RBQWzvDsjkNgiuTlXetgYFVe5/Q3YJvCCdPsbn/Fd/dc3etq5XTGJ9B1ecsI7KFdWOX8Q5HrrfWFOuo7BuW"
                        "XNnFBN7pd1RwzWI3cveysx77EgQ/fMS0Iqb9FsM4i8SIY6tW5KIj+Q0SdnGR2Fkh214QxDUwT53ThPDes9mapBpR4lxIqvROoMcM"
                        "PoMFlcvENF9S8oeE17cifQTVFrQT6xGeWL3vV7AVGVhuP7RjIX9cU/NK2qBof50qfI2BC+aqiu6WtzWQ2IdfrhCfCChccr0Jh4Il"
                        "dyF/0mW45PijT0mguv0vwceuVrtwLkYSK7gsnb6REGUmQ4WXWy9VwrCmcYzmX5igAN/ml0sML0+k1pDHtYVcflnjAT1s/wiQFJ4P"
                        "g1vUXmpjGSOUb7KiyKeiUvgTTdqD1EXHAWxnFJtgwwz/6p+VhkiZvvwOJY/GSciX14pMtA57VsGd+2eUTRdpuEbffVWjuuyeyZd6"
                        "Xgm2PNLnCIvP8sS0FLJAKhOYau1OmGwkl5gsteZqSgjg2l7hXJf3lHhXv4UrRRj/YwX8SEg/YPSzhurb3bBdLKBUPfQs3h/DlOnG"
                        "bQqi7efhq5seYJBldPpiEtwChm7o6mJ6J4cr1AqLTtDmQR6XF5ZN/oE6XGHpa2XY9N4tJmSrnIq6kuunngTfpD7yejAg9bHVy8r4"
                        "FWjU9wU2AgYHBZneDXTiVKDaJDzYPdoufiADokGyWntWksp+B3jH+El+kI7c3uu/tL28HVep3u8Jz/XlLqEgh4oViCrcsyqqjLsU"
                        "rfQ8oDmiDJZfbxECR7GHDxJ9yA=="
                    ),
                },
            },
        ],
        "usage": {
            "input_tokens": 242,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 242,
                "video_tokens": None,
            },
            "output_tokens": 329,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 265, "text_tokens": 64},
            "total_tokens": 571,
            "cost": None,
        },
        "provider_specific_fields": {"traffic_type": "ON_DEMAND"},
    },
)


GEMINI_3_1_PRO_PREVIEW_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "vertex_ai/gemini-3.1-pro-preview"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/google/models/gemini-3.1-pro-preview:generateContent",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer scripted-token"},
    expected_provider_request={
        **VERTEX_AI_GEMINI_GENERATE_CONTENT_REQUEST,
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
                        "Explain which file you need next, then call a read-only tool. Never change files. Use the responses "
                        "client for this project review. Consider zero, negative and non-integer quantities before choosing "
                        "the next file to inspect."
                    ),
                },
            ],
        },
    },
    mock_provider_response={
        **VERTEX_AI_GEMINI_GENERATE_CONTENT_RESPONSE,
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": (
                                "**My Thought Process on Validating Quantities**\n\nAlright, so my primary objective here is to get a "
                                "solid grasp on how the system validates quantities. I need to specifically understand its behavior "
                                "when faced with zero or negative values, as I've observed that the checkout process currently "
                                "rejects zero quantities, and I want to confirm if that's a deliberate design choice or something to "
                                "be revisited.\n\nMy first step is to pinpoint the exact location of this validation logic. Based on "
                                "the information I have, it seems the relevant function, `validate_quantity`, resides within the "
                                "`shop/validation.py` file, specifically around line 18.\n\nTherefore, my immediate action is to dive "
                                "into `shop/validation.py`. I'll focus on the vicinity of line 18, reading a few lines before and "
                                "after – let's say lines 10 through 30 – to ensure I have sufficient context to understand the "
                                "function's implementation and any surrounding logic.\n\nOnce I've examined the code, I'll be able to "
                                "articulate precisely how `validate_quantity` handles zero, negative, and potentially even "
                                "non-integer quantities. Then, I can proceed with the next steps, whether that involves reporting my "
                                "findings, proposing modifications, or simply confirming the current state of affairs. So, my next "
                                "move is to execute that `read_file` operation.\n\n"
                            ),
                            "thought": True,
                        },
                        {
                            "text": (
                                "I need to look at `shop/validation.py` to see the `validate_quantity` function and understand its "
                                "logic for zero, negative, and non-integer quantities."
                            ),
                        },
                        {
                            "functionCall": {
                                "name": "read_file",
                                "args": {"path": "shop/validation.py", "start_line": 10, "end_line": 30},
                                "id": "call_226688",
                            },
                            "thoughtSignature": GEMINI_3_1_PRO_PREVIEW_RESPONSES_THOUGHT_SIGNATURE_0,
                        },
                    ],
                },
                "finishReason": "STOP",
            },
        ],
        "usageMetadata": {
            "promptTokenCount": 242,
            "candidatesTokenCount": 70,
            "totalTokenCount": 483,
            "trafficType": "ON_DEMAND",
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 242}],
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 70}],
            "thoughtsTokenCount": 171,
        },
        "modelVersion": "gemini-3.1-pro-preview",
        "createTime": "2026-10-15T01:14:11.647331Z",
        "responseId": "4yjQaqPBJ72apucPwee_YQ",
    },
    expected_litellm_response={
        **RESPONSES_RESPONSE,
        "model": "vertex_ai/gemini-3.1-pro-preview",
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            "**My Thought Process on Validating Quantities**\n\nAlright, so my primary objective here is to get a "
                            "solid grasp on how the system validates quantities. I need to specifically understand its behavior "
                            "when faced with zero or negative values, as I've observed that the checkout process currently "
                            "rejects zero quantities, and I want to confirm if that's a deliberate design choice or something to "
                            "be revisited.\n\nMy first step is to pinpoint the exact location of this validation logic. Based on "
                            "the information I have, it seems the relevant function, `validate_quantity`, resides within the "
                            "`shop/validation.py` file, specifically around line 18.\n\nTherefore, my immediate action is to dive "
                            "into `shop/validation.py`. I'll focus on the vicinity of line 18, reading a few lines before and "
                            "after – let's say lines 10 through 30 – to ensure I have sufficient context to understand the "
                            "function's implementation and any surrounding logic.\n\nOnce I've examined the code, I'll be able to "
                            "articulate precisely how `validate_quantity` handles zero, negative, and potentially even "
                            "non-integer quantities. Then, I can proceed with the next steps, whether that involves reporting my "
                            "findings, proposing modifications, or simply confirming the current state of affairs. So, my next "
                            "move is to execute that `read_file` operation.\n\n"
                        ),
                        "annotations": [],
                    },
                ],
                "phase": None,
            },
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            "I need to look at `shop/validation.py` to see the `validate_quantity` function and understand its "
                            "logic for zero, negative, and non-integer quantities."
                        ),
                        "annotations": [],
                    },
                ],
                "phase": None,
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 10, "end_line": 30}',
                "call_id": (
                    "call_226688__thought__AY89a18mH4gPpBuF8sZq3BxEt7fVfeky9tDk6WrFiAYqUNfy1TwKL5UEuc60FQFzZzoY6BhtmNYM/Y"
                    "nvWntgJG06tpD+R69L+px8wJWN2jf96nYOxDNe5xM3Z6yz7QO1XikWun/bqStP3GVvN2xCTRBeQxkN0n5+NPDWYXC94zo2uikRYf"
                    "1XZ5IsCrbwhDgFhqzyv+3CpEmBpN7ghzonZp6GL23N0ELNi95QEzZMcGrD31Z3xx6Db+KblQL9yVKVj7STHJ4Bno3sU6heJx9NMJ"
                    "bM55wI5SNcZqi8YqkS3KliWRxcPdWbkekAOCWUFYYFG0JEB0iOibkqcCV6j7bZaJD3v9qdqpaSbG1Q3yl5Ncb1KMMGLwbMyjN9t3"
                    "XrFQuqtM1x3OE8hofM3bNPMYGna3DDEmlporSsQlE6W8Cxa70IsBaFY5h7eQjKCkxLSOhZmGfmuukLLpxXWG8ZHzdtOB7vmpIcBo"
                    "lW2TcQCamojGpvWdxKz/JhU9WB9xac6A5luWXnIA+YC/Pu793XzkDrST8jT4bei8Wi37sfAhRUkVeUw6VAAYbZKONNM43U/ih3Bw"
                    "JvpJoY5voZVyhm4JTT66q9vSTUSiR4oQz/ipXDUKVfWFhqEWoh3Nh5elhFe1oZyx8q+lGwXm4VcrGNtb5xwWGnAsQN9GV30egFGN"
                    "cEKFq3rnODehteSTTaHF6YaqklMypgS/K09SfshF0waKvX0GDouVMbnO3bFfMxGU1CofWB3b3q6RbIdv+O37WekKo0O8S6OwQRdp"
                    "fOCNXcwpwsRczX4s9IoGRK5IIKr1/6cQze7LA2Gi00gBIe9RpbSMJHo/XJyDsP6KT1IAAtoEkyGBzsVjBEU4ggVsG6n8ka5y4myF"
                    "gwMQeNrN5DNqFL+nVzxqUF97pRU7Xw37SlNNi9DgGnqhyWY/T6Vvr4yhCsYR2BElGCq0C1YUF9OQ8+gp2CBrtd+bRLHz5iNyUMoP"
                    "umSAn6B1+GyU5JFPQwC9YvNK6ug+9s2akQgYYgNAnDS7aXj2djQ57nwPqnsg=="
                ),
                "name": "read_file",
                "type": "function_call",
                "id": (
                    "fc_call_226688__thought__AY89a18mH4gPpBuF8sZq3BxEt7fVfeky9tDk6WrFiAYqUNfy1TwKL5UEuc60FQFzZzoY6BhtmNY"
                    "M/YnvWntgJG06tpD+R69L+px8wJWN2jf96nYOxDNe5xM3Z6yz7QO1XikWun/bqStP3GVvN2xCTRBeQxkN0n5+NPDWYXC94zo2uik"
                    "RYf1XZ5IsCrbwhDgFhqzyv+3CpEmBpN7ghzonZp6GL23N0ELNi95QEzZMcGrD31Z3xx6Db+KblQL9yVKVj7STHJ4Bno3sU6heJx9"
                    "NMJbM55wI5SNcZqi8YqkS3KliWRxcPdWbkekAOCWUFYYFG0JEB0iOibkqcCV6j7bZaJD3v9qdqpaSbG1Q3yl5Ncb1KMMGLwbMyjN"
                    "9t3XrFQuqtM1x3OE8hofM3bNPMYGna3DDEmlporSsQlE6W8Cxa70IsBaFY5h7eQjKCkxLSOhZmGfmuukLLpxXWG8ZHzdtOB7vmpI"
                    "cBolW2TcQCamojGpvWdxKz/JhU9WB9xac6A5luWXnIA+YC/Pu793XzkDrST8jT4bei8Wi37sfAhRUkVeUw6VAAYbZKONNM43U/ih"
                    "3BwJvpJoY5voZVyhm4JTT66q9vSTUSiR4oQz/ipXDUKVfWFhqEWoh3Nh5elhFe1oZyx8q+lGwXm4VcrGNtb5xwWGnAsQN9GV30eg"
                    "FGNcEKFq3rnODehteSTTaHF6YaqklMypgS/K09SfshF0waKvX0GDouVMbnO3bFfMxGU1CofWB3b3q6RbIdv+O37WekKo0O8S6OwQ"
                    "RdpfOCNXcwpwsRczX4s9IoGRK5IIKr1/6cQze7LA2Gi00gBIe9RpbSMJHo/XJyDsP6KT1IAAtoEkyGBzsVjBEU4ggVsG6n8ka5y4"
                    "myFgwMQeNrN5DNqFL+nVzxqUF97pRU7Xw37SlNNi9DgGnqhyWY/T6Vvr4yhCsYR2BElGCq0C1YUF9OQ8+gp2CBrtd+bRLHz5iNyU"
                    "MoPumSAn6B1+GyU5JFPQwC9YvNK6ug+9s2akQgYYgNAnDS7aXj2djQ57nwPqnsg=="
                ),
                "namespace": None,
                "status": "completed",
                "provider_specific_fields": {
                    "thought_signature": (
                        "AY89a18mH4gPpBuF8sZq3BxEt7fVfeky9tDk6WrFiAYqUNfy1TwKL5UEuc60FQFzZzoY6BhtmNYM/YnvWntgJG06tpD+R69L+px8"
                        "wJWN2jf96nYOxDNe5xM3Z6yz7QO1XikWun/bqStP3GVvN2xCTRBeQxkN0n5+NPDWYXC94zo2uikRYf1XZ5IsCrbwhDgFhqzyv+3C"
                        "pEmBpN7ghzonZp6GL23N0ELNi95QEzZMcGrD31Z3xx6Db+KblQL9yVKVj7STHJ4Bno3sU6heJx9NMJbM55wI5SNcZqi8YqkS3Kli"
                        "WRxcPdWbkekAOCWUFYYFG0JEB0iOibkqcCV6j7bZaJD3v9qdqpaSbG1Q3yl5Ncb1KMMGLwbMyjN9t3XrFQuqtM1x3OE8hofM3bNP"
                        "MYGna3DDEmlporSsQlE6W8Cxa70IsBaFY5h7eQjKCkxLSOhZmGfmuukLLpxXWG8ZHzdtOB7vmpIcBolW2TcQCamojGpvWdxKz/Jh"
                        "U9WB9xac6A5luWXnIA+YC/Pu793XzkDrST8jT4bei8Wi37sfAhRUkVeUw6VAAYbZKONNM43U/ih3BwJvpJoY5voZVyhm4JTT66q9"
                        "vSTUSiR4oQz/ipXDUKVfWFhqEWoh3Nh5elhFe1oZyx8q+lGwXm4VcrGNtb5xwWGnAsQN9GV30egFGNcEKFq3rnODehteSTTaHF6Y"
                        "aqklMypgS/K09SfshF0waKvX0GDouVMbnO3bFfMxGU1CofWB3b3q6RbIdv+O37WekKo0O8S6OwQRdpfOCNXcwpwsRczX4s9IoGRK"
                        "5IIKr1/6cQze7LA2Gi00gBIe9RpbSMJHo/XJyDsP6KT1IAAtoEkyGBzsVjBEU4ggVsG6n8ka5y4myFgwMQeNrN5DNqFL+nVzxqUF"
                        "97pRU7Xw37SlNNi9DgGnqhyWY/T6Vvr4yhCsYR2BElGCq0C1YUF9OQ8+gp2CBrtd+bRLHz5iNyUMoPumSAn6B1+GyU5JFPQwC9Yv"
                        "NK6ug+9s2akQgYYgNAnDS7aXj2djQ57nwPqnsg=="
                    ),
                },
            },
        ],
        "usage": {
            "input_tokens": 242,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 242,
                "video_tokens": None,
            },
            "output_tokens": 241,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 171, "text_tokens": 70},
            "total_tokens": 483,
            "cost": None,
        },
        "provider_specific_fields": {"traffic_type": "ON_DEMAND"},
    },
)


GEMINI_4_ARGON_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/responses",
    litellm_request={**RESPONSES_REQUEST, "model": "vertex_ai/gemini-4-argon"},
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/google/models/gemini-4-argon:generateContent",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer scripted-token"},
    expected_provider_request={
        **VERTEX_AI_GEMINI_GENERATE_CONTENT_REQUEST,
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
                        "Explain which file you need next, then call a read-only tool. Never change files. Use the responses "
                        "client for this project review. Consider zero, negative and non-integer quantities before choosing "
                        "the next file to inspect."
                    ),
                },
            ],
        },
    },
    mock_provider_response={
        **VERTEX_AI_GEMINI_GENERATE_CONTENT_RESPONSE,
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": (
                                "**My Thought Process for Analyzing `shop/validation.py`**\n\nMy immediate priority is to gain a "
                                "thorough understanding of how the system handles invalid product quantities. Specifically, I need "
                                "to know how it's designed to process zero, negative, and non-integer values. To achieve this, my "
                                "next step is to examine the `shop/validation.py` file. The developer's prompt explicitly states "
                                'that I should "Explain which file you need next, then call a read-only tool. Never change files." '
                                "Therefore, I will proceed by identifying `shop/validation.py` as the crucial file for this analysis "
                                "and then use the `read_file` tool to access its contents.\n\n"
                            ),
                            "thought": True,
                        },
                        {
                            "text": (
                                "I will read `shop/validation.py` next to inspect how `validate_quantity` handles zero, negative, "
                                "and non-integer quantities."
                            ),
                        },
                        {
                            "functionCall": {
                                "name": "read_file",
                                "args": {"path": "shop/validation.py", "end_line": 50, "start_line": 1},
                                "id": "call_268046",
                            },
                            "thoughtSignature": GEMINI_4_ARGON_RESPONSES_THOUGHT_SIGNATURE_0,
                        },
                    ],
                },
                "finishReason": "STOP",
            },
        ],
        "usageMetadata": {
            "promptTokenCount": 412,
            "candidatesTokenCount": 69,
            "totalTokenCount": 573,
            "trafficType": "ON_DEMAND",
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 412}],
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 69}],
            "thoughtsTokenCount": 92,
        },
        "modelVersion": "gemini-4-argon",
        "createTime": "2026-10-15T01:14:21.864566Z",
        "responseId": "7SjQarbiNNqX9LsPrO6x6QE",
    },
    expected_litellm_response={
        **RESPONSES_RESPONSE,
        "model": "vertex_ai/gemini-4-argon",
        "output": [
            {
                "type": "reasoning",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            "**My Thought Process for Analyzing `shop/validation.py`**\n\nMy immediate priority is to gain a "
                            "thorough understanding of how the system handles invalid product quantities. Specifically, I need "
                            "to know how it's designed to process zero, negative, and non-integer values. To achieve this, my "
                            "next step is to examine the `shop/validation.py` file. The developer's prompt explicitly states "
                            'that I should "Explain which file you need next, then call a read-only tool. Never change files." '
                            "Therefore, I will proceed by identifying `shop/validation.py` as the crucial file for this analysis "
                            "and then use the `read_file` tool to access its contents.\n\n"
                        ),
                        "annotations": [],
                    },
                ],
                "phase": None,
            },
            {
                "type": "message",
                "id": ANY,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            "I will read `shop/validation.py` next to inspect how `validate_quantity` handles zero, negative, "
                            "and non-integer quantities."
                        ),
                        "annotations": [],
                    },
                ],
                "phase": None,
            },
            {
                "arguments": '{"path": "shop/validation.py", "end_line": 50, "start_line": 1}',
                "call_id": (
                    "call_268046__thought__AY89a180ZmYN6Zq/ICj0BCu+eBS75TWGlATeX2fsIhoKaFuGLfTjmizARO/YOhSzt4TK8d1jl7skKb"
                    "Dq23TdvRVOJfjPsI85UiVCF8DZC4gSdHQbTXiDtBUsue7BxMBnoNfPJe34Y1p6WqjbaK44zsfJybZVsWJSmQOW5Ky8qnBZXfEL9d"
                    "yz2j0rQbN5IMGUV36wrA4vKLfUj6vS1dwzz2LETuK1rT/EFKXDG2Pwgo11yMzoZuPvC8VvNrQXqJ/Tizd/cuuc+Fiht7V4EvrEDn"
                    "DH0ZFMW4gyS52mfjE8/hnIFIgzaFkhP7mWHBOwEt3ZSPZPQ2d6ZEOXD+1clQXSOWMv9Ks4Iwvy8hrcri60cdqbFdbHxF48OgfWUy"
                    "2dUycP6kuKwWUJ+sDNT1pNlb6pfhKdmP7wd6QvjudKrrunGZbWr9CjkiAGVOD8RfcQK8Z7LUC11RyHqGDXLxJA1kzR4W4wct1Y6u"
                    "WF3TRN8QsSLpuXLS+sITYsvEaRtT/wcCBVYu+dHephEAo2UXmaMfkZ8QxrWtf/VOwDZegQiVpBkxOZyAoByiDJrCwFa/osvXXcj0"
                    "BtweIKpKKUdLy77BVE7IHaJikXSIRmpwYZRnA="
                ),
                "name": "read_file",
                "type": "function_call",
                "id": (
                    "fc_call_268046__thought__AY89a180ZmYN6Zq/ICj0BCu+eBS75TWGlATeX2fsIhoKaFuGLfTjmizARO/YOhSzt4TK8d1jl7s"
                    "kKbDq23TdvRVOJfjPsI85UiVCF8DZC4gSdHQbTXiDtBUsue7BxMBnoNfPJe34Y1p6WqjbaK44zsfJybZVsWJSmQOW5Ky8qnBZXfE"
                    "L9dyz2j0rQbN5IMGUV36wrA4vKLfUj6vS1dwzz2LETuK1rT/EFKXDG2Pwgo11yMzoZuPvC8VvNrQXqJ/Tizd/cuuc+Fiht7V4Evr"
                    "EDnDH0ZFMW4gyS52mfjE8/hnIFIgzaFkhP7mWHBOwEt3ZSPZPQ2d6ZEOXD+1clQXSOWMv9Ks4Iwvy8hrcri60cdqbFdbHxF48Ogf"
                    "WUy2dUycP6kuKwWUJ+sDNT1pNlb6pfhKdmP7wd6QvjudKrrunGZbWr9CjkiAGVOD8RfcQK8Z7LUC11RyHqGDXLxJA1kzR4W4wct1"
                    "Y6uWF3TRN8QsSLpuXLS+sITYsvEaRtT/wcCBVYu+dHephEAo2UXmaMfkZ8QxrWtf/VOwDZegQiVpBkxOZyAoByiDJrCwFa/osvXX"
                    "cj0BtweIKpKKUdLy77BVE7IHaJikXSIRmpwYZRnA="
                ),
                "namespace": None,
                "status": "completed",
                "provider_specific_fields": {
                    "thought_signature": (
                        "AY89a180ZmYN6Zq/ICj0BCu+eBS75TWGlATeX2fsIhoKaFuGLfTjmizARO/YOhSzt4TK8d1jl7skKbDq23TdvRVOJfjPsI85UiVC"
                        "F8DZC4gSdHQbTXiDtBUsue7BxMBnoNfPJe34Y1p6WqjbaK44zsfJybZVsWJSmQOW5Ky8qnBZXfEL9dyz2j0rQbN5IMGUV36wrA4v"
                        "KLfUj6vS1dwzz2LETuK1rT/EFKXDG2Pwgo11yMzoZuPvC8VvNrQXqJ/Tizd/cuuc+Fiht7V4EvrEDnDH0ZFMW4gyS52mfjE8/hnI"
                        "FIgzaFkhP7mWHBOwEt3ZSPZPQ2d6ZEOXD+1clQXSOWMv9Ks4Iwvy8hrcri60cdqbFdbHxF48OgfWUy2dUycP6kuKwWUJ+sDNT1pN"
                        "lb6pfhKdmP7wd6QvjudKrrunGZbWr9CjkiAGVOD8RfcQK8Z7LUC11RyHqGDXLxJA1kzR4W4wct1Y6uWF3TRN8QsSLpuXLS+sITYs"
                        "vEaRtT/wcCBVYu+dHephEAo2UXmaMfkZ8QxrWtf/VOwDZegQiVpBkxOZyAoByiDJrCwFa/osvXXcj0BtweIKpKKUdLy77BVE7IHa"
                        "JikXSIRmpwYZRnA="
                    ),
                },
            },
        ],
        "usage": {
            "input_tokens": 412,
            "input_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "cached_tokens_details": None,
                "image_tokens": None,
                "text_tokens": 412,
                "video_tokens": None,
            },
            "output_tokens": 161,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 92, "text_tokens": 69},
            "total_tokens": 573,
            "cost": None,
        },
        "provider_specific_fields": {"traffic_type": "ON_DEMAND"},
    },
)
