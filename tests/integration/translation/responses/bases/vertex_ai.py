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
CLAUDE_HAIKU_5_5_RESPONSES_THINKING_SIGNATURE: Final = "CAQS0gYKEggSEAIYAjgBQgh0aGlua2luZxIMcbnLkNvj2D6NoBl7Ggw+Qgax2/68TAR1maEiMH6XEeKEqR1ewAiuj01fMNuD1KI7pCTXUsWWr6H7nSzP5BiRGXRrDCIxPYu2Xbvx7yrtBbbyId4ReZzx21wZ2+d0vkkpZM82PXY72wQRJOHLR4JNLRjiR2Jn+FrZ8mBo2mofFrlTSuioOVHc1hheiH4InRK57w2X9xaTXa+mwoWW5epPieHD26+NNJirF9vB832fvqDVOd+f8N0ZDUXUut8zl/Jxx+0BMOktMqyfehIOGumQmyv49E3JHVQGrx+Dd3KZjaQXrsRFf1K3AJ7p5XElR+cYis40W7EL3DE9EAtriZuDEt0fTct79fzZIZgHd6Dnua/rGRnjNjaaZKxGjnhfkVHwsZW4HqTNgwWQiD3lt2ncoLfsfd2IWwMEI5uWpN3GtvRkQvfzidmgGmx3dI8g0nBvLPuq0jzVIuuoKIKqzT7A5ae3Gs9eTlSDr2wOGfdyNf6Q2TLw0w7EegI4wYSExKhFkx0EytCGHJcbN0WClXXmHsRtrXnP7sxCMMY7B8TIpAoN7SXyNjJizOGgzoYL7PNUgqp5nVz/C9vTvudhSWbjLITdxFJ5hearL2rMhryfHBKvRGYC9RMy/EjjtINNwYpxdVX29mUE+8KT+41HymtJ1Ewjtoo8V/7XQnQk3cB7X2RFHcR89hAM/pSsB14ofMFoDk+jJitS+sCDZI0Xh7/ySU0o/OI4l+oD/yctZTHx/mIOiWRtnAOrveNT1ms0/2+4tcO0wi3pRrOB2s5ugYc1I3+GcRwILEIM3a4ABd5Dw2nW2fgUhfSTTGURm4Z91ciih8pKUS1ZPatFTOJ1ehKeUCXflkqY2PHnftiNB/8GzlomsQ2NoRw8UXD00oMmBvSvZpPN7khFLxWHtxLjleKkeKCzN7QZkOqvjAnUwy9TacT2ZvSXIxrvaep7dOyrlMEta/1KsYEcWuBxLr87Va4txXnmoV4RqGgaauNcR2wOGyRFJppxKAgYbmMnxams7cv/7IY2RVSgSQf2k4hFQYzRwrHngjapuZzgNS7UCMjvJJs4VdIypVOATsZdPzJToMgWaiEoRZ3zApQDgYhhGAE="

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
        "id": "msg_vrtx_011CfjuySsPewSSfasuPiaHg",
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
        "id": "msg_vrtx_011CfjuxUbJsRvQZCN21qNQq",
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
        "id": "msg_vrtx_011CfjuygPVDgzuMXdyyW9UP",
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
        "id": "msg_vrtx_011CfjuyofoMgSaV8Pbt16N9",
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
    "CAQSqgcKEggSEAIYAjgBQgh0aGlua2luZxIMHBaYAp8JrEKnRBnTGgwijBpg0NpoxIwa1N4iMPW6lp+ZNZcZC05RTtExmUy/5WyQ"
    "mqBpk9fNiE9PKmQSIWh/l9NOxKOPAqBJ3uWJQSrFBjHaiQlVkJyGooin2mlHXfqZeGtr6F/3yL7fhsBTnYiZlZfLa9GwmzIZqPY4"
    "YhrWpo3hLZA9+LdoJ3sOeow336Voe/8M/eGeZDyVV33rgVVS1CVAanKfGn5vekIpLgR3EesIdw9Jx/NPOv1CQ8HMAnNZpHQYYt0P"
    "5uwr+BVS3GwAl1Rr7qlaFHj98JB6Stz+qOVGGJsLCY3JRX4j1D6rY3YQUErP1NUpUz/CD1ugOCkEzDY0/EHGaqTqenWq93YLtsdk"
    "9qqovMFXBPzsi7JhWFmlzqDoNc7MWGQHVsXvGwN678l9r28dD2A20NUJWWLPqa27KEwEihsBkXIEqFDAKP5e6tdaQWzuHqrPws3/"
    "M/L8+mf5RzO/ThtJZucdqMBAxC3HKqutVKArQym4dyqDj+V+vborLKcPNf1dEpY2aiGCDPrDhp5r+p0Fz/4++9V+vdZIy8Irun4l"
    "ILkAJOmjgmu8B+B+inbR/bn1sd0GYE752RqIWUYk9o1QzHfQ2XuiFZy4MZ3KaJHZr3eopTaPrdigJey2s369t5ACnIbEjFKINjgz"
    "eMfEHMrn95VaViTOIJJyYSN+GDEmcDobpN5OXrQtk5xSeF2CZ3uGOsUNESL76sFqt7pN69uxSGrLHohdQdQP3BLQ78ss6VcbskyR"
    "71/ZosbZDxA/vHIh59vFu7tR9V4E4C2xkVUHa5jndYCO/KIhp/Ce2HGU7PIGarsxw2MEFt3+eh8XrS6h2fbQvhB3MDquv1sP6sVb"
    "SZJ0tCqLWbpBO2Iq+3RAroS0+96eLiX3KKwuX1HVSVCgU0xzM2wobAh2NIGWuI5Ip262NArowCEAKhTDuc2zYle5pjyQ5s30Cmr/"
    "xcIXxEPcmBhIHxEWq2/Jt2ejbjLgzSZiA3SLU3SgTjSI0syfWJUMMl/9rKSyXV8hv36rk6zu4U1nnBrlOHuXKQ4+kVgfNylFaGRb"
    "g0XbHas+1nUGtH7RHbnFDJJBwzVRIlyq+wuPN4NqxzJYwrjDp1AqsFo3VJvbOx9u02lXHXRk4IovIQ2o2oA+HW9ZN3b1wDfewWnm"
    "aSu1wmkYteDmWQntg+hCxhRvAjWwdWR6nE9G5MxjPWRIRmt5OP3NFtBJFBgB"
)

CLAUDE_SONNET_5_5_RESPONSES_THINKING_SIGNATURE_1: Final = (
    "CAQSjQYKEwgSEAIYAjgBQgluYXJyYXRpb24SDNxOsiEvvTpgXLFh7BoMc2y30nGjlC6Jr/BeIjDZghnhINTjICZv6Mz/M2Vs53Cg"
    "ukg2cKEsEdShw2izKv7NanGF/Nl2ua9qsvKyxW0qpwXQ0xd5Tr4TUPYyZAy3GVQasB504SWCzSbA26LwrWtoQ1M9klTwLoAzj9S+"
    "h25IG2wbG7JbhCHgRUDQ+HlnwLRmkHisxfrWprRmXqms6EOhoU7gUM7mGtPR9Ttxln1xkZWZRA7V1N7RUb1rJA7itM4zX7nFwk3Q"
    "G4r9zkyd1BhiDV20j+u5+vEoNyeqT6XB95lgB2zvNZGUbtqdYlagOBS/bZWYQgzOGIsXKid6fbCDA0Na2EzNcG9T6fqx+EDBEGIZ"
    "JMP/zDxXVjXH/wUUi5YjZfGNPTncg6x/TT06RqRVn8phMU1l31sh+xcz6PS4tm9k4ysckmo/SxPsaFDAdgqepldXCs6Tl7viY/6Z"
    "abgJgkxtijBp9BsslKJO8hbualPuhHuBZkYJQk0HXm+4OyL4MNyswyxLP813NILt3L5txnxpjFbHWenn1hiwvcbYEUyWogxD79hD"
    "X5MZ3f86B92vk/BMGRM9TUSjphHO73X00hL4ZOCwqghvjvorRyyj5tjVnbFrAFCayaPF/QAwBIuqGFgNOG9wMF2USzioTCN1LHGM"
    "d7uvg5tAmU6k2/p1Ea4tmsVZF3h5rAw/bAUsF/msqOlw16NSakaGzpYYs++UFF4gYcR4rpROqg94/8z+up+P0Z2vrVVbn0Na7ObD"
    "RB7gwr7x1M9XQKXMp3Jtkj4V6zmAtci2PBjpIAX13y6zpR6xyVQECefg1bZmTjWPo2Ev9XNdXC8UlhersNTg9dTcnk6I1jfofBVQ"
    "sOPu5+mFG3rsZQCs3vKv5IS3j8zBFu3z95o8RNKURECQ5VcH8a1css/lNqhHhUdlRhqx6MlI1ZUd8eyI1hjjSzy3gn7VuZyOgAqM"
    "oHonAlebFZ8ARFGYGnWpid4+QyiYUyv5W7EnoOD5OpQoq2YOGAE="
)

CLAUDE_OPUS_5_5_RESPONSES_THINKING_SIGNATURE_0: Final = (
    "CAQSiAUKEggSEAIYAjgBQgh0aGlua2luZxIM5V4+5UUDIuxI11UEGgyO9SbnX7cSxGWVS+IiMD/+JMqCQUMJv+EWD6oB6YZwwIG6"
    "O5aIuop3rY1uYL76W3SEEhu5PI8DTabb70ki6yqjBLQCRri5xHeVVrEpRiH91RBZghREaOtv/TE8aiJYQ8njspRXIUbTj7PGxmgQ"
    "6xZ7bAK1WyljtD0zG3VTxFx1dSw5a/25/WJuYz/6D8gMXPQhG0j8lZzKfIvxhOq2mB5KBmnB+2agzXsyyws8nQ+tT7s/Yfyi7SbY"
    "dH+jysWNoKj/0pLiuGQE8gmmhkYfHdyOl518sVn/C/yJ6mB3i2Klkg8WifS0l1crRejKy3Er1nD2HBvwVpoTiXAe1v5ugB9NmSve"
    "pgg/5E6sewMCAnGLvL0Z9lfNPxpRTp9n0yP61Q49Z4kvSB9lcbZC5wN5/fjZTMNRQNDAEqhtXkWnA5SSW3N9r4Pcls67XcYGLI6f"
    "kjmJU6GOXiRbFIbptxjv/UBHmXPNBqqnIRioPi2RE8eGBbsNIjYrxBD/oJHb8Zd2RYeaM19UtkkYoOh/NaX9iZ0Nek2rYf55YOlo"
    "GowYdz6aYoo6E2xPQOFh8HAqF0Jzj6KCYodjidlI5jQHZlaE7901Q/u9y7AP0aQXrfAgeVxEPdPUfnp1ROVzK7A6K5O63Ozh+3jB"
    "KeZRW44jj5vA/+H2+uEVyv1x3eED2eK9j1wcCw2vIE3vTDvPbh3g7RkraTcrkdkuRHu2KS4kkQWppEeHqiJEmEFwwbCFHj41dsj1"
    "G1pWc9elR+oogW3sYKHQriwykjTCD5IaZi0H+tb8cNegqngv6kGWF10HCSM/4yaFqeATeWEYAQ=="
)

CLAUDE_OPUS_5_5_RESPONSES_THINKING_SIGNATURE_1: Final = (
    "CAQS1AUKEwgSEAIYAjgBQgluYXJyYXRpb24SDE/QqOK1lGVBzGebvRoMg39CM/sTiL2sWeIEIjDK82bOn+vjF6wbbaDm2uaqcCAx"
    "GlVUbC3CIx+rc3Wzsy7+0FhVIMf0D1k4GYnCN5Iq7gREgQE5RcQoQ7LQ8HO2ato0KeXeJosd/8ChrqGphPMNQPepQDjydUBgG8Z9"
    "BdfyxAaoAjxJwSgPe515bk9xajqoTBOlQWELydQyoSo93+SCnHQcCquftqsHbx05i1mgJZTm4L/8ON0jodnY6+Pg7yHW2QixEAht"
    "fpaLYEH0+a9/rYEiScsBls7gGPxcm4PuXJNt0+ymg3JmbcQ0KQ1sWW0FoLbfVSr0hQV64TahV6FfOVjug/hht/EyPJcSscrC6tbc"
    "e6VuW/n/LCRub62yHgJHQ0bqiDGi+uQgyhe5UgeugSFoSzmgLq8yf8TeKyA9+Pgy2VHFN+BOlW+M3s/9UFze0BBKgESOS9AxxwEF"
    "AEFWMG7UxksDvsbJ9s+rlXDpR6Zm8SfvfvQR60BAWQjyntxSd+F5RsmaFb369VRkVxUtCU6zrZpdkmAqmjHTgdBksBIEz+u/otTa"
    "TcdDNPsPVCqf+wSpTqY56X5eWWDXXimQGGzKZM7ELLHr7tvTDKfBmRsOqTisRf7EBnxRgD3IAw+nDaVweu+hVOiMdF/vLlc9MOSg"
    "75fnDILOJcmd/QiCrB2gAoUFzqLpNkyLUPOiqWaDkyWgpeD/34/hKZ0OmPZDuF55pEWQ0YwBxcw9YOx0yT5vqOhp+g4DP2dm4OUu"
    "ndhJRSGLBFJSCVdWdezujMQKacV+fOlidp7s797i7efNUzMyZqg056dN9l29fcU/0GKrpBXv7+Ne3vl2RE+4TcPuN9i2SWDsku9N"
    "yfwIS2Qy/LluH9ROHmoFTv5CSKxmskP2265nOWrdWbv4YUedA7spiQG/OxMuqF9NtriPexQeGAE="
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
            },
        ],
    },
    mock_provider_response={
        **VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-sonnet-5-5",
        "id": "msg_vrtx_011Cg38ropcTjx8YNUg3LxXa",
        "content": [
            {
                "type": "thinking",
                "thinking": (
                    "Since the validation and tests files are independent of each other, I can go ahead and read both in "
                    "the same call rather than waiting sequentially—that still satisfies the request to look at "
                    "validation first while being efficient.\n\n"
                ),
                "signature": CLAUDE_SONNET_5_5_RESPONSES_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": (
                    "I'm reading `shop/validation.py` and `tests/test_validation.py` together to see how quantity "
                    "validation handles zero, negative, and non-integer values, and what the tests expect.\n\n"
                ),
                "signature": CLAUDE_SONNET_5_5_RESPONSES_THINKING_SIGNATURE_1,
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_014c2eZQXGiRtzh9KvDXRhRh",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01VoiC4J2kb7yivWHYGqvRRm",
                "name": "read_file",
                "input": {"path": "tests/test_validation.py", "start_line": 1, "end_line": 50},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 414,
            "output_tokens_details": {"thinking_tokens": 218},
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
                            "Since the validation and tests files are independent of each other, I can go ahead and read both in "
                            "the same call rather than waiting sequentially—that still satisfies the request to look at "
                            "validation first while being efficient.\n\nI'm reading `shop/validation.py` and "
                            "`tests/test_validation.py` together to see how quantity validation handles zero, negative, and "
                            "non-integer values, and what the tests expect.\n\n"
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
                                "Since the validation and tests files are independent of each other, I can go ahead and read both in "
                                "the same call rather than waiting sequentially—that still satisfies the request to look at "
                                "validation first while being efficient.\n\n"
                            ),
                            "signature": CLAUDE_SONNET_5_5_RESPONSES_THINKING_SIGNATURE_0,
                        },
                        {
                            "type": "thinking",
                            "thinking": (
                                "I'm reading `shop/validation.py` and `tests/test_validation.py` together to see how quantity "
                                "validation handles zero, negative, and non-integer values, and what the tests expect.\n\n"
                            ),
                            "signature": CLAUDE_SONNET_5_5_RESPONSES_THINKING_SIGNATURE_1,
                        },
                    ],
                    separators=(",", ":"),
                ),
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                "call_id": "toolu_vrtx_014c2eZQXGiRtzh9KvDXRhRh",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_vrtx_014c2eZQXGiRtzh9KvDXRhRh",
                "namespace": None,
                "status": "completed",
            },
            {
                "arguments": '{"path": "tests/test_validation.py", "start_line": 1, "end_line": 50}',
                "call_id": "toolu_vrtx_01VoiC4J2kb7yivWHYGqvRRm",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_vrtx_01VoiC4J2kb7yivWHYGqvRRm",
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
            "output_tokens": 414,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 218, "text_tokens": 196},
            "total_tokens": 1291,
            "cost": None,
        },
        "provider_specific_fields": {
            "citations": None,
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": (
                        "Since the validation and tests files are independent of each other, I can go ahead and read both in "
                        "the same call rather than waiting sequentially—that still satisfies the request to look at "
                        "validation first while being efficient.\n\n"
                    ),
                    "signature": CLAUDE_SONNET_5_5_RESPONSES_THINKING_SIGNATURE_0,
                },
                {
                    "type": "thinking",
                    "thinking": (
                        "I'm reading `shop/validation.py` and `tests/test_validation.py` together to see how quantity "
                        "validation handles zero, negative, and non-integer values, and what the tests expect.\n\n"
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
            },
        ],
    },
    mock_provider_response={
        **VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-haiku-5-5",
        "id": "msg_vrtx_011Cg8BhBrWmiX5ZtMxpM9zA",
        "content": [
            {
                "type": "thinking",
                "thinking": (
                    "I should read shop/validation.py to check how it handles zero, negative, and non-integer values "
                    "before moving to the next file—I'll grab the first 60 lines since I don't know the full length yet.\n\n"
                ),
                "signature": CLAUDE_HAIKU_5_5_RESPONSES_THINKING_SIGNATURE,
            },
            {
                "type": "text",
                "text": (
                    "I'll read the validation function in `shop/validation.py`, starting around the match on line 18."
                ),
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01Jxxsojf34b9skuGH47XkJd",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 60},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 255,
            "output_tokens_details": {"thinking_tokens": 126},
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
                            "I should read shop/validation.py to check how it handles zero, negative, and non-integer "
                            "values before moving to the next file—I'll grab the first 60 lines since I don't know the "
                            "full length yet.\n\n"
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
                                "I should read shop/validation.py to check how it handles zero, negative, and non-integer "
                                "values before moving to the next file—I'll grab the first 60 lines since I don't know the "
                                "full length yet.\n\n"
                            ),
                            "signature": CLAUDE_HAIKU_5_5_RESPONSES_THINKING_SIGNATURE,
                        },
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
                        "text": "I'll read the validation function in `shop/validation.py`, starting around the match on line 18.",
                        "annotations": [],
                    },
                ],
                "phase": None,
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 60}',
                "call_id": "toolu_vrtx_01Jxxsojf34b9skuGH47XkJd",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_vrtx_01Jxxsojf34b9skuGH47XkJd",
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
            "output_tokens": 255,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 126, "text_tokens": 129},
            "total_tokens": 1132,
            "cost": None,
        },
        "provider_specific_fields": {
            "citations": None,
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": (
                        "I should read shop/validation.py to check how it handles zero, negative, and non-integer "
                        "values before moving to the next file—I'll grab the first 60 lines since I don't know the "
                        "full length yet.\n\n"
                    ),
                    "signature": CLAUDE_HAIKU_5_5_RESPONSES_THINKING_SIGNATURE,
                },
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
            },
        ],
    },
    mock_provider_response={
        **VERTEX_AI_ANTHROPIC_MESSAGES_RESPONSE,
        "model": "claude-opus-5-5",
        "id": "msg_vrtx_011Cg38rY7YVYxBPGMbH9jaS",
        "content": [
            {
                "type": "thinking",
                "thinking": (
                    "The user wants me to read the validation file first, so I'll open validation.py and focus around "
                    "line 18 to see what's there.\n\n"
                ),
                "signature": CLAUDE_OPUS_5_5_RESPONSES_THINKING_SIGNATURE_0,
            },
            {
                "type": "thinking",
                "thinking": (
                    "I'll check `validate_quantity` in `shop/validation.py` around line 18 to see how it handles zero, "
                    "negative, and non-integer inputs before looking at the tests.\n\n"
                ),
                "signature": CLAUDE_OPUS_5_5_RESPONSES_THINKING_SIGNATURE_1,
            },
            {
                "type": "tool_use",
                "id": "toolu_vrtx_01GFi5nAbXNTTfSxopXwNNPp",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 45},
            },
        ],
        "usage": {
            "input_tokens": 4,
            "cache_creation_input_tokens": 873,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 873, "ephemeral_1h_input_tokens": 0},
            "output_tokens": 224,
            "output_tokens_details": {"thinking_tokens": 126},
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
                            "The user wants me to read the validation file first, so I'll open validation.py and focus around "
                            "line 18 to see what's there.\n\nI'll check `validate_quantity` in `shop/validation.py` around line 18 "
                            "to see how it handles zero, negative, and non-integer inputs before looking at the tests.\n\n"
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
                                "The user wants me to read the validation file first, so I'll open validation.py and focus around "
                                "line 18 to see what's there.\n\n"
                            ),
                            "signature": CLAUDE_OPUS_5_5_RESPONSES_THINKING_SIGNATURE_0,
                        },
                        {
                            "type": "thinking",
                            "thinking": (
                                "I'll check `validate_quantity` in `shop/validation.py` around line 18 to see how it handles zero, "
                                "negative, and non-integer inputs before looking at the tests.\n\n"
                            ),
                            "signature": CLAUDE_OPUS_5_5_RESPONSES_THINKING_SIGNATURE_1,
                        },
                    ],
                    separators=(",", ":"),
                ),
            },
            {
                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 45}',
                "call_id": "toolu_vrtx_01GFi5nAbXNTTfSxopXwNNPp",
                "name": "read_file",
                "type": "function_call",
                "id": "fc_toolu_vrtx_01GFi5nAbXNTTfSxopXwNNPp",
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
            "output_tokens": 224,
            "output_tokens_details": {"audio_tokens": None, "reasoning_tokens": 126, "text_tokens": 98},
            "total_tokens": 1101,
            "cost": None,
        },
        "provider_specific_fields": {
            "citations": None,
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": (
                        "The user wants me to read the validation file first, so I'll open validation.py and focus around "
                        "line 18 to see what's there.\n\n"
                    ),
                    "signature": CLAUDE_OPUS_5_5_RESPONSES_THINKING_SIGNATURE_0,
                },
                {
                    "type": "thinking",
                    "thinking": (
                        "I'll check `validate_quantity` in `shop/validation.py` around line 18 to see how it handles zero, "
                        "negative, and non-integer inputs before looking at the tests.\n\n"
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
