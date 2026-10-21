from typing import Final
from unittest.mock import ANY

from integration.translation.case import TranslationTestCase

GEMINI_3_5_FLASH_THOUGHT_SIGNATURE: Final = "AY89a1+kriNs9YsPegJQxgjTWsMlxsqPU/UkHDyz4EK+3FRrE52ygqPIHTvkblgKKiO5Fn8R9D3iu//i+YnuuZ+e7W0yxUoFScas89b5rn3vZHsihnUA7VTe3b7beth1IftteoTGZ3F8OSLrSIP8HKn4xTIx35eeNEt5oiD3K5rTQrdklqO277HvTDhu/LuZhZ0pzaWciGQEc61epIBBpC2ygu8V4L8TekUVZnz6qzSzxipdiQStQ8dZou43334hMnxOR6JjJCGxqt/1K+C0vXX6YP6f+Bph5VhPSnvy2Z1uApi3TUxAcsDDyVewIr+ha2G6DPhN1Oi5uZZNhmbvH01Uq5OOquF1uEICq2m/35LhJIbDNM9N7oI1Z6L94P9xdugeFXM2S7wdJfc8sdg09vEzNeA++GK25zRuB4hU7XyaG4XnWjmu2kswi4sf+EyrO/K0P2tb/ZCJu7kJTtWTiD1wgSIl1eqS8ORKKsXDN2hIbFrtsNR50WIvVzd6"

GEMINI_3_8_FLASH_THOUGHT_SIGNATURE: Final = "AY89a1+gmp4bVJhwGOCtbUalmyfhwvgE4EsZOwS4bNR02vurTt4AZC72ypTNLJDRwK8I518tGYBiDnFqD6jTxkjWTQTPGwUZ5IOYr1IskTb3pPKfvZWZ/XunV7kfb9VmEU0TBlgwjVqxXziOxvjediJyb2gbFnCVjGTSVsq0ipmpFznF22lV1I6N9GI7RKBFgYXaSzGU1+fGbLLBigaUD0qbgquAnFv6R3JGpaLcW7EUrhKDinHoFukmUwBLx07S7SfuWBT8hMLMs32vEVzwpns9JmbVtuQG24zs5qQx9sJqtO9E07k6SFL3f1tvaKZsQ6Yjb08Xcuvnh9Pcz4ChCdRJ4vRkmtRW3AgOli8/43rrrAMmvIOqEdAOOvZW3W9D8lnOC+Nfq7S0FTNWPYkIJhrtskWoQkUwDrY6TOzMjIA6Jblt2f9kKU3VJzq7O1hPjbHsJKmLBYP+De8rKx7Ufq9Shs5qLLu4aBBgHleiyFiHGQSD18RDjnyv0DLRyrp7LOkjIYWBYoIA2+BSb0ySMyeLhk8B2U8XZ6AXiw3aHoNUG/wwuE0EhXBhOUBM14FmchGfptXUr+Dn9kZvPAy8mfbq3Pk="

GEMINI_3_1_PRO_PREVIEW_THOUGHT_SIGNATURE: Final = "AY89a198sdR5Cy3qgtvOq/o5xsmH1g3WymzyZyB/bv3XeyqNZAgXbCocCAqUS913U8FvVUodNrm3y2B5P/mDsn3A3pINhbEiwJcWA/o8bNJq+Z/1O2UYdViO0FMhOOS1aRaCbza+6aSBLPVQ81QnyxrwH6rdDCuQlwyZ+5/Evc/EOXLVxdUF9PYwTwNJcZPREl4UMcarKvtKsNofkSfa09EWQNVVMDkasNl3YHtQJqYLu5rL7vSUrbMtLLA0sMEMh2/4zMj4gIvtjoOwiI4CBdLLH0Xp8SGGco8UljxikaIp6YvQW4cHsfkaqTBQmLgREudbnKKTZ0MvMR52G/++7r8Iv6BL29+73igMMyHoQad40OM8RY3E6akpSE5uPXH/CgYTmSLXQONtRMXzrXm996JYLaHLOo2dP1RtaJANcjNoDB4eN5vj4GOpTUYqZqRRaB5jMX2od8QcAnaXk2AYs/2rNMaLV4zgraz45gbA0It465nN5hb4JbRH2toJ9K/TvsZY/YyoLmV9RjyAmIhbq6bG0fGatsXgmDaXj/vEsBmqfRyBN31k"

CLAUDE_HAIKU_4_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "vertex_ai/claude-haiku-4-5@20251001",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
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
        "created": ANY,
        "model": "vertex_ai/claude-haiku-4-5@20251001",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
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
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
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
        "model": "vertex_ai/claude-sonnet-4-6@default",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
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
        "created": ANY,
        "model": "vertex_ai/claude-sonnet-4-6@default",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello!",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
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
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
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
        "model": "vertex_ai/claude-sonnet-5@default",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
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
        "created": ANY,
        "model": "vertex_ai/claude-sonnet-5@default",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
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
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
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
        "model": "vertex_ai/claude-opus-4-8@default",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
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
        "created": ANY,
        "model": "vertex_ai/claude-opus-4-8@default",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
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
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
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
        "model": "vertex_ai/claude-sonnet-5-5@default",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-sonnet-5-5@default:rawPredict",
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
        "model": "claude-sonnet-5-5",
        "id": "msg_vrtx_011CfjuywExcu3BdZ3x1KzVJ",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 23,
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
        "created": ANY,
        "model": "vertex_ai/claude-sonnet-5-5@default",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
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
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
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
        "model": "vertex_ai/claude-opus-5-5@default",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/anthropic/models/claude-opus-5-5@default:rawPredict",
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
        "model": "claude-opus-5-5",
        "id": "msg_vrtx_011Cfjuz6qugj8FbC3fcXdun",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 23,
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
        "created": ANY,
        "model": "vertex_ai/claude-opus-5-5@default",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello.",
                    "role": "assistant",
                    "provider_specific_fields": {"citations": None, "thinking_blocks": None},
                },
            }
        ],
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
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
)

GEMINI_3_5_FLASH_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "vertex_ai/gemini-3.5-flash",
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/google/models/gemini-3.5-flash:generateContent",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer scripted-token"},
    expected_provider_request={
        "contents": [{"role": "user", "parts": [{"text": "Say hello."}]}],
        "system_instruction": {"parts": [{"text": "You are a terse assistant."}]},
        "generationConfig": {"max_output_tokens": 1024, "temperature": 1.0},
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
        "id": "uirEaqqwMc-U9LsPjJTYuQU",
        "created": ANY,
        "model": "vertex_ai/gemini-3.5-flash",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello.",
                    "role": "assistant",
                    "images": [],
                    "thinking_blocks": [],
                    "provider_specific_fields": {"thought_signatures": [GEMINI_3_5_FLASH_THOUGHT_SIGNATURE]},
                },
                "provider_specific_fields": {"native_finish_reason": "STOP"},
            }
        ],
        "usage": {
            "completion_tokens": 86,
            "prompt_tokens": 9,
            "total_tokens": 95,
            "completion_tokens_details": {"reasoning_tokens": 84, "text_tokens": 2},
            "prompt_tokens_details": {"text_tokens": 9},
        },
        "vertex_ai_grounding_metadata": [],
        "vertex_ai_url_context_metadata": [],
        "vertex_ai_safety_results": [],
        "vertex_ai_citation_metadata": [],
    },
)

GEMINI_3_8_FLASH_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "vertex_ai/gemini-3.8-flash",
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/google/models/gemini-3.8-flash:generateContent",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer scripted-token"},
    expected_provider_request={
        "contents": [{"role": "user", "parts": [{"text": "Say hello."}]}],
        "system_instruction": {"parts": [{"text": "You are a terse assistant."}]},
        "generationConfig": {"max_output_tokens": 1024, "temperature": 1.0},
    },
    mock_provider_response={
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello.", "thoughtSignature": GEMINI_3_8_FLASH_THOUGHT_SIGNATURE}],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 9,
            "candidatesTokenCount": 2,
            "totalTokenCount": 118,
            "trafficType": "ON_DEMAND",
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 9}],
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 2}],
            "thoughtsTokenCount": 107,
        },
        "modelVersion": "gemini-3.8-flash",
        "createTime": "2026-10-05T22:55:12.739855Z",
        "responseId": "0CrEao-ULbuU9LsPk-mQ2A8",
    },
    expected_litellm_status_code=200,
    expected_litellm_response={
        "id": "0CrEao-ULbuU9LsPk-mQ2A8",
        "created": ANY,
        "model": "vertex_ai/gemini-3.8-flash",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello.",
                    "role": "assistant",
                    "images": [],
                    "thinking_blocks": [],
                    "provider_specific_fields": {"thought_signatures": [GEMINI_3_8_FLASH_THOUGHT_SIGNATURE]},
                },
                "provider_specific_fields": {"native_finish_reason": "STOP"},
            }
        ],
        "usage": {
            "completion_tokens": 109,
            "prompt_tokens": 9,
            "total_tokens": 118,
            "completion_tokens_details": {"reasoning_tokens": 107, "text_tokens": 2},
            "prompt_tokens_details": {"text_tokens": 9},
        },
        "vertex_ai_grounding_metadata": [],
        "vertex_ai_url_context_metadata": [],
        "vertex_ai_safety_results": [],
        "vertex_ai_citation_metadata": [],
    },
)

GEMINI_3_1_PRO_PREVIEW_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "vertex_ai/gemini-3.1-pro-preview",
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/projects/synthetic-vertex-project/locations/global/publishers/google/models/gemini-3.1-pro-preview:generateContent",
    expected_provider_headers={"content-type": "application/json", "authorization": "Bearer scripted-token"},
    expected_provider_request={
        "contents": [{"role": "user", "parts": [{"text": "Say hello."}]}],
        "system_instruction": {"parts": [{"text": "You are a terse assistant."}]},
        "generationConfig": {"max_output_tokens": 1024, "temperature": 1.0},
    },
    mock_provider_response={
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello.", "thoughtSignature": GEMINI_3_1_PRO_PREVIEW_THOUGHT_SIGNATURE}],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 9,
            "candidatesTokenCount": 1,
            "totalTokenCount": 99,
            "trafficType": "ON_DEMAND",
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 9}],
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 1}],
            "thoughtsTokenCount": 89,
        },
        "modelVersion": "gemini-3.1-pro-preview",
        "createTime": "2026-10-05T22:55:14.874538Z",
        "responseId": "0irEaqqwNbCdq8YP3L2mqQ4",
    },
    expected_litellm_status_code=200,
    expected_litellm_response={
        "id": "0irEaqqwNbCdq8YP3L2mqQ4",
        "created": ANY,
        "model": "vertex_ai/gemini-3.1-pro-preview",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "Hello.",
                    "role": "assistant",
                    "images": [],
                    "thinking_blocks": [],
                    "provider_specific_fields": {"thought_signatures": [GEMINI_3_1_PRO_PREVIEW_THOUGHT_SIGNATURE]},
                },
                "provider_specific_fields": {"native_finish_reason": "STOP"},
            }
        ],
        "usage": {
            "completion_tokens": 90,
            "prompt_tokens": 9,
            "total_tokens": 99,
            "completion_tokens_details": {"reasoning_tokens": 89, "text_tokens": 1},
            "prompt_tokens_details": {"text_tokens": 9},
        },
        "vertex_ai_grounding_metadata": [],
        "vertex_ai_url_context_metadata": [],
        "vertex_ai_safety_results": [],
        "vertex_ai_citation_metadata": [],
    },
)
