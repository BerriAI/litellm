from typing import Final
from unittest.mock import ANY

from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.templates import CHAT_COMPLETIONS_REQUEST, CHAT_COMPLETIONS_RESPONSE
from integration.translation.templates.bedrock_mantle import (
    BEDROCK_MANTLE_ANTHROPIC_MESSAGES_REQUEST,
    BEDROCK_MANTLE_ANTHROPIC_MESSAGES_RESPONSE,
    BEDROCK_MANTLE_OPENAI_RESPONSES_REQUEST,
    BEDROCK_MANTLE_OPENAI_RESPONSES_RESPONSE,
    CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
)

CLAUDE_HAIKU_4_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "bedrock_mantle/anthropic.claude-haiku-4-5",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "content-type": "application/json",
        "authorization": "Bearer synthetic-bedrock-mantle-key",
    },
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "anthropic_version": "bedrock-2023-05-31",
        "model": "anthropic.claude-haiku-4-5",
    },
    mock_provider_response={
        "model": "claude-haiku-4-5-20251001",
        "id": "msg_bdrk_vmd4jw2bdh7lsarmxtb244m4pw3umtkcnlhschksadqqcpqfcmiq",
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
        },
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "bedrock_mantle/anthropic.claude-haiku-4-5",
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
            "service_tier": "standard",
        },
    },
)


CLAUDE_SONNET_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "bedrock_mantle/anthropic.claude-sonnet-5",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "content-type": "application/json",
        "authorization": "Bearer synthetic-bedrock-mantle-key",
    },
    expected_provider_request={
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "anthropic_version": "bedrock-2023-05-31",
        "model": "anthropic.claude-sonnet-5",
    },
    mock_provider_response={
        "model": "claude-sonnet-5",
        "id": "msg_bdrk_fxpfvo6z6ua32kp776ux7hv2ultohujjrfemqzl34vfdfaihltma",
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
        },
    },
    expected_litellm_response={
        "id": ANY,
        "created": ANY,
        "model": "bedrock_mantle/anthropic.claude-sonnet-5",
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
            "service_tier": "standard",
        },
    },
)


GPT_5_4_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "bedrock_mantle/openai.gpt-5.4",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/openai/v1/responses",
    expected_provider_headers={
        "authorization": "Bearer synthetic-bedrock-mantle-key",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "openai.gpt-5.4",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Say hello."}]}],
        "instructions": "You are a terse assistant.",
        "max_output_tokens": 64,
        "stream": False,
    },
    mock_provider_response={
        "background": False,
        "billing": {"payer": "developer"},
        "completed_at": 1791240602,
        "created_at": 1791240602,
        "error": None,
        "frequency_penalty": 0.0,
        "id": "resp_h5f65ggx7fstww26djcqgint5e7gwa2h6q43xjkxhggr2ftxybnq",
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "max_output_tokens": 64,
        "max_tool_calls": None,
        "metadata": {},
        "model": "openai.gpt-5.4",
        "object": "response",
        "output": [
            {
                "content": [{"annotations": [], "logprobs": [], "text": "Hello.", "type": "output_text"}],
                "id": "msg_1abfa71ec4495d38a341706f85e7ef0b",
                "phase": "final_answer",
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "presence_penalty": 0.0,
        "previous_response_id": None,
        "prompt_cache_key": None,
        "prompt_cache_retention": "in_memory",
        "reasoning": {"summary": None, "effort": "none"},
        "safety_identifier": None,
        "service_tier": "default",
        "status": "completed",
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "auto",
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
        "moderation": None,
    },
    expected_litellm_response={
        "id": "resp_h5f65ggx7fstww26djcqgint5e7gwa2h6q43xjkxhggr2ftxybnq",
        "created": ANY,
        "model": "bedrock_mantle/openai.gpt-5.4",
        "object": "chat.completion",
        "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "Hello.", "role": "assistant"}}],
        "usage": {
            "completion_tokens": 6,
            "prompt_tokens": 19,
            "total_tokens": 25,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "billing": {"payer": "developer"},
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
    },
)


GPT_5_6_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "bedrock_mantle/openai.gpt-5.6-sol",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/openai/v1/responses",
    expected_provider_headers={
        "authorization": "Bearer synthetic-bedrock-mantle-key",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "openai.gpt-5.6-sol",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Say hello."}]}],
        "instructions": "You are a terse assistant.",
        "max_output_tokens": 64,
        "stream": False,
    },
    mock_provider_response={
        "background": False,
        "billing": {"payer": "developer"},
        "completed_at": 1791240604,
        "created_at": 1791240604,
        "error": None,
        "frequency_penalty": 0.0,
        "id": "resp_7s4e6m623k3hkjqx4fexvwqodxn3zxtfpkqrz4hdl7qe2wcyftsa",
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "max_output_tokens": 64,
        "max_tool_calls": None,
        "metadata": {},
        "model": "openai.gpt-5.6-sol",
        "object": "response",
        "output": [
            {
                "content": [{"annotations": [], "logprobs": [], "text": "Hello!", "type": "output_text"}],
                "id": "msg_d44132d374595fe789bdc2db8e00fdc0",
                "phase": "final_answer",
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "presence_penalty": 0.0,
        "previous_response_id": None,
        "prompt_cache_key": None,
        "prompt_cache_retention": "in_memory",
        "reasoning": {"summary": None, "effort": "medium", "context": "all_turns"},
        "safety_identifier": None,
        "service_tier": "default",
        "status": "completed",
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "auto",
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
        "moderation": None,
    },
    expected_litellm_response={
        "id": "resp_7s4e6m623k3hkjqx4fexvwqodxn3zxtfpkqrz4hdl7qe2wcyftsa",
        "created": ANY,
        "model": "bedrock_mantle/openai.gpt-5.6-sol",
        "object": "chat.completion",
        "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "Hello!", "role": "assistant"}}],
        "usage": {
            "completion_tokens": 6,
            "prompt_tokens": 19,
            "total_tokens": 25,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "billing": {"payer": "developer"},
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
    },
)


GPT_5_6_LUNA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": "bedrock_mantle/openai.gpt-5.6-luna",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/openai/v1/responses",
    expected_provider_headers={
        "authorization": "Bearer synthetic-bedrock-mantle-key",
        "content-type": "application/json",
    },
    expected_provider_request={
        "model": "openai.gpt-5.6-luna",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Say hello."}]}],
        "instructions": "You are a terse assistant.",
        "max_output_tokens": 64,
        "stream": False,
    },
    mock_provider_response={
        "background": False,
        "billing": {"payer": "developer"},
        "completed_at": 1791240607,
        "created_at": 1791240606,
        "error": None,
        "frequency_penalty": 0.0,
        "id": "resp_gnit33ia2bne452i37ag5m5jgatjnnqog5a25yjrymielrfwokra",
        "incomplete_details": None,
        "instructions": "You are a terse assistant.",
        "max_output_tokens": 64,
        "max_tool_calls": None,
        "metadata": {},
        "model": "openai.gpt-5.6-luna",
        "object": "response",
        "output": [
            {
                "content": [{"annotations": [], "logprobs": [], "text": "Hello!", "type": "output_text"}],
                "id": "msg_b216bd6781d252db9607a7b2c25af58f",
                "phase": "final_answer",
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "presence_penalty": 0.0,
        "previous_response_id": None,
        "prompt_cache_key": None,
        "prompt_cache_retention": "in_memory",
        "reasoning": {"summary": None, "effort": "medium", "context": "all_turns"},
        "safety_identifier": None,
        "service_tier": "default",
        "status": "completed",
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "auto",
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
        "moderation": None,
    },
    expected_litellm_response={
        "id": "resp_gnit33ia2bne452i37ag5m5jgatjnnqog5a25yjrymielrfwokra",
        "created": ANY,
        "model": "bedrock_mantle/openai.gpt-5.6-luna",
        "object": "chat.completion",
        "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "Hello!", "role": "assistant"}}],
        "usage": {
            "completion_tokens": 6,
            "prompt_tokens": 19,
            "total_tokens": 25,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "billing": {"payer": "developer"},
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
    },
)

CLAUDE_OPUS_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "bedrock_mantle/anthropic.claude-opus-5-5"},
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "content-type": "application/json",
        "authorization": "Bearer synthetic-bedrock-mantle-key",
    },
    expected_provider_request={**BEDROCK_MANTLE_ANTHROPIC_MESSAGES_REQUEST},
    mock_provider_response={**BEDROCK_MANTLE_ANTHROPIC_MESSAGES_RESPONSE},
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "model": "bedrock_mantle/anthropic.claude-opus-5-5",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "Next I'll read `shop/validation.py` around `validate_quantity` (line 18) to see how it "
                    "handles zero, negative and non-integer values.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 2,
                            "function": {
                                "arguments": '{"path": "shop/validation.py", "start_line": 1, "end_line": 50}',
                                "name": "read_file",
                            },
                            "id": "toolu_bdrk_018J8SJwAgjhRkkmtm3xdJx2",
                            "type": "function",
                        },
                    ],
                    "reasoning_content": "I should check the validation file first, starting around line 18.\n\n",
                    "thinking_blocks": [
                        {
                            "type": "thinking",
                            "thinking": "I should check the validation file first, starting around line 18.\n\n",
                            "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
                        },
                    ],
                    "provider_specific_fields": {
                        "citations": None,
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "I should check the validation file first, starting around line 18.\n\n",
                                "signature": CLAUDE_OPUS_5_5_THINKING_SIGNATURE_0,
                            },
                        ],
                    },
                },
            },
        ],
        "usage": {
            "completion_tokens": 166,
            "prompt_tokens": 880,
            "total_tokens": 1046,
            "completion_tokens_details": {"reasoning_tokens": 19, "text_tokens": 147},
            "prompt_tokens_details": {
                "cached_tokens": 876,
                "text_tokens": 4,
                "cache_write_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_creation_token_details": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            },
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 876,
            "service_tier": "standard",
        },
    },
)


GPT_6_1_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "bedrock_mantle/openai.gpt-6.1-sol"},
    expected_provider_endpoint="/openai/v1/responses",
    expected_provider_headers={
        "authorization": "Bearer synthetic-bedrock-mantle-key",
        "content-type": "application/json",
    },
    expected_provider_request={
        **BEDROCK_MANTLE_OPENAI_RESPONSES_REQUEST,
        "stream": False,
    },
    mock_provider_response={**BEDROCK_MANTLE_OPENAI_RESPONSES_RESPONSE},
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "id": "resp_izakuft4jrxwmodefqee2jjt6gkmayywwuuc242pduoydywb3v4a",
        "model": "bedrock_mantle/openai.gpt-6.1-sol",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "I’ll read `shop/validation.py` first to see how it handles zero, negative, and "
                    "non-integer quantities before inspecting the tests.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                                "name": "read_file",
                            },
                            "id": "call_399ba0e7ac865a798b46a6cabe6b6bde",
                            "type": "function",
                        },
                    ],
                    "reasoning_items": [
                        {
                            "type": "reasoning",
                            "id": "rs_7a90cfc98ff653aeb165c2b845bc1e4b",
                            "encrypted_content": "rsn_rru6BQHgUc/VWS5vRIAlHwIFeJ7rFK0vbfOLCW3lZUKZX7nZu5wsyKm4u2DeoLRbILW0AF8AAQAVYXdzLWNyeXB0by1wdWJsaWMta2V5AERBN1N2LzZrWDd1QWV1WkpjQ1dEOFQvSDR2Y0ZaY2ZWemtpR2ZWNGRncEgwZmJqMVBNd3FIV2NnbFlTcHVOcVMyZWc9PQABABFhd3Mta21zLWhpZXJhcmNoeQAKYmVybS1zdG9yZQBcmpoCBv5IPHctwun5UTXrXrEmFVDe7MUULd6PZATyexPQO0PepR8jRQ3zVYSjkLkI8eYSjtPsT9kCKrIpsJlAuid+IimJvnTBHGplsaTC/53IpOSzB87nJlx+m4kCAAAQANtzzELZF5mKNBTvW/ca/WZ6iJI42OwsDigg6J5ZOIdJaC35TDfpQsssoDFQIHtI/v////8AAAABAAAAAAAAAAAAAAABAAABSyq/hcdJ8rImbQvE2c8qmEFM3skJI4sMoBBIdIZm8ljKX6zuu9kBGRln5StGWzlUOVYemoWeHBoDDuZkGDWAeTcQYdNeiCuTOQBg1kfIvhSGpGZHyKIfg+XnWIjrExKbZhmAC/GMebMNNuJ1/ZUFnq0UtD8X1r2xEkIWCUsQHysv9E6D6AstjrAgpo5arvZG0Gi5gryi/eE9Q7IGwilIvzeslUlM7sVcmD5RWe83nMeXmCUR41N50TK11F1j/H314CtKbJbMQWNWGLUUSRiKjqDFiC+o4bioQCRzvka6NUkIBitgjjkJ+etpL5ibB065xADWxaRc6WoUe1VkNVKXlYq4kOBNhnkJSHuasIJiD8/suYWXshD9XMF+SffKyKk3zscZwAuZ7N/ZOayYqGrhOMryCFYXvSo7v8PJ1hvg64z1MknImFG/6rSEVox9yeyfSEQmeDX/ABwO8PAsAGcwZQIwIpngWR/1wARV01RTUI+B4dV7XaR1AC0aOv5KfuQw5UJOnfGVp8E7MS379SPiA+SEAjEA0TMJ1LJ3cXitGqWvPATglnscU7UW9Mhny4elQ+cw7pFm5kRlYA/lkAyDdNyjZFVU",
                            "summary": [],
                        },
                    ],
                },
            },
        ],
        "usage": {
            "completion_tokens": 108,
            "prompt_tokens": 312,
            "total_tokens": 420,
            "completion_tokens_details": {"reasoning_tokens": 39},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "billing": {"payer": "developer"},
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
    },
)

GPT_6_LUNA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "bedrock_mantle/openai.gpt-6-luna"},
    expected_provider_endpoint="/openai/v1/responses",
    expected_provider_headers={
        "authorization": "Bearer synthetic-bedrock-mantle-key",
        "content-type": "application/json",
    },
    expected_provider_request={
        **BEDROCK_MANTLE_OPENAI_RESPONSES_REQUEST,
        "model": "openai.gpt-6-luna",
        "stream": False,
    },
    mock_provider_response={
        **BEDROCK_MANTLE_OPENAI_RESPONSES_RESPONSE,
        "completed_at": 1792027279,
        "created_at": 1792027278,
        "id": "resp_pbro2cu7wxoshets5f2v3mmv22dzjjepi3ztmm6qceyuuveilumq",
        "model": "openai.gpt-6-luna",
        "output": [
            {
                "content": [],
                "encrypted_content": "rsn_rru6BQHgUc/VWS5vRIAlHwIFeO/Kvntq5madicvQnPlNuvNXUggyOe9VUAXpEAJIioejAF8AAQAVYXdzLWNyeXB0by1wdWJsaWMta2V5AERBd3VOZ05NSUlPMW5NbkYrOTRMbVdXb2tXcEpDRE9VaUl0eFF0UzFtUjNBUGlvcUVIWlNlYm1TMFZDQmlqdEJ5WkE9PQABABFhd3Mta21zLWhpZXJhcmNoeQAKYmVybS1zdG9yZQBczj0IBqgjEgu5ZgKVdTj3OPcpzd2fIh34hEYxgwTyexPQO0PepR8jRQ3zVYTReEQNipLtt5xyBLb3uYPRCtqi6ry4MIerha2orIM/kNwSud8QD55iZRsgKj/1XbcCAAAQAJdjRdvVBzuOT4dEabwB/y7OUCvt4aMTO2tMpab00xmq/PMDdaAtcFiOCt2np+72If////8AAAABAAAAAAAAAAAAAAABAAABU4/yBtf3Lh0ztBhEM9+zVRuLBf3cQi+TBEvKzqKrCRcFAt+D762ck8n+qzDT/Kun8V8tLn5f1CQMMOdn2cu7aDV+4yYud8rII/NzEYzht+XQy0doyMj4GLfIW8jJ81IBh02MAaK+b1dZilKsA6448TsvE+58lIPPnJ7vScz2sRQzotxYqb0EIj6V+zl980Ucs+Ge1SGRwMVeUj/5mEspkmtJt0Tjkyku2jdqokDqpDXFBQDLAIKv+sRfpO6BQL64bovGzIhP+FSo1QFPW+NSpfILrWhB8WwUzRnoqrKcSTmqkDAx7v/wzUWeN0rl1XjbQgcJDs4kjgcBTR/4nhVX7rlAmiDaeZkZHDTKYYq67BaAwF35TuWAfz3cvusF1FZ6CQETRnkXXfRqQITDG6CEADQDrUtp+8kF5vIsFGX3ivYEDtlrKRcriWlq30dHn2ewRh8B1Z+k+GheeEyoDt3SC2KhsvQAZzBlAjAENp2ZItMXMogj4+bqZ1DFA/VBQb4fEwuIJlAEXCi2NWbzA7GFtwVfyqcx3nGPs8ICMQDPeXyziTwCViPMplaixJa2Z+6m8FsynJE6Gt4joeNS1Gm36RHhrdmv7Nx68QTw7RA=",
                "id": "rs_41fd1f946fd95402ac51feb270547528",
                "summary": [],
                "type": "reasoning",
            },
            {
                "content": [
                    {
                        "annotations": [],
                        "logprobs": [],
                        "text": "I need `shop/validation.py` next; it contains the quantity validator, so it will "
                        "show how zero, negative, and non-integer values are handled.",
                        "type": "output_text",
                    },
                ],
                "id": "msg_bd4b22ee05d75ecc863a79cadb0abc97",
                "phase": "commentary",
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
            {
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":100}',
                "call_id": "call_2bb9788b89cf56e595d03edd897cf96d",
                "id": "fc_6cec15107b1055dd90bdbf343add57e6",
                "name": "read_file",
                "status": "completed",
                "type": "function_call",
            },
        ],
        "usage": {
            "input_tokens": 312,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 115,
            "output_tokens_details": {"reasoning_tokens": 42},
            "total_tokens": 427,
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "id": "resp_pbro2cu7wxoshets5f2v3mmv22dzjjepi3ztmm6qceyuuveilumq",
        "model": "bedrock_mantle/openai.gpt-6-luna",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "I need `shop/validation.py` next; it contains the quantity validator, so it will show "
                    "how zero, negative, and non-integer values are handled.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":100}',
                                "name": "read_file",
                            },
                            "id": "call_2bb9788b89cf56e595d03edd897cf96d",
                            "type": "function",
                        },
                    ],
                    "reasoning_items": [
                        {
                            "type": "reasoning",
                            "id": "rs_41fd1f946fd95402ac51feb270547528",
                            "encrypted_content": "rsn_rru6BQHgUc/VWS5vRIAlHwIFeO/Kvntq5madicvQnPlNuvNXUggyOe9VUAXpEAJIioejAF8AAQAVYXdzLWNyeXB0by1wdWJsaWMta2V5AERBd3VOZ05NSUlPMW5NbkYrOTRMbVdXb2tXcEpDRE9VaUl0eFF0UzFtUjNBUGlvcUVIWlNlYm1TMFZDQmlqdEJ5WkE9PQABABFhd3Mta21zLWhpZXJhcmNoeQAKYmVybS1zdG9yZQBczj0IBqgjEgu5ZgKVdTj3OPcpzd2fIh34hEYxgwTyexPQO0PepR8jRQ3zVYTReEQNipLtt5xyBLb3uYPRCtqi6ry4MIerha2orIM/kNwSud8QD55iZRsgKj/1XbcCAAAQAJdjRdvVBzuOT4dEabwB/y7OUCvt4aMTO2tMpab00xmq/PMDdaAtcFiOCt2np+72If////8AAAABAAAAAAAAAAAAAAABAAABU4/yBtf3Lh0ztBhEM9+zVRuLBf3cQi+TBEvKzqKrCRcFAt+D762ck8n+qzDT/Kun8V8tLn5f1CQMMOdn2cu7aDV+4yYud8rII/NzEYzht+XQy0doyMj4GLfIW8jJ81IBh02MAaK+b1dZilKsA6448TsvE+58lIPPnJ7vScz2sRQzotxYqb0EIj6V+zl980Ucs+Ge1SGRwMVeUj/5mEspkmtJt0Tjkyku2jdqokDqpDXFBQDLAIKv+sRfpO6BQL64bovGzIhP+FSo1QFPW+NSpfILrWhB8WwUzRnoqrKcSTmqkDAx7v/wzUWeN0rl1XjbQgcJDs4kjgcBTR/4nhVX7rlAmiDaeZkZHDTKYYq67BaAwF35TuWAfz3cvusF1FZ6CQETRnkXXfRqQITDG6CEADQDrUtp+8kF5vIsFGX3ivYEDtlrKRcriWlq30dHn2ewRh8B1Z+k+GheeEyoDt3SC2KhsvQAZzBlAjAENp2ZItMXMogj4+bqZ1DFA/VBQb4fEwuIJlAEXCi2NWbzA7GFtwVfyqcx3nGPs8ICMQDPeXyziTwCViPMplaixJa2Z+6m8FsynJE6Gt4joeNS1Gm36RHhrdmv7Nx68QTw7RA=",
                            "summary": [],
                        },
                    ],
                },
            },
        ],
        "usage": {
            "completion_tokens": 115,
            "prompt_tokens": 312,
            "total_tokens": 427,
            "completion_tokens_details": {"reasoning_tokens": 42},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "billing": {"payer": "developer"},
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
    },
)

GPT_6_ASTRA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "bedrock_mantle/openai.gpt-6-astra"},
    expected_provider_endpoint="/openai/v1/responses",
    expected_provider_headers={
        "authorization": "Bearer synthetic-bedrock-mantle-key",
        "content-type": "application/json",
    },
    expected_provider_request={
        **BEDROCK_MANTLE_OPENAI_RESPONSES_REQUEST,
        "model": "openai.gpt-6-astra",
        "stream": False,
    },
    mock_provider_response={
        **BEDROCK_MANTLE_OPENAI_RESPONSES_RESPONSE,
        "completed_at": 1792027285,
        "created_at": 1792027282,
        "id": "resp_jpvjizqjx4f4u6nyz7deamnfdzbqxgxqm7nirzdlaxm2mbp5r7yq",
        "model": "openai.gpt-6-astra",
        "output": [
            {
                "content": [],
                "encrypted_content": "rsn_rru6BQHgUc/VWS5vRIAlHwIFeOrR57yIGw3WXzOCHxReanHc4wZJmaqSvXVqP0qGFGOIAF8AAQAVYXdzLWNyeXB0by1wdWJsaWMta2V5AERBK1FoNHBXMk1DRU5Vdis5d3M3OVBTVDBuU09VTkhlUDViMDkwejZPeWZpOEhOWnBjYXNKNmJVL0J5VWxHZDA2bFE9PQABABFhd3Mta21zLWhpZXJhcmNoeQAKYmVybS1zdG9yZQBcM8fQN00KGzoCPxlReE9xaFkOBsPUkSLsg/xl/wTyexPQO0PepR8jRQ3zVYQk8QfLD0eucuatvVchgKNLGMVoDPckkpY6SgFElbU1FqyUMQRvNUXBscq6GwAQezoCAAAQAGSi7URwBzZfrKqE7cF8zQKcbSay4uxRdsBmjsgYzhEssEjRy/XHvWoc05wPhp6Fb/////8AAAABAAAAAAAAAAAAAAABAAABsukDvrtRmG7Y5bL3xQiRmAPLsY2Wf14e5hnSAPYx/GMaB6IatF4KG1RSZK14/tlh+/kB5vAcOrXrllEzvZHIFDMWgRWexPSGO/WTBZ8GbLNmr5CZo9bMu0wxt86lxm+clENYV8HgEIZC7Gg8gQLw1IgT97PiOaYqg63/uHN6VME19CtW68lifkgtupCAP9ewU0/W1BG9TggARvgfL6Rhn087DjY8F0ievrVLHQRujVnEZa+Yfsk+WVCT156LYo2wGLwkcZDycPX7/2Qn6hiHm7/vUq2OYGSWDOabutgooMUhX/vA+0u3cEHQu+Vv0QbwaUH5+MLB0L85bX58U3HlF6KG02vuGl8TiI74zvzYx00YFoP+n6NmaH2VoY8OaaTr6hQKEpHqiesAPWXdAA1+TmJ9mMjTATYyxdsj7E3X0+djoKENYqaNqA7tlEdm9EvKnMPtHWtsSUsO4nB38nsQyHKJqyUQP/MauHrq+YDx4GKwB45EnvGADOCRf7uuub7mACOhersjo1T7SGCTq9OP73x7/tYtSKeYVGZlMMAJGTmc2o/l982eGRf6B+/o605lqvCYX3ZBTuAr+823Wx/1uuvLNgBnMGUCMQDrd4nbQJvNcl/n8tW+ibBjExgdHHqO14vyB+oBDToTD5XeX/nI8UZKrE2aFLHsXo8CMBaXARx3KSuVYzI8Dsqn5Br0DPqZx5n7NmRNhqUvL0y4ZhMpivrpq1xc2uW3vUyYxQ==",
                "id": "rs_5ddaaaa907d7599da226348b310ffaa1",
                "summary": [],
                "type": "reasoning",
            },
            {
                "content": [
                    {
                        "annotations": [],
                        "logprobs": [],
                        "text": "I’ll read `shop/validation.py` first to check how it handles zero, negative, "
                        "and non-integer quantities.\n",
                        "type": "output_text",
                    },
                ],
                "id": "msg_7ee57d84b2515199bcfd28428434f307",
                "phase": "commentary",
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
            {
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                "call_id": "call_48b391faab505f0e90dfc7c451b87c2b",
                "id": "fc_c8b90cef33d256a986e7a9229db5017a",
                "name": "read_file",
                "status": "completed",
                "type": "function_call",
            },
        ],
        "usage": {
            "input_tokens": 312,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 148,
            "output_tokens_details": {"reasoning_tokens": 83},
            "total_tokens": 460,
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "id": "resp_jpvjizqjx4f4u6nyz7deamnfdzbqxgxqm7nirzdlaxm2mbp5r7yq",
        "model": "bedrock_mantle/openai.gpt-6-astra",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "I’ll read `shop/validation.py` first to check how it handles zero, negative, and "
                    "non-integer quantities.\n",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                                "name": "read_file",
                            },
                            "id": "call_48b391faab505f0e90dfc7c451b87c2b",
                            "type": "function",
                        },
                    ],
                    "reasoning_items": [
                        {
                            "type": "reasoning",
                            "id": "rs_5ddaaaa907d7599da226348b310ffaa1",
                            "encrypted_content": "rsn_rru6BQHgUc/VWS5vRIAlHwIFeOrR57yIGw3WXzOCHxReanHc4wZJmaqSvXVqP0qGFGOIAF8AAQAVYXdzLWNyeXB0by1wdWJsaWMta2V5AERBK1FoNHBXMk1DRU5Vdis5d3M3OVBTVDBuU09VTkhlUDViMDkwejZPeWZpOEhOWnBjYXNKNmJVL0J5VWxHZDA2bFE9PQABABFhd3Mta21zLWhpZXJhcmNoeQAKYmVybS1zdG9yZQBcM8fQN00KGzoCPxlReE9xaFkOBsPUkSLsg/xl/wTyexPQO0PepR8jRQ3zVYQk8QfLD0eucuatvVchgKNLGMVoDPckkpY6SgFElbU1FqyUMQRvNUXBscq6GwAQezoCAAAQAGSi7URwBzZfrKqE7cF8zQKcbSay4uxRdsBmjsgYzhEssEjRy/XHvWoc05wPhp6Fb/////8AAAABAAAAAAAAAAAAAAABAAABsukDvrtRmG7Y5bL3xQiRmAPLsY2Wf14e5hnSAPYx/GMaB6IatF4KG1RSZK14/tlh+/kB5vAcOrXrllEzvZHIFDMWgRWexPSGO/WTBZ8GbLNmr5CZo9bMu0wxt86lxm+clENYV8HgEIZC7Gg8gQLw1IgT97PiOaYqg63/uHN6VME19CtW68lifkgtupCAP9ewU0/W1BG9TggARvgfL6Rhn087DjY8F0ievrVLHQRujVnEZa+Yfsk+WVCT156LYo2wGLwkcZDycPX7/2Qn6hiHm7/vUq2OYGSWDOabutgooMUhX/vA+0u3cEHQu+Vv0QbwaUH5+MLB0L85bX58U3HlF6KG02vuGl8TiI74zvzYx00YFoP+n6NmaH2VoY8OaaTr6hQKEpHqiesAPWXdAA1+TmJ9mMjTATYyxdsj7E3X0+djoKENYqaNqA7tlEdm9EvKnMPtHWtsSUsO4nB38nsQyHKJqyUQP/MauHrq+YDx4GKwB45EnvGADOCRf7uuub7mACOhersjo1T7SGCTq9OP73x7/tYtSKeYVGZlMMAJGTmc2o/l982eGRf6B+/o605lqvCYX3ZBTuAr+823Wx/1uuvLNgBnMGUCMQDrd4nbQJvNcl/n8tW+ibBjExgdHHqO14vyB+oBDToTD5XeX/nI8UZKrE2aFLHsXo8CMBaXARx3KSuVYzI8Dsqn5Br0DPqZx5n7NmRNhqUvL0y4ZhMpivrpq1xc2uW3vUyYxQ==",
                            "summary": [],
                        },
                    ],
                },
            },
        ],
        "usage": {
            "completion_tokens": 148,
            "prompt_tokens": 312,
            "total_tokens": 460,
            "completion_tokens_details": {"reasoning_tokens": 83},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "billing": {"payer": "developer"},
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
    },
)

GPT_5_6_TERRA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={**CHAT_COMPLETIONS_REQUEST, "model": "bedrock_mantle/openai.gpt-5.6-terra"},
    expected_provider_endpoint="/openai/v1/responses",
    expected_provider_headers={
        "authorization": "Bearer synthetic-bedrock-mantle-key",
        "content-type": "application/json",
    },
    expected_provider_request={
        **BEDROCK_MANTLE_OPENAI_RESPONSES_REQUEST,
        "model": "openai.gpt-5.6-terra",
        "stream": False,
    },
    mock_provider_response={
        **BEDROCK_MANTLE_OPENAI_RESPONSES_RESPONSE,
        "completed_at": 1792027294,
        "created_at": 1792027292,
        "id": "resp_rzoladobotq7d5aitvspg3wwysoow5b436v6zaymozikilgh4uwa",
        "model": "openai.gpt-5.6-terra",
        "output": [
            {
                "content": [],
                "encrypted_content": "rsn_rru6BQHgUc/VWS5vRIAlHwIFeJSjZG0YBxqkNyCzhib06j32zRnkxAVgKyx5FlM5EQsGAF8AAQAVYXdzLWNyeXB0by1wdWJsaWMta2V5AERBblF2WjlWWElaWDBLWXR0RDE3djIxTzd2bWJVdjZSQlhOYW1sMmxjTHRrVGFuVU4vc09LRDJlUzFGMENIQzdOa0E9PQABABFhd3Mta21zLWhpZXJhcmNoeQAKYmVybS1zdG9yZQBcL/APOsU1Boid3k+fiCN6FGnqlouHrRGycfpWSQTyexPQO0PepR8jRQ3zVYQqZPsD7v2uGJDXTNoMQ/DNJEPAmyJkLXtVRDtxK+LIpvBebVGlBLXtrij6Z/oBFI8CAAAQABUJCnt2FvwOmW31K7V/wAUU+CK5nMfo/9o1zvOTBV0aDdylXF3v56EtlCEoo8vw6/////8AAAABAAAAAAAAAAAAAAABAAABPxjRAPZm81KRyzjmvtQM3zhh5qZSePYBGrVVjgFK2e9hY/yVx8S8S70YklxeYaWmFkKyinCLXqUH0IFnsyUaBQGpvnrVixo+QDyeu4wV6WP5hyp7j/GGWaAlDM3zF4JkOIqpXYU2XbHwr1/wRzimQl3BcBxBpygEBHteEuAWENg16dfo76pI4Q+PpGOKfwaSXz9Cj2vprUCxGcsa9IY6tq6pr7hYdhtS4Zg8tYGrSoGm6/9ljF51LZ1TSvfHnKmNFgpp86naM08zO1Jp+fDU37fKH8nazA0xUB23BWMWWu63pvQaquAtx1rj5PisEWwemlY/5v/Wo4d4Q9W+nrGzXf2o9kScWnDpfCfR7G72u19J0SXHcfrFwj5oVN8Ek1eYKY8Apv2lxZAf+ULokPmUGk/lXkz9hxa52txJ/3+bjy7cJ4S9HoUuBRHt3l7PpJJ0AGcwZQIxAP/SkcKkldJ07XgfZ+neD7X9h8kLD3hHG4axXwP8LPDmZ6HKrnXbOoSZLT6Nf4JddAIwEte72+56IOzdbE/k0Zvq/5ILQB5G+FhUdu0tHd6YDhHqPuq4nzXJJ/+0dFHRWzcw",
                "id": "rs_2985de4698f95e958cd7c7cd377a76a2",
                "summary": [],
                "type": "reasoning",
            },
            {
                "content": [
                    {
                        "annotations": [],
                        "logprobs": [],
                        "text": "Next I need `shop/validation.py`, which contains `validate_quantity`; I’ll "
                        "inspect it to verify the handling of zero, negative, and non-integer inputs before "
                        "reviewing the tests.",
                        "type": "output_text",
                    },
                ],
                "id": "msg_afefa5c808a159a1965dc7c7986d0693",
                "phase": "commentary",
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
            {
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                "call_id": "call_fc5f6bc78e0b5332b7866a98d5470b46",
                "id": "fc_1e6d0bcf394d5b64badb08659c4c6028",
                "name": "read_file",
                "status": "completed",
                "type": "function_call",
            },
        ],
        "usage": {
            "input_tokens": 312,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 114,
            "output_tokens_details": {"reasoning_tokens": 36},
            "total_tokens": 426,
        },
    },
    expected_litellm_response={
        **CHAT_COMPLETIONS_RESPONSE,
        "id": "resp_rzoladobotq7d5aitvspg3wwysoow5b436v6zaymozikilgh4uwa",
        "model": "bedrock_mantle/openai.gpt-5.6-terra",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "message": {
                    "content": "Next I need `shop/validation.py`, which contains `validate_quantity`; I’ll "
                    "inspect it to verify the handling of zero, negative, and non-integer inputs before "
                    "reviewing the tests.",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                                "name": "read_file",
                            },
                            "id": "call_fc5f6bc78e0b5332b7866a98d5470b46",
                            "type": "function",
                        },
                    ],
                    "reasoning_items": [
                        {
                            "type": "reasoning",
                            "id": "rs_2985de4698f95e958cd7c7cd377a76a2",
                            "encrypted_content": "rsn_rru6BQHgUc/VWS5vRIAlHwIFeJSjZG0YBxqkNyCzhib06j32zRnkxAVgKyx5FlM5EQsGAF8AAQAVYXdzLWNyeXB0by1wdWJsaWMta2V5AERBblF2WjlWWElaWDBLWXR0RDE3djIxTzd2bWJVdjZSQlhOYW1sMmxjTHRrVGFuVU4vc09LRDJlUzFGMENIQzdOa0E9PQABABFhd3Mta21zLWhpZXJhcmNoeQAKYmVybS1zdG9yZQBcL/APOsU1Boid3k+fiCN6FGnqlouHrRGycfpWSQTyexPQO0PepR8jRQ3zVYQqZPsD7v2uGJDXTNoMQ/DNJEPAmyJkLXtVRDtxK+LIpvBebVGlBLXtrij6Z/oBFI8CAAAQABUJCnt2FvwOmW31K7V/wAUU+CK5nMfo/9o1zvOTBV0aDdylXF3v56EtlCEoo8vw6/////8AAAABAAAAAAAAAAAAAAABAAABPxjRAPZm81KRyzjmvtQM3zhh5qZSePYBGrVVjgFK2e9hY/yVx8S8S70YklxeYaWmFkKyinCLXqUH0IFnsyUaBQGpvnrVixo+QDyeu4wV6WP5hyp7j/GGWaAlDM3zF4JkOIqpXYU2XbHwr1/wRzimQl3BcBxBpygEBHteEuAWENg16dfo76pI4Q+PpGOKfwaSXz9Cj2vprUCxGcsa9IY6tq6pr7hYdhtS4Zg8tYGrSoGm6/9ljF51LZ1TSvfHnKmNFgpp86naM08zO1Jp+fDU37fKH8nazA0xUB23BWMWWu63pvQaquAtx1rj5PisEWwemlY/5v/Wo4d4Q9W+nrGzXf2o9kScWnDpfCfR7G72u19J0SXHcfrFwj5oVN8Ek1eYKY8Apv2lxZAf+ULokPmUGk/lXkz9hxa52txJ/3+bjy7cJ4S9HoUuBRHt3l7PpJJ0AGcwZQIxAP/SkcKkldJ07XgfZ+neD7X9h8kLD3hHG4axXwP8LPDmZ6HKrnXbOoSZLT6Nf4JddAIwEte72+56IOzdbE/k0Zvq/5ILQB5G+FhUdu0tHd6YDhHqPuq4nzXJJ/+0dFHRWzcw",
                            "summary": [],
                        },
                    ],
                },
            },
        ],
        "usage": {
            "completion_tokens": 114,
            "prompt_tokens": 312,
            "total_tokens": 426,
            "completion_tokens_details": {"reasoning_tokens": 36},
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "cache_creation_tokens": 0},
        },
        "billing": {"payer": "developer"},
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "service_tier": "default",
    },
)
