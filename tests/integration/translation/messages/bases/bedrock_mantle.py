from collections.abc import Mapping
from typing import Final

from integration.translation.case import TranslationTestCase
from integration.translation.messages.templates import MESSAGES_REQUEST, MESSAGES_RESPONSE
from integration.translation.templates.bedrock_mantle import (
    BEDROCK_MANTLE_OPENAI_RESPONSES_REQUEST,
    BEDROCK_MANTLE_OPENAI_RESPONSES_RESPONSE,
)
from pydantic import JsonValue

CLAUDE_HAIKU_4_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "bedrock_mantle/anthropic.claude-haiku-4-5",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "authorization": "Bearer synthetic-bedrock-mantle-key",
    },
    expected_provider_request={
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "model": "anthropic.claude-haiku-4-5",
    },
    mock_provider_response={
        "model": "claude-haiku-4-5-20251001",
        "id": "msg_bdrk_vup3t4zou7dn7wcgq7cpekk5ra22doolpkubeg53d4nsr3e3nytq",
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
        "model": "bedrock_mantle/anthropic.claude-haiku-4-5",
        "id": "msg_bdrk_vup3t4zou7dn7wcgq7cpekk5ra22doolpkubeg53d4nsr3e3nytq",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 17,
            "output_tokens": 5,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "service_tier": "standard",
        },
    },
)


CLAUDE_SONNET_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "bedrock_mantle/anthropic.claude-sonnet-5",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "authorization": "Bearer synthetic-bedrock-mantle-key",
    },
    expected_provider_request={
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "model": "anthropic.claude-sonnet-5",
    },
    mock_provider_response={
        "model": "claude-sonnet-5",
        "id": "msg_bdrk_6vuplyqolvblp5dquv5mwrkdqbysmuymlyndmzjg2btbtdnavceq",
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
        "model": "bedrock_mantle/anthropic.claude-sonnet-5",
        "id": "msg_bdrk_6vuplyqolvblp5dquv5mwrkdqbysmuymlyndmzjg2btbtdnavceq",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 21,
            "output_tokens": 6,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
        },
    },
)


CLAUDE_OPUS_5_5_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "bedrock_mantle/anthropic.claude-opus-5-5",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/anthropic/v1/messages",
    expected_provider_headers={
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "authorization": "Bearer synthetic-bedrock-mantle-key",
    },
    expected_provider_request={
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "You are a terse assistant."}],
        "model": "anthropic.claude-opus-5-5",
    },
    mock_provider_response={
        "model": "claude-opus-5-5",
        "id": "msg_bdrk_52cr74cucwfccadhge37ynypl3e7cgfisudqlyjv4xyrjpa6j7ta",
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
            "service_tier": "standard",
        },
    },
    expected_litellm_response={
        "model": "bedrock_mantle/anthropic.claude-opus-5-5",
        "id": "msg_bdrk_52cr74cucwfccadhge37ynypl3e7cgfisudqlyjv4xyrjpa6j7ta",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "container": None,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {
            "input_tokens": 23,
            "output_tokens": 6,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "output_tokens_details": {"thinking_tokens": 0},
            "service_tier": "standard",
        },
    },
)


GPT_5_4_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "bedrock_mantle/openai.gpt-5.4",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
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
    },
    mock_provider_response={
        "background": False,
        "billing": {"payer": "developer"},
        "completed_at": 1791240601,
        "created_at": 1791240601,
        "error": None,
        "frequency_penalty": 0.0,
        "id": "resp_mjtxgz2qvle22u6pbq4q54kmg2eeht2xgmplcgn4q43a4rxw6yia",
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
                "id": "msg_73ae0510d47b58ac894fc429f4e37f57",
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
        "id": "resp_mjtxgz2qvle22u6pbq4q54kmg2eeht2xgmplcgn4q43a4rxw6yia",
        "type": "message",
        "role": "assistant",
        "model": "bedrock_mantle/openai.gpt-5.4",
        "stop_sequence": None,
        "usage": {"input_tokens": 19, "output_tokens": 6},
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)


GPT_5_6_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "bedrock_mantle/openai.gpt-5.6-sol",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
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
    },
    mock_provider_response={
        "background": False,
        "billing": {"payer": "developer"},
        "completed_at": 1791240604,
        "created_at": 1791240603,
        "error": None,
        "frequency_penalty": 0.0,
        "id": "resp_7tsj6j6oi6de5rzq2lqrazvkayhfbtelrtb2x3sk7ozfu6xk2gfq",
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
                "id": "msg_049815c6c03253f9983f121c752f04b4",
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
        "id": "resp_7tsj6j6oi6de5rzq2lqrazvkayhfbtelrtb2x3sk7ozfu6xk2gfq",
        "type": "message",
        "role": "assistant",
        "model": "bedrock_mantle/openai.gpt-5.6-sol",
        "stop_sequence": None,
        "usage": {"input_tokens": 19, "output_tokens": 6},
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)


GPT_5_6_LUNA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={
        "model": "bedrock_mantle/openai.gpt-5.6-luna",
        "max_tokens": 64,
        "system": "You are a terse assistant.",
        "messages": [{"role": "user", "content": "Say hello."}],
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
    },
    mock_provider_response={
        "background": False,
        "billing": {"payer": "developer"},
        "completed_at": 1791240606,
        "created_at": 1791240606,
        "error": None,
        "frequency_penalty": 0.0,
        "id": "resp_euq6yo6tnjzhkrw56ri24yqzcsr6yyt5k4a75ppn7iclefp34gxa",
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
                "id": "msg_765b9dcbdc385d74b07a4b93cd748e6c",
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
        "id": "resp_euq6yo6tnjzhkrw56ri24yqzcsr6yyt5k4a75ppn7iclefp34gxa",
        "type": "message",
        "role": "assistant",
        "model": "bedrock_mantle/openai.gpt-5.6-luna",
        "stop_sequence": None,
        "usage": {"input_tokens": 19, "output_tokens": 6},
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "stop_details": None,
    },
)

BEDROCK_MANTLE_OPENAI_RESPONSES_MESSAGES_RESPONSE: Final[Mapping[str, JsonValue]] = {
    key: value for key, value in MESSAGES_RESPONSE.items() if key not in ("container", "diagnostics")
}


GPT_6_1_SOL_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "bedrock_mantle/openai.gpt-6.1-sol"},
    expected_provider_endpoint="/openai/v1/responses",
    expected_provider_headers={
        "authorization": "Bearer synthetic-bedrock-mantle-key",
        "content-type": "application/json",
    },
    expected_provider_request={
        **BEDROCK_MANTLE_OPENAI_RESPONSES_REQUEST,
        "instructions": MESSAGES_REQUEST["system"],
    },
    mock_provider_response={
        **BEDROCK_MANTLE_OPENAI_RESPONSES_RESPONSE,
        "completed_at": 1792027274,
        "created_at": 1792027272,
        "id": "resp_n4higs7hpzdjowbay2ujeesbgcxavsl4pbkzhbh7pjpimjjaciaa",
        "instructions": "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
        "Explain which file you need next, then call a read-only tool. Never change files. Use the messages "
        "client for this project review. Consider zero, negative and non-integer quantities before choosing "
        "the next file to inspect.",
        "output": [
            {
                "content": [],
                "encrypted_content": "rsn_rru6BQHgUc/VWS5vRIAlHwIFeB+Lw/sASAOaRTS5iOdUyDiwGqyLb6EJDqxBCdFOQvGfAF8AAQAVYXdzLWNyeXB0by1wdWJsaWMta2V5AERBaTJyR044dmNxUGt3TndhZkNmeTBpOEFUWlBHSGN4L2s3U1E0UGpyUEp0bUdlZVhXVDREVy9tS2FPdStQUFd6OGc9PQABABFhd3Mta21zLWhpZXJhcmNoeQAKYmVybS1zdG9yZQBcJ5O6OH7BzhGiCKEzUwANq0bcBtwE7aXPWBjYzgTyexPQO0PepR8jRQ3zVYRhZpm6K2TOk2lvrp72PZ5N5jG3KGRkWT2Y2jsL9pnkWwE5kN16+aY2OKOY9ZlzQukCAAAQACDHXld1blcGog/Y0exPb0n+8Vo0dT1X9pL401/Nv3/uQlT6BC0wwixOAufgfaY+ev////8AAAABAAAAAAAAAAAAAAABAAABrRdgMGsGAKubmJ7dCj1A9UQgYVFb1O6+NB+Aom8IGUy3Y6E3c63eWNkWXxN8APSjWFwq012VWZ9Q11DzJmGSwM8gtV25ckYezPfNn46VmSj2idcLs2TxTSxP89NAOfSMzWRnW6Dr3lSlx72uQKBrHWYPyAskoHTw4E6CfnFhXotCshAmT5dgImjniqdLFu4CWh2xpQcmYkFbxXakTBsK80TEI8fpJVjGYQbomlrZTwAb8bdyEp0/05xVeMVYC47F0tBY5zFOeZ0Nk0ps2PFteD1HQ5GEt7MRMb+VTGGSbq0wmjY7pvdC0f31cs+0GhEZ1N/3707cGjuMevEMn17YYxJ0oelPOatqdAeSSxBGkEKLmzos0tDcisqwWU/94DL2hmTJWacHamboEN74xZ+KwOjKEoyEr0HOL7n5+B9Oojl7MPdWgBnyGUwCsB3BH8t+h5MwMtMJY451PvO4hgl3qK2cN4WfgoYHRvWM2+rj2iJqYBXFPLo5yqWd3XIIimuTGltinK/0sr1uC/AMQ2Qa0qEB0J4Zxla83H7HCEBg4e8wde2qlXZCG9AxVpPQVd1LijTvn7sPIJgYsmSMwKYAZzBlAjBvywiP8Kq5WqQ+kknoQv+bn4OtwISL+sXhXE+9JK6hhjn02Xd9iSkC/h6IGxAc22UCMQCBBpc3UEvXQSZAxGDy1xUr3JoG4X3sj2Y4yB3SHvBVcxmAWoVo+QAUVMNulgqXbwU=",
                "id": "rs_187e1b0bc4995fc9b7fd2796a1889637",
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
                "id": "msg_56724f55f0ce5b10a7909c73300f34df",
                "phase": "commentary",
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
            {
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":120}',
                "call_id": "call_c70cf00600395be0bd35d0db3b1cfa42",
                "id": "fc_8bca96735a2c548c9f7777553cf8e5d0",
                "name": "read_file",
                "status": "completed",
                "type": "function_call",
            },
        ],
        "usage": {
            "input_tokens": 309,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 141,
            "output_tokens_details": {"reasoning_tokens": 76},
            "total_tokens": 450,
        },
    },
    expected_litellm_response={
        **BEDROCK_MANTLE_OPENAI_RESPONSES_MESSAGES_RESPONSE,
        "id": "resp_n4higs7hpzdjowbay2ujeesbgcxavsl4pbkzhbh7pjpimjjaciaa",
        "model": "bedrock_mantle/openai.gpt-6.1-sol",
        "usage": {"input_tokens": 309, "output_tokens": 141},
        "content": [
            {
                "type": "text",
                "text": "I’ll read `shop/validation.py` first to check how it handles zero, negative, and "
                "non-integer quantities.\n",
            },
            {
                "type": "tool_use",
                "id": "call_c70cf00600395be0bd35d0db3b1cfa42",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 120},
            },
        ],
    },
)

GPT_6_LUNA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "bedrock_mantle/openai.gpt-6-luna"},
    expected_provider_endpoint="/openai/v1/responses",
    expected_provider_headers={
        "authorization": "Bearer synthetic-bedrock-mantle-key",
        "content-type": "application/json",
    },
    expected_provider_request={
        **BEDROCK_MANTLE_OPENAI_RESPONSES_REQUEST,
        "model": "openai.gpt-6-luna",
        "instructions": MESSAGES_REQUEST["system"],
    },
    mock_provider_response={
        **BEDROCK_MANTLE_OPENAI_RESPONSES_RESPONSE,
        "completed_at": 1792027280,
        "created_at": 1792027279,
        "id": "resp_di3jhnf3mehuuht6smddkuygfeo7rzxdxmwvrc52aigdygbts5ca",
        "instructions": "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
        "Explain which file you need next, then call a read-only tool. Never change files. Use the messages "
        "client for this project review. Consider zero, negative and non-integer quantities before choosing "
        "the next file to inspect.",
        "model": "openai.gpt-6-luna",
        "output": [
            {
                "content": [],
                "encrypted_content": "rsn_rru6BQHgUc/VWS5vRIAlHwIFeJWGzSslRfiCKfPUzG9PfxThM9nB8cV14wc1iKvcbztDAF8AAQAVYXdzLWNyeXB0by1wdWJsaWMta2V5AERBb2ZiS1dnQ24waEpudTVSZGtlLyt3emJKY2VIUXlCeU9rWGd6enJDYU5ydXZYMEFmV1NwK01vVEUwZzhHWnpWb0E9PQABABFhd3Mta21zLWhpZXJhcmNoeQAKYmVybS1zdG9yZQBcJhapxvTAKRUzTqbKGUoTlXJqSZjSljHkimXkxwTyexPQO0PepR8jRQ3zVYQcb1ucdG8HiPjq86SsgccbqbIEmYllOUwDws5V0DtxxRnqoVEhkF3h+EG4I5uuB2kCAAAQABDKdaE231TqbLUInCwHGBD9+U9x7lXzuoVQNhj0KATZ6U5+Ki+ZDt6KLjrNb/LM2v////8AAAABAAAAAAAAAAAAAAABAAABdglH8KRxssIDHD4kcMUX0rX0yogE2V0AOj4vmUzgN+tpKEUhA0/BNphpi42nYRZK61EgLzZk42sWn6sJhR5/WKHHn/7N7FppZkt6kFCDTyhRfPimhQT0S7NiACE4511FHXE/WbRGd2/3uuLOx6PfMFlxuKgJHYQuMEttkYRZz1lwzMrtQTyTtghKKz0QaNtp7umhKHlaFVetqVsKqLD7JpKyBxjbe5VojoKmMQEY3dExNYgvCnOBPb8uLhZCcBemLq1ZQzinbMRyRO1bhQmfGsx4GgFbuA4cyJlyzHe+Af4VqcRM94bELBMXiU64UveyqRkWBdocsPfPFpVWa+Hnvl37fgn4jBTnhG1xqNuoVCXX2nV/GAMYHdPwsKweIhxg9I8EcrsXOcTWv26hwDRpmdoHHtJkSqKSbudxEs1F5kr5ELytEFS2OWJwwsMvDfBAlC4boaYtxUBG5sREazNwDNRkhFWHb8MwsXBpOqwOBtu1L2n953WUP2I2tHE6WcZ3DpoD2dP3pwBnMGUCMQDonP31z1HPHjm2mGf7SINbrdFZ5xl5ztMBj3g8hJmIpdWtN/W5dJkc7kfNSErvVrQCMDGAL4pNuVt19O4BfjAmLKnNyz5x/z1FPLGJDz2j+9b0Y0FYBIqafypg88wJ6XSc0Q==",
                "id": "rs_8de42923e05d5521b0dbd93c1adcea91",
                "summary": [],
                "type": "reasoning",
            },
            {
                "content": [
                    {
                        "annotations": [],
                        "logprobs": [],
                        "text": "I’ll read `shop/validation.py` first to check the behavior for zero, "
                        "negative, and non-integer quantities.",
                        "type": "output_text",
                    },
                ],
                "id": "msg_6a34dc1447f85d8c8a2f9010dd4bee2a",
                "phase": "commentary",
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
            {
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":120}',
                "call_id": "call_316578feac6a57098276bb1f2671b00b",
                "id": "fc_c76fa3372e0257699637cc6777775026",
                "name": "read_file",
                "status": "completed",
                "type": "function_call",
            },
        ],
        "usage": {
            "input_tokens": 309,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 119,
            "output_tokens_details": {"reasoning_tokens": 54},
            "total_tokens": 428,
        },
    },
    expected_litellm_response={
        **BEDROCK_MANTLE_OPENAI_RESPONSES_MESSAGES_RESPONSE,
        "id": "resp_di3jhnf3mehuuht6smddkuygfeo7rzxdxmwvrc52aigdygbts5ca",
        "model": "bedrock_mantle/openai.gpt-6-luna",
        "usage": {"input_tokens": 309, "output_tokens": 119},
        "content": [
            {
                "type": "text",
                "text": "I’ll read `shop/validation.py` first to check the behavior for zero, negative, and "
                "non-integer quantities.",
            },
            {
                "type": "tool_use",
                "id": "call_316578feac6a57098276bb1f2671b00b",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 120},
            },
        ],
    },
)

GPT_6_ASTRA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "bedrock_mantle/openai.gpt-6-astra"},
    expected_provider_endpoint="/openai/v1/responses",
    expected_provider_headers={
        "authorization": "Bearer synthetic-bedrock-mantle-key",
        "content-type": "application/json",
    },
    expected_provider_request={
        **BEDROCK_MANTLE_OPENAI_RESPONSES_REQUEST,
        "model": "openai.gpt-6-astra",
        "instructions": MESSAGES_REQUEST["system"],
    },
    mock_provider_response={
        **BEDROCK_MANTLE_OPENAI_RESPONSES_RESPONSE,
        "completed_at": 1792027287,
        "created_at": 1792027285,
        "id": "resp_eznficwibpblmnocp4vuvti6d5bxboyxws4vqf5bk2lboyvvbx7q",
        "instructions": "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
        "Explain which file you need next, then call a read-only tool. Never change files. Use the messages "
        "client for this project review. Consider zero, negative and non-integer quantities before choosing "
        "the next file to inspect.",
        "model": "openai.gpt-6-astra",
        "output": [
            {
                "content": [],
                "encrypted_content": "rsn_rru6BQHgUc/VWS5vRIAlHwIFeK6rtxWPF9TjmgBuzPaPZhcqwiqUtYeqNCEyD/fgEZAfAF8AAQAVYXdzLWNyeXB0by1wdWJsaWMta2V5AERBaUE5eUozT3MzTE1rRFhLWGRuWUFYRjdZU2Q3OStjT3V5LzBIRUNnZXBUWDloQXZncVppTC9qVUc3Y2h3dlQ5QVE9PQABABFhd3Mta21zLWhpZXJhcmNoeQAKYmVybS1zdG9yZQBcuTxMV80kCzRfLH6n87syGDT8XBcjqEaE8tDuSQTyexPQO0PepR8jRQ3zVYTdphRKCdq3CJr491CVE1gdVwCTcbqsK/1k7shfoHg93E2fCZXYKq0tnNLGBKhnl90CAAAQAHNHMusCy4BS4gUSPEf8T3DwPC4nfHsYIw/8I2M8N4Z4dxH0a/6ULM8Zaf5qgZInLv////8AAAABAAAAAAAAAAAAAAABAAABmFno3HYlCh8skcb8d7DvpK045XzP/rPTF9J1TXQEj+Oam1ybdDFFN5bbUrMUhvhaAKoUdhp0TPWFBQnQD7rM1QSBtZY60vbMXdscVeDkmAPh7MsZ6OLAY1olCWPsid7OqATdHKm+S6UfnqnvNLJQ0Hp12YYilLOhW5+BLnmqdm04wkU2kvH4G4vxgRK0hFGMGaoKlCIKRvAmwre2DGXdWx85E9jsq2j1ta1XT6/2eCr7zyXn8+tRI69tNwFVRTcaHu8DvSS1LBzdO7ZwNwUr6PbksO44r/sXeNstFU4u3ZQnUhWz8JNym0ydaia57zXQG3/pl8rJaA3IshV+Gm+Ie7N3qySlcIEonqN+cYhQN669G4oBuvCKARc2LFdRB1OXhErskZ8c2c3NA9Xt/dXBbclKY9i/I3Usp46t46mALqz7rA2ft/cGLtX3gNoPPEKL2+i1ZDa0EB1Y7agcESa5pYLKgajhUDbaxupsVQ93wZiCcDTr7SlEfMrBkNFGijc3kpdn1avRFEzVC3H/Ys1cPPYJ93hOUYO42b/RibFpBlkLnKtNf//v81kAZzBlAjAdmUn/WQ5cOGyMLQF2CQxoUrAO9HhKULmwznaURUbjukpBRw8v6PwflSA+mmU8iv0CMQCASBvNjVVmEd4/s4jH27TFub4sikcdi7+LR5HeitoRhyEJIfB41mRciDxhY80hp6s=",
                "id": "rs_e6237911ea3e5775acee304296484fab",
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
                "id": "msg_f652378c5e52535290a315cff41a7694",
                "phase": "commentary",
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
            {
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                "call_id": "call_3986bff59655522d95c62139330145fd",
                "id": "fc_aec0dbb1671153029ab817c03715462f",
                "name": "read_file",
                "status": "completed",
                "type": "function_call",
            },
        ],
        "usage": {
            "input_tokens": 309,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 138,
            "output_tokens_details": {"reasoning_tokens": 73},
            "total_tokens": 447,
        },
    },
    expected_litellm_response={
        **BEDROCK_MANTLE_OPENAI_RESPONSES_MESSAGES_RESPONSE,
        "id": "resp_eznficwibpblmnocp4vuvti6d5bxboyxws4vqf5bk2lboyvvbx7q",
        "model": "bedrock_mantle/openai.gpt-6-astra",
        "usage": {"input_tokens": 309, "output_tokens": 138},
        "content": [
            {
                "type": "text",
                "text": "I’ll read `shop/validation.py` first to check how it handles zero, negative, and "
                "non-integer quantities.\n",
            },
            {
                "type": "tool_use",
                "id": "call_3986bff59655522d95c62139330145fd",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 200},
            },
        ],
    },
)

GPT_5_6_TERRA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/messages",
    litellm_request={**MESSAGES_REQUEST, "model": "bedrock_mantle/openai.gpt-5.6-terra"},
    expected_provider_endpoint="/openai/v1/responses",
    expected_provider_headers={
        "authorization": "Bearer synthetic-bedrock-mantle-key",
        "content-type": "application/json",
    },
    expected_provider_request={
        **BEDROCK_MANTLE_OPENAI_RESPONSES_REQUEST,
        "model": "openai.gpt-5.6-terra",
        "instructions": MESSAGES_REQUEST["system"],
    },
    mock_provider_response={
        **BEDROCK_MANTLE_OPENAI_RESPONSES_RESPONSE,
        "completed_at": 1792027295,
        "created_at": 1792027294,
        "id": "resp_fqwklh5v6lnfftfaq6ddfbclshh3vkgmth75hufjjxjrhn726b2a",
        "instructions": "You help maintain a synthetic Python project. Read the relevant files before suggesting a fix. "
        "Explain which file you need next, then call a read-only tool. Never change files. Use the messages "
        "client for this project review. Consider zero, negative and non-integer quantities before choosing "
        "the next file to inspect.",
        "model": "openai.gpt-5.6-terra",
        "output": [
            {
                "content": [],
                "encrypted_content": "rsn_rru6BQHgUc/VWS5vRIAlHwIFeCIDnqsrWiO+hx6u0oBst+OnJIA0RIm1eTWC5ORgkFFNAF8AAQAVYXdzLWNyeXB0by1wdWJsaWMta2V5AERBMEZqYXRRMkhMSEw1clFlcythQm0wc1Bid2hBWHJhN3ZZOFdGenRwOXlsd0NqSkxBOGxWd2loU2dleHhmTFN0Snc9PQABABFhd3Mta21zLWhpZXJhcmNoeQAKYmVybS1zdG9yZQBcWxvahbwUWb5zEZ4mn/Ri1V5Z4KCuYAzvlyiI2ATyexPQO0PepR8jRQ3zVYTwOcBo7nb2ZZfI77C2nwtaEjD275c1LF19dmrCUZ7VPKIYQCMlismkx7f2X1LE4xUCAAAQAPc8v4U/p6fjgu2ZpZgPQUfpPsrIs2TWrckQcNPOhJmhKYDB5gthEp93h1Hf2+ad3P////8AAAABAAAAAAAAAAAAAAABAAABHw7y8mi4YLItPYN81mZNbuQhK0FTG04aQuvQ0966UQxFIMfbeJe+qtyO7uHmyKoz58NNb0ud67KhOPoip6wfSxVcw2RTBCAljj+gkLciZdZlGDiLyutdaHfCRh6StNZDtXG0xPgbn0mN93xnE0hDa05mjcNsNckDaS2KkTMzoXF0tqWoi+Vl0fMSK3CF/49QKyGXYsDeOPB0b8XghpfF1PvMqMI/Ua9Sd98l999mYu71TaBKvgRS9yLWBXg9YcmfZx2hulOUkK2a9aCQ5aN661B0MTdykIm+CokaKogZCv6e/MXncHPPtBeqtuQtsggedinoW/qD09HU+rLUysC+vbHT65YqoVL9wW42XZa79V5QBFYIQL787eVy54qbPxrryWKuTOsN9xRFqaSOv1L10ABnMGUCMFnJHyhVr8ujzWTwnSQCuS+zKrvPgXQfEePxJOR/KFHa7CUXZ8kBuEoW0kVxx3biLgIxAK+8PazVOuepKMy5cuKKsJChOnvQ1Npz7Gpmrx0CXO7X2sIN1WJTlhdNHEclLuwoSQ==",
                "id": "rs_a4e6e9898f225c88b4d948b6488bfacc",
                "summary": [],
                "type": "reasoning",
            },
            {
                "content": [
                    {
                        "annotations": [],
                        "logprobs": [],
                        "text": "I’ll read `shop/validation.py` first to verify the current handling of zero, "
                        "negative, and non-integer quantities; after that, `tests/test_validation.py` is "
                        "the next file needed to confirm the expected coverage.",
                        "type": "output_text",
                    },
                ],
                "id": "msg_a6c79276e3bd594f99585696b01148d2",
                "phase": "commentary",
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
            {
                "arguments": '{"path":"shop/validation.py","start_line":1,"end_line":200}',
                "call_id": "call_ea239067d1ea5b4f8b96b9240bca0523",
                "id": "fc_fb9f552a9d7b5f80917959284de3e9aa",
                "name": "read_file",
                "status": "completed",
                "type": "function_call",
            },
        ],
        "usage": {
            "input_tokens": 309,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens": 109,
            "output_tokens_details": {"reasoning_tokens": 23},
            "total_tokens": 418,
        },
    },
    expected_litellm_response={
        **BEDROCK_MANTLE_OPENAI_RESPONSES_MESSAGES_RESPONSE,
        "id": "resp_fqwklh5v6lnfftfaq6ddfbclshh3vkgmth75hufjjxjrhn726b2a",
        "model": "bedrock_mantle/openai.gpt-5.6-terra",
        "usage": {"input_tokens": 309, "output_tokens": 109},
        "content": [
            {
                "type": "text",
                "text": "I’ll read `shop/validation.py` first to verify the current handling of zero, "
                "negative, and non-integer quantities; after that, `tests/test_validation.py` is the next "
                "file needed to confirm the expected coverage.",
            },
            {
                "type": "tool_use",
                "id": "call_ea239067d1ea5b4f8b96b9240bca0523",
                "name": "read_file",
                "input": {"path": "shop/validation.py", "start_line": 1, "end_line": 200},
            },
        ],
    },
)
