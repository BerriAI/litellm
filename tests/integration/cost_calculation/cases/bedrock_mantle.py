"""Bedrock Mantle cost tracking cases for openai.gpt-6-sol, a mode responses row that LiteLLM serves on /openai/v1/responses for both /v1/responses and chat clients."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Final

from integration.cost_calculation.cases.openai import with_usage
from integration.cost_calculation.cases.anthropic import (
    CLAUDE_OPUS_5_5_CACHE_READ,
    CLAUDE_OPUS_5_5_CACHE_WRITE_1H,
    CLAUDE_OPUS_5_5_CACHE_WRITE_5M,
    CLAUDE_OPUS_5_5_INPUT_TEXT,
    CLAUDE_OPUS_5_5_MESSAGES_INPUT_TEXT,
    CLAUDE_OPUS_5_5_MESSAGES_STREAM,
    CLAUDE_OPUS_5_5_STREAM,
    CLAUDE_OPUS_5_5_THINKING,
    CLAUDE_OPUS_5_5_TOOL_CALL,
)
from integration.cost_calculation.bridge_cost_parity.case import BridgeCostParityTestCase, bridge_pairs
from integration.cost_calculation.cost_tracking_case import (
    CostTrackingTestCase,
    Deployment,
    ExactExpected,
    JsonResponse,
    SseResponse,
)
from integration.cost_calculation.stream_parity.case import StreamParityTestCase, sse_frames
from pydantic import JsonValue


BEDROCK_MANTLE_OPENAI_GPT_6_SOL_RESPONSES_JSON: Final = CostTrackingTestCase(
    name="bedrock_mantle-openai.gpt-6-sol-responses_json",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="bedrock_mantle/openai.gpt-6-sol",
    endpoint="/v1/responses",
    request={"model": "$MODEL", "input": "cost tracking fixture", "max_output_tokens": 412, "stream": False},
    response=JsonResponse(
        content_type="application/json",
        body={
            "background": False,
            "billing": {"payer": "developer"},
            "completed_at": 1791861771,
            "created_at": 1791861771,
            "error": None,
            "frequency_penalty": 0.0,
            "id": "resp_$REQUEST_ID",
            "incomplete_details": None,
            "instructions": None,
            "max_output_tokens": 64,
            "max_tool_calls": None,
            "metadata": {},
            "model": "openai.gpt-6-sol",
            "object": "response",
            "output": [
                {
                    "content": [{"annotations": [], "logprobs": [], "text": "Hello!", "type": "output_text"}],
                    "id": "msg_$REQUEST_ID",
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
                "input_tokens": 1840,
                "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                "output_tokens": 412,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 2252,
            },
            "user": None,
            "moderation": None,
        },
    ),
    expected=ExactExpected(
        spend=0.00858, input_cost=0.004048, output_cost=0.004532, prompt_tokens=1840, completion_tokens=412
    ),
)

BEDROCK_MANTLE_OPENAI_GPT_6_SOL_RESPONSES_STREAM: Final = CostTrackingTestCase(
    name="bedrock_mantle-openai.gpt-6-sol-responses_stream",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="bedrock_mantle/openai.gpt-6-sol",
    endpoint="/v1/responses",
    request={"model": "$MODEL", "input": "cost tracking fixture", "max_output_tokens": 412, "stream": True},
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "response": {
                    "background": False,
                    "completed_at": None,
                    "created_at": 1791861771,
                    "error": None,
                    "frequency_penalty": 0.0,
                    "id": "resp_$REQUEST_ID",
                    "incomplete_details": None,
                    "instructions": None,
                    "max_output_tokens": 64,
                    "max_tool_calls": None,
                    "metadata": {},
                    "model": "openai.gpt-6-sol",
                    "object": "response",
                    "output": [],
                    "parallel_tool_calls": True,
                    "presence_penalty": 0.0,
                    "previous_response_id": None,
                    "prompt_cache_key": None,
                    "prompt_cache_retention": "in_memory",
                    "reasoning": {"summary": None, "effort": "medium", "context": "all_turns"},
                    "safety_identifier": None,
                    "service_tier": "auto",
                    "status": "in_progress",
                    "store": True,
                    "temperature": 1.0,
                    "text": {"format": {"type": "text"}, "verbosity": "medium"},
                    "tool_choice": "auto",
                    "tools": [],
                    "top_logprobs": 0,
                    "top_p": 0.98,
                    "truncation": "disabled",
                    "usage": None,
                    "user": None,
                    "moderation": None,
                },
                "sequence_number": 0,
                "type": "response.created",
            },
            {
                "response": {
                    "background": False,
                    "completed_at": None,
                    "created_at": 1791861771,
                    "error": None,
                    "frequency_penalty": 0.0,
                    "id": "resp_$REQUEST_ID",
                    "incomplete_details": None,
                    "instructions": None,
                    "max_output_tokens": 64,
                    "max_tool_calls": None,
                    "metadata": {},
                    "model": "openai.gpt-6-sol",
                    "object": "response",
                    "output": [],
                    "parallel_tool_calls": True,
                    "presence_penalty": 0.0,
                    "previous_response_id": None,
                    "prompt_cache_key": None,
                    "prompt_cache_retention": "in_memory",
                    "reasoning": {"summary": None, "effort": "medium", "context": "all_turns"},
                    "safety_identifier": None,
                    "service_tier": "auto",
                    "status": "in_progress",
                    "store": True,
                    "temperature": 1.0,
                    "text": {"format": {"type": "text"}, "verbosity": "medium"},
                    "tool_choice": "auto",
                    "tools": [],
                    "top_logprobs": 0,
                    "top_p": 0.98,
                    "truncation": "disabled",
                    "usage": None,
                    "user": None,
                    "moderation": None,
                },
                "sequence_number": 1,
                "type": "response.in_progress",
            },
            {
                "item": {
                    "content": [],
                    "id": "msg_$REQUEST_ID",
                    "phase": "final_answer",
                    "role": "assistant",
                    "status": "in_progress",
                    "type": "message",
                },
                "output_index": 0,
                "sequence_number": 2,
                "type": "response.output_item.added",
            },
            {
                "content_index": 0,
                "item_id": "msg_$REQUEST_ID",
                "output_index": 0,
                "part": {"annotations": [], "logprobs": [], "text": "", "type": "output_text"},
                "sequence_number": 3,
                "type": "response.content_part.added",
            },
            {
                "content_index": 0,
                "delta": "Hello",
                "item_id": "msg_$REQUEST_ID",
                "logprobs": [],
                "obfuscation": "abcdefghijklmnopqr",
                "output_index": 0,
                "sequence_number": 4,
                "type": "response.output_text.delta",
            },
            {
                "content_index": 0,
                "delta": "!",
                "item_id": "msg_$REQUEST_ID",
                "logprobs": [],
                "obfuscation": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                "output_index": 0,
                "sequence_number": 5,
                "type": "response.output_text.delta",
            },
            {
                "content_index": 0,
                "item_id": "msg_$REQUEST_ID",
                "logprobs": [],
                "output_index": 0,
                "sequence_number": 6,
                "text": "Hello!",
                "type": "response.output_text.done",
            },
            {
                "content_index": 0,
                "item_id": "msg_$REQUEST_ID",
                "output_index": 0,
                "part": {"annotations": [], "logprobs": [], "text": "Hello!", "type": "output_text"},
                "sequence_number": 7,
                "type": "response.content_part.done",
            },
            {
                "item": {
                    "content": [{"annotations": [], "logprobs": [], "text": "Hello!", "type": "output_text"}],
                    "id": "msg_$REQUEST_ID",
                    "phase": "final_answer",
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                },
                "output_index": 0,
                "sequence_number": 8,
                "type": "response.output_item.done",
            },
            {
                "response": {
                    "background": False,
                    "completed_at": 1791861772,
                    "created_at": 1791861771,
                    "error": None,
                    "frequency_penalty": 0.0,
                    "id": "resp_$REQUEST_ID",
                    "incomplete_details": None,
                    "instructions": None,
                    "max_output_tokens": 64,
                    "max_tool_calls": None,
                    "metadata": {},
                    "model": "openai.gpt-6-sol",
                    "object": "response",
                    "output": [
                        {
                            "content": [{"annotations": [], "logprobs": [], "text": "Hello!", "type": "output_text"}],
                            "id": "msg_$REQUEST_ID",
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
                        "input_tokens": 1840,
                        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                        "output_tokens": 412,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 2252,
                    },
                    "user": None,
                    "moderation": None,
                },
                "sequence_number": 9,
                "type": "response.completed",
            },
            done=False,
        ),
    ),
    expected=ExactExpected(
        spend=0.00858, input_cost=0.004048, output_cost=0.004532, prompt_tokens=1840, completion_tokens=412
    ),
)

BEDROCK_MANTLE_OPENAI_GPT_6_SOL_RESPONSES_REASONING: Final = CostTrackingTestCase(
    name="bedrock_mantle-openai.gpt-6-sol-responses_reasoning",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="bedrock_mantle/openai.gpt-6-sol",
    endpoint="/v1/responses",
    request={
        "model": "$MODEL",
        "input": "cost tracking fixture",
        "max_output_tokens": 412,
        "stream": False,
        "reasoning": {"effort": "high"},
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "background": False,
            "billing": {"payer": "developer"},
            "completed_at": 1791861773,
            "created_at": 1791861772,
            "error": None,
            "frequency_penalty": 0.0,
            "id": "resp_$REQUEST_ID",
            "incomplete_details": None,
            "instructions": None,
            "max_output_tokens": 2048,
            "max_tool_calls": None,
            "metadata": {},
            "model": "openai.gpt-6-sol",
            "object": "response",
            "output": [
                {
                    "content": [],
                    "encrypted_content": "rsn_rru6BQHgUc/VWS5vRIAlHwIFeG7RzPj6s6ycKY/RZlUM5dqF0VNyuAaJUtQ/+O5IKuX/AF8AAQAVYXdzLWNyeXB0by1wdWJsaWMta2V5AERBaFdiZnh2QVRaZFNMS0ZCcTJVNFU1WEs0ZHpxQWpmQjJWQ3lrSW05TU15NWtpdERENHdVWC9uMlk1Ukk4OWpLUGc9PQABABFhd3Mta21zLWhpZXJhcmNoeQAKYmVybS1zdG9yZQBcXPp4b5qq1rMOAJCEMPOm1n1nzGavCPmKLYo+FgTyexPQO0PepR8jRQ3zVYSaLucS0FaGd1vcWrKrWKX8eIKD7wfHZpbMRbHMetv9w7xqCa4AE7hEPeWZthEccJMCAAAQACD+kEmiADTYVTf4H4U4QPblaMH/uE6BZviKmxyR4w3Xmwt5lcAEk0qVMG8al0lAk/////8AAAABAAAAAAAAAAAAAAABAAAA79DoUN9BMZV9CqY2mV0QW7ClRkkz38xKlz6LsDiZhV7/t7HA5FtCgxQNoW7vsVlddML725aymJQaphYk5S6FlvHrCiTlXBN+rVIy4HladKMznUZ/ObJd5wd/PdnQPur169UkSOlMa7UrZ7LP3spUG+mX7VS/yIoBKEDj7gK0MEblJ/AOJUtMKKcPh+AbuTXhLEMLl5bbsIozmRPj+w/kN3btLUIeWXijbzeoaO4//v8ia9wXUS4QUTduf9IJm7RgGy2qfyKQf4Ieiw2ZXEsFL5uuCR5P2IkFZ+gSktZtEV4S5rrvmpVmCuUdpmlYsQ/bLAO/7LgpKYHfx/v//f0seABnMGUCMCra6yiTJwzvp7p/t98exwMQnyJV8XMw6kCkn1LxNZbdBAL8z22Cu6BTkbAqgFS/DAIxAPwlq5ScUMLeMMuY5gotOj8u5jmQGG6RL7JxaUQCu/cNsZIa0oDUK5GlZkHGhkI54w==",
                    "id": "rs_$REQUEST_ID",
                    "summary": [],
                    "type": "reasoning",
                },
                {
                    "content": [{"annotations": [], "logprobs": [], "text": "757", "type": "output_text"}],
                    "id": "msg_$REQUEST_ID",
                    "phase": "final_answer",
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                },
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
                "input_tokens": 1840,
                "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                "output_tokens": 412,
                "output_tokens_details": {"reasoning_tokens": 300},
                "total_tokens": 2252,
            },
            "user": None,
            "moderation": None,
        },
    ),
    expected=ExactExpected(
        spend=0.00858, input_cost=0.004048, output_cost=0.004532, prompt_tokens=1840, completion_tokens=412
    ),
)

BEDROCK_MANTLE_OPENAI_GPT_6_SOL_CHAT_JSON: Final = CostTrackingTestCase(
    name="bedrock_mantle-openai.gpt-6-sol-chat_json",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="bedrock_mantle/openai.gpt-6-sol",
    endpoint="/v1/chat/completions",
    request={"model": "$MODEL", "messages": [{"role": "user", "content": "cost tracking fixture"}], "max_tokens": 412},
    response=JsonResponse(
        content_type="application/json",
        body={
            "background": False,
            "billing": {"payer": "developer"},
            "completed_at": 1791861772,
            "created_at": 1791861772,
            "error": None,
            "frequency_penalty": 0.0,
            "id": "resp_$REQUEST_ID",
            "incomplete_details": None,
            "instructions": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
            "max_output_tokens": 64,
            "max_tool_calls": None,
            "metadata": {},
            "model": "openai.gpt-6-sol",
            "object": "response",
            "output": [
                {
                    "content": [{"annotations": [], "logprobs": [], "text": "Hello.", "type": "output_text"}],
                    "id": "msg_$REQUEST_ID",
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
                "input_tokens": 1840,
                "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                "output_tokens": 412,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 2252,
            },
            "user": None,
            "moderation": None,
        },
    ),
    expected=ExactExpected(
        spend=0.00858, input_cost=0.004048, output_cost=0.004532, prompt_tokens=1840, completion_tokens=412
    ),
)

BEDROCK_MANTLE_OPENAI_GPT_6_SOL_RESPONSES_CACHE_READ: Final = with_usage(
    BEDROCK_MANTLE_OPENAI_GPT_6_SOL_RESPONSES_JSON,
    "bedrock_mantle-openai.gpt-6-sol-responses_cache_read",
    {
        "input_tokens": 1840,
        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 640},
        "output_tokens": 412,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 2252,
    },
    ExactExpected(
        spend=0.0073128, input_cost=0.0027808, output_cost=0.004532, prompt_tokens=1840, completion_tokens=412
    ),
)

BEDROCK_MANTLE_OPENAI_GPT_6_SOL_TIERED_INPUT_ABOVE_272K: Final = with_usage(
    BEDROCK_MANTLE_OPENAI_GPT_6_SOL_RESPONSES_JSON,
    "bedrock_mantle-openai.gpt-6-sol-tiered_input_above_272k",
    {
        "input_tokens": 300000,
        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
        "output_tokens": 412,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 300412,
    },
    ExactExpected(spend=1.326798, input_cost=1.32, output_cost=0.006798, prompt_tokens=300000, completion_tokens=412),
)

BEDROCK_MANTLE_OPENAI_GPT_6_SOL_TIERED_CACHE_READ_ABOVE_272K: Final = with_usage(
    BEDROCK_MANTLE_OPENAI_GPT_6_SOL_RESPONSES_JSON,
    "bedrock_mantle-openai.gpt-6-sol-tiered_cache_read_above_272k",
    {
        "input_tokens": 300000,
        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 200000},
        "output_tokens": 412,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 300412,
    },
    ExactExpected(spend=0.534798, input_cost=0.528, output_cost=0.006798, prompt_tokens=300000, completion_tokens=412),
)

BEDROCK_MANTLE_OPENAI_GPT_6_SOL_TIERED_AT_272K_BOUNDARY: Final = with_usage(
    BEDROCK_MANTLE_OPENAI_GPT_6_SOL_RESPONSES_JSON,
    "bedrock_mantle-openai.gpt-6-sol-tiered_at_272k_boundary",
    {
        "input_tokens": 272000,
        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
        "output_tokens": 412,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 272412,
    },
    ExactExpected(spend=0.602932, input_cost=0.5984, output_cost=0.004532, prompt_tokens=272000, completion_tokens=412),
)

_MANTLE_GPT_RESPONSES_REQUEST: Final = {
    "model": "$MODEL",
    "input": "cost tracking fixture",
    "max_output_tokens": 64,
    "stream": False,
}
_MANTLE_GPT_REASONING_REQUEST: Final = {
    **_MANTLE_GPT_RESPONSES_REQUEST,
    "max_output_tokens": 128,
    "reasoning": {"effort": "medium"},
}
_MANTLE_GPT_STREAM_REQUEST: Final = {**_MANTLE_GPT_RESPONSES_REQUEST, "stream": True}
_MANTLE_GPT_CHAT_REQUEST: Final = {
    "model": "$MODEL",
    "messages": [{"role": "user", "content": "cost tracking fixture"}],
    "max_completion_tokens": 64,
}
_MANTLE_GPT_TOOL_REQUEST: Final = {
    **_MANTLE_GPT_RESPONSES_REQUEST,
    "tools": [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ],
    "tool_choice": "required",
}
_MANTLE_GPT_CACHE_WRITE_REQUEST: Final = {
    **_MANTLE_GPT_RESPONSES_REQUEST,
    "input": "cacheable pricing-harness prefix " * 1000,
    "prompt_cache_key": "cost-cases-20261017",
    "max_output_tokens": 64,
}


def _mantle_gpt_usage(
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> dict[str, JsonValue]:
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cache_write_tokens": cache_write_tokens, "cached_tokens": cached_tokens},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": input_tokens + output_tokens,
    }


def _mantle_gpt_json_case(
    base: CostTrackingTestCase,
    provider_model: str,
    shape: str,
    usage: Mapping[str, JsonValue],
    expected: ExactExpected,
    request: Mapping[str, JsonValue],
) -> CostTrackingTestCase:
    case: Final = with_usage(base, f"bedrock_mantle-{provider_model}-{shape}", usage, expected)
    if not isinstance(case.response, JsonResponse):
        raise TypeError(f"{case.name}: expected a JSON response fixture")
    response: Final = case.response.model_copy(update={"body": {**case.response.body, "model": provider_model}})
    return case.model_copy(
        update={"model": f"bedrock_mantle/{provider_model}", "request": dict(request), "response": response}
    )


def _mantle_gpt_stream_case(
    provider_model: str, usage: Mapping[str, JsonValue], expected: ExactExpected
) -> CostTrackingTestCase:
    base: Final = BEDROCK_MANTLE_OPENAI_GPT_6_SOL_RESPONSES_STREAM
    if not isinstance(base.response, SseResponse):
        raise TypeError(f"{base.name}: expected an SSE response fixture")
    old_usage: Final = json.dumps(_mantle_gpt_usage(1840, 412, 0))
    new_usage: Final = json.dumps(dict(usage))
    frames: Final = tuple(
        frame.replace("openai.gpt-6-sol", provider_model).replace(old_usage, new_usage)
        for frame in base.response.frames
    )
    response: Final = base.response.model_copy(update={"frames": frames})
    return base.model_copy(
        update={
            "name": f"bedrock_mantle-{provider_model}-responses_stream",
            "model": f"bedrock_mantle/{provider_model}",
            "request": dict(_MANTLE_GPT_STREAM_REQUEST),
            "response": response,
            "expected": expected,
        }
    )


def _mantle_gpt_cases(
    provider_model: str,
    response_reasoning_tokens: int,
    response_expected: ExactExpected,
    reasoning_usage: Mapping[str, JsonValue],
    reasoning_expected: ExactExpected,
    cache_reasoning_tokens: int,
    cache_expected: ExactExpected,
    tiered_input_expected: ExactExpected,
    tiered_cache_expected: ExactExpected,
    tiered_boundary_expected: ExactExpected,
) -> tuple[CostTrackingTestCase, ...]:
    response_usage: Final = _mantle_gpt_usage(9, 64, response_reasoning_tokens)
    response_case: Final = _mantle_gpt_json_case(
        BEDROCK_MANTLE_OPENAI_GPT_6_SOL_RESPONSES_JSON,
        provider_model,
        "responses_json",
        response_usage,
        response_expected,
        _MANTLE_GPT_RESPONSES_REQUEST,
    )
    stream_case: Final = _mantle_gpt_stream_case(provider_model, response_usage, response_expected)
    reasoning_case: Final = _mantle_gpt_json_case(
        BEDROCK_MANTLE_OPENAI_GPT_6_SOL_RESPONSES_REASONING,
        provider_model,
        "responses_reasoning",
        reasoning_usage,
        reasoning_expected,
        _MANTLE_GPT_REASONING_REQUEST,
    )
    chat_case: Final = _mantle_gpt_json_case(
        BEDROCK_MANTLE_OPENAI_GPT_6_SOL_CHAT_JSON,
        provider_model,
        "chat_json",
        response_usage,
        response_expected,
        _MANTLE_GPT_CHAT_REQUEST,
    )
    chat_stream_case: Final = chat_case.model_copy(
        update={
            "name": f"bedrock_mantle-{provider_model}-chat_stream",
            "request": {**chat_case.request, "stream": True},
            "response": stream_case.response,
        }
    )
    messages_pairs: Final = bridge_pairs((response_case, stream_case), ("/v1/messages",))
    messages_json: Final = messages_pairs[0].bridged.model_copy(
        update={"name": f"bedrock_mantle-{provider_model}-messages_bridged_json"}
    )
    messages_stream: Final = messages_pairs[1].bridged.model_copy(
        update={"name": f"bedrock_mantle-{provider_model}-messages_bridged_stream"}
    )
    cache_case: Final = _mantle_gpt_json_case(
        BEDROCK_MANTLE_OPENAI_GPT_6_SOL_RESPONSES_JSON,
        provider_model,
        "responses_cache_read",
        _mantle_gpt_usage(3144, 16, cache_reasoning_tokens, cached_tokens=3142),
        cache_expected,
        {**_MANTLE_GPT_RESPONSES_REQUEST, "max_output_tokens": 16},
    )
    tiered_input: Final = _mantle_gpt_json_case(
        BEDROCK_MANTLE_OPENAI_GPT_6_SOL_RESPONSES_JSON,
        provider_model,
        "tiered_input_above_272k",
        _mantle_gpt_usage(300000, 412, 0),
        tiered_input_expected,
        {**_MANTLE_GPT_RESPONSES_REQUEST, "max_output_tokens": 412},
    )
    tiered_cache: Final = _mantle_gpt_json_case(
        BEDROCK_MANTLE_OPENAI_GPT_6_SOL_RESPONSES_JSON,
        provider_model,
        "tiered_cache_read_above_272k",
        _mantle_gpt_usage(300000, 412, 0, cached_tokens=200000),
        tiered_cache_expected,
        {**_MANTLE_GPT_RESPONSES_REQUEST, "max_output_tokens": 412},
    )
    tiered_boundary: Final = _mantle_gpt_json_case(
        BEDROCK_MANTLE_OPENAI_GPT_6_SOL_RESPONSES_JSON,
        provider_model,
        "tiered_at_272k_boundary",
        _mantle_gpt_usage(272000, 412, 0),
        tiered_boundary_expected,
        {**_MANTLE_GPT_RESPONSES_REQUEST, "max_output_tokens": 412},
    )
    return (
        response_case,
        stream_case,
        reasoning_case,
        chat_case,
        cache_case,
        tiered_input,
        tiered_cache,
        tiered_boundary,
        chat_stream_case,
        messages_json,
        messages_stream,
    )


def _mantle_gpt_audit3_cases(
    provider_model: str,
    tool_input_tokens: int,
    tool_output_tokens: int,
    tool_reasoning_tokens: int,
    tool_expected: ExactExpected,
    cache_output_tokens: int,
    cache_reasoning_tokens: int,
    cache_expected: ExactExpected,
) -> tuple[CostTrackingTestCase, ...]:
    tool_case: Final = _mantle_gpt_json_case(
        BEDROCK_MANTLE_OPENAI_GPT_6_SOL_RESPONSES_JSON,
        provider_model,
        "tool_call",
        _mantle_gpt_usage(tool_input_tokens, tool_output_tokens, tool_reasoning_tokens),
        tool_expected,
        _MANTLE_GPT_TOOL_REQUEST,
    )
    if not isinstance(tool_case.response, JsonResponse):
        raise TypeError(f"{tool_case.name}: expected a JSON response fixture")
    tool_response: Final = tool_case.response.model_copy(
        update={
            "body": {
                **tool_case.response.body,
                "output": [
                    {
                        "arguments": '{"city":"Paris"}',
                        "call_id": "call_$REQUEST_ID",
                        "id": "fc_$REQUEST_ID",
                        "name": "get_weather",
                        "status": "completed",
                        "type": "function_call",
                    }
                ],
            }
        }
    )
    tool_case_with_response: Final = tool_case.model_copy(update={"response": tool_response})
    cache_case: Final = _mantle_gpt_json_case(
        BEDROCK_MANTLE_OPENAI_GPT_6_SOL_RESPONSES_JSON,
        provider_model,
        "cache_write_5m",
        _mantle_gpt_usage(
            1667,
            cache_output_tokens,
            cache_reasoning_tokens,
            cache_write_tokens=1665,
        ),
        cache_expected,
        _MANTLE_GPT_CACHE_WRITE_REQUEST,
    )
    return (tool_case_with_response, cache_case)


def _mantle_gpt_bridge(case: CostTrackingTestCase) -> BridgeCostParityTestCase:
    bridged: Final = case.model_copy(
        update={
            "name": case.name.removesuffix("responses_json") + "chat_bridged_json",
            "endpoint": "/v1/chat/completions",
            "request": dict(_MANTLE_GPT_CHAT_REQUEST),
        }
    )
    return BridgeCostParityTestCase(native=case, bridged=bridged)


_MANTLE_GPT_61_SOL_CASES: Final = _mantle_gpt_cases(
    "openai.gpt-6.1-sol",
    48,
    ExactExpected(spend=0.00079618, input_cost=0.00002178, output_cost=0.0007744, prompt_tokens=9, completion_tokens=64),
    _mantle_gpt_usage(16, 5, 0),
    ExactExpected(spend=0.00009922, input_cost=0.00003872, output_cost=0.0000605, prompt_tokens=16, completion_tokens=5),
    0,
    ExactExpected(spend=0.000578622, input_cost=0.000385022, output_cost=0.0001936, prompt_tokens=3144, completion_tokens=16),
    ExactExpected(spend=1.4594778, input_cost=1.452, output_cost=0.0074778, prompt_tokens=300000, completion_tokens=412),
    ExactExpected(spend=0.5398778, input_cost=0.5324, output_cost=0.0074778, prompt_tokens=300000, completion_tokens=412),
    ExactExpected(spend=0.6632252, input_cost=0.65824, output_cost=0.0049852, prompt_tokens=272000, completion_tokens=412),
)
_MANTLE_GPT_6_LUNA_CASES: Final = _mantle_gpt_cases(
    "openai.gpt-6-luna",
    64,
    ExactExpected(spend=0.000043428, input_cost=0.000001188, output_cost=0.00004224, prompt_tokens=9, completion_tokens=64),
    _mantle_gpt_usage(16, 18, 11),
    ExactExpected(spend=0.000013992, input_cost=0.000002112, output_cost=0.00001188, prompt_tokens=16, completion_tokens=18),
    16,
    ExactExpected(spend=0.0000522984, input_cost=0.0000417384, output_cost=0.00001056, prompt_tokens=3144, completion_tokens=16),
    ExactExpected(spend=0.07960788, input_cost=0.0792, output_cost=0.00040788, prompt_tokens=300000, completion_tokens=412),
    ExactExpected(spend=0.03208788, input_cost=0.03168, output_cost=0.00040788, prompt_tokens=300000, completion_tokens=412),
    ExactExpected(spend=0.03617592, input_cost=0.035904, output_cost=0.00027192, prompt_tokens=272000, completion_tokens=412),
)
_MANTLE_GPT_6_ASTRA_CASES: Final = _mantle_gpt_cases(
    "openai.gpt-6-astra",
    29,
    ExactExpected(spend=0.0047047, input_cost=0.0001287, output_cost=0.004576, prompt_tokens=9, completion_tokens=64),
    _mantle_gpt_usage(16, 5, 0),
    ExactExpected(spend=0.0005863, input_cost=0.0002288, output_cost=0.0003575, prompt_tokens=16, completion_tokens=5),
    0,
    ExactExpected(spend=0.00566566, input_cost=0.00452166, output_cost=0.001144, prompt_tokens=3144, completion_tokens=16),
    ExactExpected(spend=8.624187, input_cost=8.58, output_cost=0.044187, prompt_tokens=300000, completion_tokens=412),
    ExactExpected(spend=3.476187, input_cost=3.432, output_cost=0.044187, prompt_tokens=300000, completion_tokens=412),
    ExactExpected(spend=3.919058, input_cost=3.8896, output_cost=0.029458, prompt_tokens=272000, completion_tokens=412),
)
_MANTLE_GPT_56_TERRA_CASES: Final = _mantle_gpt_cases(
    "openai.gpt-5.6-terra",
    33,
    ExactExpected(spend=0.00121044, input_cost=0.00002772, output_cost=0.00118272, prompt_tokens=9, completion_tokens=64),
    _mantle_gpt_usage(16, 17, 10),
    ExactExpected(spend=0.00036344, input_cost=0.00004928, output_cost=0.00031416, prompt_tokens=16, completion_tokens=17),
    0,
    ExactExpected(spend=0.001269576, input_cost=0.000973896, output_cost=0.00029568, prompt_tokens=3144, completion_tokens=16),
    ExactExpected(spend=1.85942064, input_cost=1.848, output_cost=0.01142064, prompt_tokens=300000, completion_tokens=412),
    ExactExpected(spend=0.75062064, input_cost=0.7392, output_cost=0.01142064, prompt_tokens=300000, completion_tokens=412),
    ExactExpected(spend=0.84537376, input_cost=0.83776, output_cost=0.00761376, prompt_tokens=272000, completion_tokens=412),
)

_MANTLE_GPT_AUDIT3_CASES: Final = (
    *_mantle_gpt_audit3_cases(
        "openai.gpt-6.1-sol",
        50,
        20,
        0,
        ExactExpected(spend=0.000363, input_cost=0.000121, output_cost=0.000242, prompt_tokens=50, completion_tokens=20),
        28,
        0,
        ExactExpected(
            spend=0.005381765,
            input_cost=0.005042965,
            output_cost=0.0003388,
            prompt_tokens=1667,
            completion_tokens=28,
            cache_creation_cost=0.005038125,
        ),
    ),
    *_mantle_gpt_audit3_cases(
        "openai.gpt-6-luna",
        51,
        18,
        0,
        ExactExpected(
            spend=0.000018612, input_cost=0.000006732, output_cost=0.00001188, prompt_tokens=51, completion_tokens=18
        ),
        64,
        40,
        ExactExpected(
            spend=0.000317229,
            input_cost=0.000274989,
            output_cost=0.00004224,
            prompt_tokens=1667,
            completion_tokens=64,
            cache_creation_cost=0.000274725,
        ),
    ),
    *_mantle_gpt_audit3_cases(
        "openai.gpt-6-astra",
        51,
        18,
        0,
        ExactExpected(
            spend=0.0020163, input_cost=0.0007293, output_cost=0.001287, prompt_tokens=51, completion_tokens=18
        ),
        27,
        0,
        ExactExpected(
            spend=0.031720975,
            input_cost=0.029790475,
            output_cost=0.0019305,
            prompt_tokens=1667,
            completion_tokens=27,
            cache_creation_cost=0.029761875,
        ),
    ),
    *_mantle_gpt_audit3_cases(
        "openai.gpt-5.6-terra",
        51,
        28,
        8,
        ExactExpected(
            spend=0.00067452, input_cost=0.00015708, output_cost=0.00051744, prompt_tokens=51, completion_tokens=28
        ),
        28,
        0,
        ExactExpected(
            spend=0.00693385,
            input_cost=0.00641641,
            output_cost=0.00051744,
            prompt_tokens=1667,
            completion_tokens=28,
            cache_creation_cost=0.00641025,
        ),
    ),
)

_MANTLE_CLAUDE_OPUS_MODEL: Final = "bedrock_mantle/anthropic.claude-opus-5-5"
_ADAPTIVE_USAGE: Final = {
    "input_tokens": 24,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
    "output_tokens": 18,
    "output_tokens_details": {"thinking_tokens": 6},
    "service_tier": "standard",
}
_ADAPTIVE_EXPECTED: Final = ExactExpected(
    spend=0.0007524, input_cost=0.0001584, output_cost=0.000594, prompt_tokens=24, completion_tokens=18
)


def _mantle_without_usage_fields(case: CostTrackingTestCase) -> CostTrackingTestCase:
    if not isinstance(case.response, JsonResponse):
        raise TypeError(f"{case.name}: expected a JSON response fixture")
    usage: Final = case.response.body.get("usage")
    if not isinstance(usage, dict):
        raise TypeError(f"{case.name}: expected a usage object")
    response: Final = case.response.model_copy(
        update={
            "body": {
                **case.response.body,
                "usage": {key: value for key, value in usage.items() if key not in {"inference_geo", "speed"}},
            }
        }
    )
    return case.model_copy(update={"response": response})


BEDROCK_MANTLE_CLAUDE_OPUS_5_5_INPUT_TEXT: Final = _mantle_without_usage_fields(with_usage(
    CLAUDE_OPUS_5_5_INPUT_TEXT,
    "bedrock_mantle-anthropic.claude-opus-5-5-input_text",
    _ADAPTIVE_USAGE,
    _ADAPTIVE_EXPECTED,
).model_copy(
    update={
        "model": _MANTLE_CLAUDE_OPUS_MODEL,
        "request": {
            **CLAUDE_OPUS_5_5_INPUT_TEXT.request,
            "reasoning_effort": "medium",
            "allowed_openai_params": ["reasoning_effort"],
        },
    }
))

_MANTLE_CLAUDE_STREAM_RESPONSE: Final = SseResponse(
    content_type="text/event-stream",
    frames=sse_frames(
        {
            "type": "message_start",
            "message": {
                "id": "msg_$REQUEST_ID",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 24,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
                    "output_tokens": 5,
                    "service_tier": "standard",
                },
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": ""}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "4"}},
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {
                "input_tokens": 24,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 18,
                "output_tokens_details": {"thinking_tokens": 6},
            },
        },
        {"type": "message_stop"},
    ),
)

BEDROCK_MANTLE_CLAUDE_OPUS_5_5_STREAM: Final = CLAUDE_OPUS_5_5_STREAM.model_copy(
    update={
        "name": "bedrock_mantle-anthropic.claude-opus-5-5-stream",
        "model": _MANTLE_CLAUDE_OPUS_MODEL,
        "request": {
            **CLAUDE_OPUS_5_5_STREAM.request,
            "reasoning_effort": "medium",
            "allowed_openai_params": ["reasoning_effort"],
        },
        "response": _MANTLE_CLAUDE_STREAM_RESPONSE,
        "expected": _ADAPTIVE_EXPECTED,
    }
)

BEDROCK_MANTLE_CLAUDE_OPUS_5_5_MESSAGES_INPUT_TEXT: Final = with_usage(
    CLAUDE_OPUS_5_5_MESSAGES_INPUT_TEXT,
    "bedrock_mantle-anthropic.claude-opus-5-5-messages_input_text",
    _ADAPTIVE_USAGE,
    _ADAPTIVE_EXPECTED,
).model_copy(
    update={
        "model": _MANTLE_CLAUDE_OPUS_MODEL,
        "request": {
            **CLAUDE_OPUS_5_5_MESSAGES_INPUT_TEXT.request,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"},
        },
    }
)

BEDROCK_MANTLE_CLAUDE_OPUS_5_5_MESSAGES_STREAM: Final = CLAUDE_OPUS_5_5_MESSAGES_STREAM.model_copy(
    update={
        "name": "bedrock_mantle-anthropic.claude-opus-5-5-messages_stream",
        "model": _MANTLE_CLAUDE_OPUS_MODEL,
        "request": {
            **CLAUDE_OPUS_5_5_MESSAGES_STREAM.request,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"},
        },
        "response": _MANTLE_CLAUDE_STREAM_RESPONSE,
        "expected": _ADAPTIVE_EXPECTED,
    }
)

BEDROCK_MANTLE_CLAUDE_OPUS_5_5_CACHE_READ: Final = _mantle_without_usage_fields(with_usage(
    CLAUDE_OPUS_5_5_CACHE_READ,
    "bedrock_mantle-anthropic.claude-opus-5-5-cache_read",
    {
        "input_tokens": 22,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 4203,
        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
        "output_tokens": 47,
        "output_tokens_details": {"thinking_tokens": 35},
        "service_tier": "standard",
    },
    ExactExpected(
        spend=0.00308319, input_cost=0.00153219, output_cost=0.001551, prompt_tokens=4225, completion_tokens=47
    ),
).model_copy(update={"model": _MANTLE_CLAUDE_OPUS_MODEL}))

BEDROCK_MANTLE_CLAUDE_OPUS_5_5_CACHE_WRITE_5M: Final = with_usage(
    CLAUDE_OPUS_5_5_CACHE_WRITE_5M,
    "bedrock_mantle-anthropic.claude-opus-5-5-cache_write_5m",
    {
        "input_tokens": 22,
        "cache_creation_input_tokens": 4203,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 4203, "ephemeral_1h_input_tokens": 0},
        "output_tokens": 44,
        "output_tokens_details": {"thinking_tokens": 32},
        "service_tier": "standard",
    },
    ExactExpected(
        spend=0.03627195, input_cost=0.03481995, output_cost=0.001452, prompt_tokens=4225, completion_tokens=44
    ),
).model_copy(update={"model": _MANTLE_CLAUDE_OPUS_MODEL})

BEDROCK_MANTLE_CLAUDE_OPUS_5_5_CACHE_WRITE_1H: Final = with_usage(
    CLAUDE_OPUS_5_5_CACHE_WRITE_1H,
    "bedrock_mantle-anthropic.claude-opus-5-5-cache_write_1h",
    {
        "input_tokens": 22,
        "cache_creation_input_tokens": 9003,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 9003},
        "output_tokens": 12,
        "output_tokens_details": {"thinking_tokens": 0},
        "service_tier": "standard",
    },
    ExactExpected(
        spend=0.1193808, input_cost=0.1189848, output_cost=0.000396, prompt_tokens=9025, completion_tokens=12
    ),
).model_copy(update={"model": _MANTLE_CLAUDE_OPUS_MODEL})

BEDROCK_MANTLE_CLAUDE_OPUS_5_5_THINKING: Final = _mantle_without_usage_fields(with_usage(
    CLAUDE_OPUS_5_5_THINKING,
    "bedrock_mantle-anthropic.claude-opus-5-5-thinking",
    _ADAPTIVE_USAGE,
    _ADAPTIVE_EXPECTED,
).model_copy(update={"model": _MANTLE_CLAUDE_OPUS_MODEL}))

BEDROCK_MANTLE_CLAUDE_OPUS_5_5_TOOL_CALL: Final = with_usage(
    CLAUDE_OPUS_5_5_TOOL_CALL,
    "bedrock_mantle-anthropic.claude-opus-5-5-tool_call",
    {
        "input_tokens": 376,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
        "output_tokens": 12,
        "output_tokens_details": {"thinking_tokens": 0},
        "service_tier": "standard",
    },
    ExactExpected(spend=0.0028776, input_cost=0.0024816, output_cost=0.000396, prompt_tokens=376, completion_tokens=12),
).model_copy(update={"model": _MANTLE_CLAUDE_OPUS_MODEL})

BEDROCK_MANTLE_CLAUDE_OPUS_5_5_CHAT_BRIDGED_JSON: Final = BEDROCK_MANTLE_CLAUDE_OPUS_5_5_INPUT_TEXT.model_copy(
    update={
        "name": "bedrock_mantle-anthropic.claude-opus-5-5-chat_bridged_json",
        "response": BEDROCK_MANTLE_CLAUDE_OPUS_5_5_MESSAGES_INPUT_TEXT.response,
    }
)

BEDROCK_MANTLE_CLAUDE_OPUS_5_5_RESPONSES_JSON: Final = BEDROCK_MANTLE_CLAUDE_OPUS_5_5_INPUT_TEXT.model_copy(
    update={
        "name": "bedrock_mantle-anthropic.claude-opus-5-5-responses_json",
        "endpoint": "/v1/responses",
        "request": {"model": "$MODEL", "input": "summarize the attached material in one line"},
    }
)
BEDROCK_MANTLE_CLAUDE_OPUS_5_5_RESPONSES_STREAM: Final = BEDROCK_MANTLE_CLAUDE_OPUS_5_5_STREAM.model_copy(
    update={
        "name": "bedrock_mantle-anthropic.claude-opus-5-5-responses_stream",
        "endpoint": "/v1/responses",
        "request": {"model": "$MODEL", "input": "summarize the attached material in one line", "stream": True},
    }
)

BRIDGE_COST_PARITY: Final[tuple[BridgeCostParityTestCase, ...]] = (
    _mantle_gpt_bridge(_MANTLE_GPT_61_SOL_CASES[0]),
    _mantle_gpt_bridge(_MANTLE_GPT_6_LUNA_CASES[0]),
    _mantle_gpt_bridge(_MANTLE_GPT_6_ASTRA_CASES[0]),
    _mantle_gpt_bridge(_MANTLE_GPT_56_TERRA_CASES[0]),
    BridgeCostParityTestCase(
        native=BEDROCK_MANTLE_CLAUDE_OPUS_5_5_MESSAGES_INPUT_TEXT,
        bridged=BEDROCK_MANTLE_CLAUDE_OPUS_5_5_CHAT_BRIDGED_JSON,
    ),
    *tuple(
        BridgeCostParityTestCase(native=cases[0], bridged=cases[9])
        for cases in (_MANTLE_GPT_61_SOL_CASES, _MANTLE_GPT_6_LUNA_CASES, _MANTLE_GPT_6_ASTRA_CASES, _MANTLE_GPT_56_TERRA_CASES)
    ),
    *tuple(
        BridgeCostParityTestCase(native=cases[1], bridged=cases[10])
        for cases in (_MANTLE_GPT_61_SOL_CASES, _MANTLE_GPT_6_LUNA_CASES, _MANTLE_GPT_6_ASTRA_CASES, _MANTLE_GPT_56_TERRA_CASES)
    ),
)

CASES: Final[tuple[CostTrackingTestCase, ...]] = (
    BEDROCK_MANTLE_OPENAI_GPT_6_SOL_RESPONSES_JSON,
    BEDROCK_MANTLE_OPENAI_GPT_6_SOL_RESPONSES_STREAM,
    BEDROCK_MANTLE_OPENAI_GPT_6_SOL_RESPONSES_REASONING,
    BEDROCK_MANTLE_OPENAI_GPT_6_SOL_CHAT_JSON,
    BEDROCK_MANTLE_OPENAI_GPT_6_SOL_RESPONSES_CACHE_READ,
    BEDROCK_MANTLE_OPENAI_GPT_6_SOL_TIERED_INPUT_ABOVE_272K,
    BEDROCK_MANTLE_OPENAI_GPT_6_SOL_TIERED_CACHE_READ_ABOVE_272K,
    BEDROCK_MANTLE_OPENAI_GPT_6_SOL_TIERED_AT_272K_BOUNDARY,
    *_MANTLE_GPT_61_SOL_CASES,
    *_MANTLE_GPT_6_LUNA_CASES,
    *_MANTLE_GPT_6_ASTRA_CASES,
    *_MANTLE_GPT_56_TERRA_CASES,
    *_MANTLE_GPT_AUDIT3_CASES,
    BEDROCK_MANTLE_CLAUDE_OPUS_5_5_INPUT_TEXT,
    BEDROCK_MANTLE_CLAUDE_OPUS_5_5_STREAM,
    BEDROCK_MANTLE_CLAUDE_OPUS_5_5_MESSAGES_INPUT_TEXT,
    BEDROCK_MANTLE_CLAUDE_OPUS_5_5_MESSAGES_STREAM,
    BEDROCK_MANTLE_CLAUDE_OPUS_5_5_CHAT_BRIDGED_JSON,
    BEDROCK_MANTLE_CLAUDE_OPUS_5_5_RESPONSES_JSON,
    BEDROCK_MANTLE_CLAUDE_OPUS_5_5_RESPONSES_STREAM,
    BEDROCK_MANTLE_CLAUDE_OPUS_5_5_CACHE_READ,
    BEDROCK_MANTLE_CLAUDE_OPUS_5_5_CACHE_WRITE_5M,
    BEDROCK_MANTLE_CLAUDE_OPUS_5_5_CACHE_WRITE_1H,
    BEDROCK_MANTLE_CLAUDE_OPUS_5_5_THINKING,
    BEDROCK_MANTLE_CLAUDE_OPUS_5_5_TOOL_CALL,
)

PARITY: Final[tuple[StreamParityTestCase, ...]] = (
    StreamParityTestCase(
        plain=BEDROCK_MANTLE_OPENAI_GPT_6_SOL_RESPONSES_JSON, streamed=BEDROCK_MANTLE_OPENAI_GPT_6_SOL_RESPONSES_STREAM
    ),
    StreamParityTestCase(plain=_MANTLE_GPT_61_SOL_CASES[0], streamed=_MANTLE_GPT_61_SOL_CASES[1]),
    StreamParityTestCase(plain=_MANTLE_GPT_6_LUNA_CASES[0], streamed=_MANTLE_GPT_6_LUNA_CASES[1]),
    StreamParityTestCase(plain=_MANTLE_GPT_6_ASTRA_CASES[0], streamed=_MANTLE_GPT_6_ASTRA_CASES[1]),
    StreamParityTestCase(plain=_MANTLE_GPT_56_TERRA_CASES[0], streamed=_MANTLE_GPT_56_TERRA_CASES[1]),
    StreamParityTestCase(
        plain=BEDROCK_MANTLE_CLAUDE_OPUS_5_5_INPUT_TEXT, streamed=BEDROCK_MANTLE_CLAUDE_OPUS_5_5_STREAM
    ),
    StreamParityTestCase(
        plain=BEDROCK_MANTLE_CLAUDE_OPUS_5_5_MESSAGES_INPUT_TEXT,
        streamed=BEDROCK_MANTLE_CLAUDE_OPUS_5_5_MESSAGES_STREAM,
    ),
    StreamParityTestCase(
        plain=BEDROCK_MANTLE_CLAUDE_OPUS_5_5_RESPONSES_JSON,
        streamed=BEDROCK_MANTLE_CLAUDE_OPUS_5_5_RESPONSES_STREAM,
    ),
    *tuple(
        StreamParityTestCase(plain=cases[3], streamed=cases[8])
        for cases in (_MANTLE_GPT_61_SOL_CASES, _MANTLE_GPT_6_LUNA_CASES, _MANTLE_GPT_6_ASTRA_CASES, _MANTLE_GPT_56_TERRA_CASES)
    ),
    *tuple(
        StreamParityTestCase(plain=cases[9], streamed=cases[10])
        for cases in (_MANTLE_GPT_61_SOL_CASES, _MANTLE_GPT_6_LUNA_CASES, _MANTLE_GPT_6_ASTRA_CASES, _MANTLE_GPT_56_TERRA_CASES)
    ),
)
