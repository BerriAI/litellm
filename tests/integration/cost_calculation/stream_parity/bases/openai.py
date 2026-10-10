from dataclasses import replace
from typing import Final

from integration._support.provider import PROVIDER_URL
from integration.cost_calculation.stream_parity.bases.openai_responses import GPT_6_1_SOL_RESPONSES_TEST_CASE
from integration.cost_calculation.stream_parity.case import MODEL, RESPONSE_ID, StreamParityTestCase

_USAGE: Final = {"prompt_tokens": 30, "completion_tokens": 40, "total_tokens": 70}
_CHUNK: Final = {"id": RESPONSE_ID, "object": "chat.completion.chunk", "created": 1, "model": "gpt-6.1-sol"}

GPT_6_1_SOL_CHAT_COMPLETIONS_TEST_CASE: Final = StreamParityTestCase(
    scenario="chat_completions",
    litellm_endpoint="/v1/chat/completions",
    deployment={
        "model": "openai/gpt-6.1-sol",
        "api_key": "synthetic-openai-key",
        "api_base": f"{PROVIDER_URL}/v1",
        "input_cost_per_token": 0.001,
        "output_cost_per_token": 0.002,
    },
    litellm_request={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    stream_parameters={"stream": True, "stream_options": {"include_usage": True}},
    expected_provider_endpoint="/v1/chat/completions",
    mock_provider_response={
        "id": RESPONSE_ID,
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-6.1-sol",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello."}, "finish_reason": "stop"}],
        "usage": _USAGE,
    },
    mock_provider_stream=(
        {**_CHUNK, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello."}}]},
        {**_CHUNK, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {**_CHUNK, "choices": [], "usage": _USAGE},
    ),
    expected_spend_row={
        "model": "openai/gpt-6.1-sol",
        "model_group": MODEL,
        "custom_llm_provider": "openai",
        "prompt_tokens": 30,
        "completion_tokens": 40,
        "total_tokens": 70,
        "spend": 0.11,
    },
)

GPT_6_1_SOL_MESSAGES_TEST_CASE: Final = replace(
    GPT_6_1_SOL_RESPONSES_TEST_CASE,
    scenario="messages",
    litellm_endpoint="/v1/messages",
    deployment=GPT_6_1_SOL_CHAT_COMPLETIONS_TEST_CASE.deployment,
    litellm_request={
        "model": MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    expected_spend_row={**GPT_6_1_SOL_RESPONSES_TEST_CASE.expected_spend_row, "model": "openai/gpt-6.1-sol"},
)
