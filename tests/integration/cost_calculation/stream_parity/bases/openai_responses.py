from dataclasses import replace
from typing import Final

from integration._support.provider import PROVIDER_URL
from integration.cost_calculation.stream_parity.case import MODEL, RESPONSE_ID, StreamParityTestCase

_ITEM: Final = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "Hello.", "annotations": []}],
}
_RESPONSE: Final = {
    "id": RESPONSE_ID,
    "object": "response",
    "created_at": 1,
    "status": "completed",
    "model": "gpt-6.1-sol",
    "output": [_ITEM],
    "usage": {"input_tokens": 30, "output_tokens": 40, "total_tokens": 70},
}

GPT_6_1_SOL_RESPONSES_TEST_CASE: Final = StreamParityTestCase(
    scenario="responses",
    litellm_endpoint="/v1/responses",
    deployment={
        "model": "openai/responses/gpt-6.1-sol",
        "api_key": "synthetic-openai-key",
        "api_base": f"{PROVIDER_URL}/v1",
        "input_cost_per_token": 0.001,
        "output_cost_per_token": 0.002,
    },
    litellm_request={"model": MODEL, "input": "Say hello.", "cache": {"no-cache": True}},
    expected_provider_endpoint="/v1/responses",
    mock_provider_response=_RESPONSE,
    mock_provider_stream=(
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**_RESPONSE, "status": "in_progress", "output": []},
        },
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {**_ITEM, "status": "in_progress", "content": []},
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 2,
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello.",
        },
        {"type": "response.output_item.done", "sequence_number": 3, "output_index": 0, "item": _ITEM},
        {"type": "response.completed", "sequence_number": 4, "response": _RESPONSE},
    ),
    expected_spend_row={
        "model": "openai/responses/gpt-6.1-sol",
        "model_group": MODEL,
        "custom_llm_provider": "openai",
        "prompt_tokens": 30,
        "completion_tokens": 40,
        "total_tokens": 70,
        "spend": 0.11,
    },
)

GPT_6_1_SOL_CHAT_COMPLETIONS_TEST_CASE: Final = replace(
    GPT_6_1_SOL_RESPONSES_TEST_CASE,
    scenario="chat_completions",
    litellm_endpoint="/v1/chat/completions",
    litellm_request={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    stream_parameters={"stream": True, "stream_options": {"include_usage": True}},
)
