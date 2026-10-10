from typing import Final

from integration.cost_calculation.cost_tracking_case import (
    CostTrackingTestCase,
    Deployment,
    ExactExpected,
    JsonResponse,
    SseResponse,
)
from integration.cost_calculation.stream_parity.bases.openai_responses import (
    GPT_5_3_CODEX_RESPONSES,
    GPT_5_3_CODEX_RESPONSES_PARITY_STREAM_RESPONSE,
)
from integration.cost_calculation.stream_parity.case import COVERS, MODEL, REQUEST_ID, StreamParityTestCase, sse_frames

_USAGE: Final = {"prompt_tokens": 30, "completion_tokens": 40, "total_tokens": 70}
_CHUNK: Final = {
    "id": f"chatcmpl-{REQUEST_ID}",
    "object": "chat.completion.chunk",
    "created": 1,
    "model": "gpt-5.6",
}

GPT_5_6_CHAT_COMPLETIONS: Final = CostTrackingTestCase(
    name="gpt-5.6-chat-parity",
    covers=COVERS,
    model="gpt-5.6",
    endpoint="/v1/chat/completions",
    deployment=Deployment(input_cost_per_token=0.001, output_cost_per_token=0.002),
    request={"model": MODEL, "messages": [{"role": "user", "content": "Say hello."}], "cache": {"no-cache": True}},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": f"chatcmpl-{REQUEST_ID}",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-5.6",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello."}, "finish_reason": "stop"}],
            "usage": _USAGE,
        },
    ),
    expected=ExactExpected(spend=0.11, input_cost=0.03, output_cost=0.08, prompt_tokens=30, completion_tokens=40),
)

GPT_5_6_CHAT_COMPLETIONS_PARITY: Final = StreamParityTestCase(
    plain=GPT_5_6_CHAT_COMPLETIONS,
    streamed=GPT_5_6_CHAT_COMPLETIONS.model_copy(
        update={
            "name": "gpt-5.6-chat-parity-stream",
            "request": {
                **GPT_5_6_CHAT_COMPLETIONS.request,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            "response": SseResponse(
                content_type="text/event-stream",
                frames=sse_frames(
                    {**_CHUNK, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello."}}]},
                    {**_CHUNK, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                    {**_CHUNK, "choices": [], "usage": _USAGE},
                    done=True,
                ),
            ),
        }
    ),
)

GPT_5_6_MESSAGES: Final = GPT_5_3_CODEX_RESPONSES.model_copy(
    update={
        "name": "gpt-5.6-messages-parity",
        "model": "gpt-5.6",
        "endpoint": "/v1/messages",
        "request": {
            "model": MODEL,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Say hello."}],
            "cache": {"no-cache": True},
        },
    }
)

GPT_5_6_MESSAGES_PARITY: Final = StreamParityTestCase(
    plain=GPT_5_6_MESSAGES,
    streamed=GPT_5_6_MESSAGES.model_copy(
        update={
            "name": "gpt-5.6-messages-parity-stream",
            "request": {**GPT_5_6_MESSAGES.request, "stream": True},
            "response": GPT_5_3_CODEX_RESPONSES_PARITY_STREAM_RESPONSE,
        }
    ),
)
