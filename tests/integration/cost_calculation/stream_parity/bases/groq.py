"""/v1/messages on a chat-completions provider: the messages -> chat/completions bridge."""

from typing import Final

from integration.cost_calculation.cost_tracking_case import (
    CostTrackingTestCase,
    Deployment,
    ExactExpected,
    JsonResponse,
    SseResponse,
)
from integration.cost_calculation.stream_parity.case import COVERS, MODEL, REQUEST_ID, StreamParityTestCase, sse_frames

_USAGE: Final = {"prompt_tokens": 30, "completion_tokens": 40, "total_tokens": 70}
_CHUNK: Final = {
    "id": f"chatcmpl-{REQUEST_ID}",
    "object": "chat.completion.chunk",
    "created": 1,
    "model": "qwen/qwen3.8-27b",
}

QWEN_3_8_MESSAGES: Final = CostTrackingTestCase(
    name="groq-qwen-3.8-messages-parity",
    covers=COVERS,
    model="groq/qwen/qwen3.8-27b",
    endpoint="/v1/messages",
    deployment=Deployment(input_cost_per_token=0.001, output_cost_per_token=0.002),
    request={
        "model": MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Say hello."}],
        "cache": {"no-cache": True},
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": f"chatcmpl-{REQUEST_ID}",
            "object": "chat.completion",
            "created": 1,
            "model": "qwen/qwen3.8-27b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello."}, "finish_reason": "stop"}],
            "usage": _USAGE,
            "service_tier": "on_demand",
        },
    ),
    expected=ExactExpected(spend=0.11, input_cost=0.03, output_cost=0.08, prompt_tokens=30, completion_tokens=40),
)

QWEN_3_8_MESSAGES_PARITY: Final = StreamParityTestCase(
    plain=QWEN_3_8_MESSAGES,
    streamed=QWEN_3_8_MESSAGES.model_copy(
        update={
            "name": "groq-qwen-3.8-messages-parity-stream",
            "request": {**QWEN_3_8_MESSAGES.request, "stream": True},
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
