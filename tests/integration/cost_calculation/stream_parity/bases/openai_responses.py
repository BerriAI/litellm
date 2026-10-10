from typing import Final

from integration.cost_calculation.cost_tracking_case import (
    CostTrackingTestCase,
    Deployment,
    ExactExpected,
    JsonResponse,
    SseResponse,
)
from integration.cost_calculation.stream_parity.case import COVERS, MODEL, REQUEST_ID, StreamParityTestCase, sse_frames

_ITEM: Final = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "Hello.", "annotations": []}],
}
_RESPONSE: Final = {
    "id": f"resp_{REQUEST_ID}",
    "object": "response",
    "created_at": 1,
    "status": "completed",
    "model": "gpt-5.3-codex",
    "output": [_ITEM],
    "usage": {"input_tokens": 30, "output_tokens": 40, "total_tokens": 70},
}

GPT_5_3_CODEX_RESPONSES: Final = CostTrackingTestCase(
    name="gpt-5.3-codex-responses-parity",
    covers=COVERS,
    model="gpt-5.3-codex",
    endpoint="/v1/responses",
    deployment=Deployment(input_cost_per_token=0.001, output_cost_per_token=0.002),
    request={"model": MODEL, "input": "Say hello.", "cache": {"no-cache": True}},
    response=JsonResponse(content_type="application/json", body=_RESPONSE),
    expected=ExactExpected(spend=0.11, input_cost=0.03, output_cost=0.08, prompt_tokens=30, completion_tokens=40),
)

GPT_5_3_CODEX_RESPONSES_PARITY_STREAM_RESPONSE: Final = SseResponse(
    content_type="text/event-stream",
    frames=sse_frames(
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
)

GPT_5_3_CODEX_RESPONSES_PARITY: Final = StreamParityTestCase(
    plain=GPT_5_3_CODEX_RESPONSES,
    streamed=GPT_5_3_CODEX_RESPONSES.model_copy(
        update={
            "name": "gpt-5.3-codex-responses-parity-stream",
            "request": {**GPT_5_3_CODEX_RESPONSES.request, "stream": True},
            "response": GPT_5_3_CODEX_RESPONSES_PARITY_STREAM_RESPONSE,
        }
    ),
)

GPT_5_3_CODEX_CHAT_COMPLETIONS: Final = GPT_5_3_CODEX_RESPONSES.model_copy(
    update={
        "name": "gpt-5.3-codex-chat-parity",
        "endpoint": "/v1/chat/completions",
        "request": {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Say hello."}],
            "cache": {"no-cache": True},
        },
    }
)

GPT_5_3_CODEX_CHAT_COMPLETIONS_PARITY: Final = StreamParityTestCase(
    plain=GPT_5_3_CODEX_CHAT_COMPLETIONS,
    streamed=GPT_5_3_CODEX_CHAT_COMPLETIONS.model_copy(
        update={
            "name": "gpt-5.3-codex-chat-parity-stream",
            "request": {
                **GPT_5_3_CODEX_CHAT_COMPLETIONS.request,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            "response": GPT_5_3_CODEX_RESPONSES_PARITY_STREAM_RESPONSE,
        }
    ),
)
