from typing import Final

from integration.cost_calculation.cost_tracking_case import (
    CostTrackingTestCase,
    Deployment,
    ExactExpected,
    JsonResponse,
    SseResponse,
)
from integration.cost_calculation.stream_parity.case import COVERS, MODEL, REQUEST_ID, StreamParityTestCase, sse_frames

CLAUDE_SONNET_5_MESSAGES: Final = CostTrackingTestCase(
    name="claude-sonnet-5-messages-parity",
    covers=COVERS,
    model="claude-sonnet-5",
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
            "id": f"msg_{REQUEST_ID}",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "Hello."}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 30, "output_tokens": 40},
        },
    ),
    expected=ExactExpected(spend=0.11, input_cost=0.03, output_cost=0.08, prompt_tokens=30, completion_tokens=40),
)

CLAUDE_SONNET_5_STREAM_RESPONSE: Final = SseResponse(
    content_type="text/event-stream",
    frames=sse_frames(
        {
            "type": "message_start",
            "message": {
                "id": f"msg_{REQUEST_ID}",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 30, "output_tokens": 1},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello."}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 40},
        },
        {"type": "message_stop"},
    ),
)

CLAUDE_SONNET_5_MESSAGES_PARITY: Final = StreamParityTestCase(
    plain=CLAUDE_SONNET_5_MESSAGES,
    streamed=CLAUDE_SONNET_5_MESSAGES.model_copy(
        update={
            "name": "claude-sonnet-5-messages-parity-stream",
            "request": {**CLAUDE_SONNET_5_MESSAGES.request, "stream": True},
            "response": CLAUDE_SONNET_5_STREAM_RESPONSE,
        }
    ),
)

CLAUDE_SONNET_5_CHAT_COMPLETIONS: Final = CLAUDE_SONNET_5_MESSAGES.model_copy(
    update={
        "name": "claude-sonnet-5-chat-parity",
        "endpoint": "/v1/chat/completions",
        "request": {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Say hello."}],
            "cache": {"no-cache": True},
        },
    }
)

CLAUDE_SONNET_5_CHAT_COMPLETIONS_PARITY: Final = StreamParityTestCase(
    plain=CLAUDE_SONNET_5_CHAT_COMPLETIONS,
    streamed=CLAUDE_SONNET_5_CHAT_COMPLETIONS.model_copy(
        update={
            "name": "claude-sonnet-5-chat-parity-stream",
            "request": {
                **CLAUDE_SONNET_5_CHAT_COMPLETIONS.request,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            "response": CLAUDE_SONNET_5_STREAM_RESPONSE,
        }
    ),
)

CLAUDE_SONNET_5_RESPONSES: Final = CLAUDE_SONNET_5_MESSAGES.model_copy(
    update={
        "name": "claude-sonnet-5-responses-parity",
        "endpoint": "/v1/responses",
        "request": {"model": MODEL, "input": "Say hello.", "cache": {"no-cache": True}},
    }
)

CLAUDE_SONNET_5_RESPONSES_PARITY: Final = StreamParityTestCase(
    plain=CLAUDE_SONNET_5_RESPONSES,
    streamed=CLAUDE_SONNET_5_RESPONSES.model_copy(
        update={
            "name": "claude-sonnet-5-responses-parity-stream",
            "request": {**CLAUDE_SONNET_5_RESPONSES.request, "stream": True},
            "response": CLAUDE_SONNET_5_STREAM_RESPONSE,
        }
    ),
)
