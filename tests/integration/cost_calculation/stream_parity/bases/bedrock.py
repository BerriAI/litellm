"""/v1/messages on a Bedrock Converse deployment: the messages -> converse bridge."""

from typing import Final

from integration.cost_calculation.cost_tracking_case import (
    CostTrackingTestCase,
    Deployment,
    EventStreamEvent,
    EventStreamResponse,
    ExactExpected,
    JsonResponse,
)
from integration.cost_calculation.stream_parity.case import COVERS, MODEL, StreamParityTestCase

_USAGE: Final = {"inputTokens": 30, "outputTokens": 40, "totalTokens": 70}

CLAUDE_OPUS_5_5_CONVERSE_MESSAGES: Final = CostTrackingTestCase(
    name="bedrock-converse-claude-opus-5-5-messages-parity",
    covers=COVERS,
    model="us.anthropic.claude-opus-5-5",
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
            "output": {"message": {"role": "assistant", "content": [{"text": "Hello."}]}},
            "stopReason": "end_turn",
            "usage": _USAGE,
            "metrics": {"latencyMs": 1},
        },
    ),
    expected=ExactExpected(spend=0.11, input_cost=0.03, output_cost=0.08, prompt_tokens=30, completion_tokens=40),
)

CLAUDE_OPUS_5_5_CONVERSE_MESSAGES_PARITY: Final = StreamParityTestCase(
    plain=CLAUDE_OPUS_5_5_CONVERSE_MESSAGES,
    streamed=CLAUDE_OPUS_5_5_CONVERSE_MESSAGES.model_copy(
        update={
            "name": "bedrock-converse-claude-opus-5-5-messages-parity-stream",
            "request": {**CLAUDE_OPUS_5_5_CONVERSE_MESSAGES.request, "stream": True},
            "response": EventStreamResponse(
                content_type="application/vnd.amazon.eventstream",
                framing="converse",
                events=(
                    EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
                    EventStreamEvent(
                        event_type="contentBlockDelta", payload={"delta": {"text": "Hello."}, "contentBlockIndex": 0}
                    ),
                    EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
                    EventStreamEvent(event_type="messageStop", payload={"stopReason": "end_turn"}),
                    EventStreamEvent(event_type="metadata", payload={"usage": _USAGE, "metrics": {"latencyMs": 1}}),
                ),
            ),
        }
    ),
)
