"""Live e2e: the /openai passthrough injects usage.cost into streaming usage frames.

Pins #36503: with the proxy running `include_cost_in_streaming_usage: true`, a
streamed call through the provider passthrough surface must carry the computed
cost inside the final usage-only SSE frame, the same contract the native
/chat/completions stream has. Providers never send `cost` themselves, so a
nonzero value proves the proxy computed and injected it on the passthrough path.

The row-side spend accounting for passthrough calls is covered elsewhere; this
test pins only the in-stream cost surface, which clients read without ever
touching /spend/logs.
"""

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from models import ChatBody, ChatMessage, StreamOptions, Usage
from spend_e2e_client import SpendClient

pytestmark = pytest.mark.e2e

OPENAI_MODEL = "gpt-5.6-luna"


class _StreamFrame(BaseModel):
    usage: Usage | None = None


class TestPassthroughStreamCost:
    @pytest.mark.covers("quota_management.spend_tracking.passthrough_stream.injects_usage_cost")
    def test_passthrough_stream_final_usage_frame_carries_cost(
        self, client: SpendClient, scoped_key: str
    ) -> None:
        result = client.proxy.transport.send(
            "/openai/v1/chat/completions",
            headers=client.proxy.transport.bearer(scoped_key),
            json=ChatBody(
                model=OPENAI_MODEL,
                messages=[
                    ChatMessage(
                        role="user",
                        content=f"{unique_marker()} Reply with the single word passthrough.",
                    )
                ],
                stream=True,
                stream_options=StreamOptions(),
            ),
            stream=True,
        )
        assert result.ok and result.stream_events, (
            f"passthrough stream failed (status {result.status_code}): {result.body[:300]}"
        )

        usage_frames = [
            frame.usage
            for frame in (_StreamFrame.model_validate_json(event) for event in result.stream_events)
            if frame.usage is not None
        ]
        assert usage_frames, (
            f"no usage frame in the passthrough stream despite stream_options.include_usage; "
            f"last event: {result.stream_events[-1][:300]}"
        )

        final_usage = usage_frames[-1]
        assert final_usage.cost is not None and final_usage.cost > 0, (
            f"final passthrough usage frame carries no injected cost: {final_usage}"
        )
