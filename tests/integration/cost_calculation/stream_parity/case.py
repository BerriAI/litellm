import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from integration.cost_calculation.cost_tracking_case import CostTrackingTestCase
from pydantic import JsonValue

COVERS: Final = "quota_management.spend_tracking.scripted_wire.logs_cost"
REQUEST_ID: Final = "$REQUEST_ID"
MODEL: Final = "$MODEL"


def sse_frames(*events: Mapping[str, JsonValue], done: bool = False) -> tuple[str, ...]:
    frames: Final = tuple(
        f"event: {event['type']}\ndata: {json.dumps(event)}" if "type" in event else f"data: {json.dumps(event)}"
        for event in events
    )
    return (*frames, "data: [DONE]") if done else frames


@dataclass(frozen=True, slots=True, kw_only=True)
class StreamParityTestCase:
    """The same request as two CostTrackingTestCases, `plain` answered with JSON and `streamed` answered with SSE,
    that must both bill the one `expected` row."""

    plain: CostTrackingTestCase
    streamed: CostTrackingTestCase

    def __post_init__(self) -> None:
        if self.plain.expected != self.streamed.expected:
            raise ValueError(f"{self.id}: plain and streamed cases expect different rows")
        if self.streamed.request.get("stream") is not True or self.plain.request.get("stream") is True:
            raise ValueError(f"{self.id}: streamed case must set stream: true and the plain case must not")
        if (self.plain.model, self.plain.endpoint, self.plain.deployment) != (
            self.streamed.model,
            self.streamed.endpoint,
            self.streamed.deployment,
        ):
            raise ValueError(f"{self.id}: plain and streamed cases must share model, endpoint and deployment")

    @property
    def id(self) -> str:
        """The plain case's name, or the streamed case's own name when it is a variant beyond a `-stream` twin."""
        return self.plain.name if self.streamed.name == f"{self.plain.name}-stream" else self.streamed.name
