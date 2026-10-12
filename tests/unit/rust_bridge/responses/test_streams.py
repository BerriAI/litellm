from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Final

import pytest

from litellm.rust_bridge.lifecycle import Complete, Open, Step, Yield
from litellm.rust_bridge.responses import streams
from litellm.types.llms.openai import (
    OutputTextDeltaEvent,
    ResponseCompletedEvent,
    ResponseIncompleteEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)

RESPONSE: Final = ResponsesAPIResponse.model_validate(
    {"id": "resp_1", "object": "response", "created_at": 1, "model": "gpt", "status": "completed", "output": []}
)
DELTA: Final = OutputTextDeltaEvent(
    type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA, item_id="msg", output_index=0, content_index=0, delta="hi"
)
COMPLETED: Final = ResponseCompletedEvent(type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED, response=RESPONSE)
INCOMPLETE: Final = ResponseIncompleteEvent(type=ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE, response=RESPONSE)


class ScriptedExecution:
    def __init__(self, steps: Sequence[Step]) -> None:
        self._steps: Final = list(steps)
        self.closed = False

    def start(self) -> Step:
        return self._steps.pop(0)

    def resume_value(self, value: object) -> Step:
        return self._steps.pop(0)

    def resume_error(self, error: BaseException) -> Step:
        return self._steps.pop(0)

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("terminal", (COMPLETED, INCOMPLETE), ids=("completed", "incomplete"))
def test_the_async_stream_exposes_the_terminal_event_as_it_hands_it_over(terminal: object) -> None:
    async def run() -> list[tuple[object, object | None]]:
        stream: Final = await streams.drive(
            ScriptedExecution([Open({}), Yield(DELTA), Yield(terminal), Complete(None)])
        )
        assert isinstance(stream, streams.Stream)
        assert stream.completed_response is None
        seen: Final = []
        async for event in stream:
            seen.append((event, stream.completed_response))
        return seen

    assert asyncio.run(run()) == [(DELTA, None), (terminal, terminal)]


def test_the_sync_stream_exposes_the_terminal_event_as_it_hands_it_over() -> None:
    stream: Final = streams.SyncStream(ScriptedExecution([Yield(DELTA), Yield(COMPLETED), Complete(None)]), {})

    seen: Final = [(event, stream.completed_response) for event in stream]

    assert seen == [(DELTA, None), (COMPLETED, COMPLETED)]


@pytest.mark.parametrize(
    ("event", "expected"), ((COMPLETED, True), (INCOMPLETE, True), (DELTA, False), (b"raw", False))
)
def test_is_terminal_event_recognises_only_the_terminal_events(event: object, expected: bool) -> None:
    assert streams.is_terminal_event(event) is expected
