"""The Responses stream wrappers the native route hands its callers.

The Rust driver binds this module: `drive` and `SyncStream` wrap each execution, and every
lifecycle step class is re-exported so the driver finds them here. The wrappers track the
terminal event's response as `completed_response`, which the proxy reads off a Responses
stream before the terminal chunk goes out.
"""

from __future__ import annotations

from typing import Final

from litellm.rust_bridge import lifecycle, streams
from litellm.rust_bridge.lifecycle import Execution
from litellm.types.llms.openai import ResponseCompletedEvent, ResponseFailedEvent, ResponseIncompleteEvent

Await: Final = streams.Await
Complete: Final = streams.Complete
Keepalive: Final = streams.Keepalive
Open: Final = streams.Open
Yield: Final = streams.Yield

TERMINAL_EVENTS: Final = (ResponseCompletedEvent, ResponseIncompleteEvent, ResponseFailedEvent)


def is_terminal_event(event: object) -> bool:
    return isinstance(event, TERMINAL_EVENTS)


async def drive(execution: Execution) -> object:
    return await lifecycle.drive(execution, Stream)


class Stream(streams.Stream):
    def __init__(self, execution: Execution, hidden_params: object = None) -> None:
        super().__init__(execution, hidden_params)
        self.completed_response: object | None = None

    async def __anext__(self) -> object:
        event: Final = await super().__anext__()
        self._note(event)
        return event

    def _note(self, event: object) -> None:
        if isinstance(event, TERMINAL_EVENTS):
            self.completed_response = event


class SyncStream(streams.SyncStream):
    def __init__(self, execution: Execution, hidden_params: object = None) -> None:
        super().__init__(execution, hidden_params)
        self.completed_response: object | None = None

    def __next__(self) -> object:
        event: Final = super().__next__()
        if isinstance(event, TERMINAL_EVENTS):
            self.completed_response = event
        return event
