from functools import cache
from typing import Final

import httpcore
import httpcore._async.http2
import httpcore._sync.http2
from h2.config import H2Configuration
from h2.connection import H2Connection
from h2.errors import ErrorCodes
from h2.events import ConnectionTerminated, Event
from h2.exceptions import NoAvailableStreamIDError
from h2.stream import StreamState
from hyperframe.frame import Frame, GoAwayFrame
from typing_extensions import override

_AWAITING_AN_ANSWER: Final = frozenset({StreamState.OPEN, StreamState.HALF_CLOSED_LOCAL})


def _terminated_by(goaway: GoAwayFrame) -> ConnectionTerminated:
    event: Final = ConnectionTerminated()
    event.error_code = ErrorCodes.NO_ERROR
    event.last_stream_id = goaway.last_stream_id
    event.additional_data = goaway.additional_data or None
    return event


class DrainingH2Connection(H2Connection):
    def __init__(self, config: H2Configuration) -> None:
        super().__init__(config=config)
        self._graceful_goaway: GoAwayFrame | None = None
        self._goaway_reported = False

    @property
    def draining(self) -> bool:
        return self._graceful_goaway is not None

    def report_goaway(self) -> ConnectionTerminated | None:
        goaway: Final = self._graceful_goaway
        if goaway is None or self._goaway_reported:
            return None
        self._goaway_reported = True
        return _terminated_by(goaway)

    @override
    def get_next_available_stream_id(self) -> int:
        if self.draining:
            raise NoAvailableStreamIDError("The server is draining this connection with a GOAWAY")
        return super().get_next_available_stream_id()

    @override
    def _receive_goaway_frame(self, frame: GoAwayFrame) -> tuple[list[Frame], list[Event]]:  # mutable-ok: h2 API
        if frame.error_code != ErrorCodes.NO_ERROR or self._goaway_reported:
            self._goaway_reported = True
            return super()._receive_goaway_frame(frame)
        self._graceful_goaway = frame
        return [], []

    @override
    def _receive_frame(self, frame: Frame) -> list[Event]:  # mutable-ok: h2 API
        events: Final = super()._receive_frame(frame)
        goaway: Final = self._graceful_goaway
        if goaway is None or self._awaits_an_answer_up_to(goaway.last_stream_id):
            return events
        report: Final = self.report_goaway()
        return events if report is None else [*events, report]

    def _awaits_an_answer_up_to(self, last_stream_id: int) -> bool:
        return any(
            stream_id <= last_stream_id and stream.state_machine.state in _AWAITING_AN_ANSWER
            for stream_id, stream in self.streams.items()
        )


class _DrainingAsyncHTTP2Connection(httpcore.AsyncHTTP2Connection):
    def __init__(
        self, origin: httpcore.Origin, stream: httpcore.AsyncNetworkStream, keepalive_expiry: float | None = None
    ) -> None:
        super().__init__(origin=origin, stream=stream, keepalive_expiry=keepalive_expiry)
        self._draining_h2_state: Final = DrainingH2Connection(config=self.CONFIG)
        self._h2_state = self._draining_h2_state

    @override
    async def _receive_events(self, request: httpcore.Request, stream_id: int | None = None) -> None:
        try:
            await super()._receive_events(request, stream_id)
        except (httpcore.RemoteProtocolError, httpcore.NetworkError):
            if stream_id is not None and self._events.get(stream_id):
                return
            goaway: Final = self._draining_h2_state.report_goaway()
            if goaway is None:
                raise
            self._connection_terminated = goaway
            await super()._receive_events(request, stream_id)

    @override
    def is_available(self) -> bool:
        return super().is_available() and not self._draining_h2_state.draining


class _DrainingHTTP2Connection(httpcore.HTTP2Connection):
    def __init__(
        self, origin: httpcore.Origin, stream: httpcore.NetworkStream, keepalive_expiry: float | None = None
    ) -> None:
        super().__init__(origin=origin, stream=stream, keepalive_expiry=keepalive_expiry)
        self._draining_h2_state: Final = DrainingH2Connection(config=self.CONFIG)
        self._h2_state = self._draining_h2_state

    @override
    def _receive_events(self, request: httpcore.Request, stream_id: int | None = None) -> None:
        try:
            super()._receive_events(request, stream_id)
        except (httpcore.RemoteProtocolError, httpcore.NetworkError):
            if stream_id is not None and self._events.get(stream_id):
                return
            goaway: Final = self._draining_h2_state.report_goaway()
            if goaway is None:
                raise
            self._connection_terminated = goaway
            super()._receive_events(request, stream_id)

    @override
    def is_available(self) -> bool:
        return super().is_available() and not self._draining_h2_state.draining


@cache
def drain_http2_connections_on_graceful_goaway() -> None:
    httpcore._async.http2.AsyncHTTP2Connection = _DrainingAsyncHTTP2Connection
    httpcore._sync.http2.HTTP2Connection = _DrainingHTTP2Connection
