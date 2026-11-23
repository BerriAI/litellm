from typing import Final

import anyio
import pytest

from litellm.proxy.mcp_server.events.frames import Overflow, StreamFrame, StreamRoute


def _frame(cursor: str) -> StreamFrame:
    return StreamFrame("notifications/events/event", {"cursor": cursor})


def _drain(route: StreamRoute) -> tuple[StreamFrame | Overflow, ...]:
    def items():  # noqa: ANN202  # local generator
        while True:
            try:
                yield route.receive.receive_nowait()
            except (anyio.EndOfStream, anyio.WouldBlock):
                return

    return tuple(items())


def test_a_full_route_reports_dropped_frames_as_one_overflow_before_the_next_frame() -> None:
    route: Final = StreamRoute()
    for index in range(259):
        route.deliver(_frame(str(index)))
    assert route.receive.receive_nowait() == _frame("0")
    route.deliver(_frame("next"))
    drained: Final = _drain(route)
    assert drained[-1] == Overflow(cursor="258")
    assert _frame("next") not in drained and _frame("257") not in drained
    route.deliver(_frame("after"))
    assert _drain(route) == (Overflow(cursor="next"), _frame("after"))


def test_an_overflow_still_pending_at_the_end_is_flushed_when_the_route_finishes() -> None:
    route: Final = StreamRoute()
    for index in range(257):
        route.deliver(_frame(str(index)))
    route.receive.receive_nowait()
    route.finish(None)
    assert _drain(route)[-1] == Overflow(cursor="256")


@pytest.mark.asyncio
async def test_an_overflow_that_cannot_fit_at_the_end_is_still_read_after_the_last_frame() -> None:
    route: Final = StreamRoute()
    for index in range(257):
        route.deliver(_frame(str(index)))
    route.finish(None)
    reader: Final = route.reader()
    received: Final = [await reader.receive() for _ in range(257)]
    assert received[-2:] == [_frame("255"), Overflow(cursor="256")]
    with pytest.raises(anyio.EndOfStream):
        await reader.receive()


@pytest.mark.asyncio
async def test_a_terminated_notice_survives_a_full_buffer_and_arrives_after_the_overflow() -> None:
    route: Final = StreamRoute()
    for index in range(257):
        route.deliver(_frame(str(index)))
    terminated: Final = StreamFrame(
        "notifications/events/terminated", {"error": {"code": -32024, "message": "Forbidden"}}
    )
    route.deliver(terminated)
    route.finish(None)
    reader: Final = route.reader()
    received: Final = [await reader.receive() for _ in range(258)]
    assert received[-2:] == [Overflow(cursor="256"), terminated]
    with pytest.raises(anyio.EndOfStream):
        await reader.receive()


@pytest.mark.asyncio
async def test_draining_a_full_buffer_releases_the_overflow_without_new_upstream_frames() -> None:
    route: Final = StreamRoute()
    for index in range(257):
        route.deliver(_frame(str(index)))
    reader: Final = route.reader()
    received: Final = [await reader.receive() for _ in range(257)]
    assert received[-1] == Overflow(cursor="256")


@pytest.mark.asyncio
async def test_a_reader_that_falls_too_far_behind_is_terminated_with_bounded_memory() -> None:
    route: Final = StreamRoute()
    for index in range(256):
        route.deliver(StreamFrame("notifications/events/event", {"cursor": str(index)}))
    for index in range(500):
        route.deliver(StreamFrame("notifications/events/error", {"error": {"code": -32603, "message": str(index)}}))
    route.finish(None)
    methods: Final = [frame.method if isinstance(frame, StreamFrame) else "overflow" async for frame in route.reader()]
    assert len(methods) == 256 + 32 + 1
    assert methods[-1] == "notifications/events/terminated"


def test_a_new_overflow_with_no_cursor_does_not_reuse_a_previous_overflow_cursor() -> None:
    route: Final = StreamRoute()
    for index in range(257):
        route.deliver(_frame(str(index)))
    route.receive.receive_nowait()
    route.deliver(StreamFrame("notifications/events/heartbeat", {}))
    assert _drain(route)[-1] == Overflow(cursor="256")
    route.finish(None)
    assert _drain(route) == (Overflow(cursor=None),)
