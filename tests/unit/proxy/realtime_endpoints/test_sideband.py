import asyncio
import json
from typing import Final

import pytest
from starlette.websockets import WebSocketDisconnect

from litellm.proxy.realtime_endpoints.sideband import SidebandClientWebSocket, join_sideband, realtime_call_id


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("/v1/realtime/calls/rtc_abc123", "rtc_abc123"),
        ("https://api.openai.com/v1/realtime/calls/rtc_abc123/", "rtc_abc123"),
        ("/v1/realtime/calls/other_abc123", None),
        ("", None),
        (None, None),
    ],
)
def test_realtime_call_id_reads_only_a_call_id_from_the_location_header(
    location: str | None, expected: str | None
) -> None:
    assert realtime_call_id(location) == expected


@pytest.mark.asyncio
async def test_sideband_client_never_speaks_and_disconnects_once_the_session_closes_it() -> None:
    client: Final = SidebandClientWebSocket("/v1/realtime")
    pending_receive: Final = asyncio.create_task(client.receive_text())
    await client.send_text('{"type":"response.done"}')
    await asyncio.sleep(0)
    assert not pending_receive.done(), "the proxy's sideband must stay silent while the call runs"

    await client.close()

    with pytest.raises(WebSocketDisconnect):
        await pending_receive


_NEVER: Final = 3600.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_frame", "reads_its_client", "joined"),
    [
        (None, True, True),
        ('{"type":"session.created"}', False, True),
        ('{"type":"error","error":{"message":"rate limited"}}', True, False),
        (None, False, False),
    ],
    ids=["relay-started", "first-session-event", "route-refused-the-session", "ended-before-the-relay"],
)
async def test_a_sideband_has_joined_once_the_relay_to_the_provider_runs(
    first_frame: str | None, reads_its_client: bool, joined: bool
) -> None:
    async def _session(client: SidebandClientWebSocket, _metered_seconds: float) -> None:
        if first_frame is not None:
            await client.send_text(first_frame)
        if reads_its_client:
            with pytest.raises(WebSocketDisconnect):
                await client.receive_text()

    assert await join_sideband("/v1/realtime", _session, rejoin_delays=(_NEVER,)) is joined


@pytest.mark.asyncio
async def test_a_sideband_that_never_reaches_the_provider_is_given_up_on_and_stopped() -> None:
    outcome: Final[asyncio.Future[str]] = asyncio.get_running_loop().create_future()

    async def _session(_client: SidebandClientWebSocket, _metered_seconds: float) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            outcome.set_result("stopped")
            raise

    assert await join_sideband("/v1/realtime", _session, timeout=0, wind_down=0) is False
    assert await asyncio.wait_for(outcome, 1) == "stopped"


@pytest.mark.asyncio
async def test_a_refused_sideband_finishes_releasing_its_holds_before_it_is_stopped() -> None:
    outcome: Final[asyncio.Future[str]] = asyncio.get_running_loop().create_future()

    async def _session(client: SidebandClientWebSocket, _metered_seconds: float) -> None:
        try:
            await client.send_text(json.dumps({"type": "error", "error": {"message": "rate limited"}}))
            await asyncio.sleep(0.05)
            outcome.set_result("released")
        except asyncio.CancelledError:
            outcome.set_result("stopped")
            raise

    assert await join_sideband("/v1/realtime", _session) is False
    assert await asyncio.wait_for(outcome, 1) == "released"


_DROPPED: Final = ('{"type":"session.created"}',)
_REFUSED: Final = ('{"type":"error","error":{"message":"Upstream realtime handshake rejected with HTTP 404"}}',)
_RATE_LIMITED: Final = ('{"type":"error","error":{"message":"Upstream realtime handshake rejected with HTTP 429"}}',)
_UNREACHABLE: Final = ('{"type":"error","error":{"message":"Cannot connect to host"}}',)
_NOT_ADMITTED: Final = ('{"type":"error","error":{"message":"Rate limit exceeded","code":"session_not_admitted"}}',)
_CLOSED_BY_PROVIDER: Final = ('{"type":"session.started"}', '{"type":"session.closed"}')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sessions", "sessions_opened"),
    [
        ((_DROPPED, _DROPPED, _CLOSED_BY_PROVIDER, _DROPPED), 3),
        ((_DROPPED, _REFUSED, _DROPPED), 2),
        ((_CLOSED_BY_PROVIDER, _DROPPED), 1),
        ((_DROPPED, _RATE_LIMITED, _UNREACHABLE, _DROPPED, _CLOSED_BY_PROVIDER), 5),
        ((_DROPPED, _UNREACHABLE, _UNREACHABLE, _UNREACHABLE, _UNREACHABLE, _UNREACHABLE, _DROPPED), 6),
        ((_DROPPED, _NOT_ADMITTED, _NOT_ADMITTED, _NOT_ADMITTED, _DROPPED), 4),
        ((_DROPPED, _NOT_ADMITTED, _UNREACHABLE, _NOT_ADMITTED, _DROPPED, _CLOSED_BY_PROVIDER), 6),
    ],
    ids=[
        "rejoined-after-each-drop-until-the-session-closes",
        "call-gone-at-the-provider-is-not-tried-again",
        "session-closed",
        "a-rejoin-that-fails-is-tried-again",
        "given-up-on-after-the-last-delay",
        "given-up-on-after-the-proxy-refused-three-rejoins",
        "a-rejoin-the-proxy-refused-twice-still-joins",
    ],
)
async def test_a_joined_sideband_is_reopened_when_it_drops_while_the_call_still_runs(
    sessions: tuple[tuple[str, ...], ...], sessions_opened: int
) -> None:
    remaining: Final = iter(sessions)
    opened: Final[list[str]] = []
    idle: Final = asyncio.Event()

    async def _session(client: SidebandClientWebSocket, _metered_seconds: float) -> None:
        frames: Final = next(remaining)
        opened.append(client.url.path)
        for frame in frames:
            await client.send_text(frame)
        if len(opened) == sessions_opened:
            idle.set()

    assert await join_sideband("/v1/realtime", _session, rejoin_delays=(0, 0, 0, 0, 0)) is True
    await asyncio.wait_for(idle.wait(), 1)
    for _ in range(20):
        await asyncio.sleep(0)

    assert opened == ["/v1/realtime"] * sessions_opened


@pytest.mark.asyncio
async def test_a_rejoined_session_is_told_the_seconds_the_sessions_before_it_billed() -> None:
    remaining: Final = iter(((_DROPPED, 6.5), (_DROPPED, 0.0), (_DROPPED, 9.25), (_CLOSED_BY_PROVIDER, 4.0)))
    told: Final[list[float]] = []
    idle: Final = asyncio.Event()

    async def _session(client: SidebandClientWebSocket, metered_seconds: float) -> None:
        frames, billed_seconds = next(remaining)
        told.append(metered_seconds)
        for frame in frames:
            await client.send_text(frame)
        client.bill_seconds(billed_seconds)
        if client.session_ended:
            idle.set()

    assert await join_sideband("/v1/live/sessions", _session, rejoin_delays=(0, 0, 0)) is True
    await asyncio.wait_for(idle.wait(), 1)

    assert told == [0.0, 6.5, 6.5, 15.75]


@pytest.mark.asyncio
async def test_a_sideband_given_up_on_while_it_connects_is_stopped_before_it_can_join_and_bill() -> None:
    outcome: Final[asyncio.Future[str]] = asyncio.get_running_loop().create_future()

    async def _session(client: SidebandClientWebSocket, _metered_seconds: float) -> None:
        try:
            await asyncio.sleep(0.05)
            await client.send_text('{"type":"session.started"}')
            outcome.set_result("joined late")
        except asyncio.CancelledError:
            outcome.set_result("stopped")
            raise

    assert await join_sideband("/v1/live/sessions", _session, timeout=0) is False
    assert await asyncio.wait_for(outcome, 1) == "stopped"
