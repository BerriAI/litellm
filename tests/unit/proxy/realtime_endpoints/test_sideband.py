import asyncio
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
    client: Final = SidebandClientWebSocket("/v1/realtime")

    async def _session() -> None:
        if first_frame is not None:
            await client.send_text(first_frame)
        if reads_its_client:
            with pytest.raises(WebSocketDisconnect):
                await client.receive_text()

    assert await join_sideband(client, _session()) is joined
    await client.close()


@pytest.mark.asyncio
async def test_a_sideband_that_never_reaches_the_provider_is_given_up_on_and_stopped() -> None:
    client: Final = SidebandClientWebSocket("/v1/realtime")
    outcome: Final[asyncio.Future[str]] = asyncio.get_running_loop().create_future()

    async def _session() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            outcome.set_result("stopped")
            raise

    assert await join_sideband(client, _session(), timeout=0) is False
    with pytest.raises(WebSocketDisconnect):
        await client.receive_text()
    assert await asyncio.wait_for(outcome, 1) == "stopped"
