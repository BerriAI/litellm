import asyncio
from typing import Final

import pytest
from starlette.websockets import WebSocketDisconnect

from litellm.proxy.realtime_endpoints.sideband import SidebandClientWebSocket, realtime_call_id


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
