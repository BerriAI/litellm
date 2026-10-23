from collections.abc import Mapping
from types import TracebackType
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.llms.elevenlabs.realtime.handler import ElevenLabsRealtime
from litellm.types.realtime import RealtimeQueryParams


@pytest.mark.parametrize(
    ("api_base", "expected_url"),
    [
        (
            "https://api.elevenlabs.io",
            "wss://api.elevenlabs.io/v1/convai/realtime?model=eleven_realtime_v1_mini",
        ),
        (
            "https://api.elevenlabs.io/",
            "wss://api.elevenlabs.io/v1/convai/realtime?model=eleven_realtime_v1_mini",
        ),
        (
            "http://127.0.0.1:8081",
            "ws://127.0.0.1:8081/v1/convai/realtime?model=eleven_realtime_v1_mini",
        ),
    ],
)
def test_construct_url_targets_the_convai_realtime_route(api_base: str, expected_url: str) -> None:
    handler: Final = ElevenLabsRealtime()

    url: Final = handler.construct_url(api_base=api_base, query_params={"model": "eleven_realtime_v1_mini"})

    assert url == expected_url


def test_construct_url_forwards_every_client_query_param() -> None:
    handler: Final = ElevenLabsRealtime()
    query_params: Final[RealtimeQueryParams] = {"model": "eleven_realtime_v1_max", "intent": "chat"}

    url: Final = handler.construct_url(api_base="https://api.elevenlabs.io", query_params=query_params)

    assert url == "wss://api.elevenlabs.io/v1/convai/realtime?model=eleven_realtime_v1_max&intent=chat"


@pytest.mark.parametrize("openai_beta_realtime", [False, True])
def test_backend_handshake_authenticates_with_xi_api_key_only(openai_beta_realtime: bool) -> None:
    handler: Final = ElevenLabsRealtime()

    headers: Final = handler._get_additional_headers("xi-secret", openai_beta_realtime=openai_beta_realtime)

    assert headers == {"xi-api-key": "xi-secret"}


class _ConnectRecordingTheHandshake:
    def __init__(self) -> None:
        self.url: str | None = None
        self.additional_headers: Mapping[str, str] | None = None
        self.ssl: object = None

    def __call__(
        self, url: str, *, additional_headers: Mapping[str, str], max_size: int, ssl: object
    ) -> "_ConnectRecordingTheHandshake":
        self.url = url
        self.additional_headers = additional_headers
        self.ssl = ssl
        return self

    async def __aenter__(self) -> AsyncMock:
        return AsyncMock()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


@pytest.mark.asyncio
async def test_async_realtime_opens_the_elevenlabs_socket_with_the_vendor_key() -> None:
    handler: Final = ElevenLabsRealtime()
    client_websocket: Final = MagicMock()
    client_websocket.scope = {"headers": [(b"openai-beta", b"realtime=v1")]}
    connect: Final = _ConnectRecordingTheHandshake()

    with (
        patch("websockets.connect", connect),
        patch("litellm.llms.openai.realtime.handler.RealTimeStreaming") as streaming_cls,
    ):
        streaming_cls.return_value.bidirectional_forward = AsyncMock()
        await handler.async_realtime(
            model="eleven_realtime_v1_mini",
            websocket=client_websocket,
            logging_obj=MagicMock(),
            api_key="xi-secret",
            query_params={"model": "eleven_realtime_v1_mini"},
        )

    assert connect.url == "wss://api.elevenlabs.io/v1/convai/realtime?model=eleven_realtime_v1_mini"
    assert connect.additional_headers == {"xi-api-key": "xi-secret"}
    assert connect.ssl not in (None, False)
    streaming_cls.return_value.bidirectional_forward.assert_awaited_once()
