from collections.abc import Iterable, Iterator, Sequence
from typing import Final

import httpcore
import pytest
from h2.config import H2Configuration
from h2.connection import H2Connection
from h2.errors import ErrorCodes
from h2.events import ConnectionTerminated
from h2.exceptions import NoAvailableStreamIDError
from httpcore._backends.base import SOCKET_OPTION
from hyperframe.frame import GoAwayFrame, PingFrame
from typing_extensions import override

from litellm.llms.custom_httpx.http2_graceful_goaway import (
    DrainingH2Connection,
    drain_http2_connections_on_graceful_goaway,
)

URL: Final = "https://search.example/v1/search"
ANSWER: Final = b'{"results": [{"id": "answered"}]}'
LAST_POSSIBLE_STREAM_ID: Final = 2**31 - 1
REQUEST_HEADERS: Final = (
    (":method", "POST"),
    (":scheme", "https"),
    (":authority", "search.example"),
    (":path", "/v1/search"),
)


def _goaway(last_stream_id: int, error_code: ErrorCodes = ErrorCodes.NO_ERROR) -> bytes:
    return GoAwayFrame(stream_id=0, last_stream_id=last_stream_id, error_code=error_code).serialize()


def _two_step_goaway() -> tuple[bytes, ...]:
    return (
        _goaway(LAST_POSSIBLE_STREAM_ID) + PingFrame(stream_id=0, opaque_data=b"drain-01").serialize(),
        _goaway(1),
    )


class _Peer:
    def __init__(self) -> None:
        client: Final = H2Connection()
        client.initiate_connection()
        client.send_headers(1, REQUEST_HEADERS, end_stream=True)
        self._server: Final = H2Connection(config=H2Configuration(client_side=False))
        self._server.initiate_connection()
        _ = self._server.receive_data(client.data_to_send())

    def preface(self) -> bytes:
        return self._server.data_to_send()

    def answer_headers(self) -> bytes:
        self._server.send_headers(1, [(":status", "200"), ("content-type", "application/json")])
        return self._server.data_to_send()

    def answer_body(self) -> bytes:
        self._server.send_data(1, ANSWER, end_stream=True)
        return self._server.data_to_send()


def _script(*middle: bytes, answered: bool = True) -> tuple[bytes, ...]:
    peer: Final = _Peer()
    preface: Final = peer.preface()
    answer: Final = (peer.answer_headers(), peer.answer_body()) if answered else ()
    return (preface, *middle, *answer)


@pytest.fixture(autouse=True)
def _draining_connections() -> Iterator[None]:
    drain_http2_connections_on_graceful_goaway()
    yield


class _ResetAfterScriptStream(httpcore.AsyncMockStream):
    @override
    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        chunk: Final = await super().read(max_bytes, timeout)
        if chunk:
            return chunk
        raise httpcore.ReadError("Connection reset by peer")


class _ResetAfterScriptBackend(httpcore.AsyncMockBackend):
    def __init__(self, script: Sequence[bytes]) -> None:
        super().__init__(list(script), http2=True)
        self._script: Final = script

    @override
    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        return _ResetAfterScriptStream(list(self._script), http2=True)


async def _async_search(script: Sequence[bytes], backend: httpcore.AsyncNetworkBackend | None = None) -> bytes:
    async with httpcore.AsyncConnectionPool(
        http2=True, network_backend=backend or httpcore.AsyncMockBackend(list(script), http2=True)
    ) as pool:
        response: Final = await pool.request("POST", URL, content=b"{}")
        return response.content


def _sync_search(script: Sequence[bytes]) -> bytes:
    with httpcore.ConnectionPool(http2=True, network_backend=httpcore.MockBackend(list(script), http2=True)) as pool:
        return pool.request("POST", URL, content=b"{}").content


@pytest.mark.parametrize(
    "goaway",
    [pytest.param(_two_step_goaway(), id="google-two-step"), pytest.param((_goaway(1),), id="single")],
)
async def test_a_search_sent_before_a_graceful_goaway_gets_its_answer(goaway: tuple[bytes, ...]) -> None:
    assert await _async_search(_script(*goaway)) == ANSWER
    assert _sync_search(_script(*goaway)) == ANSWER


async def test_a_draining_connection_takes_no_new_requests() -> None:
    async with httpcore.AsyncConnectionPool(
        http2=True, network_backend=httpcore.AsyncMockBackend(list(_script(*_two_step_goaway())), http2=True)
    ) as pool:
        async with pool.stream("POST", URL, content=b"{}") as response:
            assert not pool.connections[0].is_available()
            assert await response.aread() == ANSWER


def test_a_client_that_saw_a_graceful_goaway_opens_no_new_stream() -> None:
    client: Final = DrainingH2Connection(config=H2Configuration())
    client.initiate_connection()
    client.send_headers(client.get_next_available_stream_id(), REQUEST_HEADERS, end_stream=True)
    _ = client.receive_data(_Peer().preface() + _goaway(1))

    with pytest.raises(NoAvailableStreamIDError):
        _ = client.get_next_available_stream_id()


def _terminated_by(error: httpcore.RemoteProtocolError) -> ConnectionTerminated:
    terminated: Final = error.args[0]
    assert isinstance(terminated, ConnectionTerminated)
    return terminated


async def test_a_server_that_hangs_up_after_a_graceful_goaway_reports_the_goaway() -> None:
    with pytest.raises(httpcore.RemoteProtocolError) as async_error:
        _ = await _async_search(_script(*_two_step_goaway(), answered=False))
    with pytest.raises(httpcore.RemoteProtocolError) as sync_error:
        _ = _sync_search(_script(*_two_step_goaway(), answered=False))

    assert _terminated_by(async_error.value).last_stream_id == 1
    assert _terminated_by(sync_error.value).last_stream_id == 1


async def test_a_server_that_resets_the_connection_after_a_graceful_goaway_reports_the_goaway() -> None:
    script: Final = _script(*_two_step_goaway(), answered=False)

    with pytest.raises(httpcore.RemoteProtocolError) as error:
        _ = await _async_search(script, _ResetAfterScriptBackend(script))

    assert _terminated_by(error.value).last_stream_id == 1


async def test_an_error_goaway_ends_the_stream_without_waiting_for_an_answer() -> None:
    with pytest.raises(httpcore.RemoteProtocolError) as error:
        _ = await _async_search(_script(_goaway(1, ErrorCodes.INTERNAL_ERROR)))

    assert _terminated_by(error.value).error_code == ErrorCodes.INTERNAL_ERROR
