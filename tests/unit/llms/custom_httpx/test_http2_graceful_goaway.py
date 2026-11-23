import asyncio
from collections.abc import Iterable, Iterator, Sequence
from typing import Final

import httpcore
import httpcore._async.http2
import httpcore._sync.http2
import pytest
from h2.config import H2Configuration
from h2.connection import H2Connection
from h2.errors import ErrorCodes
from h2.events import ConnectionTerminated, StreamEnded
from h2.exceptions import NoAvailableStreamIDError
from httpcore._backends.base import SOCKET_OPTION
from hyperframe.frame import GoAwayFrame, PingFrame, RstStreamFrame, WindowUpdateFrame
from typing_extensions import override

from litellm.llms.custom_httpx.http2_graceful_goaway import (
    DrainingH2Connection,
    drain_http2_connections_on_graceful_goaway,
)

URL: Final = "https://search.example/v1/search"
ANSWER: Final = b'{"results": [{"id": "answered"}]}'
LAST_POSSIBLE_STREAM_ID: Final = 2**31 - 1
UPLOAD_WINDOW: Final = H2Connection().remote_settings.initial_window_size
REQUEST_HEADERS: Final = (
    (":method", "POST"),
    (":scheme", "https"),
    (":authority", "search.example"),
    (":path", "/v1/search"),
)


def _goaway(last_stream_id: int, error_code: ErrorCodes = ErrorCodes.NO_ERROR) -> bytes:
    return GoAwayFrame(stream_id=0, last_stream_id=last_stream_id, error_code=error_code).serialize()


def _reset(stream_id: int) -> bytes:
    return RstStreamFrame(stream_id=stream_id, error_code=ErrorCodes.CANCEL).serialize()


def _window_update(stream_id: int) -> bytes:
    return WindowUpdateFrame(stream_id=stream_id, window_increment=UPLOAD_WINDOW).serialize()


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
def _draining_connections(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(httpcore._async.http2, "AsyncHTTP2Connection", httpcore.AsyncHTTP2Connection)
    monkeypatch.setattr(httpcore._sync.http2, "HTTP2Connection", httpcore.HTTP2Connection)
    drain_http2_connections_on_graceful_goaway.cache_clear()
    drain_http2_connections_on_graceful_goaway()
    yield
    drain_http2_connections_on_graceful_goaway.cache_clear()


class _FailAfterScriptStream(httpcore.AsyncMockStream):
    def __init__(self, script: Sequence[bytes], error: Exception) -> None:
        super().__init__(list(script), http2=True)
        self._error: Final = error

    @override
    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        chunk: Final = await super().read(max_bytes, timeout)
        if chunk:
            return chunk
        raise self._error


class _FailAfterScriptBackend(httpcore.AsyncMockBackend):
    def __init__(self, script: Sequence[bytes], error: Exception) -> None:
        super().__init__(list(script), http2=True)
        self._script: Final = script
        self._error: Final = error

    @override
    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        return _FailAfterScriptStream(self._script, self._error)


class _FailAfterScriptSyncStream(httpcore.MockStream):
    def __init__(self, script: Sequence[bytes], error: Exception) -> None:
        super().__init__(list(script), http2=True)
        self._error: Final = error

    @override
    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        chunk: Final = super().read(max_bytes, timeout)
        if chunk:
            return chunk
        raise self._error


class _FailAfterScriptSyncBackend(httpcore.MockBackend):
    def __init__(self, script: Sequence[bytes], error: Exception) -> None:
        super().__init__(list(script), http2=True)
        self._script: Final = script
        self._error: Final = error

    @override
    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        return _FailAfterScriptSyncStream(self._script, self._error)


class _OneConnectionBackend(httpcore.AsyncMockBackend):
    def __init__(self, script: Sequence[bytes]) -> None:
        super().__init__(list(script), http2=True)
        self._connected = False

    @override
    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        if self._connected:
            raise httpcore.ConnectError("the search opened a second connection")
        self._connected = True
        return await super().connect_tcp(host, port, timeout, local_address, socket_options)


class _OneConnectionSyncBackend(httpcore.MockBackend):
    def __init__(self, script: Sequence[bytes]) -> None:
        super().__init__(list(script), http2=True)
        self._connected = False

    @override
    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        if self._connected:
            raise httpcore.ConnectError("the search opened a second connection")
        self._connected = True
        return super().connect_tcp(host, port, timeout, local_address, socket_options)


async def _async_search(
    script: Sequence[bytes], backend: httpcore.AsyncNetworkBackend | None = None, content: bytes = b"{}"
) -> bytes:
    async with httpcore.AsyncConnectionPool(
        http2=True, network_backend=backend or httpcore.AsyncMockBackend(list(script), http2=True)
    ) as pool:
        response: Final = await pool.request("POST", URL, content=content)
        return response.content


def _sync_search(script: Sequence[bytes], backend: httpcore.NetworkBackend | None = None) -> bytes:
    with httpcore.ConnectionPool(
        http2=True, network_backend=backend or httpcore.MockBackend(list(script), http2=True)
    ) as pool:
        return pool.request("POST", URL, content=b"{}").content


@pytest.mark.parametrize(
    "goaway",
    [pytest.param(_two_step_goaway(), id="google-two-step"), pytest.param((_goaway(1),), id="single")],
)
async def test_a_search_sent_before_a_graceful_goaway_gets_its_answer(goaway: tuple[bytes, ...]) -> None:
    assert await _async_search(_script(*goaway)) == ANSWER
    assert _sync_search(_script(*goaway)) == ANSWER


def _refusing_script() -> list[bytes]:
    peer: Final = _Peer()
    return [peer.preface(), peer.answer_headers(), _goaway(1), peer.answer_body()]


async def test_a_search_a_graceful_goaway_refused_is_answered_while_the_accepted_one_still_streams() -> None:
    async with httpcore.AsyncConnectionPool(
        http2=True, network_backend=httpcore.AsyncMockBackend(_refusing_script(), http2=True)
    ) as async_pool:
        async with async_pool.stream("POST", URL, content=b"{}") as accepted:
            refused: Final = await async_pool.request("POST", URL, content=b"{}")
            assert (refused.content, await accepted.aread()) == (ANSWER, ANSWER)

    with httpcore.ConnectionPool(
        http2=True, network_backend=httpcore.MockBackend(_refusing_script(), http2=True)
    ) as sync_pool:
        with sync_pool.stream("POST", URL, content=b"{}") as sync_accepted:
            sync_refused: Final = sync_pool.request("POST", URL, content=b"{}")
            assert (sync_refused.content, sync_accepted.read()) == (ANSWER, ANSWER)


async def test_a_search_still_uploading_at_a_graceful_goaway_gets_its_answer() -> None:
    peer: Final = _Peer()
    script: Final = (
        peer.preface(),
        _goaway(1),
        _window_update(0) + _window_update(1),
        peer.answer_headers(),
        peer.answer_body(),
    )

    assert await _async_search(script, content=b"x" * (UPLOAD_WINDOW + 1)) == ANSWER


class _AnswersOnceBothSearchesArriveStream(httpcore.AsyncMockStream):
    def __init__(self) -> None:
        super().__init__([], http2=True)
        self._server: Final = H2Connection(config=H2Configuration(client_side=False))
        self._server.initiate_connection()
        self._both_arrived: Final = asyncio.Event()
        self._arrived = 0
        self._answered = False

    @override
    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self._arrived += sum(isinstance(event, StreamEnded) for event in self._server.receive_data(buffer))
        if self._arrived == 2:
            self._both_arrived.set()

    @override
    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        pending: Final = self._server.data_to_send()
        if pending or self._answered:
            return pending
        await self._both_arrived.wait()
        self._answered = True
        return _goaway(3) + b"".join(self._answer(stream_id) for stream_id in (1, 3))

    def _answer(self, stream_id: int) -> bytes:
        self._server.send_headers(stream_id, [(":status", "200"), ("content-type", "application/json")])
        self._server.send_data(stream_id, ANSWER, end_stream=True)
        return self._server.data_to_send()


class _AnswersOnceBothSearchesArriveBackend(httpcore.AsyncMockBackend):
    def __init__(self) -> None:
        super().__init__([], http2=True)

    @override
    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        return _AnswersOnceBothSearchesArriveStream()


async def test_two_searches_answered_in_one_read_with_a_graceful_goaway_both_get_their_answers() -> None:
    async with httpcore.AsyncConnectionPool(
        http2=True, network_backend=_AnswersOnceBothSearchesArriveBackend()
    ) as pool:
        responses: Final = await asyncio.gather(*(pool.request("POST", URL, content=b"{}") for _ in range(2)))

    assert [response.content for response in responses] == [ANSWER, ANSWER]


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


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(httpcore.ReadError("Connection reset by peer"), id="reset"),
        pytest.param(httpcore.ReadTimeout("The read operation timed out"), id="silent"),
    ],
)
async def test_a_server_that_resets_or_goes_silent_after_a_graceful_goaway_reports_the_goaway(
    error: Exception,
) -> None:
    script: Final = _script(*_two_step_goaway(), answered=False)

    with pytest.raises(httpcore.RemoteProtocolError) as async_error:
        _ = await _async_search(script, _FailAfterScriptBackend(script, error))
    with pytest.raises(httpcore.RemoteProtocolError) as sync_error:
        _ = _sync_search(script, _FailAfterScriptSyncBackend(script, error))

    assert _terminated_by(async_error.value).last_stream_id == 1
    assert _terminated_by(sync_error.value).last_stream_id == 1


@pytest.mark.parametrize("last_stream_id", [LAST_POSSIBLE_STREAM_ID, 1])
async def test_a_search_reset_after_a_graceful_goaway_reports_the_goaway(last_stream_id: int) -> None:
    script: Final = _script(_goaway(last_stream_id), _reset(1), answered=False)

    with pytest.raises(httpcore.RemoteProtocolError) as async_error:
        _ = await _async_search(script)
    with pytest.raises(httpcore.RemoteProtocolError) as sync_error:
        _ = _sync_search(script)

    assert _terminated_by(async_error.value).last_stream_id == last_stream_id
    assert _terminated_by(sync_error.value).last_stream_id == last_stream_id


async def test_an_error_goaway_ends_the_stream_without_waiting_for_an_answer() -> None:
    with pytest.raises(httpcore.RemoteProtocolError) as error:
        _ = await _async_search(_script(_goaway(1, ErrorCodes.INTERNAL_ERROR)))

    assert _terminated_by(error.value).error_code == ErrorCodes.INTERNAL_ERROR


async def test_a_graceful_goaway_that_accepted_no_stream_fails_the_search_instead_of_reconnecting() -> None:
    script: Final = _script(_goaway(0), answered=False)

    with pytest.raises(httpcore.RemoteProtocolError) as async_error:
        _ = await _async_search(script, _OneConnectionBackend(script))
    with pytest.raises(httpcore.RemoteProtocolError) as sync_error:
        _ = _sync_search(script, _OneConnectionSyncBackend(script))

    assert _terminated_by(async_error.value).last_stream_id == 0
    assert _terminated_by(sync_error.value).last_stream_id == 0
