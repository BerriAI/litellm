import asyncio
import queue
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
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
from h2.settings import SettingCodes, Settings
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
TIME_FOR_THE_OTHER_SEARCH_TO_WAIT_ON_THE_READ: Final = 0.1
TIME_FOR_THE_OTHER_SEARCHES_TO_WAIT_FOR_A_STREAM: Final = 0.1
SEARCHES_FINISH_WITHIN: Final = 2.0
UPLOAD_WINDOW: Final = H2Connection().remote_settings.initial_window_size
REQUEST_HEADERS: Final = (
    (":method", "POST"),
    (":scheme", "https"),
    (":authority", "search.example"),
    (":path", "/v1/search"),
)


def _goaway(
    last_stream_id: int, error_code: ErrorCodes = ErrorCodes.NO_ERROR, additional_data: bytes = b""
) -> bytes:
    return GoAwayFrame(
        stream_id=0, last_stream_id=last_stream_id, error_code=error_code, additional_data=additional_data
    ).serialize()


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


def _answer(server: H2Connection, stream_id: int) -> bytes:
    server.send_headers(stream_id, [(":status", "200"), ("content-type", "application/json")])
    server.send_data(stream_id, ANSWER, end_stream=True)
    return server.data_to_send()


def _both_answered_after_a_graceful_goaway(server: H2Connection) -> bytes:
    return _goaway(3) + _answer(server, 1) + _answer(server, 3)


def _second_answered_before_an_error_goaway(server: H2Connection) -> bytes:
    return _answer(server, 3) + _goaway(3, ErrorCodes.ENHANCE_YOUR_CALM)


class _RepliesOnceBothSearchesArriveStream(httpcore.AsyncMockStream):
    def __init__(self, reply: Callable[[H2Connection], bytes]) -> None:
        super().__init__([], http2=True)
        self._server: Final = H2Connection(config=H2Configuration(client_side=False))
        self._server.initiate_connection()
        self._reply: Final = reply
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
        return self._reply(self._server)


class _RepliesOnceBothSearchesArriveBackend(httpcore.AsyncMockBackend):
    def __init__(self, reply: Callable[[H2Connection], bytes]) -> None:
        super().__init__([], http2=True)
        self._reply: Final = reply

    @override
    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        return _RepliesOnceBothSearchesArriveStream(self._reply)


class _RepliesOnceBothSearchesArriveSyncStream(httpcore.MockStream):
    def __init__(self, reply: Callable[[H2Connection], bytes]) -> None:
        super().__init__([], http2=True)
        self._server: Final = H2Connection(config=H2Configuration(client_side=False))
        self._server.initiate_connection()
        self._reply: Final = reply
        self.waiting_for_the_second_search: Final = threading.Event()
        self._both_arrived: Final = threading.Event()
        self._arrived = 0
        self._answered = False

    @override
    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self._arrived += sum(isinstance(event, StreamEnded) for event in self._server.receive_data(buffer))
        if self._arrived == 2:
            self._both_arrived.set()

    @override
    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        pending: Final = self._server.data_to_send()
        if pending or self._answered:
            return pending
        self.waiting_for_the_second_search.set()
        _ = self._both_arrived.wait()
        time.sleep(TIME_FOR_THE_OTHER_SEARCH_TO_WAIT_ON_THE_READ)
        self._answered = True
        return self._reply(self._server)


class _OneStreamSyncBackend(httpcore.MockBackend):
    def __init__(self, stream: httpcore.NetworkStream) -> None:
        super().__init__([], http2=True)
        self._stream: Final = stream

    @override
    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        return self._stream


def _server_allowing_one_stream_at_a_time() -> H2Connection:
    server: Final = H2Connection(config=H2Configuration(client_side=False))
    server.local_settings = Settings(client=False, initial_values={SettingCodes.MAX_CONCURRENT_STREAMS: 1})
    server.initiate_connection()
    return server


def _answer_with_a_graceful_goaway(server: H2Connection, stream_id: int) -> bytes:
    return _goaway(stream_id) + _answer(server, stream_id)


class _OneStreamAtATimeStream(httpcore.AsyncMockStream):
    def __init__(self, answer: Callable[[H2Connection, int], bytes]) -> None:
        super().__init__([], http2=True)
        self._server: Final = _server_allowing_one_stream_at_a_time()
        self._answer: Final = answer
        self._arrived: Final = asyncio.Queue[int]()

    @override
    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        for event in self._server.receive_data(buffer):
            if isinstance(event, StreamEnded) and event.stream_id:
                self._arrived.put_nowait(event.stream_id)

    @override
    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        pending: Final = self._server.data_to_send()
        if pending:
            return pending
        stream_id: Final = await self._arrived.get()
        await asyncio.sleep(TIME_FOR_THE_OTHER_SEARCHES_TO_WAIT_FOR_A_STREAM)
        return self._answer(self._server, stream_id)


class _DrainsItsFirstConnectionBackend(httpcore.AsyncMockBackend):
    def __init__(self) -> None:
        super().__init__([], http2=True)
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
        answer: Final = _answer if self._connected else _answer_with_a_graceful_goaway
        self._connected = True
        return _OneStreamAtATimeStream(answer)


class _OneStreamAtATimeSyncStream(httpcore.MockStream):
    def __init__(self, answer: Callable[[H2Connection, int], bytes]) -> None:
        super().__init__([], http2=True)
        self._server: Final = _server_allowing_one_stream_at_a_time()
        self._answer: Final = answer
        self._arrived: Final = queue.Queue[int]()

    @override
    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        for event in self._server.receive_data(buffer):
            if isinstance(event, StreamEnded) and event.stream_id:
                self._arrived.put_nowait(event.stream_id)

    @override
    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        pending: Final = self._server.data_to_send()
        if pending:
            return pending
        stream_id: Final = self._arrived.get()
        time.sleep(TIME_FOR_THE_OTHER_SEARCHES_TO_WAIT_FOR_A_STREAM)
        return self._answer(self._server, stream_id)


class _DrainsItsFirstConnectionSyncBackend(httpcore.MockBackend):
    def __init__(self) -> None:
        super().__init__([], http2=True)
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
        answer: Final = _answer if self._connected else _answer_with_a_graceful_goaway
        self._connected = True
        return _OneStreamAtATimeSyncStream(answer)


def _search_on_a_daemon_thread(pool: httpcore.ConnectionPool) -> Future[bytes]:
    content: Final[Future[bytes]] = Future()

    def search() -> None:
        try:
            content.set_result(pool.request("POST", URL, content=b"{}").content)
        except httpcore.HTTPCoreException as error:
            content.set_exception(error)

    threading.Thread(target=search, daemon=True).start()
    return content


async def test_two_searches_answered_in_one_read_with_a_graceful_goaway_both_get_their_answers() -> None:
    async with httpcore.AsyncConnectionPool(
        http2=True, network_backend=_RepliesOnceBothSearchesArriveBackend(_both_answered_after_a_graceful_goaway)
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

    with httpcore.ConnectionPool(
        http2=True, network_backend=httpcore.MockBackend(list(_script(*_two_step_goaway())), http2=True)
    ) as sync_pool:
        with sync_pool.stream("POST", URL, content=b"{}") as sync_response:
            assert not sync_pool.connections[0].is_available()
            assert sync_response.read() == ANSWER


async def test_searches_waiting_for_a_stream_at_a_graceful_goaway_move_to_a_new_connection() -> None:
    async with httpcore.AsyncConnectionPool(http2=True, network_backend=_DrainsItsFirstConnectionBackend()) as pool:
        responses: Final = await asyncio.wait_for(
            asyncio.gather(*(pool.request("POST", URL, content=b"{}") for _ in range(3))), SEARCHES_FINISH_WITHIN
        )

    assert [response.content for response in responses] == [ANSWER, ANSWER, ANSWER]

    with httpcore.ConnectionPool(http2=True, network_backend=_DrainsItsFirstConnectionSyncBackend()) as sync_pool:
        searches: Final = [_search_on_a_daemon_thread(sync_pool) for _ in range(3)]
        assert [search.result(timeout=SEARCHES_FINISH_WITHIN) for search in searches] == [ANSWER, ANSWER, ANSWER]


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


@pytest.mark.parametrize(
    "before",
    [pytest.param((), id="alone"), pytest.param(_two_step_goaway(), id="after-a-graceful-goaway")],
)
async def test_an_error_goaway_ends_the_stream_with_its_error_code(before: tuple[bytes, ...]) -> None:
    script: Final = _script(*before, _goaway(1, ErrorCodes.INTERNAL_ERROR))

    with pytest.raises(httpcore.RemoteProtocolError) as async_error:
        _ = await _async_search(script)
    with pytest.raises(httpcore.RemoteProtocolError) as sync_error:
        _ = _sync_search(script)

    assert _terminated_by(async_error.value).error_code == ErrorCodes.INTERNAL_ERROR
    assert _terminated_by(sync_error.value).error_code == ErrorCodes.INTERNAL_ERROR


async def test_a_search_answered_in_full_before_an_error_goaway_keeps_its_answer() -> None:
    async with httpcore.AsyncConnectionPool(
        http2=True, network_backend=_RepliesOnceBothSearchesArriveBackend(_second_answered_before_an_error_goaway)
    ) as pool:
        unanswered, answered = await asyncio.gather(
            *(pool.request("POST", URL, content=b"{}") for _ in range(2)), return_exceptions=True
        )

    assert isinstance(unanswered, httpcore.RemoteProtocolError)
    assert _terminated_by(unanswered).error_code == ErrorCodes.ENHANCE_YOUR_CALM
    assert isinstance(answered, httpcore.Response)
    assert answered.content == ANSWER

    stream: Final = _RepliesOnceBothSearchesArriveSyncStream(_second_answered_before_an_error_goaway)
    with httpcore.ConnectionPool(http2=True, network_backend=_OneStreamSyncBackend(stream)) as sync_pool:
        with ThreadPoolExecutor(max_workers=2) as searches:
            first: Final = searches.submit(sync_pool.request, "POST", URL, content=b"{}")
            _ = stream.waiting_for_the_second_search.wait()
            second: Final = searches.submit(sync_pool.request, "POST", URL, content=b"{}")

            with pytest.raises(httpcore.RemoteProtocolError) as sync_error:
                _ = first.result()
            assert second.result().content == ANSWER

    assert _terminated_by(sync_error.value).error_code == ErrorCodes.ENHANCE_YOUR_CALM


async def test_a_graceful_goaway_that_accepted_no_stream_fails_the_search_instead_of_reconnecting() -> None:
    script: Final = _script(_goaway(0, additional_data=b"max_age"), answered=False)

    with pytest.raises(httpcore.RemoteProtocolError) as async_error:
        _ = await _async_search(script, _OneConnectionBackend(script))
    with pytest.raises(httpcore.RemoteProtocolError) as sync_error:
        _ = _sync_search(script, _OneConnectionSyncBackend(script))

    assert (_terminated_by(async_error.value).last_stream_id, _terminated_by(async_error.value).additional_data) == (
        0,
        b"max_age",
    )
    assert (_terminated_by(sync_error.value).last_stream_id, _terminated_by(sync_error.value).additional_data) == (
        0,
        b"max_age",
    )
