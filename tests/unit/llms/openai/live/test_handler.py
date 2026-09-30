import asyncio
import json
from collections.abc import Coroutine
from typing import Final, NoReturn, cast

import pytest
from pydantic import TypeAdapter
from starlette.websockets import WebSocketDisconnect
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedOK, InvalidStatus
from websockets.frames import Close
from websockets.http11 import Response

import litellm
from litellm.constants import REALTIME_SESSION_SUCCESS_LOGGED_KEY, REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
from litellm.cost_calculator import handle_realtime_stream_cost_calculation
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.openai.live import handler as live_handler
from litellm.llms.openai.live.handler import OpenAILiveSessions
from litellm.types.utils import LiteLLMRealtimeStreamLoggingObject, Usage

_LIVE_EVENT_FRAME_ADAPTER: Final = TypeAdapter(dict[str, object])


def _event_type(message: str) -> str | None:
    event_type: Final = _LIVE_EVENT_FRAME_ADAPTER.validate_json(message).get("type")
    return event_type if isinstance(event_type, str) else None


class _FakeClientWebSocket:
    def __init__(
        self,
        messages: tuple[str, ...],
        *,
        disconnect_after_send_type: str | None = None,
        disconnect_after_receive_type: str | None = None,
        fail_on_send_type: str | None = None,
    ) -> None:
        self._messages = iter(messages)
        self._disconnect_after_send_type = disconnect_after_send_type
        self._disconnect_after_receive_type = disconnect_after_receive_type
        self._fail_on_send_type = fail_on_send_type
        self.sent_text: asyncio.Queue[str] = asyncio.Queue()
        self.closed = asyncio.Event()
        self.close_info: tuple[int, str | None] | None = None

    async def receive_text(self) -> str:
        try:
            message: Final = next(self._messages)
            if _event_type(message) == self._disconnect_after_receive_type:
                self.closed.set()
            return message
        except StopIteration:
            await self.closed.wait()
            raise WebSocketDisconnect(code=1000)

    async def send_text(self, data: str) -> None:
        event_type: Final = _event_type(data)
        if event_type == self._fail_on_send_type:
            raise RuntimeError("client is disconnected")
        await self.sent_text.put(data)
        if event_type == self._disconnect_after_send_type:
            self.closed.set()

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.close_info = (code, reason)
        self.closed.set()


class _FakeUpstream:
    def __init__(
        self,
        client_frames: tuple[str, ...],
        upstream_frames: tuple[str, ...],
        *,
        close_response: str | None = None,
        never_respond_after_close: bool = False,
        block_session_close_send: bool = False,
        block_close: bool = False,
    ) -> None:
        self.client_frames = client_frames
        self._upstream_frames = iter(upstream_frames)
        self._close_response = close_response
        self._close_response_sent = False
        self._never_respond_after_close = never_respond_after_close
        self._block_session_close_send = block_session_close_send
        self._block_close = block_close
        self.sent_text: asyncio.Queue[str] = asyncio.Queue()
        self.client_frames_sent = asyncio.Event()
        self.session_close_sent = asyncio.Event()
        self.closed_event = asyncio.Event()
        self.closed = False

    async def send(self, data: str) -> None:
        event_type: Final = _event_type(data)
        if event_type == "session.close" and self._block_session_close_send:
            await asyncio.Event().wait()
        await self.sent_text.put(data)
        if event_type == "session.close":
            self.session_close_sent.set()
        if self.sent_text.qsize() == len(self.client_frames) + 1:
            self.client_frames_sent.set()

    async def recv(self) -> str:
        await self.client_frames_sent.wait()
        try:
            return next(self._upstream_frames)
        except StopIteration as error:
            if self._close_response is not None and not self._close_response_sent:
                await self.session_close_sent.wait()
                self._close_response_sent = True
                return self._close_response
            if self._never_respond_after_close:
                await self.session_close_sent.wait()
                await self.closed_event.wait()
            raise ConnectionClosedOK(
                Close(code=1000, reason="upstream closed"),
                Close(code=1000, reason="upstream closed"),
                True,
            ) from error

    async def close(self) -> None:
        if self._block_close:
            await asyncio.Event().wait()
        self.closed = True
        self.closed_event.set()

    async def __aenter__(self) -> "_FakeUpstream":
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.closed = True
        self.closed_event.set()


class _FailingConnection:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def __aenter__(self) -> NoReturn:
        raise self.error

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        return None


class _FakeLogging:
    def __init__(self, pre_call_error: Exception | None = None) -> None:
        self.model_call_details: dict[str, object] = {}
        self.pre_call_kwargs: dict[str, object] = {}
        self.dispatch_count = 0
        self.dispatched_result: object | None = None
        self.prefer_async_handlers = False
        self._pre_call_error = pre_call_error

    def pre_call(self, **kwargs: object) -> None:
        if self._pre_call_error is not None:
            raise self._pre_call_error
        self.pre_call_kwargs = kwargs

    async def dispatch_success_handlers(
        self,
        result: object,
        *,
        prefer_async_handlers: bool = False,
    ) -> None:
        self.dispatch_count += 1
        self.dispatched_result = result
        self.prefer_async_handlers = prefer_async_handlers


class _FakeLoggingWorker:
    def __init__(self) -> None:
        self.coroutine: Coroutine[object, object, None] | None = None

    def ensure_initialized_and_enqueue(self, async_coroutine: Coroutine[object, object, None]) -> None:
        self.coroutine = async_coroutine


class _FakeConnector:
    def __init__(self, upstream: _FakeUpstream) -> None:
        self.upstream = upstream
        self.url: str | None = None
        self.kwargs: dict[str, object] = {}

    def __call__(self, url: str, **kwargs: object) -> _FakeUpstream:
        self.url = url
        self.kwargs = kwargs
        return self.upstream


@pytest.mark.parametrize(
    ("api_base", "expected_url"),
    [
        ("https://api.openai.com/", "wss://api.openai.com/v1/live/sessions"),
        ("https://api.openai.com/v1", "wss://api.openai.com/v1/live/sessions"),
        ("http://localhost:8080", "ws://localhost:8080/v1/live/sessions"),
        ("https://api.openai.com/v1?api-version=1", "wss://api.openai.com/v1/live/sessions"),
    ],
)
def test_construct_live_url_uses_exact_path_and_drops_query(api_base: str, expected_url: str) -> None:
    assert OpenAILiveSessions._construct_live_url(api_base) == expected_url


@pytest.mark.asyncio
async def test_async_live_session_rewrites_model_and_relays_frames_without_mutating_input() -> None:
    client_frames: Final = ('{"type":"input_audio.append","audio":"client-a"}', '{"type":"session.close"}')
    upstream_frames: Final = (
        '{"type":"session.started","session":{"model":"gpt-live-1"}}',
        '{"type":"response.audio.delta","delta":"server-a"}',
        '{"type":"session.closed","usage":{"seconds":16}}',
    )
    upstream: Final = _FakeUpstream(client_frames=client_frames, upstream_frames=upstream_frames)
    connector: Final = _FakeConnector(upstream)
    client: Final = _FakeClientWebSocket(messages=client_frames)
    logger: Final = _FakeLogging()
    worker: Final = _FakeLoggingWorker()
    session_start: Final[dict[str, object]] = {
        "type": "session.start",
        "session": {
            "model": "proxy-live-alias",
            "instructions": "Be concise",
            "audio": {"input": {"format": "pcm16"}},
        },
    }
    original_session_start: Final = json.loads(json.dumps(session_start))

    await OpenAILiveSessions(logging_worker=worker, websocket_connector=connector).async_live_session(
        model="gpt-live-1",
        websocket=client,
        logging_obj=cast(Logging, logger),
        session_start=session_start,
        api_base="https://api.openai.com/",
        api_key="live-test-key",
    )

    sent_frames: Final = tuple(
        await asyncio.gather(*(upstream.sent_text.get() for _ in range(upstream.sent_text.qsize())))
    )
    forwarded_frames: Final = tuple(
        await asyncio.gather(*(client.sent_text.get() for _ in range(client.sent_text.qsize())))
    )
    rewritten_start: Final = json.loads(sent_frames[0])
    assert connector.url == "wss://api.openai.com/v1/live/sessions"
    assert connector.kwargs["additional_headers"] == {"Authorization": "Bearer live-test-key"}
    assert connector.kwargs["max_size"] == REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
    assert "ssl" in connector.kwargs
    assert rewritten_start == {
        "type": "session.start",
        "session": {
            "model": "gpt-live-1",
            "instructions": "Be concise",
            "audio": {"input": {"format": "pcm16"}},
        },
    }
    assert sent_frames[1:] == client_frames
    assert forwarded_frames == upstream_frames
    assert session_start == original_session_start
    assert client.close_info == (1000, "upstream closed")
    assert logger.model_call_details[REALTIME_SESSION_SUCCESS_LOGGED_KEY] is True
    assert worker.coroutine is not None
    await worker.coroutine
    assert logger.dispatch_count == 1
    assert logger.prefer_async_handlers is True
    logged_result: Final = cast(LiteLLMRealtimeStreamLoggingObject, logger.dispatched_result)
    assert tuple(event["type"] for event in logged_result.results) == (
        "session.started",
        "session.closed",
    )


@pytest.mark.asyncio
async def test_async_live_session_coalesces_cumulative_usage_updates() -> None:
    client_frames: Final = ('{"type":"session.close"}',)
    upstream_frames: Final = (
        '{"type":"session.started","session":{"model":"gpt-live-1"}}',
        '{"type":"response.event","event":{"type":"response.completed","response":{"id":"resp-1"}}}',
        '{"type":"response.event","event":{"type":"response.failed","response":{"id":"resp-2"}}}',
        '{"type":"session.usage.updated","usage":{"seconds":1}}',
        '{"type":"session.usage.updated","usage":{"seconds":2}}',
        '{"type":"session.usage.updated","usage":{"seconds":3}}',
        '{"type":"session.closed","usage":{"seconds":3}}',
    )
    upstream: Final = _FakeUpstream(client_frames=client_frames, upstream_frames=upstream_frames)
    connector: Final = _FakeConnector(upstream)
    worker: Final = _FakeLoggingWorker()
    logger: Final = _FakeLogging()

    await OpenAILiveSessions(logging_worker=worker, websocket_connector=connector).async_live_session(
        model="gpt-live-1",
        websocket=_FakeClientWebSocket(messages=client_frames),
        logging_obj=cast(Logging, logger),
        session_start={"type": "session.start", "session": {"model": "gpt-live-1"}},
        api_base="https://api.openai.com/",
        api_key="live-test-key",
    )

    assert worker.coroutine is not None
    await worker.coroutine
    logged_result: Final = cast(LiteLLMRealtimeStreamLoggingObject, logger.dispatched_result)
    assert tuple((event["type"], event.get("usage")) for event in logged_result.results) == (
        ("session.started", None),
        ("response.event", None),
        ("response.event", None),
        ("session.usage.updated", {"seconds": 3}),
        ("session.closed", {"seconds": 3}),
    )
    assert tuple(logged_result.results[1:3]) == (
        {
            "type": "response.event",
            "event": {
                "type": "response.completed",
                "response": {"id": "resp-1", "model": None, "usage": None},
            },
        },
        {
            "type": "response.event",
            "event": {
                "type": "response.failed",
                "response": {"id": "resp-2", "model": None, "usage": None},
            },
        },
    )


@pytest.mark.asyncio
async def test_async_live_session_requires_api_key() -> None:
    with pytest.raises(ValueError, match="api_key"):
        await OpenAILiveSessions().async_live_session(
            model="gpt-live-1",
            websocket=_FakeClientWebSocket(messages=()),
            logging_obj=cast(Logging, _FakeLogging()),
            session_start={"type": "session.start", "session": {"model": "proxy-live-alias"}},
            api_base="https://api.openai.com/",
            api_key=None,
        )


@pytest.mark.asyncio
async def test_async_live_session_does_not_log_success_on_handshake_refusal() -> None:
    response: Final = Response(401, "Unauthorized", Headers())
    invalid_status: Final = InvalidStatus(response)

    def refuse_connection(url: str, **kwargs: object) -> _FailingConnection:
        return _FailingConnection(invalid_status)

    client: Final = _FakeClientWebSocket(messages=())
    logger: Final = _FakeLogging()
    worker: Final = _FakeLoggingWorker()

    await OpenAILiveSessions(logging_worker=worker, websocket_connector=refuse_connection).async_live_session(
        model="gpt-live-1",
        websocket=client,
        logging_obj=cast(Logging, logger),
        session_start={"type": "session.start", "session": {"model": "gpt-live-1"}},
        api_base="https://api.openai.com/",
        api_key="live-test-key",
    )

    try:
        assert worker.coroutine is None
        assert REALTIME_SESSION_SUCCESS_LOGGED_KEY not in logger.model_call_details
        assert logger.dispatch_count == 0
    finally:
        if worker.coroutine is not None:
            await worker.coroutine


@pytest.mark.asyncio
async def test_async_live_session_does_not_log_success_when_connection_fails_before_connect() -> None:
    def fail_connection(url: str, **kwargs: object) -> _FailingConnection:
        return _FailingConnection(RuntimeError("connection failed"))

    logger: Final = _FakeLogging()
    worker: Final = _FakeLoggingWorker()

    await OpenAILiveSessions(logging_worker=worker, websocket_connector=fail_connection).async_live_session(
        model="gpt-live-1",
        websocket=_FakeClientWebSocket(messages=()),
        logging_obj=cast(Logging, logger),
        session_start={"type": "session.start", "session": {"model": "gpt-live-1"}},
        api_base="https://api.openai.com/",
        api_key="live-test-key",
    )

    assert worker.coroutine is None
    assert REALTIME_SESSION_SUCCESS_LOGGED_KEY not in logger.model_call_details
    assert logger.dispatch_count == 0


@pytest.mark.asyncio
async def test_async_live_session_retains_final_usage_after_client_disconnect_and_bills_voice() -> None:
    session_closed: Final = json.dumps({"type": "session.closed", "usage": {"seconds": 7}})
    upstream: Final = _FakeUpstream(
        client_frames=(),
        upstream_frames=('{"type":"session.started","session":{"model":"gpt-live-1"}}',),
        close_response=session_closed,
    )
    connector: Final = _FakeConnector(upstream)
    client: Final = _FakeClientWebSocket(messages=(), disconnect_after_send_type="session.started")
    logger: Final = _FakeLogging()
    worker: Final = _FakeLoggingWorker()

    await OpenAILiveSessions(logging_worker=worker, websocket_connector=connector).async_live_session(
        model="gpt-live-1",
        websocket=client,
        logging_obj=cast(Logging, logger),
        session_start={"type": "session.start", "session": {"model": "gpt-live-1"}},
        api_base="https://api.openai.com/",
        api_key="live-test-key",
    )

    assert worker.coroutine is not None
    await worker.coroutine
    logged_result: Final = cast(LiteLLMRealtimeStreamLoggingObject, logger.dispatched_result)
    assert logged_result.results == [
        {"type": "session.started", "session": {"model": "gpt-live-1"}},
        {"type": "session.closed", "usage": {"seconds": 7}},
    ]
    sent_frames: Final = tuple(
        await asyncio.gather(*(upstream.sent_text.get() for _ in range(upstream.sent_text.qsize())))
    )
    session_close: Final = json.dumps({"type": "session.close"})
    assert tuple(frame for frame in sent_frames if _event_type(frame) == "session.close") == (session_close,)

    rate: Final = cast(float, litellm.model_cost["gpt-live-1"]["input_cost_per_second"])
    cost: Final = handle_realtime_stream_cost_calculation(
        results=logged_result.results,
        combined_usage_object=Usage(),
        custom_llm_provider="openai",
        litellm_model_name="gpt-live-1",
    )
    assert cost == pytest.approx(7 * rate)


@pytest.mark.asyncio
async def test_async_live_session_keeps_finite_usage_after_invalid_snapshot() -> None:
    upstream: Final = _FakeUpstream(
        client_frames=(),
        upstream_frames=(
            '{"type":"session.started","session":{"model":"gpt-live-1"}}',
            '{"type":"session.usage.updated","usage":{"seconds":7}}',
            '{"type":"session.usage.updated","usage":{"seconds":"nan"}}',
        ),
        never_respond_after_close=True,
    )
    connector: Final = _FakeConnector(upstream)
    client: Final = _FakeClientWebSocket(messages=(), disconnect_after_send_type="session.started")
    logger: Final = _FakeLogging()
    worker: Final = _FakeLoggingWorker()

    await OpenAILiveSessions(logging_worker=worker, websocket_connector=connector).async_live_session(
        model="gpt-live-1",
        websocket=client,
        logging_obj=cast(Logging, logger),
        session_start={"type": "session.start", "session": {"model": "gpt-live-1"}},
        api_base="https://api.openai.com/",
        api_key="live-test-key",
    )

    assert worker.coroutine is not None
    await worker.coroutine
    logged_result: Final = cast(LiteLLMRealtimeStreamLoggingObject, logger.dispatched_result)
    assert logged_result.results == [
        {"type": "session.started", "session": {"model": "gpt-live-1"}},
        {"type": "session.usage.updated", "usage": {"seconds": 7}},
    ]
    rate: Final = cast(float, litellm.model_cost["gpt-live-1"]["input_cost_per_second"])
    cost: Final = handle_realtime_stream_cost_calculation(
        results=logged_result.results,
        combined_usage_object=Usage(),
        custom_llm_provider="openai",
        litellm_model_name="gpt-live-1",
    )
    assert cost == pytest.approx(7 * rate)


@pytest.mark.asyncio
async def test_async_live_session_does_not_send_a_second_close_after_client_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_handler, "OPENAI_LIVE_SESSION_CLOSE_TIMEOUT_SECONDS", 0.01)
    client_frames: Final = (json.dumps({"type": "session.close"}),)
    upstream: Final = _FakeUpstream(
        client_frames=client_frames,
        upstream_frames=(),
        never_respond_after_close=True,
    )
    connector: Final = _FakeConnector(upstream)
    worker: Final = _FakeLoggingWorker()
    logger: Final = _FakeLogging()

    await OpenAILiveSessions(logging_worker=worker, websocket_connector=connector).async_live_session(
        model="gpt-live-1",
        websocket=_FakeClientWebSocket(
            messages=client_frames,
            disconnect_after_receive_type="session.close",
        ),
        logging_obj=cast(Logging, logger),
        session_start={"type": "session.start", "session": {"model": "gpt-live-1"}},
        api_base="https://api.openai.com/",
        api_key="live-test-key",
    )

    sent_frames: Final = tuple(
        await asyncio.gather(*(upstream.sent_text.get() for _ in range(upstream.sent_text.qsize())))
    )
    assert tuple(frame for frame in sent_frames if _event_type(frame) == "session.close") == client_frames
    assert worker.coroutine is not None
    await worker.coroutine


@pytest.mark.asyncio
async def test_async_live_session_bounds_a_blocking_upstream_close_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_handler, "OPENAI_LIVE_SESSION_CLOSE_TIMEOUT_SECONDS", 0.05)
    upstream: Final = _FakeUpstream(
        client_frames=(),
        upstream_frames=('{"type":"session.started","session":{"model":"gpt-live-1"}}',),
        never_respond_after_close=True,
        block_session_close_send=True,
    )
    connector: Final = _FakeConnector(upstream)
    client: Final = _FakeClientWebSocket(messages=(), disconnect_after_send_type="session.started")
    logger: Final = _FakeLogging()
    worker: Final = _FakeLoggingWorker()

    await asyncio.wait_for(
        OpenAILiveSessions(logging_worker=worker, websocket_connector=connector).async_live_session(
            model="gpt-live-1",
            websocket=client,
            logging_obj=cast(Logging, logger),
            session_start={"type": "session.start", "session": {"model": "gpt-live-1"}},
            api_base="https://api.openai.com/",
            api_key="live-test-key",
        ),
        timeout=1,
    )

    assert worker.coroutine is not None
    await worker.coroutine
    assert logger.dispatch_count == 1


@pytest.mark.asyncio
async def test_async_live_session_bounds_a_blocking_upstream_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_handler, "OPENAI_LIVE_SESSION_CLOSE_TIMEOUT_SECONDS", 0.05)
    upstream: Final = _FakeUpstream(
        client_frames=(),
        upstream_frames=('{"type":"session.started","session":{"model":"gpt-live-1"}}',),
        close_response='{"type":"session.closed","usage":{"seconds":7}}',
        block_close=True,
    )
    connector: Final = _FakeConnector(upstream)
    client: Final = _FakeClientWebSocket(messages=(), disconnect_after_send_type="session.started")
    logger: Final = _FakeLogging()
    worker: Final = _FakeLoggingWorker()

    await asyncio.wait_for(
        OpenAILiveSessions(logging_worker=worker, websocket_connector=connector).async_live_session(
            model="gpt-live-1",
            websocket=client,
            logging_obj=cast(Logging, logger),
            session_start={"type": "session.start", "session": {"model": "gpt-live-1"}},
            api_base="https://api.openai.com/",
            api_key="live-test-key",
        ),
        timeout=1,
    )

    assert worker.coroutine is not None
    await worker.coroutine
    assert logger.dispatch_count == 1


@pytest.mark.asyncio
async def test_async_live_session_retains_closed_event_when_client_send_fails() -> None:
    upstream: Final = _FakeUpstream(
        client_frames=(),
        upstream_frames=(
            '{"type":"session.started","session":{"model":"gpt-live-1"}}',
            '{"type":"session.closed","usage":{"seconds":7}}',
        ),
    )
    connector: Final = _FakeConnector(upstream)
    client: Final = _FakeClientWebSocket(messages=(), fail_on_send_type="session.closed")
    logger: Final = _FakeLogging()
    worker: Final = _FakeLoggingWorker()

    await OpenAILiveSessions(logging_worker=worker, websocket_connector=connector).async_live_session(
        model="gpt-live-1",
        websocket=client,
        logging_obj=cast(Logging, logger),
        session_start={"type": "session.start", "session": {"model": "gpt-live-1"}},
        api_base="https://api.openai.com/",
        api_key="live-test-key",
    )

    assert worker.coroutine is not None
    await worker.coroutine
    logged_result: Final = cast(LiteLLMRealtimeStreamLoggingObject, logger.dispatched_result)
    assert {"type": "session.closed", "usage": {"seconds": 7}} in logged_result.results


@pytest.mark.asyncio
async def test_async_live_session_bounds_wait_for_usage_after_client_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_handler, "OPENAI_LIVE_SESSION_CLOSE_TIMEOUT_SECONDS", 0.01)
    upstream: Final = _FakeUpstream(
        client_frames=(),
        upstream_frames=('{"type":"session.started","session":{"model":"gpt-live-1"}}',),
        never_respond_after_close=True,
    )
    connector: Final = _FakeConnector(upstream)
    client: Final = _FakeClientWebSocket(messages=(), disconnect_after_send_type="session.started")
    logger: Final = _FakeLogging()
    worker: Final = _FakeLoggingWorker()

    await asyncio.wait_for(
        OpenAILiveSessions(logging_worker=worker, websocket_connector=connector).async_live_session(
            model="gpt-live-1",
            websocket=client,
            logging_obj=cast(Logging, logger),
            session_start={"type": "session.start", "session": {"model": "gpt-live-1"}},
            api_base="https://api.openai.com/",
            api_key="live-test-key",
        ),
        timeout=1,
    )

    assert upstream.closed
    assert worker.coroutine is not None
    await worker.coroutine
    logged_result: Final = cast(LiteLLMRealtimeStreamLoggingObject, logger.dispatched_result)
    assert all(event["type"] != "session.closed" for event in logged_result.results)


@pytest.mark.asyncio
async def test_async_live_session_compacts_and_deduplicates_terminal_response_events() -> None:
    large_output: Final = "x" * 4096
    large_instructions: Final = "y" * 4096
    response_usage: Final[dict[str, object]] = {"input_tokens": 10, "output_tokens": 5}
    duplicate_response_usage: Final[dict[str, object]] = {"input_tokens": 100, "output_tokens": 50}
    first_response: Final = json.dumps(
        {
            "type": "response.event",
            "event": {
                "type": "response.completed",
                "response": {
                    "id": "resp-1",
                    "model": "gpt-live-1",
                    "usage": response_usage,
                    "output": large_output,
                    "instructions": large_instructions,
                },
            },
        }
    )
    duplicate_response: Final = json.dumps(
        {
            "type": "response.event",
            "event": {
                "type": "response.completed",
                "response": {
                    "id": "resp-1",
                    "model": "duplicate-model",
                    "usage": duplicate_response_usage,
                    "output": large_output,
                    "instructions": large_instructions,
                },
            },
        }
    )
    response_without_id: Final = json.dumps(
        {
            "type": "response.event",
            "event": {
                "type": "response.completed",
                "response": {"model": "gpt-live-1", "usage": response_usage},
            },
        }
    )
    upstream: Final = _FakeUpstream(
        client_frames=(),
        upstream_frames=(
            '{"type":"session.started","session":{"model":"gpt-live-1"}}',
            first_response,
            duplicate_response,
            response_without_id,
        ),
    )
    connector: Final = _FakeConnector(upstream)
    logger: Final = _FakeLogging()
    worker: Final = _FakeLoggingWorker()

    await OpenAILiveSessions(logging_worker=worker, websocket_connector=connector).async_live_session(
        model="gpt-live-1",
        websocket=_FakeClientWebSocket(messages=()),
        logging_obj=cast(Logging, logger),
        session_start={"type": "session.start", "session": {"model": "gpt-live-1"}},
        api_base="https://api.openai.com/",
        api_key="live-test-key",
    )

    assert worker.coroutine is not None
    await worker.coroutine
    logged_result: Final = cast(LiteLLMRealtimeStreamLoggingObject, logger.dispatched_result)
    response_events: Final = tuple(event for event in logged_result.results if event["type"] == "response.event")
    assert response_events == (
        {
            "type": "response.event",
            "event": {
                "type": "response.completed",
                "response": {
                    "id": "resp-1",
                    "model": "gpt-live-1",
                    "usage": response_usage,
                },
            },
        },
    )


@pytest.mark.asyncio
async def test_async_live_session_reports_redacted_server_error_and_bounds_close_reason() -> None:
    error_message: Final = "x" * 300
    client: Final = _FakeClientWebSocket(messages=())
    logger: Final = _FakeLogging(pre_call_error=RuntimeError(error_message))
    worker: Final = _FakeLoggingWorker()

    await OpenAILiveSessions(logging_worker=worker).async_live_session(
        model="gpt-live-1",
        websocket=client,
        logging_obj=cast(Logging, logger),
        session_start={"type": "session.start", "session": {"model": "proxy-live-alias"}},
        api_base="https://api.openai.com/",
        api_key="live-test-key",
    )

    error_event: Final = json.loads(client.sent_text.get_nowait())
    assert error_event == {"type": "error", "error": {"type": "server_error", "message": error_message}}
    assert client.close_info is not None
    assert client.close_info[0] == 1011
    assert client.close_info[1] is not None
    assert len(client.close_info[1].encode("utf-8")) <= 123
    assert client.close_info[1] == "x" * 123
    try:
        assert worker.coroutine is None
        assert REALTIME_SESSION_SUCCESS_LOGGED_KEY not in logger.model_call_details
        assert logger.dispatch_count == 0
    finally:
        if worker.coroutine is not None:
            await worker.coroutine
