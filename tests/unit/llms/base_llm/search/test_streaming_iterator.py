import json
from collections.abc import AsyncIterator, Iterator
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.llms.base_llm.search.streaming_iterator import SearchStreamWrapper

URL: Final = "https://api.exa.ai/search"
SCHEMA: Final = {"type": "text", "description": "A concise answer"}
RESULT: Final = {"title": "Source", "url": "https://example.com", "highlights": ["evidence"]}
OUTPUT: Final = {"content": "answer", "grounding": []}
EVENTS: Final = (
    {"type": "results", "results": [RESULT], "requestId": "id"},
    {"type": "text-delta", "delta": "discarded"},
    {"type": "stream-reset", "streamReset": True},
    {
        "type": "text-delta",
        "delta": "answer",
        "choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": None}],
    },
    {"type": "grounding", "grounding": [], "citations": []},
    {
        "type": "done",
        "output": OUTPUT,
        "searchTime": 12,
        "costDollars": {"total": 0.01},
        "requestId": "id",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    },
)


class FragmentedStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, body: bytes):
        self.body = body
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        for offset in range(0, len(self.body), 7):
            yield self.body[offset : offset + 7]

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for part in self:
            yield part

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.close()


def wire(*events: object) -> bytes:
    return (
        ": heartbeat\r\n\r\n" + "".join("event: message\r\ndata: " + json.dumps(event) + "\r\n\r\n" for event in events)
    ).encode()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_search_stream_preserves_events_and_final_response(respx_mock: respx.MockRouter, asynchronous: bool):
    body: Final = FragmentedStream(wire(*EVENTS))
    route: Final = respx_mock.post(URL).mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream; charset=utf-8"}, stream=body)
    )
    params: Final = {
        "query": "question",
        "search_provider": "exa_ai",
        "api_key": "test",
        "stream": True,
        "outputSchema": SCHEMA,
        "contents": {"highlights": True},
    }
    stream: Final = await litellm.asearch(**params) if asynchronous else litellm.search(**params)
    assert isinstance(stream, SearchStreamWrapper)
    assert not body.closed
    received: Final = (
        [event.model_dump() async for event in stream] if asynchronous else [event.model_dump() for event in stream]
    )
    assert received == list(EVENTS)
    assert body.closed
    assert json.loads(route.calls.last.request.content)["stream"] is True
    assert stream.final_response is not None
    assert stream.final_response.model_dump()["output"] == OUTPUT
    assert stream.final_response.results[0].snippet == "evidence"
    assert stream.final_response.model_dump()["costDollars"] == EVENTS[-1]["costDollars"]
    assert stream.final_response.model_dump()["choices"] == EVENTS[-1]["choices"]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "body",
    [
        wire({"type": "error", "error": {"message": "provider failed"}}),
        wire({"type": "text-delta", "delta": "incomplete"}),
        b"data: not-json\n\n",
        b"data: [DONE]\n\n",
    ],
)
@pytest.mark.asyncio
async def test_stream_failure_is_not_silent_and_closes_connection(
    respx_mock: respx.MockRouter, asynchronous: bool, body: bytes
):
    transport: Final = FragmentedStream(body)
    respx_mock.post(URL).mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=transport)
    )
    params: Final = {
        "query": "question",
        "search_provider": "exa_ai",
        "api_key": "test",
        "stream": True,
        "outputSchema": SCHEMA,
    }
    stream: Final = await litellm.asearch(**params) if asynchronous else litellm.search(**params)
    if asynchronous:
        with pytest.raises(litellm.APIError):
            _ = [event async for event in stream]
    else:
        with pytest.raises(litellm.APIError):
            list(stream)
    assert transport.closed


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_context_manager_closes_abandoned_stream(respx_mock: respx.MockRouter, asynchronous: bool):
    body: Final = FragmentedStream(wire(*EVENTS))
    respx_mock.post(URL).mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
    )
    params: Final = {
        "query": "question",
        "search_provider": "exa_ai",
        "api_key": "test",
        "stream": True,
        "outputSchema": SCHEMA,
    }
    if asynchronous:
        async with await litellm.asearch(**params) as stream:
            assert (await anext(stream)).type == "results"
    else:
        with litellm.search(**params) as stream:
            assert next(stream).type == "results"
    assert body.closed


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_stream_flag_without_schema_returns_regular_json(respx_mock: respx.MockRouter, asynchronous: bool):
    respx_mock.post(URL).respond(json={"results": [RESULT], "requestId": "id"})
    params: Final = {"query": "question", "search_provider": "exa_ai", "api_key": "test", "stream": True}
    response: Final = await litellm.asearch(**params) if asynchronous else litellm.search(**params)
    assert response.model_dump()["requestId"] == "id"
    assert response.results[0].snippet == "evidence"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_complete_response_uses_search_terminal_payload(respx_mock: respx.MockRouter, asynchronous: bool):
    respx_mock.post(URL).mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=FragmentedStream(wire(*EVENTS))
        )
    )
    params: Final = {
        "query": "question",
        "search_provider": "exa_ai",
        "api_key": "test",
        "stream": True,
        "outputSchema": SCHEMA,
        "complete_response": True,
    }
    response: Final = await litellm.asearch(**params) if asynchronous else litellm.search(**params)
    assert response.model_dump()["output"] == OUTPUT
    assert response.results[0].snippet == "evidence"


@pytest.fixture(autouse=True)
def use_httpx_for_respx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_sse_multiline_unicode_and_unknown_events(respx_mock: respx.MockRouter, asynchronous: bool):
    body: Final = FragmentedStream(
        'data: {"type":"text-delta",\r\ndata: "delta":"東京"}\r\n\r\n'.encode()
        + wire({"type": "future-event", "value": {"nested": None}}, EVENTS[-1])
    )
    respx_mock.post(URL).mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
    )
    params: Final = {
        "query": "question",
        "search_provider": "exa_ai",
        "api_key": "test",
        "stream": True,
        "outputSchema": SCHEMA,
    }
    stream: Final = await litellm.asearch(**params) if asynchronous else litellm.search(**params)
    events: Final = [e.model_dump() async for e in stream] if asynchronous else [e.model_dump() for e in stream]
    assert events[0] == {"type": "text-delta", "delta": "東京"}
    assert events[1] == {"type": "future-event", "value": {"nested": None}}
    assert events[-1] == EVENTS[-1]
    assert body.closed


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("error_type", [httpx.ReadTimeout, httpx.RemoteProtocolError])
@pytest.mark.asyncio
async def test_stream_transport_failure_closes_connection(
    respx_mock: respx.MockRouter, asynchronous: bool, error_type: type[httpx.TransportError]
):
    class BrokenStream(FragmentedStream):
        def __iter__(self) -> Iterator[bytes]:
            yield from super().__iter__()
            raise error_type("interrupted")

    body: Final = BrokenStream(wire({"type": "text-delta", "delta": "partial"}))
    respx_mock.post(URL).mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
    )
    params: Final = {
        "query": "question",
        "search_provider": "exa_ai",
        "api_key": "test",
        "stream": True,
        "outputSchema": SCHEMA,
    }
    stream: Final = await litellm.asearch(**params) if asynchronous else litellm.search(**params)
    expected: Final = litellm.Timeout if error_type is httpx.ReadTimeout else litellm.APIError
    if asynchronous:
        assert (await anext(stream)).delta == "partial"
        with pytest.raises(expected):
            await anext(stream)
    else:
        assert next(stream).delta == "partial"
        with pytest.raises(expected):
            next(stream)
    assert body.closed


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("output", [None, {}])
@pytest.mark.asyncio
async def test_done_without_results_preserves_empty_output(
    respx_mock: respx.MockRouter, asynchronous: bool, output: object
):
    body: Final = FragmentedStream(
        wire(
            {"type": "results", "results": [], "requestId": "early-id"},
            {"type": "done", "output": output, "searchTime": 0},
        )
    )
    respx_mock.post(URL).mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
    )
    params: Final = {
        "query": "question",
        "search_provider": "exa_ai",
        "api_key": "test",
        "stream": True,
        "outputSchema": SCHEMA,
        "complete_response": True,
    }
    response: Final = await litellm.asearch(**params) if asynchronous else litellm.search(**params)
    assert response.results == []
    assert response.model_dump()["output"] == output
    assert response.model_dump()["requestId"] == "early-id"
    assert body.closed


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_provider_stream_error_retains_request_id_and_payload(respx_mock: respx.MockRouter, asynchronous: bool):
    payload: Final = {"type": "error", "requestId": "failed-request", "error": {"message": "upstream failed"}}
    body: Final = FragmentedStream(wire(payload))
    respx_mock.post(URL).mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
    )
    params: Final = {
        "query": "question",
        "search_provider": "exa_ai",
        "api_key": "test",
        "stream": True,
        "outputSchema": SCHEMA,
    }
    stream: Final = await litellm.asearch(**params) if asynchronous else litellm.search(**params)
    if asynchronous:
        with pytest.raises(litellm.APIError) as caught:
            await anext(stream)
    else:
        with pytest.raises(litellm.APIError) as caught:
            next(stream)
    assert caught.value.body == payload
    assert str(caught.value.request.url) == URL
    assert body.closed


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.asyncio
async def test_requested_stream_accepts_json_response(respx_mock: respx.MockRouter, asynchronous: bool, complete: bool):
    body: Final = FragmentedStream(json.dumps({"results": [RESULT], "output": OUTPUT}).encode())
    respx_mock.post(URL).mock(
        return_value=httpx.Response(200, headers={"content-type": "application/json"}, stream=body)
    )
    params: Final = {
        "query": "question",
        "search_provider": "exa_ai",
        "api_key": "test",
        "stream": True,
        "outputSchema": SCHEMA,
        "complete_response": complete,
    }
    response: Final = await litellm.asearch(**params) if asynchronous else litellm.search(**params)
    assert response.model_dump()["output"] == OUTPUT
    assert response.results[0].snippet == "evidence"
    assert body.closed


@pytest.mark.asyncio
async def test_cancelled_stream_read_closes_connection(respx_mock: respx.MockRouter):
    import asyncio

    waiting: Final = asyncio.Event()
    release: Final = asyncio.Event()

    class PausedStream(FragmentedStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield wire({"type": "text-delta", "delta": "partial"})
            waiting.set()
            await release.wait()

    body: Final = PausedStream(b"")
    respx_mock.post(URL).mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
    )
    stream: Final = await litellm.asearch(
        query="question", search_provider="exa_ai", api_key="test", stream=True, outputSchema=SCHEMA
    )
    assert (await anext(stream)).delta == "partial"
    pending: Final = asyncio.create_task(anext(stream))
    await waiting.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert body.closed
