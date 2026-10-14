import json
import threading
from typing import Final

import pytest

import litellm
from tests.integration._support.wire import Reply, Request, wire_server

PAYLOAD: Final = {
    "results": [{"title": "Source", "url": "https://example.com", "highlights": ["evidence"], "author": None}],
    "output": {"content": {"answer": "verified"}, "grounding": []},
    "requestId": "wire-request",
    "searchTime": 14,
    "costDollars": {"total": 0.01},
}
PARAMS: Final = {
    "contents": {"highlights": {"dynamic": True, "verbosity": "low"}},
    "outputSchema": {"type": "object", "properties": {"answer": {"type": "string"}}},
    "systemPrompt": "Use evidence",
    "type": "auto",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_exa_search_request_and_full_response_cross_real_http(asynchronous: bool) -> None:
    def respond(request: Request) -> Reply:
        assert request.target == "/search"
        assert request.headers["x-api-key"] == "test-key"
        assert request.headers["exa-beta"] == "dynamic-highlights-2026-08-28"
        assert json.loads(request.body) == {"query": "question", **PARAMS}
        return Reply(body=json.dumps(PAYLOAD).encode())

    with wire_server(respond) as server:
        params: Final = {
            "query": "question",
            "search_provider": "exa_ai",
            "api_base": server.url,
            "api_key": "test-key",
            "extra_headers": {"Exa-Beta": "dynamic-highlights-2026-08-28"},
            **PARAMS,
        }
        response: Final = await litellm.asearch(**params) if asynchronous else litellm.search(**params)
        result: Final = json.loads(response.model_dump_json())
        assert result["output"] == PAYLOAD["output"]
        assert result["requestId"] == PAYLOAD["requestId"]
        assert result["costDollars"] == PAYLOAD["costDollars"]
        assert result["results"][0]["highlights"] == ["evidence"]
        assert result["results"][0]["author"] is None
        assert result["results"][0]["snippet"] == "evidence"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_exa_sse_delivers_first_event_before_server_finishes(asynchronous: bool) -> None:
    release: Final = threading.Event()
    first: Final = {"type": "results", "results": PAYLOAD["results"]}
    last: Final = {"type": "done", "output": PAYLOAD["output"], "searchTime": 14, "costDollars": PAYLOAD["costDollars"]}

    def respond(request: Request) -> Reply:
        assert json.loads(request.body)["stream"] is True
        return Reply(
            content_type="text/event-stream",
            chunks=tuple(("data: " + json.dumps(event) + "\n\n").encode() for event in (first, last)),
            gate_after_first=release,
        )

    with wire_server(respond) as server:
        params: Final = {
            "query": "question",
            "search_provider": "exa_ai",
            "api_base": server.url,
            "api_key": "test-key",
            "stream": True,
            "timeout": 3,
            **PARAMS,
        }
        try:
            if asynchronous:
                async with await litellm.asearch(**params) as stream:
                    assert (await anext(stream)).model_dump() == first
                    release.set()
                    assert [event.model_dump() async for event in stream] == [last]
            else:
                with litellm.search(**params) as stream:
                    assert next(stream).model_dump() == first
                    release.set()
                    assert [event.model_dump() for event in stream] == [last]
        finally:
            release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(("sse", "schema"), [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("price_override", [None, 0.0, 0.123])
async def test_search_logs_one_completed_response_and_cost(
    asynchronous: bool, sse: bool, schema: bool, price_override: float | None
) -> None:
    import asyncio
    from collections.abc import Mapping

    from litellm.integrations.custom_logger import CustomLogger

    class Recorder(CustomLogger):
        def __init__(self) -> None:
            self.finished = threading.Event()
            self.records = ()

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            self.records = (*self.records, (kwargs, response_obj))
            self.finished.set()

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            self.log_success_event(kwargs, response_obj, start_time, end_time)

    recorder: Final = Recorder()

    def respond(request: Request) -> Reply:
        assert "input_cost_per_query" not in json.loads(request.body)
        if not sse:
            return Reply(body=json.dumps(PAYLOAD).encode())
        events: Final = (
            {"type": "results", "results": PAYLOAD["results"]},
            {"type": "done", "output": PAYLOAD["output"], "searchTime": 14, "costDollars": PAYLOAD["costDollars"]},
        )
        return Reply(
            content_type="text/event-stream",
            body="".join("data: " + json.dumps(event) + "\n\n" for event in events).encode(),
        )

    with wire_server(respond) as server:
        params: Final = {
            "query": "question",
            "search_provider": "exa_ai",
            "api_base": server.url,
            "api_key": "test-key",
            "stream": True,
            "complete_response": True,
            **({"input_cost_per_query": price_override} if price_override is not None else {}),
            "success_callback": [recorder.async_log_success_event if asynchronous else recorder],
            **{key: value for key, value in PARAMS.items() if schema or key != "outputSchema"},
        }
        response: Final = await litellm.asearch(**params) if asynchronous else litellm.search(**params)
        assert await asyncio.to_thread(recorder.finished.wait, 5)
        assert len(recorder.records) == 1
        details, logged = recorder.records[0]
        assert logged.model_dump()["output"] == response.model_dump()["output"]
        assert isinstance(details["standard_logging_object"], Mapping)
        expected_cost: Final = PAYLOAD["costDollars"]["total"] if price_override is None else price_override
        assert details["response_cost"] == expected_cost
        assert details["standard_logging_object"]["response_cost"] == expected_cost
        assert details["stream"] is sse


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("asynchronous", "consume"),
    [
        (False, "none"),
        (True, "none"),
        (False, "results"),
        (True, "results"),
        (False, "done"),
        (True, "done"),
        (True, "cancel"),
    ],
)
async def test_search_disconnect_bills_once(asynchronous: bool, consume: str) -> None:
    import asyncio

    from litellm.integrations.custom_logger import CustomLogger

    class Recorder(CustomLogger):
        def __init__(self) -> None:
            self.finished = threading.Event()
            self.records = ()

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            self.records = (*self.records, (kwargs, response_obj))
            self.finished.set()

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            self.log_success_event(kwargs, response_obj, start_time, end_time)

    recorder: Final = Recorder()
    release: Final = threading.Event()
    events: Final = (
        {"type": "results", "results": PAYLOAD["results"], "requestId": "disconnect"},
        {"type": "done", "output": PAYLOAD["output"], "searchTime": 14, "costDollars": PAYLOAD["costDollars"]},
    )

    def respond(request: Request) -> Reply:
        assert json.loads(request.body)["stream"] is True
        return Reply(
            content_type="text/event-stream",
            chunks=tuple(("data: " + json.dumps(event) + "\n\n").encode() for event in events),
            gate_after_first=release if consume == "cancel" else None,
        )

    with wire_server(respond) as server:
        params: Final = {
            "query": "question",
            "search_provider": "exa_ai",
            "api_base": server.url,
            "api_key": "test-key",
            "stream": True,
            "success_callback": [recorder.async_log_success_event if asynchronous else recorder],
            **PARAMS,
        }
        stream: Final = await litellm.asearch(**params) if asynchronous else litellm.search(**params)
        try:
            if consume != "none":
                first: Final = await anext(stream) if asynchronous else next(stream)
                assert first.type == "results"
            if consume == "done":
                if asynchronous:
                    await stream.aget_final_response()
                else:
                    stream.get_final_response()
            if consume == "cancel":
                pending: Final = asyncio.create_task(anext(stream))
                await asyncio.sleep(0)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            if asynchronous:
                await stream.aclose()
                await stream.aclose()
            else:
                stream.close()
                stream.close()
            assert await asyncio.to_thread(recorder.finished.wait, 5)
            assert len(recorder.records) == 1
            details, logged = recorder.records[0]
            expected: Final = (
                PAYLOAD["costDollars"]["total"]
                if consume == "done"
                else litellm.completion_cost(
                    completion_response={"results": []},
                    model="exa_ai/search",
                    custom_llm_provider="exa_ai",
                    call_type="search",
                )
            )
            assert details["response_cost"] == expected
            assert details["standard_logging_object"]["response_cost"] == details["response_cost"]
            assert len(logged.results) == (0 if consume == "none" else 1)
            assert (stream.final_response is not None) is (consume == "done")
        finally:
            release.set()
