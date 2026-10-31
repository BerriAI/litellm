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
        assert json.loads(response.model_dump_json()) == {
            "object": "search",
            "results": [
                {
                    "title": "Source",
                    "url": "https://example.com",
                    "snippet": "evidence",
                    "date": None,
                    "last_updated": None,
                    "highlights": ["evidence"],
                    "author": None,
                }
            ],
            "provider_metadata": {key: value for key, value in PAYLOAD.items() if key != "results"},
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("schema", [False, True])
@pytest.mark.parametrize("price_override", [None, 0.0, 0.123])
async def test_search_logs_one_completed_response_and_cost(
    asynchronous: bool, schema: bool, price_override: float | None
) -> None:
    import asyncio
    from collections.abc import Mapping
    from datetime import datetime

    from litellm.integrations.custom_logger import CustomLogger
    from litellm.llms.base_llm.search.transformation import SearchResponse

    class Recorder(CustomLogger):
        def __init__(self) -> None:
            self.finished = threading.Event()
            self.records: tuple[tuple[Mapping[str, object], SearchResponse], ...] = ()

        def log_success_event(
            self, kwargs: Mapping[str, object], response_obj: SearchResponse, start_time: datetime, end_time: datetime
        ) -> None:
            self.records = (*self.records, (kwargs, response_obj))
            self.finished.set()

        async def async_log_success_event(
            self, kwargs: Mapping[str, object], response_obj: SearchResponse, start_time: datetime, end_time: datetime
        ) -> None:
            self.log_success_event(kwargs, response_obj, start_time, end_time)

    recorder: Final = Recorder()

    def respond(request: Request) -> Reply:
        assert "input_cost_per_query" not in json.loads(request.body)
        return Reply(body=json.dumps(PAYLOAD).encode())

    with wire_server(respond) as server:
        params: Final = {
            "query": "question",
            "search_provider": "exa_ai",
            "api_base": server.url,
            "api_key": "test-key",
            **({"input_cost_per_query": price_override} if price_override is not None else {}),
            "success_callback": [recorder.async_log_success_event if asynchronous else recorder],
            **{key: value for key, value in PARAMS.items() if schema or key != "outputSchema"},
        }
        response: Final = await litellm.asearch(**params) if asynchronous else litellm.search(**params)
        assert await asyncio.to_thread(recorder.finished.wait, 5)
        assert len(recorder.records) == 1
        details, logged = recorder.records[0]
        assert logged.model_dump() == response.model_dump()
        assert isinstance(details["standard_logging_object"], Mapping)
        expected_cost: Final = PAYLOAD["costDollars"]["total"] if price_override is None else price_override
        assert details["response_cost"] == expected_cost
        assert details["standard_logging_object"]["response_cost"] == expected_cost
        assert details["stream"] is False
