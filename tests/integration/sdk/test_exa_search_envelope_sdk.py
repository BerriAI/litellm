import asyncio
import json
import threading
from collections.abc import Mapping
from datetime import datetime
from typing import Final

import pytest
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.base_llm.search.transformation import SearchResponse

pytestmark: Final = pytest.mark.timeout(60)

_REPORTED_COST: Final = 0.005
_ENVELOPE: Final[dict[str, JsonValue]] = {
    "requestId": "exa-request-sdk",
    "output": {"content": {"answer": "verified"}, "grounding": []},
    "resolvedSearchType": "neural",
    "searchTime": 211.0,
    "costDollars": {"total": _REPORTED_COST, "search": {"neural": _REPORTED_COST}},
}
_RESULT: Final[dict[str, JsonValue]] = {
    "id": "https://result.invalid/sdk",
    "title": "SDK result",
    "url": "https://result.invalid/sdk",
    "publishedDate": "2026-02-01T00:00:00.000Z",
    "score": 0.73,
    "author": None,
    "highlights": ["First line.", "Second line."],
    "highlightScores": [0.61, 0.42],
}
_EXPECTED: Final[dict[str, JsonValue]] = {
    "object": "search",
    "results": [
        {
            "id": "https://result.invalid/sdk",
            "title": "SDK result",
            "url": "https://result.invalid/sdk",
            "snippet": "First line.\n\nSecond line.",
            "date": "2026-02-01T00:00:00.000Z",
            "last_updated": None,
            "highlights": ["First line.", "Second line."],
            "highlight_scores": [0.61, 0.42],
            "score": 0.73,
            "author": None,
        }
    ],
    "provider_metadata": _ENVELOPE,
}


def _respond(request: Request) -> Reply:
    assert (request.method, request.target) == ("POST", "/exa/search"), request.target
    return Reply(body=json.dumps({**_ENVELOPE, "results": [_RESULT]}).encode())


PARAMS: Final = {
    "contents": {"highlights": {"dynamic": True, "verbosity": "low"}},
    "outputSchema": {"type": "object", "properties": {"answer": {"type": "string"}}},
    "systemPrompt": "Use evidence",
    "type": "auto",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("schema", [False, True])
@pytest.mark.parametrize("price_override", [None, 0.0, 0.123])
@pytest.mark.parametrize("global_callback", [False, True])
async def test_exa_search_request_response_and_logged_cost_cross_real_http(
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
    schema: bool,
    price_override: float | None,
    global_callback: bool,
) -> None:
    class Recorder(CustomLogger):
        def __init__(self) -> None:
            self.finished: Final = threading.Event()
            self.records: tuple[tuple[Mapping[str, object], object], ...] = ()

        def log_success_event(
            self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
        ) -> None:
            self.records = (*self.records, (kwargs, response_obj))
            self.finished.set()

        async def async_log_success_event(
            self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
        ) -> None:
            self.log_success_event(kwargs, response_obj, start_time, end_time)

    recorder: Final = Recorder()
    if global_callback:
        monkeypatch.setattr(litellm, "callbacks", [recorder])
    provider_params: Final = {
        **{key: value for key, value in PARAMS.items() if schema or key != "outputSchema"},
        "contents": PARAMS["contents"] if schema else {"highlights": {"numSentences": 2}},
    }

    def respond(request: Request) -> Reply:
        assert request.headers["x-api-key"] == "test-key"
        assert request.headers["exa-beta"] == "dynamic-highlights-2026-08-28"
        assert json.loads(request.body) == {"query": "question", "numResults": 1, **provider_params}
        return _respond(request)

    with wire_server(respond) as server:
        params: Final = {
            "query": "question",
            "search_provider": "exa_ai",
            "api_base": f"{server.url}/exa",
            "max_results": 1,
            "api_key": "test-key",
            "extra_headers": {"Exa-Beta": "dynamic-highlights-2026-08-28"},
            **({"input_cost_per_query": price_override} if price_override is not None else {}),
            **(
                {}
                if global_callback
                else {"success_callback": [recorder.async_log_success_event if asynchronous else recorder]}
            ),
            **provider_params,
        }
        response: Final = await litellm.asearch(**params) if asynchronous else litellm.search(**params)
        assert len(server.drain()) == 1
        assert isinstance(response, SearchResponse)
        assert response.model_dump(mode="json") == _EXPECTED
        assert await asyncio.to_thread(recorder.finished.wait, 5)
        assert len(recorder.records) == 1
        details, logged = recorder.records[0]
        assert isinstance(logged, SearchResponse)
        assert logged.model_dump(mode="json") == _EXPECTED
        assert isinstance(details["standard_logging_object"], Mapping)
        expected_cost: Final = _REPORTED_COST if price_override is None else price_override
        assert response.hidden_params["response_cost"] == expected_cost
        assert details["response_cost"] == expected_cost
        assert details["standard_logging_object"]["response_cost"] == expected_cost
        assert details["stream"] is False
