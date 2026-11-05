import json
import uuid
from typing import Final

import pytest
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

import litellm
from litellm.llms.base_llm.search.transformation import SearchResponse

pytestmark: Final = pytest.mark.timeout(60)

_ENVELOPE: Final[dict[str, JsonValue]] = {
    "requestId": "exa-request-sdk",
    "resolvedSearchType": "neural",
    "searchTime": 211.0,
    "costDollars": {"total": 0.005, "search": {"neural": 0.005}},
}
_RESULT: Final[dict[str, JsonValue]] = {
    "id": "https://result.invalid/sdk",
    "title": "SDK result",
    "url": "https://result.invalid/sdk",
    "publishedDate": "2026-02-01T00:00:00.000Z",
    "score": 0.73,
    "highlights": ["First line.", "Second line."],
    "highlightScores": [0.61, 0.42],
}
_EXPECTED: Final[dict[str, JsonValue]] = {
    "object": "search",
    "results": [
        {
            "title": "SDK result",
            "url": "https://result.invalid/sdk",
            "snippet": "First line.\n\nSecond line.",
            "date": "2026-02-01T00:00:00.000Z",
            "last_updated": None,
            "highlights": ["First line.", "Second line."],
            "highlight_scores": [0.61, 0.42],
            "score": 0.73,
        }
    ],
    "provider_metadata": _ENVELOPE,
}


def _respond(request: Request) -> Reply:
    assert (request.method, request.target) == ("POST", "/exa/search"), request.target
    return Reply(body=json.dumps({**_ENVELOPE, "results": [_RESULT]}).encode())


def test_sync_exa_search_returns_highlights_scores_and_the_response_envelope() -> None:
    with wire_server(_respond) as wire:
        response: Final = litellm.search(
            query=f"sdk sync {uuid.uuid4().hex}",
            search_provider="exa_ai",
            api_key="synthetic-exa-key",
            api_base=f"{wire.url}/exa",
            max_results=1,
            contents={"highlights": {"numSentences": 2}},
        )
        assert len(wire.drain()) == 1

    assert isinstance(response, SearchResponse), type(response)
    assert response.model_dump(mode="json") == _EXPECTED


@pytest.mark.asyncio
async def test_async_exa_search_returns_highlights_scores_and_the_response_envelope() -> None:
    with wire_server(_respond) as wire:
        response: Final = await litellm.asearch(
            query=f"sdk async {uuid.uuid4().hex}",
            search_provider="exa_ai",
            api_key="synthetic-exa-key",
            api_base=f"{wire.url}/exa",
            max_results=1,
            contents={"highlights": {"numSentences": 2}},
        )
        assert len(wire.drain()) == 1

    assert isinstance(response, SearchResponse), type(response)
    assert response.model_dump(mode="json") == _EXPECTED
