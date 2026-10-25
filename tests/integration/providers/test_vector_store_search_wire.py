import json
import uuid
from typing import Final
from urllib.parse import urlsplit

import httpx
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from openai import OpenAI
from pydantic import BaseModel, JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_SEARCH_PATH_SUFFIX: Final = "/vector_stores/{vector_store_id}/search"
_FILTERS: Final[dict[str, JsonValue]] = {
    "type": "and",
    "filters": [{"type": "eq", "key": "tenant", "value": "a"}],
}
_SEARCH_BODY: Final[dict[str, JsonValue]] = {
    "query": ["a", "b"],
    "filters": _FILTERS,
    "max_num_results": 3,
    "ranking_options": {"ranker": "auto", "score_threshold": 0.2},
    "rewrite_query": True,
}
_ALIAS_BODY: Final[dict[str, JsonValue]] = {
    "query": "c",
    "filters": None,
    "max_num_results": None,
    "ranking_options": None,
    "rewrite_query": None,
}
_ALIAS_REQUEST_BODY: Final[dict[str, JsonValue]] = {"query": "c"}
_SEARCH_RESULT: Final[dict[str, JsonValue]] = {
    "file_id": "file-search-wire",
    "filename": "search.txt",
    "score": 0.91,
    "attributes": {"tenant": "a"},
    "content": [{"type": "text", "text": "matching content"}],
}


class _SearchContent(BaseModel):
    type: str
    text: str


class _SearchResult(BaseModel):
    file_id: str
    filename: str
    score: float
    attributes: dict[str, str]
    content: tuple[_SearchContent, ...]


class _SearchPage(BaseModel):
    object: str
    search_query: str
    data: tuple[_SearchResult, ...]
    has_more: bool
    next_page: str | None


def _search_reply(query: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "object": "vector_store.search_results.page",
                "search_query": query,
                "data": [_SEARCH_RESULT],
                "has_more": False,
                "next_page": None,
            }
        ).encode()
    )


def test_vector_store_search_forwards_all_search_fields_on_both_routes(gateway: Gateway) -> None:
    store_id: Final = f"vs_search_{uuid.uuid4().hex}"
    registry_key: Final = f"registry-search-{uuid.uuid4().hex}"
    search_path: Final = _SEARCH_PATH_SUFFIX.format(vector_store_id=store_id)
    expected_sdk_response: Final = {
        "object": "vector_store.search_results.page",
        "search_query": "a b",
        "data": [_SEARCH_RESULT],
        "has_more": False,
        "next_page": None,
    }
    expected_alias_response: Final = {
        **expected_sdk_response,
        "search_query": "c",
    }

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert urlsplit(request.target).path == f"/v1{search_path}", request.target
        assert request.headers["authorization"] == f"Bearer {registry_key}"
        body: Final = JSON_OBJECT.validate_json(request.body)
        if body.get("query") == ["a", "b"]:
            assert body == _SEARCH_BODY
            return _search_reply("a b")
        assert body == _ALIAS_BODY
        return _search_reply("c")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        registered: Final = gateway.request(
            "POST",
            "/vector_store/new",
            {
                "vector_store_id": store_id,
                "custom_llm_provider": "openai",
                "litellm_params": {"api_base": f"{wire.url}/v1", "api_key": registry_key},
            },
        )
        assert registered.status_code == 200, registered.text
        scenario.cleanups.callback(gateway.post, "/vector_store/delete", {"vector_store_id": store_id})

        with OpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1",
            api_key=gateway.key,
            http_client=httpx.Client(trust_env=False),
            max_retries=0,
        ) as client:
            sdk_response: Final = client.vector_stores.search(
                store_id,
                query=["a", "b"],
                filters=_FILTERS,
                max_num_results=3,
                ranking_options={"ranker": "auto", "score_threshold": 0.2},
                rewrite_query=True,
            )
            assert sdk_response.model_dump(exclude_unset=True) == expected_sdk_response

        alias_response: Final = gateway.request("POST", f"{search_path}", _ALIAS_REQUEST_BODY)
        assert alias_response.status_code == 200, alias_response.text
        assert _SearchPage.model_validate_json(alias_response.content).model_dump(mode="json") == expected_alias_response

        requests: Final = wire.drain()
        assert [(request.method, request.target) for request in requests] == [
            ("POST", f"/v1{search_path}"),
            ("POST", f"/v1{search_path}"),
        ]
