from __future__ import annotations

from typing import Final
from urllib.parse import urlsplit

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
VECTOR_STORE_RESPONSE: Final = b'{"id":"vs_internal_metadata","object":"vector_store"}'


def _openai_vector_store(request: Request) -> Reply:
    path: Final = urlsplit(request.target).path
    if request.method != "POST" or not path.startswith("/v1/vector_stores"):
        return Reply(status=404)
    return Reply(body=VECTOR_STORE_RESPONSE)


def test_vector_store_create_and_update_only_forward_caller_metadata(gateway: Gateway) -> None:
    with wire_server(_openai_vector_store) as upstream, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{upstream.url}/v1")
        key: Final = scenario.key(models=[model])

        create_response: Final = gateway.request(
            "POST",
            "/v1/vector_stores",
            {"model": model, "name": "x"},
            key=key,
        )
        assert create_response.status_code == 200, create_response.text

        update_response: Final = gateway.request(
            "POST",
            "/v1/vector_stores/vs_internal_metadata",
            {"model": model, "metadata": {"team": "llmproxy"}},
            key=key,
        )
        assert update_response.status_code == 200, update_response.text

        requests: Final = upstream.drain()
        assert tuple((request.method, urlsplit(request.target).path) for request in requests) == (
            ("POST", "/v1/vector_stores"),
            ("POST", "/v1/vector_stores/vs_internal_metadata"),
        )
        create_body: Final = JSON_OBJECT.validate_json(requests[0].body)
        update_body: Final = JSON_OBJECT.validate_json(requests[1].body)
        assert create_body.get("metadata") in (None, {})
        assert update_body.get("metadata") == {"team": "llmproxy"}
