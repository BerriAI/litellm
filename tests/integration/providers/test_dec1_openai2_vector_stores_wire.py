"""Contract: OpenAI vector store and vector store file routes for a store registered on the proxy.

A store registered with `POST /vector_store/new` (`custom_llm_provider: openai` plus `litellm_params` with
`api_base` and `api_key`) is the way an admin pins credentials to an OpenAI vector store. Every later route that
names the store id picks those credentials up from the managed store row:

- `GET`, `POST` (update) and `DELETE /v1/vector_stores/{id}` through
  `_update_request_data_with_litellm_managed_vector_store_registry` (`vector_store_endpoints/endpoints.py:125`);
- `/v1/vector_stores/{id}/files[...]` through `assert_user_can_access_vector_store_id` and
  `_update_request_data_with_litellm_managed_vector_store_registry` in `vector_store_files_endpoints/endpoints.py`,
  which merges the row's `litellm_params` into the request (`endpoints.py:645`).

Each route reaches the same OpenAI path on the store's `api_base` with the store's key, keeps the raw store id,
and forwards the OpenAI fields (attributes, chunking strategy, list cursors and filters) unchanged. Create and
search are pinned in `test_vector_store_create_wire.py` and `test_vector_store_search_wire.py`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, object_value
from integration.providers.dec1_openai2_support import calls_to, resource_respond, unique
from integration.providers.dec1_openai_support import Peer, peer
from pydantic import JsonValue

_STORE_KEY: Final = "sk-dec1-openai2-store-key"
_CHUNKING: Final[dict[str, JsonValue]] = {
    "type": "static",
    "static": {"max_chunk_size_tokens": 600, "chunk_overlap_tokens": 100},
}


@contextmanager
def _registered_store(gateway: Gateway) -> Iterator[tuple[Peer, str]]:
    """A peer and a store id registered on the proxy with the peer as `api_base` and `_STORE_KEY`."""
    store: Final = unique("vs_dec1openai2_")
    with peer(resource_respond) as provider, gateway.scenario() as scenario:
        registered: Final = gateway.request(
            "POST",
            "/vector_store/new",
            {
                "vector_store_id": store,
                "custom_llm_provider": "openai",
                "litellm_params": {"api_base": f"{provider.url}/v1", "api_key": _STORE_KEY},
            },
        )
        assert registered.status_code == 200, registered.text
        scenario.cleanups.callback(gateway.post, "/vector_store/delete", {"vector_store_id": store})
        yield provider, store


def _only(provider: Peer, method: str, prefix: str) -> tuple[str, dict[str, list[str]], dict[str, JsonValue]]:
    calls: Final = [call for call in calls_to(provider.all(), prefix) if call.method == method]
    assert len(calls) == 1, [(call.method, call.path) for call in provider.all()]
    call: Final = calls[0]
    assert call.headers["authorization"] == f"Bearer {_STORE_KEY}", dict(call.headers)
    return call.path, {key: list(value) for key, value in call.query.items()}, dict(call.body)


def test_retrieve_uses_the_registered_store_credentials(gateway: Gateway) -> None:
    """The caller calls `GET /v1/vector_stores/<registered id>`.

    The provider receives `GET /v1/vector_stores/<id>` with `authorization: Bearer <store key>` and the caller
    gets 200 with `object: vector_store` and the same id.
    """
    with _registered_store(gateway) as (provider, store):
        response: Final = gateway.request("GET", f"/v1/vector_stores/{store}")
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert (body["object"], body["id"]) == ("vector_store", store), body
        path, _, _ = _only(provider, "GET", "/vector_stores/")
        assert path == f"/v1/vector_stores/{store}"


def test_update_forwards_name_and_expiry_to_the_store_path(gateway: Gateway) -> None:
    """The caller posts `{"name": "dec1 renamed", "expires_after": {"anchor": "last_active_at", "days": 3}}` to
    `/v1/vector_stores/<id>`.

    The provider receives `POST /v1/vector_stores/<id>` whose body has the same `name` and `expires_after`, and no
    `vector_store_id`, `api_base` or `api_key` field leaks into the body. UNVERIFIED: the update body is built from
    the update params only (`litellm/vector_stores/main.py`).
    """
    with _registered_store(gateway) as (provider, store):
        response: Final = gateway.request(
            "POST",
            f"/v1/vector_stores/{store}",
            {"name": "dec1 renamed", "expires_after": {"anchor": "last_active_at", "days": 3}},
        )
        assert response.status_code == 200, response.text
        path, _, body = _only(provider, "POST", "/vector_stores/")
        assert path == f"/v1/vector_stores/{store}"
        assert (body["name"], body["expires_after"]) == ("dec1 renamed", {"anchor": "last_active_at", "days": 3})
        assert not {"vector_store_id", "api_base", "api_key", "custom_llm_provider"} & body.keys(), body


def test_delete_reaches_the_store_path_and_returns_deleted(gateway: Gateway) -> None:
    """The caller calls `DELETE /v1/vector_stores/<id>`.

    The provider receives `DELETE /v1/vector_stores/<id>` with the store key and the caller gets 200 with
    `object: vector_store.deleted` and `deleted: true`.
    """
    with _registered_store(gateway) as (provider, store):
        response: Final = gateway.request("DELETE", f"/v1/vector_stores/{store}")
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert (body["object"], body["deleted"]) == ("vector_store.deleted", True), body
        path, _, _ = _only(provider, "DELETE", "/vector_stores/")
        assert path == f"/v1/vector_stores/{store}"


def test_file_attach_forwards_file_id_attributes_and_chunking(gateway: Gateway) -> None:
    """The caller posts `{"file_id": "file-dec1-a", "attributes": {"team": "blue", "year": 2026},
    "chunking_strategy": <static 600/100>}` to `/v1/vector_stores/<id>/files`.

    The provider receives `POST /v1/vector_stores/<id>/files` with the store key and exactly those three fields;
    the caller gets the `vector_store.file` object for `file-dec1-a`.
    """
    sent: Final[dict[str, JsonValue]] = {
        "file_id": "file-dec1-a",
        "attributes": {"team": "blue", "year": 2026},
        "chunking_strategy": _CHUNKING,
    }
    with _registered_store(gateway) as (provider, store):
        response: Final = gateway.request("POST", f"/v1/vector_stores/{store}/files", sent)
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["id"] == "file-dec1-a"
        path, _, body = _only(provider, "POST", "/vector_stores/")
        assert path == f"/v1/vector_stores/{store}/files"
        assert body == sent


def test_file_list_forwards_cursor_order_and_status_filter(gateway: Gateway) -> None:
    """The caller lists `/v1/vector_stores/<id>/files?limit=2&order=asc&after=file-dec1-prev&filter=completed`.

    The provider receives one `GET /v1/vector_stores/<id>/files` whose query has those four values unchanged and
    the caller gets the provider's list page.
    """
    with _registered_store(gateway) as (provider, store):
        response: Final = gateway.request(
            "GET",
            f"/v1/vector_stores/{store}/files",
            params={"limit": "2", "order": "asc", "after": "file-dec1-prev", "filter": "completed"},
        )
        assert response.status_code == 200, response.text
        path, query, _ = _only(provider, "GET", "/vector_stores/")
        assert path == f"/v1/vector_stores/{store}/files"
        assert {key: query.get(key) for key in ("limit", "order", "after", "filter")} == {
            "limit": ["2"],
            "order": ["asc"],
            "after": ["file-dec1-prev"],
            "filter": ["completed"],
        }, query


def test_file_retrieve_reaches_the_file_path(gateway: Gateway) -> None:
    """The caller calls `GET /v1/vector_stores/<id>/files/file-dec1-a`.

    The provider receives the same path with the store key and the caller gets `object: vector_store.file`
    naming the store.
    """
    with _registered_store(gateway) as (provider, store):
        response: Final = gateway.request("GET", f"/v1/vector_stores/{store}/files/file-dec1-a")
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert (body["object"], body["vector_store_id"]) == ("vector_store.file", store), body
        path, _, _ = _only(provider, "GET", "/vector_stores/")
        assert path == f"/v1/vector_stores/{store}/files/file-dec1-a"


def test_file_attribute_update_forwards_only_the_attributes(gateway: Gateway) -> None:
    """The caller posts `{"attributes": {"team": "green"}}` to `/v1/vector_stores/<id>/files/file-dec1-a`.

    The provider receives `POST` on the same path with body exactly `{"attributes": {"team": "green"}}` and the
    caller's answer shows the new attributes.
    """
    with _registered_store(gateway) as (provider, store):
        response: Final = gateway.request(
            "POST", f"/v1/vector_stores/{store}/files/file-dec1-a", {"attributes": {"team": "green"}}
        )
        assert response.status_code == 200, response.text
        assert object_value(JSON_OBJECT.validate_json(response.content)["attributes"]) == {"team": "green"}
        path, _, body = _only(provider, "POST", "/vector_stores/")
        assert path == f"/v1/vector_stores/{store}/files/file-dec1-a"
        assert body == {"attributes": {"team": "green"}}


def test_file_content_returns_the_parsed_text(gateway: Gateway) -> None:
    """The caller calls `GET /v1/vector_stores/<id>/files/file-dec1-a/content`.

    The provider receives that path and the caller gets its parsed content list with the text
    `dec1 openai2 file text`.
    """
    with _registered_store(gateway) as (provider, store):
        response: Final = gateway.request("GET", f"/v1/vector_stores/{store}/files/file-dec1-a/content")
        assert response.status_code == 200, response.text
        assert "dec1 openai2 file text" in response.text
        path, _, _ = _only(provider, "GET", "/vector_stores/")
        assert path == f"/v1/vector_stores/{store}/files/file-dec1-a/content"


def test_file_delete_detaches_the_file(gateway: Gateway) -> None:
    """The caller calls `DELETE /v1/vector_stores/<id>/files/file-dec1-a`.

    The provider receives that DELETE with the store key and the caller gets `deleted: true` for
    `file-dec1-a`.
    """
    with _registered_store(gateway) as (provider, store):
        response: Final = gateway.request("DELETE", f"/v1/vector_stores/{store}/files/file-dec1-a")
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert (body["id"], body["deleted"]) == ("file-dec1-a", True), body
        path, _, _ = _only(provider, "DELETE", "/vector_stores/")
        assert path == f"/v1/vector_stores/{store}/files/file-dec1-a"


@pytest.mark.parametrize(
    ("method", "suffix"),
    [("GET", ""), ("DELETE", ""), ("GET", "/files"), ("POST", "/files")],
    ids=["retrieve", "delete", "file_list", "file_attach"],
)
def test_a_key_outside_the_store_team_is_refused(gateway: Gateway, method: str, suffix: str) -> None:
    """A store is registered with `team_id` of team A; a key of team B calls `<method> /v1/vector_stores/<id><suffix>`.

    The key gets 403 and the provider receives nothing for that store (`assert_user_can_access_vector_store`).
    UNVERIFIED: `/vector_store/new` accepts `team_id` and the refusal status is 403, not 404.
    """
    store: Final = unique("vs_dec1openai2team_")
    with peer(resource_respond) as provider, gateway.scenario() as scenario:
        owner_team: Final = scenario.team()
        other_team: Final = scenario.team()
        registered: Final = gateway.request(
            "POST",
            "/vector_store/new",
            {
                "vector_store_id": store,
                "custom_llm_provider": "openai",
                "team_id": owner_team,
                "litellm_params": {"api_base": f"{provider.url}/v1", "api_key": _STORE_KEY},
            },
        )
        assert registered.status_code == 200, registered.text
        scenario.cleanups.callback(gateway.post, "/vector_store/delete", {"vector_store_id": store})
        outsider: Final = scenario.key(team_id=other_team)
        body: Final[dict[str, JsonValue] | None] = {"file_id": "file-dec1-a"} if method == "POST" else None
        response: Final = gateway.request(method, f"/v1/vector_stores/{store}{suffix}", body, key=outsider)
        assert response.status_code == 403, response.text
        assert not calls_to(provider.all(), "/vector_stores"), provider.all()
