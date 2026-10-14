import base64
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Final
from urllib.parse import urlsplit

import httpx
from integration._support.client import Gateway, eventually, string_value
from integration._support.database import read_rows, write_rows
from integration._support.wire import Reply, Request, Wire, wire_server
from openai import OpenAI
from pydantic import BaseModel, JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_SEARCH_BODY: Final[dict[str, JsonValue]] = {
    "query": "unified search",
    "filters": None,
    "max_num_results": None,
    "ranking_options": None,
    "rewrite_query": None,
}
_SEARCH_RESPONSE: Final[dict[str, JsonValue]] = {
    "object": "vector_store.search_results.page",
    "search_query": "unified search",
    "data": [
        {
            "file_id": "file-managed-vector-store",
            "filename": "managed-store.txt",
            "score": 0.93,
            "attributes": {"team": "a"},
            "content": [{"type": "text", "text": "managed store content"}],
        }
    ],
    "has_more": False,
    "next_page": None,
}
_MANAGED_STORE_ROW: Final = (
    'SELECT created_by, team_id FROM "LiteLLM_ManagedVectorStoreTable" WHERE unified_resource_id = %s'
)


class _Content(BaseModel):
    type: str
    text: str


class _Result(BaseModel):
    file_id: str
    filename: str
    score: float
    attributes: dict[str, str]
    content: tuple[_Content, ...]


class _SearchPage(BaseModel):
    object: str
    search_query: str
    data: tuple[_Result, ...]
    has_more: bool
    next_page: str | None


def _deployment_id(model: str) -> str:
    rows: Final = read_rows(
        'SELECT model_id FROM "LiteLLM_ProxyModelTable" WHERE model_name = %s '
        "OR model_info->>'team_public_model_name' = %s",
        (model, model),
    )
    assert len(rows) == 1, f"Expected one deployment for {model!r}, found {rows}"
    return string_value(rows[0]["model_id"])


def _worker_has_deployment(gateway: Gateway, deployment_id: str, _: int) -> bool:
    with httpx.Client(base_url=gateway.client.base_url, timeout=15, trust_env=False) as client:
        response: Final = client.get(
            "/model/info",
            params={"litellm_model_id": deployment_id},
            headers={"Authorization": f"Bearer {gateway.key}"},
        )
    if response.status_code == 200:
        return True
    assert response.status_code == 400, response.text
    assert "not found on litellm proxy" in response.text.lower(), response.text
    return False


def _wait_until_every_worker_serves(gateway: Gateway, model: str) -> None:
    deployment_id: Final = _deployment_id(model)

    def every_worker_serves() -> bool:
        with ThreadPoolExecutor(max_workers=16) as pool:
            rounds: Final = tuple(
                tuple(pool.map(partial(_worker_has_deployment, gateway, deployment_id), range(16))) for _ in range(2)
            )
        return all(has_deployment for round_ in rounds for has_deployment in round_)

    eventually(every_worker_serves, lambda served: served, seconds=90)


def _create_reply(provider_id: str, name: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": provider_id,
                "object": "vector_store",
                "created_at": 1700000000,
                "name": name,
                "file_counts": {
                    "in_progress": 0,
                    "completed": 0,
                    "failed": 0,
                    "cancelled": 0,
                    "total": 0,
                },
                "status": "completed",
                "usage_bytes": 0,
                "expires_after": None,
                "expires_at": None,
                "last_active_at": None,
                "metadata": {},
            }
        ).encode()
    )


def _search_reply() -> Reply:
    return Reply(body=json.dumps(_SEARCH_RESPONSE).encode())


def _models_reply() -> Reply:
    return Reply(body=json.dumps({"object": "list", "data": []}).encode())


def _store_creation_requests(wire: Wire) -> tuple[Request, ...]:
    requests: Final = wire.drain()
    assert all(
        (request.method, request.target) in {("GET", "/v1/models"), ("POST", "/v1/vector_stores")}
        for request in requests
    ), requests
    return tuple(request for request in requests if request.method == "POST")


def _create_request_body(name: str) -> dict[str, JsonValue]:
    return {
        "name": name,
        "file_ids": None,
        "expires_after": None,
        "chunking_strategy": None,
    }


def _assert_create_request(request: Request, name: str) -> None:
    body: Final = JSON_OBJECT.validate_json(request.body)
    assert {key: value for key, value in body.items() if key != "metadata"} == _create_request_body(name)


def _delete_managed_store_row(unified_id: str) -> None:
    write_rows(
        'DELETE FROM "LiteLLM_ManagedVectorStoreTable" WHERE unified_resource_id = %s',
        (unified_id,),
    )


def test_managed_vector_store_id_routes_to_creator_deployment_and_enforces_ownership(gateway: Gateway) -> None:
    store_name: Final = f"managed-store-{uuid.uuid4().hex}"
    provider_id_a: Final = f"vs_provider_a_{uuid.uuid4().hex}"
    provider_id_b: Final = f"vs_provider_b_{uuid.uuid4().hex}"
    provider_key_a: Final = f"provider-a-{uuid.uuid4().hex}"
    provider_key_b: Final = f"provider-b-{uuid.uuid4().hex}"

    def respond_a(request: Request) -> Reply:
        path: Final = urlsplit(request.target).path
        assert request.headers["authorization"] == f"Bearer {provider_key_a}"
        if request.method == "GET" and path == "/v1/models":
            return _models_reply()
        if request.method == "POST" and path == "/v1/vector_stores":
            _assert_create_request(request, store_name)
            return _create_reply(provider_id_a, store_name)
        assert request.method == "POST"
        assert path == f"/v1/vector_stores/{provider_id_a}/search", request.target
        assert JSON_OBJECT.validate_json(request.body) == _SEARCH_BODY
        return _search_reply()

    def respond_b(request: Request) -> Reply:
        assert request.headers["authorization"] == f"Bearer {provider_key_b}"
        if request.method == "GET" and request.target == "/v1/models":
            return _models_reply()
        assert request.method == "POST"
        assert urlsplit(request.target).path == "/v1/vector_stores", request.target
        _assert_create_request(request, store_name)
        return _create_reply(provider_id_b, store_name)

    with (
        wire_server(respond_a) as wire_a,
        wire_server(respond_b) as wire_b,
        gateway.scenario() as scenario,
    ):
        model_a: Final = scenario.model(api_base=f"{wire_a.url}/v1", api_key=provider_key_a)
        model_b: Final = scenario.model(api_base=f"{wire_b.url}/v1", api_key=provider_key_b)
        _wait_until_every_worker_serves(gateway, model_a)
        _wait_until_every_worker_serves(gateway, model_b)
        creator_team: Final = scenario.team(models=[model_a, model_b])
        creator_user: Final = scenario.member(creator_team)
        creator_key: Final = scenario.key(team_id=creator_team, user_id=creator_user, models=[model_a, model_b])
        other_team: Final = scenario.team(models=[model_a, model_b])
        other_user: Final = scenario.member(other_team)
        other_key: Final = scenario.key(team_id=other_team, user_id=other_user, models=[model_a, model_b])

        with OpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1",
            api_key=creator_key,
            http_client=httpx.Client(trust_env=False),
            max_retries=0,
        ) as client:
            created: Final = client.vector_stores.create(
                name=store_name,
                extra_body={"target_model_names": f"{model_a},{model_b}"},
            )

        unified_id: Final = created.id
        decoded_id: Final = base64.urlsafe_b64decode(unified_id + "=" * (-len(unified_id) % 4)).decode()
        assert decoded_id.startswith("litellm_proxy:vector_store;"), decoded_id
        assert f"target_model_names,{model_a},{model_b};" in decoded_id, decoded_id
        assert f"resource_id,{provider_id_a};" in decoded_id, decoded_id
        scenario.cleanups.callback(_delete_managed_store_row, unified_id)
        created_rows: Final = read_rows(_MANAGED_STORE_ROW, (unified_id,))
        assert created_rows == [{"created_by": creator_user, "team_id": creator_team}]
        assert [(request.method, request.target) for request in _store_creation_requests(wire_a)] == [
            ("POST", "/v1/vector_stores")
        ]
        assert [(request.method, request.target) for request in _store_creation_requests(wire_b)] == [
            ("POST", "/v1/vector_stores")
        ]

        search_path: Final = f"/v1/vector_stores/{unified_id}/search"
        with OpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1",
            api_key=creator_key,
            http_client=httpx.Client(trust_env=False),
            max_retries=0,
        ) as client:
            searched: Final = client.vector_stores.search(unified_id, query="unified search")
            assert searched.model_dump(exclude_unset=True) == _SEARCH_RESPONSE

        alias_search: Final = gateway.request(
            "POST",
            f"/vector_stores/{unified_id}/search",
            {"query": "unified search"},
            key=creator_key,
        )
        assert alias_search.status_code == 200, alias_search.text
        assert _SearchPage.model_validate_json(alias_search.content).model_dump(mode="json") == _SEARCH_RESPONSE
        assert [(request.method, request.target) for request in wire_a.drain()] == [
            ("POST", f"/v1/vector_stores/{provider_id_a}/search"),
            ("POST", f"/v1/vector_stores/{provider_id_a}/search"),
        ]
        assert wire_b.drain() == ()

        denied: Final = gateway.request("POST", search_path, _SEARCH_BODY, key=other_key)
        assert denied.status_code == 403, denied.text
        assert wire_a.drain() == ()
        assert wire_b.drain() == ()

        db_row_after_denial: Final = read_rows(_MANAGED_STORE_ROW, (unified_id,))
        assert db_row_after_denial == [{"created_by": creator_user, "team_id": creator_team}]
