import asyncio
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value
from integration.authorization._guardrail_opt_out import upstream_observations
from pydantic import JsonValue

CONFIG_STORE_ID: Final = "vs_integration_config_store"
SEARCH_PATH: Final = f"/vector_stores/{CONFIG_STORE_ID}/search"
JsonObject: TypeAlias = dict[str, JsonValue]


def _json_array(*values: JsonValue) -> JsonValue:
    return [*values]  # mutable-ok: request payloads require JSON arrays


def _permission_for_stores(*store_ids: str) -> JsonObject:
    permission: Final[JsonObject] = {"vector_stores": _json_array(*store_ids)}
    return permission


def _retrieval_config_without_store() -> JsonObject:
    config: Final[JsonObject] = {"custom_llm_provider": "openai", "top_k": 1}
    return config


def _key_for_scope(scenario: Scenario, model: str, scope: Literal["key", "team"], store_id: str) -> str:
    if scope == "key":
        return scenario.key(models=_json_array(model), object_permission=_permission_for_stores(store_id))
    team: Final = scenario.team(models=_json_array(model), object_permission=_permission_for_stores(store_id))
    return scenario.key(team_id=team, models=_json_array(model))


def _key_with_conflicting_allowlists(scenario: Scenario, model: str, scope: Literal["team", "key"]) -> str:
    if scope == "team":
        denied_team: Final = scenario.team(
            models=_json_array(model), object_permission=_permission_for_stores("vs_team_excluded")
        )
        return scenario.key(
            team_id=denied_team,
            models=_json_array(model),
            object_permission=_permission_for_stores(CONFIG_STORE_ID),
        )
    allowed_team: Final = scenario.team(
        models=_json_array(model), object_permission=_permission_for_stores(CONFIG_STORE_ID)
    )
    return scenario.key(
        team_id=allowed_team,
        models=_json_array(model),
        object_permission=_permission_for_stores("vs_key_excluded"),
    )


def _rag_query_body(model: str, marker: str, vector_store_id: JsonValue, *, stream: bool = False) -> JsonObject:
    body: Final[JsonObject] = {
        "model": model,
        "messages": _json_array({"role": "user", "content": marker}),
        "retrieval_config": {"vector_store_id": vector_store_id, "custom_llm_provider": "openai", "top_k": 1},
        **({"stream": True} if stream else {}),
    }
    return body


def _rag_query(
    gateway: Gateway,
    model: str,
    marker: str,
    key: str,
    *,
    vector_store_id: JsonValue = CONFIG_STORE_ID,
    stream: bool = False,
) -> httpx.Response:
    return gateway.request(
        "POST", "/v1/rag/query", _rag_query_body(model, marker, vector_store_id, stream=stream), key=key
    )


async def _async_rag_query(
    client: httpx.AsyncClient, model: str, marker: str, key: str, vector_store_id: str
) -> httpx.Response:
    headers: Final[Mapping[str, str]] = MappingProxyType({"Authorization": f"Bearer {key}"})
    return await client.post(
        "/v1/rag/query",
        json=_rag_query_body(model, marker, vector_store_id),
        headers=headers,
    )


def _requests_for_marker(gateway: Gateway, marker: str) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(observation for observation in upstream_observations(gateway) if marker in str(observation["body"]))


def _searches_in(
    observations: tuple[Mapping[str, JsonValue], ...], marker: str, path: str = SEARCH_PATH
) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(
        observation
        for observation in observations
        if observation["path"] == path and marker in str(observation["body"])
    )


def _warm_rag_query_model(gateway: Gateway, model: str, key: str) -> None:
    markers: Final = tuple(f"lit5610 rag query model warm {uuid.uuid4().hex}" for _ in range(8))
    with ThreadPoolExecutor(max_workers=8) as executor:
        responses: Final = eventually(
            lambda: tuple(executor.map(_rag_query, repeat(gateway), repeat(model), markers, repeat(key))),
            lambda values: tuple(response.status_code for response in values) == (200,) * 8,
            seconds=30,
        )
    assert tuple(response.status_code for response in responses) == (200,) * 8, tuple(
        response.text for response in responses
    )


@pytest.mark.parametrize(
    ("scope", "error_type"),
    (("key", "key_vector_store_access_denied"), ("team", "team_vector_store_access_denied")),
)
def test_rag_query_is_denied_when_key_or_team_allowlist_excludes_store(
    gateway: Gateway, scope: Literal["key", "team"], error_type: str
) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = _key_for_scope(scenario, model, scope, "vs_some_other_store")
        marker: Final = f"lit5610 rag query denied {uuid.uuid4().hex}"

        response: Final = _rag_query(gateway, model, marker, key)

        assert response.status_code == 401, response.text
        assert response.json()["error"]["type"] == error_type, response.text
        assert (
            tuple(request for request in _requests_for_marker(gateway, marker) if request["path"] == SEARCH_PATH) == ()
        )


@pytest.mark.parametrize("scope", ("key", "team"))
def test_rag_query_searches_configured_store_when_allowlist_includes_it(
    gateway: Gateway, scope: Literal["key", "team"]
) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = _key_for_scope(scenario, model, scope, CONFIG_STORE_ID)
        marker: Final = f"lit5610 rag query allowed {uuid.uuid4().hex}"

        response: Final = _rag_query(gateway, model, marker, key)
        assert response.status_code == 200, response.text

        searches: Final = tuple(
            request for request in _requests_for_marker(gateway, marker) if request["path"] == SEARCH_PATH
        )
        assert len(searches) == 1, searches
        assert marker in str(object_value(searches[0]["body"])["query"]), searches


def test_rag_query_without_key_object_permission_can_search_store(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=_json_array(model))
        marker: Final = f"lit5610 rag query no permission {uuid.uuid4().hex}"

        response: Final = _rag_query(gateway, model, marker, key)
        assert response.status_code == 200, response.text

        searches: Final = tuple(
            request for request in _requests_for_marker(gateway, marker) if request["path"] == SEARCH_PATH
        )
        assert len(searches) == 1, searches
        assert marker in str(object_value(searches[0]["body"])["query"]), searches


def test_rag_query_with_empty_key_vector_store_allowlist_can_search_store(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=_json_array(model), object_permission=_permission_for_stores())
        marker: Final = f"lit5610 rag query empty allowlist {uuid.uuid4().hex}"

        response: Final = _rag_query(gateway, model, marker, key)
        assert response.status_code == 200, response.text

        searches: Final = tuple(
            request for request in _requests_for_marker(gateway, marker) if request["path"] == SEARCH_PATH
        )
        assert len(searches) == 1, searches
        assert marker in str(object_value(searches[0]["body"])["query"]), searches


def test_rag_query_with_master_key_can_search_store(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        marker: Final = f"lit5610 rag query master key {uuid.uuid4().hex}"

        response: Final = _rag_query(gateway, model, marker, gateway.key)
        assert response.status_code == 200, response.text

        searches: Final = tuple(
            request for request in _requests_for_marker(gateway, marker) if request["path"] == SEARCH_PATH
        )
        assert len(searches) == 1, searches
        assert marker in str(object_value(searches[0]["body"])["query"]), searches


@pytest.mark.parametrize(
    ("scope", "error_type"),
    (("team", "team_vector_store_access_denied"), ("key", "key_vector_store_access_denied")),
)
def test_rag_query_denies_store_when_key_and_team_permissions_conflict(
    gateway: Gateway, scope: Literal["team", "key"], error_type: str
) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = _key_with_conflicting_allowlists(scenario, model, scope)
        marker: Final = f"lit5610 rag query conflict {scope} {uuid.uuid4().hex}"

        response: Final = _rag_query(gateway, model, marker, key)
        assert response.status_code == 401, response.text
        assert response.json()["error"]["type"] == error_type, response.text
        assert _searches_in(upstream_observations(gateway), marker) == ()


def test_async_rag_query_checks_team_vector_store_allowlist(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        denied_team: Final = scenario.team(
            models=_json_array(model), object_permission=_permission_for_stores("vs_team_excluded")
        )
        allowed_team: Final = scenario.team(
            models=_json_array(model), object_permission=_permission_for_stores(CONFIG_STORE_ID)
        )
        denied_key: Final = scenario.key(team_id=denied_team, models=_json_array(model))
        allowed_key: Final = scenario.key(team_id=allowed_team, models=_json_array(model))
        denied_marker: Final = f"lit5610 async rag query denied {uuid.uuid4().hex}"
        allowed_marker: Final = f"lit5610 async rag query allowed {uuid.uuid4().hex}"

        async def query_both() -> tuple[httpx.Response, httpx.Response]:
            async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=15, trust_env=False) as client:
                return await asyncio.gather(
                    _async_rag_query(client, model, denied_marker, denied_key, CONFIG_STORE_ID),
                    _async_rag_query(client, model, allowed_marker, allowed_key, CONFIG_STORE_ID),
                )

        denied, allowed = asyncio.run(query_both())

        assert denied.status_code == 401, denied.text
        assert denied.json()["error"]["type"] == "team_vector_store_access_denied", denied.text
        assert allowed.status_code == 200, allowed.text
        observations: Final = upstream_observations(gateway)
        assert _searches_in(observations, denied_marker) == ()
        allowed_searches: Final = _searches_in(observations, allowed_marker)
        assert len(allowed_searches) == 1, allowed_searches
        assert allowed_marker in str(object_value(allowed_searches[0]["body"])["query"]), allowed_searches


def test_streaming_rag_query_checks_allowlist_and_reads_allowed_stream(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        denied_key: Final = _key_for_scope(scenario, model, "team", "vs_team_excluded")
        allowed_key: Final = _key_for_scope(scenario, model, "team", CONFIG_STORE_ID)
        denied_marker: Final = f"lit5610 rag query stream denied {uuid.uuid4().hex}"
        allowed_marker: Final = f"lit5610 rag query stream allowed {uuid.uuid4().hex}"

        denied: Final = _rag_query(gateway, model, denied_marker, denied_key, stream=True)
        denied.read()
        allowed: Final = _rag_query(gateway, model, allowed_marker, allowed_key, stream=True)
        allowed.read()

        assert denied.status_code == 401, denied.text
        assert denied.json()["error"]["type"] == "team_vector_store_access_denied", denied.text
        assert allowed.status_code == 200, allowed.text
        assert allowed.headers["content-type"].startswith("text/event-stream"), allowed.text
        assert "data: [DONE]" in allowed.text, allowed.text
        observations: Final = upstream_observations(gateway)
        assert _searches_in(observations, denied_marker) == ()
        allowed_searches: Final = _searches_in(observations, allowed_marker)
        assert len(allowed_searches) == 1, allowed_searches
        assert allowed_marker in str(object_value(allowed_searches[0]["body"])["query"]), allowed_searches


@pytest.mark.parametrize("vector_store_id", (1, _json_array("vs_invalid"), ""))
def test_rag_query_rejects_malformed_vector_store_id(gateway: Gateway, vector_store_id: JsonValue) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=_json_array(model))
        marker: Final = f"lit5610 rag query malformed store {uuid.uuid4().hex}"

        response: Final = _rag_query(gateway, model, marker, key, vector_store_id=vector_store_id)
        assert response.status_code == 400, response.text
        assert _searches_in(upstream_observations(gateway), marker) == ()


@pytest.mark.parametrize("retrieval_config", ("invalid", _retrieval_config_without_store()))
def test_rag_query_rejects_malformed_retrieval_config(gateway: Gateway, retrieval_config: JsonValue) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=_json_array(model))
        marker: Final = f"lit5610 rag query malformed config {uuid.uuid4().hex}"
        body: Final[JsonObject] = {
            "model": model,
            "messages": _json_array({"role": "user", "content": marker}),
            "retrieval_config": retrieval_config,
        }

        response: Final = gateway.request("POST", "/v1/rag/query", body, key=key)
        assert response.status_code == 400, response.text
        assert _searches_in(upstream_observations(gateway), marker) == ()


def test_rag_query_without_authorization_header_is_unauthorized(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        marker: Final = f"lit5610 rag query missing authorization {uuid.uuid4().hex}"

        response: Final = gateway.client.post("/v1/rag/query", json=_rag_query_body(model, marker, CONFIG_STORE_ID))
        assert response.status_code == 401, response.text
        assert _searches_in(upstream_observations(gateway), marker) == ()


def test_repeated_allowed_rag_queries_search_store_once_per_request(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = _key_for_scope(scenario, model, "key", CONFIG_STORE_ID)
        marker: Final = f"lit5610 rag query repeated {uuid.uuid4().hex}"

        first: Final = _rag_query(gateway, model, marker, key)
        second: Final = _rag_query(gateway, model, marker, key)
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text

        searches: Final = _searches_in(upstream_observations(gateway), marker)
        assert tuple(object_value(search["body"])["query"] for search in searches) == (marker, marker), searches


def test_team_vector_store_allowlist_updates_apply_to_rag_queries(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(
            models=_json_array(model), object_permission=_permission_for_stores("vs_team_excluded")
        )
        key: Final = scenario.key(team_id=team, models=_json_array(model))
        denied_marker: Final = f"lit5610 rag query team update denied {uuid.uuid4().hex}"
        allowed_marker: Final = f"lit5610 rag query team update allowed {uuid.uuid4().hex}"
        denied_again_marker: Final = f"lit5610 rag query team update denied again {uuid.uuid4().hex}"

        denied: Final = _rag_query(gateway, model, denied_marker, key)
        assert denied.status_code == 401, denied.text
        assert denied.json()["error"]["type"] == "team_vector_store_access_denied", denied.text

        included_body: Final[JsonObject] = {
            "team_id": team,
            "object_permission": _permission_for_stores(CONFIG_STORE_ID),
        }
        included: Final = gateway.request(
            "POST",
            "/team/update",
            included_body,
        )
        assert included.status_code == 200, included.text
        allowed: Final = eventually(
            lambda: _rag_query(gateway, model, allowed_marker, key),
            lambda response: response.status_code == 200,
        )
        assert allowed.status_code == 200, allowed.text

        excluded_body: Final[JsonObject] = {
            "team_id": team,
            "object_permission": _permission_for_stores("vs_team_excluded"),
        }
        excluded: Final = gateway.request(
            "POST",
            "/team/update",
            excluded_body,
        )
        assert excluded.status_code == 200, excluded.text
        denied_again: Final = eventually(
            lambda: _rag_query(gateway, model, denied_again_marker, key),
            lambda response: response.status_code == 401,
        )
        assert denied_again.status_code == 401, denied_again.text
        assert denied_again.json()["error"]["type"] == "team_vector_store_access_denied", denied_again.text
        observations: Final = upstream_observations(gateway)
        assert _searches_in(observations, denied_marker) == ()
        assert len(_searches_in(observations, allowed_marker)) == 1
        assert _searches_in(observations, denied_again_marker) == ()


def test_concurrent_rag_queries_enforce_allowlists_across_proxy_workers(gateway: Gateway, peer: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        allowed_key: Final = _key_for_scope(scenario, model, "key", CONFIG_STORE_ID)
        denied_team: Final = scenario.team(
            models=_json_array(model), object_permission=_permission_for_stores("vs_team_excluded")
        )
        denied_key: Final = scenario.key(team_id=denied_team, models=_json_array(model))
        _warm_rag_query_model(gateway, model, allowed_key)
        _warm_rag_query_model(peer, model, allowed_key)
        proxies: Final = tuple(gateway if index % 2 == 0 else peer for index in range(30))
        keys: Final = tuple(allowed_key if index < 15 else denied_key for index in range(30))
        markers: Final = tuple(f"lit5610 rag query concurrent {index} {uuid.uuid4().hex}" for index in range(30))

        with ThreadPoolExecutor(max_workers=30) as executor:
            responses: Final = tuple(executor.map(_rag_query, proxies, repeat(model), markers, keys))

        observations: Final = upstream_observations(gateway)
        assert tuple(response.status_code for response in responses) == (200,) * 15 + (401,) * 15, (
            tuple(response.text for response in responses),
            observations,
        )
        assert (
            tuple(response.json()["error"]["type"] for response in responses[15:])
            == ("team_vector_store_access_denied",) * 15
        ), tuple(response.text for response in responses[15:])
        for marker in markers[:15]:
            searches: Final = _searches_in(observations, marker)
            assert len(searches) == 1, searches
            assert marker in str(object_value(searches[0]["body"])["query"]), searches
        for marker in markers[15:]:
            assert _searches_in(observations, marker) == ()
