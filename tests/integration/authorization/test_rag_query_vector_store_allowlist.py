from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import httpx
import pytest
import yaml
from integration._support.client import Gateway, Scenario, gateway_from_environment, object_value
from integration._support.process import owned_proxy
from integration.authorization._guardrail_opt_out import upstream_observations
from pydantic import JsonValue

CONFIG_STORE_ID: Final = "vs_integration_config_store"
PROXY_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
REMOVE_OPENAI_API_BASE: Final = ("OPENAI_API_BASE",)
JsonObject: TypeAlias = dict[str, JsonValue]


def _json_array(*values: JsonValue) -> JsonValue:
    return [*values]  # mutable-ok: request payloads and YAML sequences require list values


def _permission_for_stores(*store_ids: str) -> JsonObject:
    permission: Final[JsonObject] = {"vector_stores": _json_array(*store_ids)}
    return permission


def _key_for_scope(scenario: Scenario, model: str, scope: Literal["key", "team"], store_id: str) -> str:
    if scope == "key":
        return scenario.key(models=_json_array(model), object_permission=_permission_for_stores(store_id))
    team: Final = scenario.team(models=_json_array(model), object_permission=_permission_for_stores(store_id))
    return scenario.key(team_id=team, models=_json_array(model))


def _rag_query_body(model: str, marker: str, store_id: str) -> JsonObject:
    body: Final[JsonObject] = {
        "model": model,
        "messages": _json_array({"role": "user", "content": marker}),
        "retrieval_config": {"vector_store_id": store_id, "custom_llm_provider": "openai", "top_k": 1},
    }
    return body


def _rag_query(
    gateway: Gateway,
    model: str,
    marker: str,
    key: str,
    *,
    store_id: str = CONFIG_STORE_ID,
    path: str = "/v1/rag/query",
) -> httpx.Response:
    return gateway.request("POST", path, _rag_query_body(model, marker, store_id), key=key)


def _searches_for_marker(
    gateway: Gateway, marker: str, store_id: str = CONFIG_STORE_ID
) -> tuple[Mapping[str, JsonValue], ...]:
    search_path: Final = f"/vector_stores/{store_id}/search"
    return tuple(
        observation
        for observation in upstream_observations(gateway)
        if observation["path"] == search_path and marker in str(observation["body"])
    )


def _no_registry_config(directory: Path) -> Path:
    config: Final = object_value(yaml.safe_load(PROXY_CONFIG.read_text()))
    config_without_registry: Final[Mapping[str, JsonValue]] = MappingProxyType(
        {name: value for name, value in config.items() if name != "vector_store_registry"}
    )
    yaml_config: Final[JsonObject] = {**config_without_registry, "model_list": _json_array()}
    path: Final = directory / "proxy_no_vector_store_registry.yaml"
    path.write_text(yaml.safe_dump(yaml_config))
    return path


def _openai_environment(gateway: Gateway) -> Mapping[str, str]:
    return MappingProxyType({"OPENAI_BASE_URL": gateway.upstream_url, "OPENAI_API_KEY": "synthetic-openai-key"})


@pytest.fixture(scope="module")
def no_registry_gateways(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, Gateway]]:
    with gateway_from_environment() as upstream_gateway:
        directory: Final = tmp_path_factory.mktemp("rag_query_no_registry")
        config: Final = _no_registry_config(directory)
        with owned_proxy(
            upstream_gateway,
            directory,
            _openai_environment(upstream_gateway),
            config=config,
            remove_environment=REMOVE_OPENAI_API_BASE,
            workers=2,
        ) as no_registry_gateway:
            yield no_registry_gateway, upstream_gateway


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
        assert _searches_for_marker(gateway, marker) == ()


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

        searches: Final = _searches_for_marker(gateway, marker)
        assert len(searches) == 1, searches
        assert marker in str(object_value(searches[0]["body"])["query"]), searches


def test_rag_query_without_key_object_permission_can_search_store(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=_json_array(model))
        marker: Final = f"lit5610 rag query no permission {uuid.uuid4().hex}"

        response: Final = _rag_query(gateway, model, marker, key)
        assert response.status_code == 200, response.text

        searches: Final = _searches_for_marker(gateway, marker)
        assert len(searches) == 1, searches
        assert marker in str(object_value(searches[0]["body"])["query"]), searches


@pytest.mark.parametrize("scope", ("team", "key"))
def test_no_registry_rag_query_denies_unregistered_store_when_allowlist_excludes(
    no_registry_gateways: tuple[Gateway, Gateway], scope: Literal["team", "key"]
) -> None:
    no_registry_gateway, upstream_gateway = no_registry_gateways
    with no_registry_gateway.scenario() as scenario:
        model: Final = scenario.model()
        store_id: Final = f"vs_unregistered_{uuid.uuid4().hex}"
        key: Final = _key_for_scope(scenario, model, scope, "vs_some_other_store")
        marker: Final = f"lit5610 no registry denied {scope} {uuid.uuid4().hex}"
        error_type: Final = "team_vector_store_access_denied" if scope == "team" else "key_vector_store_access_denied"

        response: Final = _rag_query(no_registry_gateway, model, marker, key, store_id=store_id)
        searches: Final = _searches_for_marker(upstream_gateway, marker, store_id)

        assert response.status_code == 401, f"{response.text}; scripted_upstream_searches={searches!r}"
        assert response.json()["error"]["type"] == error_type, response.text
        assert searches == ()


def test_no_registry_rag_query_allows_team_allowlisted_unregistered_store(
    no_registry_gateways: tuple[Gateway, Gateway],
) -> None:
    no_registry_gateway, upstream_gateway = no_registry_gateways
    with no_registry_gateway.scenario() as scenario:
        model: Final = scenario.model()
        store_id: Final = f"vs_unregistered_{uuid.uuid4().hex}"
        key: Final = _key_for_scope(scenario, model, "team", store_id)
        marker: Final = f"lit5610 no registry allowed {uuid.uuid4().hex}"

        response: Final = _rag_query(no_registry_gateway, model, marker, key, store_id=store_id)
        assert response.status_code == 200, response.text

        searches: Final = _searches_for_marker(upstream_gateway, marker, store_id)
        assert len(searches) == 1, searches
        assert marker in str(object_value(searches[0]["body"])["query"]), searches


def test_chat_completions_top_level_retrieval_config_uses_team_allowlist(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = _key_for_scope(scenario, model, "team", "vs_some_other_store")
        marker: Final = f"lit5610 chat top-level retrieval config denied {uuid.uuid4().hex}"
        body: Final[JsonObject] = {
            "model": model,
            "messages": _json_array({"role": "user", "content": marker}),
            "retrieval_config": {
                "vector_store_id": CONFIG_STORE_ID,
                "custom_llm_provider": "openai",
                "top_k": 1,
            },
        }

        response: Final = gateway.request("POST", "/v1/chat/completions", body, key=key)
        observations: Final = upstream_observations(gateway)

        assert response.status_code == 401, f"{response.text}; scripted_upstream_observations={observations!r}"
        assert response.json()["error"]["type"] == "team_vector_store_access_denied", response.text


def test_rag_query_alias_denies_store_when_team_allowlist_excludes(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = _key_for_scope(scenario, model, "team", "vs_some_other_store")
        marker: Final = f"lit5610 rag query alias denied {uuid.uuid4().hex}"

        response: Final = _rag_query(gateway, model, marker, key, path="/rag/query")

        assert response.status_code == 401, response.text
        assert response.json()["error"]["type"] == "team_vector_store_access_denied", response.text
        assert _searches_for_marker(gateway, marker) == ()
