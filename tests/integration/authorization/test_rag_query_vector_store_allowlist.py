import uuid
from typing import Final, Literal

import httpx
import pytest
from integration._support.client import Gateway, Scenario, object_value
from integration.authorization._guardrail_opt_out import upstream_observations
from pydantic import JsonValue

CONFIG_STORE_ID: Final = "vs_integration_config_store"
SEARCH_PATH: Final = f"/vector_stores/{CONFIG_STORE_ID}/search"


def _key_for_scope(scenario: Scenario, model: str, scope: Literal["key", "team"], store_id: str) -> str:
    if scope == "key":
        return scenario.key(models=[model], object_permission={"vector_stores": [store_id]})
    team: Final = scenario.team(models=[model], object_permission={"vector_stores": [store_id]})
    return scenario.key(team_id=team, models=[model])


def _rag_query(gateway: Gateway, model: str, marker: str, key: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/rag/query",
        {
            "model": model,
            "messages": [{"role": "user", "content": marker}],
            "retrieval_config": {
                "vector_store_id": CONFIG_STORE_ID,
                "custom_llm_provider": "openai",
                "top_k": 1,
            },
        },
        key=key,
    )


def _requests_for_marker(gateway: Gateway, marker: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(observation for observation in upstream_observations(gateway) if marker in str(observation["body"]))


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
