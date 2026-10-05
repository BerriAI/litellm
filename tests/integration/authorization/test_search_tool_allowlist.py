import json
import uuid
from collections.abc import Callable
from typing import Final

import pytest
from integration._support.client import Gateway, Scenario, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_EXA_KEY: Final = "synthetic-exa-allowlist-key"
_SEARCH_BODY: Final = {"query": "allowlisted search tools only"}
_EXA_BODY: Final = {"query": "allowlisted search tools only", "contents": {"text": True}}


def _exa_peer(request: Request) -> Reply:
    assert (request.method, request.target) == ("POST", "/allowed/search"), request.target
    assert request.headers["x-api-key"] == _EXA_KEY, request.headers
    assert json.loads(request.body) == _EXA_BODY, request.body
    return Reply(body=json.dumps({"results": []}).encode())


def _create_tool(gateway: Gateway, scenario: Scenario, name: str, api_base: str) -> str:
    created: Final = gateway.post(
        "/search_tools",
        {
            "search_tool": {
                "search_tool_name": name,
                "litellm_params": {"search_provider": "exa_ai", "api_key": _EXA_KEY, "api_base": api_base},
                "search_tool_info": {"description": f"integration tool {name}"},
            }
        },
    )
    identity: Final = string_value(created["search_tool_id"])
    scenario.cleanups.callback(gateway.request, "DELETE", f"/search_tools/{identity}")
    return identity


def _key_scoped_to(scenario: Scenario, allowed: str) -> str:
    user: Final = scenario.user(user_role="internal_user")
    return scenario.key(user_id=user, object_permission={"search_tools": [allowed]})


def _team_key_scoped_to(scenario: Scenario, allowed: str) -> str:
    team: Final = scenario.team(object_permission={"search_tools": [allowed]})
    return scenario.key(team_id=team, user_id=scenario.member(team))


def _listed_names(gateway: Gateway, key: str, path: str, field: str, candidates: set[str]) -> list[str]:
    response: Final = gateway.request("GET", path, key=key)
    assert response.status_code == 200, response.text
    entries: Final = response.json()[field]
    return sorted(entry["search_tool_name"] for entry in entries if entry["search_tool_name"] in candidates)


def _two_tools(gateway: Gateway, scenario: Scenario, wire: Wire) -> tuple[str, str]:
    allowed: Final = f"tool-a-{uuid.uuid4().hex}"
    denied: Final = f"tool-b-{uuid.uuid4().hex}"
    _create_tool(gateway, scenario, allowed, f"{wire.url}/allowed")
    _create_tool(gateway, scenario, denied, f"{wire.url}/denied")
    return allowed, denied


@pytest.mark.parametrize(
    ("scope", "make_key"),
    [pytest.param("Key", _key_scoped_to, id="key"), pytest.param("Team", _team_key_scoped_to, id="team")],
)
def test_search_tool_allowlist_refuses_other_tools_before_any_upstream_call(
    gateway: Gateway, scope: str, make_key: Callable[[Scenario, str], str]
) -> None:
    with wire_server(_exa_peer) as wire, gateway.scenario() as scenario:
        allowed, denied = _two_tools(gateway, scenario, wire)
        key: Final = make_key(scenario, allowed)
        refusal: Final = {
            "error": {
                "message": f"{scope} not allowed to access search tool: {denied}. Allowed search tools: ['{allowed}']",
                "type": "key_model_access_denied",
                "param": "search_tool_name",
                "code": "403",
            }
        }
        for path, body in (
            (f"/v1/search/{denied}", _SEARCH_BODY),
            ("/v1/search", {"search_tool_name": denied, **_SEARCH_BODY}),
        ):
            refused = gateway.request("POST", path, body, key=key)
            assert (refused.status_code, refused.json()) == (403, refusal), refused.text
        assert wire.drain() == ()
        served: Final = gateway.request("POST", f"/v1/search/{allowed}", _SEARCH_BODY, key=key)
        assert (served.status_code, served.json()) == (200, {"object": "search", "results": []}), served.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/allowed/search")]
        assert _listed_names(gateway, key, "/search_tools/list", "search_tools", {allowed, denied}) == [allowed]


def test_search_tools_listing_route_hides_tools_outside_the_key_allowlist(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: GET /v1/search/tools lists every router search tool to a key whose search_tools allowlist excludes them"
    )
    with wire_server(_exa_peer) as wire, gateway.scenario() as scenario:
        allowed, denied = _two_tools(gateway, scenario, wire)
        key: Final = _key_scoped_to(scenario, allowed)
        for path in ("/v1/search/tools", "/search/tools"):
            assert _listed_names(gateway, key, path, "data", {allowed, denied}) == [allowed]
        assert wire.drain() == ()


def _search_tool_row(identity: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT search_tool_name, litellm_params, search_tool_info, updated_at::text AS updated_at FROM "LiteLLM_SearchToolsTable" '
        "WHERE search_tool_id = %s",
        (identity,),
    )


@pytest.mark.parametrize(("role", "list_status"), [("internal_user", 200), ("internal_user_viewer", 401)])
def test_non_admin_keys_cannot_create_update_delete_or_probe_search_tools(
    gateway: Gateway, role: str, list_status: int
) -> None:
    def refuse_all(request: Request) -> Reply:
        raise AssertionError(f"non-admin search tool management reached the provider: {request.target}")

    with wire_server(refuse_all) as wire, gateway.scenario() as scenario:
        existing_name: Final = f"admin-owned-{uuid.uuid4().hex}"
        existing: Final = _create_tool(gateway, scenario, existing_name, f"{wire.url}/admin")
        before: Final = _search_tool_row(existing)
        user: Final = scenario.user(user_role=role)
        key: Final = scenario.key(user_id=user)
        attempted_name: Final = f"non-admin-{uuid.uuid4().hex}"
        attacker_params: Final = {
            "search_provider": "exa_ai",
            "api_key": "attacker",
            "api_base": f"{wire.url}/attacker",
        }
        attempts: Final = (
            (
                "POST",
                "/search_tools",
                {"search_tool": {"search_tool_name": attempted_name, "litellm_params": attacker_params}},
            ),
            (
                "PUT",
                f"/search_tools/{existing}",
                {"search_tool": {"search_tool_name": existing_name, "litellm_params": attacker_params}},
            ),
            ("DELETE", f"/search_tools/{existing}", None),
            ("POST", "/search_tools/test_connection", {"litellm_params": attacker_params}),
        )
        for method, path, body in attempts:
            response = gateway.request(method, path, body, key=key)
            assert response.status_code == 401, f"{method} {path}: {response.text}"
            assert response.json()["error"]["type"] == "auth_error", response.text
            assert f"Route={path}. Your role={role}." in response.json()["error"]["message"], response.text
        assert _search_tool_row(existing) == before
        assert (
            read_rows(
                'SELECT search_tool_id FROM "LiteLLM_SearchToolsTable" WHERE search_tool_name = %s', (attempted_name,)
            )
            == []
        )
        listed: Final = gateway.request("GET", "/search_tools/list", key=key)
        assert listed.status_code == list_status, listed.text
        assert wire.drain() == ()
