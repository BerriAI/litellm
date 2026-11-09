from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final, Literal, TypeAlias

import httpx
import pytest
import yaml
from integration._support.client import Gateway, Scenario, gateway_from_environment, object_value
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

PROXY_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
SEARCH_TOOL: Final = "integration-search"
OTHER_SEARCH_TOOL: Final = "integration-other-search"
FAILING_SEARCH_TOOL: Final = "integration-failing-search"
SEARCH_RESULT: Final = {"title": "Synthetic result", "url": "https://example.test/result", "content": "snippet"}
JsonObject: TypeAlias = dict[str, JsonValue]


def _json_array(*values: JsonValue) -> JsonValue:
    return [*values]  # mutable-ok: request payloads and YAML sequences require list values


def _grant(*search_tools: str) -> JsonObject:
    permission: Final[JsonObject] = {"search_tools": _json_array(*search_tools)}
    return permission


def _respond(request: Request) -> Reply:
    if (request.method, request.target) == ("POST", "/failing/search"):
        return Reply(status=500, body=b'{"error": "synthetic outage"}')
    assert (request.method, request.target) == ("POST", "/tavily/search"), request
    query: Final = json.loads(request.body)["query"]
    return Reply(body=json.dumps({"query": query, "results": [SEARCH_RESULT]}).encode())


def _config(directory: Path, wire: Wire) -> Path:
    config: Final = object_value(yaml.safe_load(PROXY_CONFIG.read_text()))
    tool_params: Final[JsonObject] = {
        "search_provider": "tavily",
        "api_key": "synthetic-tavily-key",
        "api_base": f"{wire.url}/tavily",
    }
    failing_tool_params: Final[JsonObject] = {**tool_params, "api_base": f"{wire.url}/failing"}
    strict: Final[JsonObject] = {
        **config,
        "general_settings": {**object_value(config["general_settings"]), "search_tool_deny_by_default": True},
        "router_settings": {
            **object_value(config["router_settings"]),
            "fallbacks": _json_array({FAILING_SEARCH_TOOL: _json_array(OTHER_SEARCH_TOOL)}),
        },
        "search_tools": _json_array(
            {"search_tool_name": SEARCH_TOOL, "litellm_params": tool_params},
            {"search_tool_name": OTHER_SEARCH_TOOL, "litellm_params": tool_params},
            {"search_tool_name": FAILING_SEARCH_TOOL, "litellm_params": failing_tool_params},
        ),
    }
    path: Final = directory / "proxy_search_tool_deny_by_default.yaml"
    path.write_text(yaml.safe_dump(strict))
    return path


@pytest.fixture(scope="module")
def strict_search(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, Wire]]:
    with gateway_from_environment() as upstream, wire_server(_respond) as wire:
        directory: Final = tmp_path_factory.mktemp("search_tool_deny_by_default")
        with owned_proxy(upstream, directory, {}, config=_config(directory, wire)) as gateway:
            yield gateway, wire


def _search(gateway: Gateway, key: str, marker: str) -> httpx.Response:
    return gateway.request("POST", f"/v1/search/{SEARCH_TOOL}", {"query": marker}, key=key)


def _searched(wire: Wire, marker: str) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if marker in request.body.decode())


Caller: TypeAlias = Literal[
    "standalone_key_no_permission",
    "standalone_key_empty_grant",
    "standalone_key_other_tool",
    "team_key_empty_key_grant",
    "team_key_empty_team_grant",
    "proxy_admin_key_no_grant",
]


def _denied_key(scenario: Scenario, caller: Caller) -> tuple[str, str]:
    if caller == "standalone_key_no_permission":
        return scenario.key(), "key_search_tool_access_denied"
    if caller == "standalone_key_empty_grant":
        return scenario.key(object_permission=_grant()), "key_search_tool_access_denied"
    if caller == "standalone_key_other_tool":
        return scenario.key(object_permission=_grant(OTHER_SEARCH_TOOL)), "key_search_tool_access_denied"
    if caller == "team_key_empty_key_grant":
        team: Final = scenario.team(object_permission=_grant(SEARCH_TOOL))
        return scenario.key(team_id=team, object_permission=_grant()), "key_search_tool_access_denied"
    if caller == "team_key_empty_team_grant":
        empty_team: Final = scenario.team(object_permission=_grant())
        return scenario.key(team_id=empty_team, object_permission=_grant(SEARCH_TOOL)), "team_search_tool_access_denied"
    admin: Final = scenario.user(user_role="proxy_admin")
    return scenario.key(user_id=admin), "key_search_tool_access_denied"


@pytest.mark.parametrize(
    "caller",
    [
        "standalone_key_no_permission",
        "standalone_key_empty_grant",
        "standalone_key_other_tool",
        "team_key_empty_key_grant",
        "team_key_empty_team_grant",
        "proxy_admin_key_no_grant",
    ],
)
def test_deny_by_default_rejects_ungranted_search_before_the_provider_is_called(
    strict_search: tuple[Gateway, Wire], caller: Caller
) -> None:
    gateway, wire = strict_search
    with gateway.scenario() as scenario:
        key, error_type = _denied_key(scenario, caller)
        marker: Final = f"search deny {caller} {uuid.uuid4().hex}"

        response: Final = _search(gateway, key, marker)

        assert response.status_code == 403, response.text
        assert response.json()["error"]["type"] == error_type, response.text
        assert _searched(wire, marker) == ()


@pytest.mark.parametrize("scope", ["standalone_key", "team_key", "master_key"])
def test_deny_by_default_serves_explicitly_granted_search(
    strict_search: tuple[Gateway, Wire], scope: Literal["standalone_key", "team_key", "master_key"]
) -> None:
    gateway, wire = strict_search
    with gateway.scenario() as scenario:
        if scope == "standalone_key":
            key = scenario.key(object_permission=_grant(SEARCH_TOOL))
        elif scope == "team_key":
            team: Final = scenario.team(object_permission=_grant(SEARCH_TOOL))
            key = scenario.key(team_id=team, object_permission=_grant(SEARCH_TOOL))
        else:
            key = gateway.key
        marker: Final = f"search allow {scope} {uuid.uuid4().hex}"

        response: Final = _search(gateway, key, marker)

        assert response.status_code == 200, response.text
        assert response.json()["results"][0]["url"] == SEARCH_RESULT["url"]
        assert len(_searched(wire, marker)) == 1


def test_search_tools_list_shows_only_the_tools_the_key_and_its_team_both_grant(
    strict_search: tuple[Gateway, Wire],
) -> None:
    gateway, _ = strict_search
    with gateway.scenario() as scenario:
        team: Final = scenario.team(object_permission=_grant(SEARCH_TOOL, OTHER_SEARCH_TOOL))
        member: Final = scenario.member(team)
        key: Final = scenario.key(team_id=team, user_id=member, object_permission=_grant(SEARCH_TOOL))

        response: Final = gateway.request("GET", "/search_tools/list", key=key)

        assert response.status_code == 200, response.text
        assert [tool["search_tool_name"] for tool in response.json()["search_tools"]] == [SEARCH_TOOL]


def test_revoking_a_team_search_grant_takes_effect_on_the_next_request(strict_search: tuple[Gateway, Wire]) -> None:
    gateway, wire = strict_search
    with gateway.scenario() as scenario:
        team: Final = scenario.team(object_permission=_grant(SEARCH_TOOL))
        key: Final = scenario.key(team_id=team, object_permission=_grant(SEARCH_TOOL))
        assert _search(gateway, key, f"search warm {uuid.uuid4().hex}").status_code == 200

        gateway.post("/team/update", {"team_id": team, "object_permission": _grant()})
        marker: Final = f"search revoked {uuid.uuid4().hex}"
        response: Final = _search(gateway, key, marker)

        assert response.status_code == 403, response.text
        assert response.json()["error"]["type"] == "team_search_tool_access_denied", response.text
        assert _searched(wire, marker) == ()


def test_deny_by_default_authorizes_a_search_tool_named_by_the_model_field(strict_search: tuple[Gateway, Wire]) -> None:
    gateway, wire = strict_search
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        marker: Final = f"search model field {uuid.uuid4().hex}"

        response: Final = gateway.request("POST", "/v1/search", {"model": SEARCH_TOOL, "query": marker}, key=key)

        assert response.status_code == 403, response.text
        assert response.json()["error"]["type"] == "key_search_tool_access_denied", response.text
        assert _searched(wire, marker) == ()


@pytest.mark.parametrize("fallback_granted", [False, True])
def test_router_search_fallback_only_reaches_a_granted_tool(
    strict_search: tuple[Gateway, Wire], fallback_granted: bool
) -> None:
    gateway, wire = strict_search
    with gateway.scenario() as scenario:
        granted: Final = (FAILING_SEARCH_TOOL, OTHER_SEARCH_TOOL) if fallback_granted else (FAILING_SEARCH_TOOL,)
        key: Final = scenario.key(object_permission=_grant(*granted))
        marker: Final = f"search fallback {fallback_granted} {uuid.uuid4().hex}"

        response: Final = gateway.request("POST", f"/v1/search/{FAILING_SEARCH_TOOL}", {"query": marker}, key=key)

        searched: Final = tuple(request.target for request in _searched(wire, marker))
        if fallback_granted:
            assert response.status_code == 200, response.text
            assert searched == ("/failing/search", "/tavily/search")
        else:
            assert response.status_code == 500, response.text
            assert searched == ("/failing/search",)
