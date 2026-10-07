import json
import uuid
from typing import Final, Literal

import httpx
import pytest
from integration._support.client import Gateway, JsonValue, Scenario, eventually, object_value, string_value
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import TypeAdapter

ENDPOINTS: Final = TypeAdapter(list[JsonValue])
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
Owner = Literal["key", "team"]


def _echo(request: Request) -> Reply:
    return Reply(body=json.dumps({"target": request.target}).encode())


def _registered_endpoint(gateway: Gateway, scenario: Scenario, wire: Wire, *, auth: bool = True) -> str:
    path: Final = f"/integration-deny-{uuid.uuid4().hex}"
    created: Final = gateway.post(
        "/config/pass_through_endpoint",
        {"path": path, "target": f"{wire.url}/upstream", "auth": auth, "include_subpath": True},
    )
    endpoint_id: Final = object_value(ENDPOINTS.validate_python(created["endpoints"])[0])["id"]
    scenario.cleanups.callback(
        lambda: gateway.request("DELETE", "/config/pass_through_endpoint", params={"endpoint_id": str(endpoint_id)})
    )
    return path


def _call(gateway: Gateway, route: str, key: str) -> httpx.Response:
    return gateway.request("POST", route, {"probe": "denylist"}, key=key)


def _upstream_targets(wire: Wire) -> tuple[str, ...]:
    return tuple(request.target for request in wire.drain())


def _assert_denied(response: httpx.Response, denied_entry: str) -> None:
    assert response.status_code == 403, response.text
    assert f"Matched `{denied_entry}` in `denied_passthrough_routes`" in response.text, response.text


def _key_with_routes(
    scenario: Scenario, allow_on: Owner, deny_on: Owner, allowed: list[JsonValue], denied: list[JsonValue]
) -> str:
    team_fields: Final[dict[str, JsonValue]] = {
        **({"allowed_passthrough_routes": allowed} if allow_on == "team" else {}),
        **({"denied_passthrough_routes": denied} if deny_on == "team" else {}),
    }
    key_fields: Final[dict[str, JsonValue]] = {
        **({"allowed_passthrough_routes": allowed} if allow_on == "key" else {}),
        **({"denied_passthrough_routes": denied} if deny_on == "key" else {}),
    }
    return scenario.key(team_id=scenario.team(**team_fields), **key_fields)


@pytest.mark.parametrize(
    ("allow_on", "deny_on"),
    [("key", "key"), ("team", "key"), ("key", "team")],
)
def test_denied_subpath_is_blocked_even_when_allowed_while_its_sibling_still_reaches_upstream(
    gateway: Gateway, allow_on: Owner, deny_on: Owner
) -> None:
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        path: Final = _registered_endpoint(gateway, scenario, wire)
        key: Final = _key_with_routes(scenario, allow_on, deny_on, [path], [f"{path}/admin"])

        _assert_denied(_call(gateway, f"{path}/admin/users", key), f"{path}/admin")
        sibling: Final = _call(gateway, f"{path}/public", key)

        assert sibling.status_code == 200, sibling.text
        assert _upstream_targets(wire) == ("/upstream/public",)


@pytest.mark.parametrize(
    "subpath",
    [
        "public/%2e%2e/admin/users",
        "/admin/users",
        "admin%3F",
        "admin%3F/users",
        "admin%23",
        "admin%23/users",
        "public%3Fx/%2e%2e/admin%3F",
        "public%23x/%2e%2e/admin%23",
    ],
    ids=[
        "encoded_dot_dot_segment",
        "empty_segment",
        "encoded_query_mark",
        "encoded_query_mark_then_subpath",
        "encoded_fragment_mark",
        "encoded_fragment_mark_then_subpath",
        "encoded_query_mark_then_dot_dot",
        "encoded_fragment_mark_then_dot_dot",
    ],
)
def test_dot_and_empty_segments_cannot_reach_a_denied_subpath(gateway: Gateway, subpath: str) -> None:
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        path: Final = _registered_endpoint(gateway, scenario, wire)
        key: Final = scenario.key(allowed_passthrough_routes=[path], denied_passthrough_routes=[f"{path}/admin"])

        response: Final = _call(gateway, f"{path}/{subpath}", key)

        _assert_denied(response, f"{path}/admin")
        assert _upstream_targets(wire) == ()


def test_trailing_slash_deny_entry_blocks_the_route_and_everything_under_it(gateway: Gateway) -> None:
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        path: Final = _registered_endpoint(gateway, scenario, wire)
        key: Final = scenario.key(allowed_passthrough_routes=[path], denied_passthrough_routes=[f"{path}/admin/"])

        _assert_denied(_call(gateway, f"{path}/admin", key), f"{path}/admin/")
        _assert_denied(_call(gateway, f"{path}/admin/", key), f"{path}/admin/")
        _assert_denied(_call(gateway, f"{path}/admin/users", key), f"{path}/admin/")
        sibling: Final = _call(gateway, f"{path}/public", key)

        assert sibling.status_code == 200, sibling.text
        assert _upstream_targets(wire) == ("/upstream/public",)


def test_trailing_wildcard_deny_blocks_every_route_with_that_prefix(gateway: Gateway) -> None:
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        path: Final = _registered_endpoint(gateway, scenario, wire)
        key: Final = scenario.key(allowed_passthrough_routes=[path], denied_passthrough_routes=[f"{path}/adm*"])

        _assert_denied(_call(gateway, f"{path}/admin", key), f"{path}/adm*")
        _assert_denied(_call(gateway, f"{path}/adm-console/x", key), f"{path}/adm*")
        sibling: Final = _call(gateway, f"{path}/public", key)

        assert sibling.status_code == 200, sibling.text
        assert _upstream_targets(wire) == ("/upstream/public",)


def test_deny_entry_does_not_match_a_longer_segment_that_shares_its_prefix(gateway: Gateway) -> None:
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        path: Final = _registered_endpoint(gateway, scenario, wire)
        key: Final = scenario.key(allowed_passthrough_routes=[path], denied_passthrough_routes=[f"{path}/admin"])

        response: Final = _call(gateway, f"{path}/administrator", key)

        assert response.status_code == 200, response.text
        assert _upstream_targets(wire) == ("/upstream/administrator",)


def test_proxy_admin_key_reaches_a_route_its_key_and_team_both_deny(gateway: Gateway) -> None:
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        path: Final = _registered_endpoint(gateway, scenario, wire)
        admin: Final = scenario.user(user_role="proxy_admin")
        team: Final = scenario.team(denied_passthrough_routes=[path])
        gateway.post("/team/member_add", {"team_id": team, "member": {"role": "user", "user_id": admin}})
        key: Final = scenario.key(user_id=admin, team_id=team, denied_passthrough_routes=[path])

        response: Final = _call(gateway, f"{path}/ops", key)

        assert response.status_code == 200, response.text
        assert _upstream_targets(wire) == ("/upstream/ops",)


@pytest.mark.parametrize("deny_on", ["key", "team"])
def test_deny_added_and_cleared_through_update_takes_effect_on_the_next_request(
    gateway: Gateway, deny_on: Owner
) -> None:
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        path: Final = _registered_endpoint(gateway, scenario, wire)
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team, allowed_passthrough_routes=[path])

        def set_denied(routes: list[JsonValue]) -> None:
            if deny_on == "key":
                gateway.post("/key/update", {"key": key, "denied_passthrough_routes": routes})
            else:
                gateway.post("/team/update", {"team_id": team, "denied_passthrough_routes": routes})

        def probe() -> httpx.Response:
            return _call(gateway, f"{path}/admin", key)

        before: Final = probe()
        assert before.status_code == 200, before.text
        set_denied([path])
        _assert_denied(eventually(probe, lambda response: response.status_code == 403, seconds=10), path)
        set_denied([])
        restored: Final = eventually(probe, lambda response: response.status_code == 200, seconds=10)

        assert restored.status_code == 200, restored.text
        targets: Final = _upstream_targets(wire)
        assert len(targets) >= 2 and set(targets) == {"/upstream/admin"}, targets


@pytest.mark.parametrize(
    "body",
    [{"denied_passthrough_routes": ["/integration-deny-probe"]}, {"metadata": {"denied_passthrough_routes": ["/x"]}}],
    ids=["top_level", "metadata"],
)
def test_internal_user_cannot_set_denied_routes_while_proxy_admin_can(
    gateway: Gateway, body: dict[str, JsonValue]
) -> None:
    with gateway.scenario() as scenario:
        user: Final = scenario.user(user_role="internal_user")
        user_key: Final = scenario.key(user_id=user)

        refused: Final = gateway.request("POST", "/key/generate", {"user_id": user, **body}, key=user_key)
        if refused.status_code == 200:
            scenario.cleanups.callback(
                scenario.delete_key, string_value(JSON_OBJECT.validate_json(refused.content)["key"])
            )

        assert refused.status_code == 403, refused.text
        assert "denied_passthrough_routes" in refused.text, refused.text
        admin_key: Final = scenario.key(denied_passthrough_routes=["/integration-deny-probe"])
        info: Final = object_value(gateway.get("/key/info", {"key": admin_key})["info"])
        assert object_value(info["metadata"])["denied_passthrough_routes"] == ["/integration-deny-probe"], info


def test_deny_entries_leave_open_passthroughs_and_llm_routes_untouched(gateway: Gateway) -> None:
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        open_path: Final = _registered_endpoint(gateway, scenario, wire, auth=False)
        model: Final = scenario.model()
        key: Final = scenario.key(denied_passthrough_routes=[open_path, "/v1/chat/completions", "/chat/completions"])

        opened: Final = _call(gateway, open_path, key)
        chat: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "x"}]}, key=key
        )

        assert opened.status_code == 200, opened.text
        assert _upstream_targets(wire) == ("/upstream",)
        assert chat.status_code == 200, chat.text


def test_team_endpoint_listing_hides_routes_the_team_denies(gateway: Gateway) -> None:
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        denied: Final = _registered_endpoint(gateway, scenario, wire)
        visible: Final = _registered_endpoint(gateway, scenario, wire)
        team: Final = scenario.team(denied_passthrough_routes=[denied])

        listed: Final = gateway.get("/config/pass_through_endpoint", {"team_id": team})["endpoints"]

        paths: Final = {string_value(object_value(endpoint)["path"]) for endpoint in ENDPOINTS.validate_python(listed)}
        assert visible in paths, paths
        assert denied not in paths, paths


def test_team_admin_cannot_clear_or_drop_a_deny_a_proxy_admin_set_on_a_team_key(gateway: Gateway) -> None:
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        path: Final = _registered_endpoint(gateway, scenario, wire)
        team_admin: Final = scenario.user(user_role="internal_user")
        team: Final = scenario.team(members_with_roles=[{"role": "admin", "user_id": team_admin}])
        team_admin_key: Final = scenario.key(user_id=team_admin)
        denied: Final[list[JsonValue]] = [f"{path}/admin"]
        key: Final = scenario.key(team_id=team, allowed_passthrough_routes=[path], denied_passthrough_routes=denied)

        def update(body: dict[str, JsonValue]) -> httpx.Response:
            return gateway.request("POST", "/key/update", {"key": key, **body}, key=team_admin_key)

        cleared: Final = update({"denied_passthrough_routes": []})
        dropped: Final = update({"metadata": {}})
        unchanged: Final = update({"denied_passthrough_routes": denied})

        assert cleared.status_code == 403 and "denied_passthrough_routes" in cleared.text, cleared.text
        assert dropped.status_code == 403 and "metadata.denied_passthrough_routes" in dropped.text, dropped.text
        assert unchanged.status_code == 200, unchanged.text
        _assert_denied(_call(gateway, f"{path}/admin/users", key), f"{path}/admin")
        assert _upstream_targets(wire) == ()


def test_team_admin_bulk_update_cannot_drop_a_deny_a_proxy_admin_set_on_a_team_key(gateway: Gateway) -> None:
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        path: Final = _registered_endpoint(gateway, scenario, wire)
        team_admin: Final = scenario.user(user_role="internal_user")
        team: Final = scenario.team(members_with_roles=[{"role": "admin", "user_id": team_admin}])
        team_admin_key: Final = scenario.key(user_id=team_admin)
        guarded: Final = scenario.key(
            team_id=team, allowed_passthrough_routes=[path], denied_passthrough_routes=[f"{path}/admin"]
        )
        plain: Final = scenario.key(team_id=team, allowed_passthrough_routes=[path])

        response: Final = gateway.request(
            "POST",
            "/team/key/bulk_update",
            {"team_id": team, "key_ids": [guarded, plain], "update_fields": {"metadata": {}}},
            key=team_admin_key,
        )

        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_python(response.json())
        failed: Final = tuple(object_value(item) for item in ENDPOINTS.validate_python(body["failed_updates"]))
        succeeded: Final = tuple(object_value(item) for item in ENDPOINTS.validate_python(body["successful_updates"]))
        assert [string_value(item["key"]) for item in failed] == [guarded], response.text
        assert "metadata.denied_passthrough_routes" in string_value(failed[0]["failed_reason"]), response.text
        assert [string_value(item["key"]) for item in succeeded] == [plain], response.text
        _assert_denied(_call(gateway, f"{path}/admin/users", guarded), f"{path}/admin")
        assert _upstream_targets(wire) == ()


@pytest.mark.parametrize("route", ["/key/update", "/key/regenerate"])
def test_non_owner_gets_the_same_refusal_whether_or_not_another_users_key_has_a_deny(
    gateway: Gateway, route: str
) -> None:
    with gateway.scenario() as scenario:
        owner: Final = scenario.user(user_role="internal_user")
        guarded: Final = scenario.key(user_id=owner, denied_passthrough_routes=["/integration-deny-probe"])
        plain: Final = scenario.key(user_id=owner)
        outsider_key: Final = scenario.key(user_id=scenario.user(user_role="internal_user"))

        def probe(key: str) -> httpx.Response:
            return gateway.request("POST", route, {"key": key, "denied_passthrough_routes": []}, key=outsider_key)

        on_guarded: Final = probe(guarded)
        on_plain: Final = probe(plain)

        assert on_guarded.status_code == on_plain.status_code != 200, (on_guarded.text, on_plain.text)
        assert "denied_passthrough_routes" not in on_guarded.text, on_guarded.text
        assert on_guarded.text.replace(guarded, "KEY") == on_plain.text.replace(plain, "KEY")


def test_non_admin_setting_allowed_routes_on_regenerate_is_refused_before_the_key_lookup(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        user_key: Final = scenario.key(user_id=scenario.user(user_role="internal_user"))

        response: Final = gateway.request(
            "POST",
            "/key/regenerate",
            {"key": f"sk-missing-{uuid.uuid4().hex}", "allowed_passthrough_routes": ["/integration-deny-probe"]},
            key=user_key,
        )

        assert response.status_code == 403, response.text
        assert "allowed_passthrough_routes" in response.text, response.text
