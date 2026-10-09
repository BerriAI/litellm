import uuid
from typing import Final, cast

import httpx
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, object_value, string_value


def _service_account_key(scenario: Scenario, team_id: str, alias: str) -> str:
    created: Final = scenario.gateway.post(
        "/key/service-account/generate",
        {"team_id": team_id, "key_alias": alias},
    )
    key: Final = string_value(created["key"])
    scenario.cleanups.callback(scenario.delete_key, key)
    return key


def _list_response(gateway: Gateway, key: str, params: dict[str, str]) -> httpx.Response:
    return gateway.request("GET", "/key/list", key=key, params=params)


def _matches_aliases(response: httpx.Response, expected: set[str]) -> bool:
    if response.status_code != 200:
        return False
    body: Final = object_value(cast(JsonValue, response.json()))
    raw_keys: Final = body.get("keys")
    if not isinstance(raw_keys, list):
        return False
    key_objects: Final = tuple(object_value(item) for item in raw_keys if isinstance(item, dict))
    if len(key_objects) != len(raw_keys):
        return False
    aliases: Final = tuple(item.get("key_alias") for item in key_objects)
    valid_aliases: Final = tuple(alias for alias in aliases if isinstance(alias, str))
    return len(valid_aliases) == len(aliases) == len(expected) and set(valid_aliases) == expected


def test_key_list_user_id_and_email_filters_follow_shared_team_visibility(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        suffix: Final = uuid.uuid4().hex
        alice_email: Final = f"alice-{suffix}@example.com"
        bob_email: Final = f"bob-{suffix}@example.com"
        alice: Final = scenario.user(user_role="internal_user", user_email=alice_email)
        bob: Final = scenario.user(user_role="internal_user", user_email=bob_email)
        carol: Final = scenario.user(user_role="internal_user", user_email=f"carol-{suffix}@example.com")
        dave: Final = scenario.user(user_role="internal_user", user_email=f"dave-{suffix}@example.com")
        erin: Final = scenario.user(user_role="internal_user", user_email=f"erin-{suffix}@example.com")
        team_list: Final = scenario.team(
            members_with_roles=[{"user_id": alice, "role": "user"}, {"user_id": bob, "role": "user"}],
            team_member_permissions=["/key/info", "/key/list"],
        )
        team_plain: Final = scenario.team(
            members_with_roles=[
                {"user_id": dave, "role": "admin"},
                {"user_id": carol, "role": "user"},
                {"user_id": alice, "role": "user"},
            ]
        )

        alice_auth: Final = scenario.key(user_id=alice, key_alias="alice-auth")
        scenario.key(user_id=alice, team_id=team_list, key_alias="alice-in-LIST")
        scenario.key(user_id=alice, team_id=team_plain, key_alias="alice-in-PLAIN")
        scenario.key(user_id=bob, key_alias="bob-auth")
        scenario.key(user_id=bob, team_id=team_list, key_alias="bob-in-LIST")
        scenario.key(user_id=carol, key_alias="carol-auth")
        carol_plain: Final = scenario.key(user_id=carol, team_id=team_plain, key_alias="carol-in-PLAIN")
        scenario.key(user_id=dave, key_alias="dave-auth")
        dave_plain: Final = scenario.key(user_id=dave, team_id=team_plain, key_alias="dave-in-PLAIN")
        erin_auth: Final = scenario.key(user_id=erin, key_alias="erin-auth")
        _service_account_key(scenario, team_list, "svc-in-LIST")
        _service_account_key(scenario, team_plain, "svc-in-PLAIN")
        list_params: Final = {"return_full_object": "true", "size": "100"}

        cell_1: Final = _list_response(
            gateway,
            alice_auth,
            {**list_params, "user_id": alice, "include_team_keys": "true"},
        )
        cell_2: Final = _list_response(gateway, alice_auth, {**list_params, "user_id": bob})
        cell_3: Final = _list_response(gateway, dave_plain, {**list_params, "user_id": carol})
        cell_4: Final = _list_response(gateway, carol_plain, {**list_params, "user_id": dave})
        unknown_email: Final = f"missing-{suffix}@example.com"
        denied_by_user_id: Final = _list_response(gateway, erin_auth, {**list_params, "user_id": alice})
        denied_by_email: Final = _list_response(gateway, erin_auth, {**list_params, "user_email": unknown_email})
        cell_6: Final = _list_response(gateway, alice_auth, {**list_params, "user_email": bob_email.swapcase()})
        cell_7: Final = _list_response(gateway, alice_auth, {**list_params, "include_team_keys": "true"})
        cell_7_legacy: Final = _list_response(gateway, alice_auth, {**list_params, "include_created_by_keys": "true"})
        cell_8: Final = _list_response(
            gateway,
            alice_auth,
            {**list_params, "user_id": bob, "user_email": bob_email},
        )
        viewer: Final = scenario.user(user_role="proxy_admin_viewer", user_email=f"viewer-{suffix}@example.com")
        viewer_key: Final = scenario.key(user_id=viewer)
        cell_9_admin_bob: Final = _list_response(gateway, gateway.key, {**list_params, "user_email": bob_email})
        cell_9_admin_unknown: Final = _list_response(gateway, gateway.key, {**list_params, "user_email": unknown_email})
        cell_9_viewer_bob: Final = _list_response(gateway, viewer_key, {**list_params, "user_email": bob_email})
        cell_9_viewer_unknown: Final = _list_response(gateway, viewer_key, {**list_params, "user_email": unknown_email})
        checks: Final = (
            ("1", _matches_aliases(cell_1, {"alice-auth", "alice-in-LIST", "alice-in-PLAIN"})),
            ("2", _matches_aliases(cell_2, {"bob-in-LIST"})),
            ("3", _matches_aliases(cell_3, {"carol-in-PLAIN"})),
            ("4", cell_4.status_code == 403),
            (
                "5",
                denied_by_user_id.status_code == denied_by_email.status_code == 403
                and denied_by_user_id.json() == denied_by_email.json(),
            ),
            ("6", _matches_aliases(cell_6, {"bob-in-LIST"})),
            (
                "7",
                _matches_aliases(
                    cell_7,
                    {"alice-auth", "alice-in-LIST", "alice-in-PLAIN", "bob-in-LIST", "svc-in-LIST", "svc-in-PLAIN"},
                )
                and _matches_aliases(cell_7_legacy, {"alice-auth", "alice-in-LIST", "alice-in-PLAIN"}),
            ),
            ("8", cell_8.status_code == 400),
            (
                "9",
                _matches_aliases(cell_9_admin_bob, {"bob-auth", "bob-in-LIST"})
                and _matches_aliases(cell_9_viewer_bob, {"bob-auth", "bob-in-LIST"})
                and _matches_aliases(cell_9_admin_unknown, set())
                and _matches_aliases(cell_9_viewer_unknown, set()),
            ),
        )
        failed_cells: Final = tuple(cell for cell, passed in checks if not passed)
        assert failed_cells == (), f"Failed key-list contract cells: {failed_cells}"
