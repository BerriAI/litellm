from __future__ import annotations

import uuid
import warnings
from typing import Final

import httpx
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import Gateway, Scenario, string_value

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_KEY_ROWS: Final = TypeAdapter(list[dict[str, JsonValue]])


def _key_list_params(user_id: str | None) -> dict[str, str]:
    defaults: Final = user_id is None
    return {
        "page": "1",
        "size": "50",
        "sort_by": "created_at",
        "sort_order": "desc",
        "expand": "user",
        "return_full_object": "true",
        "include_team_keys": str(defaults).lower(),
        "include_created_by_keys": str(defaults).lower(),
        "substring_matching": str(defaults).lower(),
        **({"user_id": user_id} if user_id is not None else {}),
    }


def _key_rows(response: httpx.Response) -> tuple[dict[str, JsonValue], ...]:
    assert response.status_code == 200, f"GET /key/list: {response.status_code} {response.text}"
    body: Final = _JSON_OBJECT.validate_json(response.content)
    return tuple(_KEY_ROWS.validate_python(body["keys"]))


def _key_aliases(rows: tuple[dict[str, JsonValue], ...]) -> frozenset[str]:
    return frozenset(string_value(row["key_alias"]) for row in rows)


def _create_key_for_dave(
    scenario: Scenario,
    carol_key: str,
    dave: str,
    team: str,
    key_alias: str,
) -> str | None:
    response: Final = scenario.gateway.request(
        "POST",
        "/key/generate",
        {"user_id": dave, "team_id": team, "key_alias": key_alias},
        key=carol_key,
    )
    if response.status_code != 200:
        warnings.warn(
            f"Carol's /key/generate request was refused with {response.status_code}: {response.text}",
            stacklevel=2,
        )
        return None
    created: Final = _JSON_OBJECT.validate_json(response.content)
    scenario.cleanups.callback(scenario.delete_key, string_value(created["key"]))
    return key_alias


def test_my_keys_hides_teammate_keys_for_internal_user(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team(team_member_permissions=["/key/list"])
        alice: Final = scenario.member(team)
        bob: Final = scenario.member(team)

        alice_auth_alias: Final = f"alice-auth-{uuid.uuid4().hex}"
        alice_team_alias: Final = f"alice-team-{uuid.uuid4().hex}"
        bob_team_alias: Final = f"bob-team-{uuid.uuid4().hex}"
        service_account_alias: Final = f"service-account-{uuid.uuid4().hex}"

        alice_key: Final = scenario.key(user_id=alice, key_alias=alice_auth_alias)
        scenario.key(user_id=alice, team_id=team, key_alias=alice_team_alias)
        scenario.key(user_id=bob, team_id=team, key_alias=bob_team_alias)
        service_account: Final = gateway.post(
            "/key/service-account/generate",
            {"team_id": team, "key_alias": service_account_alias},
        )
        scenario.cleanups.callback(scenario.delete_key, string_value(service_account["key"]))

        defaults: Final = gateway.request(
            "GET",
            "/key/list",
            key=alice_key,
            params=_key_list_params(user_id=None),
        )
        default_aliases: Final = _key_aliases(_key_rows(defaults))
        assert {bob_team_alias, service_account_alias} <= default_aliases

        my_keys: Final = gateway.request(
            "GET",
            "/key/list",
            key=alice_key,
            params=_key_list_params(user_id=alice),
        )
        my_key_rows: Final = _key_rows(my_keys)
        assert _key_aliases(my_key_rows) == frozenset({alice_auth_alias, alice_team_alias})
        assert {string_value(row["user_id"]) for row in my_key_rows} == {alice}


def test_my_keys_hides_team_keys_for_team_admin(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        carol: Final = scenario.user(user_role="internal_user")
        team: Final = scenario.team(members_with_roles=[{"role": "admin", "user_id": carol}])
        dave: Final = scenario.member(team)

        carol_auth_alias: Final = f"carol-auth-{uuid.uuid4().hex}"
        carol_team_alias: Final = f"carol-team-{uuid.uuid4().hex}"
        dave_team_alias: Final = f"dave-team-{uuid.uuid4().hex}"
        created_for_dave_alias: Final = f"carol-created-for-dave-{uuid.uuid4().hex}"

        carol_key: Final = scenario.key(user_id=carol, key_alias=carol_auth_alias)
        scenario.key(user_id=carol, team_id=team, key_alias=carol_team_alias)
        scenario.key(user_id=dave, team_id=team, key_alias=dave_team_alias)
        created_for_dave: Final = _create_key_for_dave(scenario, carol_key, dave, team, created_for_dave_alias)

        defaults: Final = gateway.request(
            "GET",
            "/key/list",
            key=carol_key,
            params=_key_list_params(user_id=None),
        )
        default_aliases: Final = _key_aliases(_key_rows(defaults))
        assert dave_team_alias in default_aliases
        if created_for_dave is not None:
            assert created_for_dave in default_aliases

        my_keys: Final = gateway.request(
            "GET",
            "/key/list",
            key=carol_key,
            params=_key_list_params(user_id=carol),
        )
        assert _key_aliases(_key_rows(my_keys)) == frozenset({carol_auth_alias, carol_team_alias})
