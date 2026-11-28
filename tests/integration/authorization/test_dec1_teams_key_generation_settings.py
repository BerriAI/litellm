"""`litellm_settings.key_generation_settings` and the team lookup on `/key/generate`.

Contract, read from litellm/proxy/management_endpoints/key_management_endpoints.py (`generate_key_fn`,
`key_generation_check`, `_team_key_generation_check`, `_team_key_operation_team_member_check`,
`_key_generation_required_param_check`, `_personal_key_membership_check`):

* `generate_key_fn` reads the team with `get_team_object(check_db_only=True)`, the lookup #52176 moves to
  `RustTeamStore`. A non admin naming a team that does not exist gets 400 `Team not found for team_id=<id>. Non-admin
  users cannot create keys for non-existent teams.`; a proxy admin is let through and gets a key bound to a team id
  with no row, unless `key_generation_settings` is set, in which case even the admin gets 400 `Unable to find team
  object in database. Team ID: <id>`.
* `team_key_generation.allowed_team_member_roles` refuses other team roles with 400 `Team member role <role> not in
  allowed_team_member_roles=[...]`; the default when unset is `["admin", "user"]`.
* `team_key_generation.required_params` and `personal_key_generation.required_params` refuse a request that does not
  set the named field with 400 `Required param <name> not in data`.
* `personal_key_generation.allowed_user_roles` refuses other proxy roles with 400 `Personal key creation has been
  restricted by admin. Allowed roles=[...]. Your role=<role>`.
* A proxy admin skips every `team_key_generation` rule (`_team_key_generation_check` returns first).

The settings tests boot one owned proxy for the module (`owned_door` from `security/dec1_authsec_support.py`) with
`key_generation_settings` set; the other tests use the suite `gateway`. Callers other than the master key use a
personal key generated for the user by the master key, so their `user_role` and team role come from the database.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Final

import pytest

from tests.integration._support.client import Gateway, Scenario
from tests.integration.authorization.dec1_teams_support import chat_as, error_of
from tests.integration.security.dec1_authsec_support import Door, owned_door

_ADMIN_ONLY_TEAM_KEYS: Final = ["admin"]
_REQUIRED_TEAM_PARAM: Final = "tags"
_PERSONAL_ROLES: Final = ["proxy_admin"]


@pytest.fixture(scope="module")
def restricted(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Door]:
    with owned_door(
        tmp_path_factory.mktemp("dec1-teams-keygen"),
        litellm_settings={
            "key_generation_settings": {
                "team_key_generation": {
                    "allowed_team_member_roles": _ADMIN_ONLY_TEAM_KEYS,
                    "required_params": [_REQUIRED_TEAM_PARAM],
                },
                "personal_key_generation": {"allowed_user_roles": _PERSONAL_ROLES},
            }
        },
    ) as door:
        yield door


def _personal_key_of(scenario: Scenario, user_id: str) -> str:
    """A key the master key creates for `user_id`, so the master key is not subject to personal key rules."""
    return scenario.key(user_id=user_id)


def test_a_team_member_with_role_user_is_refused_when_only_admins_may_create_team_keys(restricted: Door) -> None:
    """With `allowed_team_member_roles: ["admin"]`, a team member with role `user` asks `/key/generate` for a team key
    (with `tags`, so the required param rule is satisfied).

    The answer is 400 with message `Team member role user not in allowed_team_member_roles=['admin']`. The caller's
    team role comes from the team row's `members_with_roles`, which the team store must return in full.
    """
    gateway: Final = restricted.gateway
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        member: Final = scenario.member(team)
        refused = gateway.request(
            "POST",
            "/key/generate",
            {"team_id": team, "user_id": member, "tags": ["dec1-teams"]},
            key=_personal_key_of(scenario, member),
        )
        assert refused.status_code == 400, refused.text
        assert error_of(refused)["message"] == "Team member role user not in allowed_team_member_roles=['admin']"


def test_a_team_admin_may_create_a_team_key_when_only_admins_may(restricted: Door) -> None:
    """With the same settings, a team member with role `admin` asks for a team key for themselves with `tags`.

    The answer is 200 and the key row names the team.
    """
    gateway: Final = restricted.gateway
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        admin: Final = scenario.member(team, role="admin")
        created = gateway.request(
            "POST",
            "/key/generate",
            {"team_id": team, "user_id": admin, "tags": ["dec1-teams"]},
            key=_personal_key_of(scenario, admin),
        )
        assert created.status_code == 200, created.text
        scenario.cleanups.callback(scenario.delete_key, created.json()["key"])
        assert created.json()["team_id"] == team, created.text


def test_a_team_admin_omitting_a_required_param_is_refused(restricted: Door) -> None:
    """With `required_params: ["tags"]`, a team admin asks for a team key without `tags`.

    The answer is 400 with message `Required param tags not in data`; the check uses
    `model_dump(exclude_unset=True)`, so only a field the caller actually sent counts.
    """
    gateway: Final = restricted.gateway
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        admin: Final = scenario.member(team, role="admin")
        refused = gateway.request(
            "POST", "/key/generate", {"team_id": team, "user_id": admin}, key=_personal_key_of(scenario, admin)
        )
        assert refused.status_code == 400, refused.text
        assert error_of(refused)["message"] == f"Required param {_REQUIRED_TEAM_PARAM} not in data"


def test_the_proxy_admin_skips_the_team_key_generation_rules(restricted: Door) -> None:
    """The master key asks for a team key without `tags` and with no team role at all.

    The answer is 200: `_team_key_generation_check` returns before the role and required param checks for a proxy
    admin.
    """
    gateway: Final = restricted.gateway
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team)
        assert key.startswith("sk-")


def test_an_internal_user_is_refused_a_personal_key_when_only_admins_may(restricted: Door) -> None:
    """With `personal_key_generation.allowed_user_roles: ["proxy_admin"]`, an internal user asks for a key with no
    team.

    The answer is 400 with message `Personal key creation has been restricted by admin. Allowed
    roles=['proxy_admin']. Your role=internal_user`.
    """
    gateway: Final = restricted.gateway
    with gateway.scenario() as scenario:
        user: Final = scenario.user(user_role="internal_user")
        refused = gateway.request("POST", "/key/generate", {"user_id": user}, key=_personal_key_of(scenario, user))
        assert refused.status_code == 400, refused.text
        assert error_of(refused)["message"] == (
            "Personal key creation has been restricted by admin. Allowed roles=['proxy_admin']. Your role=internal_user"
        )


def test_with_settings_even_the_proxy_admin_cannot_name_a_missing_team(restricted: Door) -> None:
    """With `key_generation_settings` set, the master key asks for a key with a `team_id` that has no row.

    The answer is 400 with message `Unable to find team object in database. Team ID: <id>`: `key_generation_check`
    raises when the team is missing and settings exist, before it considers the caller's role.
    """
    missing: Final = f"dec1-teams-missing-{uuid.uuid4().hex}"
    refused: Final = restricted.gateway.request("POST", "/key/generate", {"team_id": missing})
    assert refused.status_code == 400, refused.text
    assert error_of(refused)["message"] == f"Unable to find team object in database. Team ID: {missing}"


def test_a_non_admin_cannot_create_a_key_for_a_team_that_does_not_exist(gateway: Gateway) -> None:
    """On the suite proxy (no `key_generation_settings`), an internal user asks for a key naming a team id with no row.

    The answer is 400 with message `Team not found for team_id=<id>. Non-admin users cannot create keys for
    non-existent teams.` (LIT-1884) and no key row with that team id exists.
    """
    missing: Final = f"dec1-teams-missing-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        user: Final = scenario.user(user_role="internal_user")
        refused = gateway.request(
            "POST", "/key/generate", {"team_id": missing, "user_id": user}, key=_personal_key_of(scenario, user)
        )
        assert refused.status_code == 400, refused.text
        assert error_of(refused)["message"] == (
            f"Team not found for team_id={missing}. Non-admin users cannot create keys for non-existent teams."
        )
        assert gateway.rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE team_id=%s', (missing,)) == []


def test_without_settings_the_proxy_admin_gets_a_key_for_a_missing_team_that_is_refused_on_use(
    gateway: Gateway,
) -> None:
    """On the suite proxy, the master key creates a key with a `team_id` that has no row, then the key calls chat.

    Creation answers 200 (`admins can assign team_id without team table`), and the chat is refused with 404 whose
    message names the missing team (`Team doesn't exist in db. Team=<id>. Create team via `/team/new` call.`,
    auth_checks.py:3899). UNVERIFIED: the exact body shape of that 404 (`detail.error` versus `error.message`); only
    the status and the team id in the body are asserted.
    """
    missing: Final = f"dec1-teams-missing-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(team_id=missing)
        refused: Final = chat_as(gateway, key, model)
        assert refused.status_code == 404, refused.text
        assert missing in refused.text, refused.text
