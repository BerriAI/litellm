"""How NULL, empty and default valued columns of the auth rows are read, as the stores move to typed Rust codecs.

Contract, read from the branch:

* The key row comes back from raw SQL (`combined_view`, litellm/proxy/utils.py:6829-6919) and is validated into
  `LiteLLM_VerificationTokenView`. Only `team_models` and `team_blocked` are normalised from NULL
  (utils.py:6907-6910). The key model (litellm/models/verification_token.py:16-66) declares `models: list`, so a
  NULL `models` column fails validation, while `allowed_routes`, `allowed_cache_controls`, `access_group_ids`,
  `blocked`, `router_settings` and `budget_limits` are `Optional` and accept NULL.
* A validation failure inside the key lookup is neither "not found" nor "database unavailable", so
  `_as_proxy_exception` turns it into 401 `auth_error` with "Authentication Error, <text>"
  (litellm/proxy/auth/auth_exception_handler.py:79-84).
* `allowed_routes` NULL and `[]` both mean "no route restriction" (litellm/proxy/auth/route_checks.py:161-167);
  `blocked` NULL is not blocked (`valid_token.blocked is True`, litellm/proxy/auth/user_api_key_auth.py:3230).
* The user model turns NULL `models` into `[]` (litellm/models/user.py:54-61), so a NULL list means every model.
* The team model turns an empty `members_with_roles` object (the column default is `{}`) into `[]` and accepts
  NULL `router_settings` and `metadata` (litellm/models/team.py:72-133).
* Postgres allows NULL in every `TEXT[]` column here (the baseline migration declares `"models" TEXT[]` without
  NOT NULL), so these states are reachable by migrations, manual SQL and older proxies.

Each test writes the state with SQL on a never used key, so the first request decodes the row cold. A Rust codec
that maps NULL to "empty list" where Python refuses, or refuses where Python reads "unrestricted", fails here.
"""

from __future__ import annotations

from typing import Final, LiteralString

import pytest

from tests.integration._support.client import Gateway
from tests.integration.authorization.dec1_dbstores2_support import (
    AFTER_TTL_SECONDS,
    assert_generic_auth_refusal,
    bare_project,
    name,
    nothing_reached_upstream,
    sql,
)
from tests.integration.authorization.dec1_dbstores_support import assert_ok, chat_as, settled
from tests.integration.authorization.dec1_keys_support import digest, error_of

_PROJECT_BLOCKED_TEMPLATE: Final = (
    "Authentication Error, Project={project} is blocked. Update via `/project/update` if you're an admin."
)


def test_a_key_whose_models_column_is_null_is_not_served(gateway: Gateway) -> None:
    """A never used key gets `models = NULL` by SQL and sends a chat completion.

    The answer is 401 `auth_error` with a message starting "Authentication Error, " (the `models: list` field
    rejects None while the view is built), and nothing reaches the upstream. A codec that decodes NULL as the empty
    list would silently widen the key to every model. UNVERIFIED: the exact message text (pydantic's
    "Input should be a valid list"); only the prefix is asserted.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        text: Final = name("prompt")
        sql(gateway, 'UPDATE "LiteLLM_VerificationToken" SET models = NULL WHERE token = %s', (digest(key),))
        assert_generic_auth_refusal(chat_as(gateway, key, model, text=text))
        assert nothing_reached_upstream(gateway, text)


_ALLOWED_ROUTES_WRITES: Final[dict[str, LiteralString]] = {
    "null": 'UPDATE "LiteLLM_VerificationToken" SET allowed_routes = NULL WHERE token = %s',
    "empty": 'UPDATE "LiteLLM_VerificationToken" SET allowed_routes = ARRAY[]::TEXT[] WHERE token = %s',
}


@pytest.mark.parametrize("state", sorted(_ALLOWED_ROUTES_WRITES))
def test_null_and_empty_allowed_routes_both_leave_every_route_open(gateway: Gateway, state: str) -> None:
    """A key created with `allowed_routes: ["/v1/models"]` gets `allowed_routes` set to NULL or to `{}` by SQL.

    Its first chat completion is served with 200 and `GET /v1/models` answers 200: both states skip the route
    check (route_checks.py:161-167). Python never stores NULL here itself, so a store that rejects NULL arrays or
    reads `{}` as "nothing allowed" breaks keys written by migrations.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(allowed_routes=["/v1/models"])
        sql(gateway, _ALLOWED_ROUTES_WRITES[state], (digest(key),))
        assert_ok(chat_as(gateway, key, model))
        assert gateway.request("GET", "/v1/models", key=key).status_code == 200


_BLOCKED_WRITES: Final[dict[str, LiteralString]] = {
    "null": 'UPDATE "LiteLLM_VerificationToken" SET blocked = NULL WHERE token = %s',
    "false": 'UPDATE "LiteLLM_VerificationToken" SET blocked = false WHERE token = %s',
}


@pytest.mark.parametrize("state", sorted(_BLOCKED_WRITES))
def test_a_null_or_false_blocked_column_serves_the_key(gateway: Gateway, state: str) -> None:
    """A key created with `blocked: true` gets `blocked` set to NULL or false by SQL before its first request.

    The request is served with 200 in both cases: the key column is `Boolean?` and only `blocked is True` refuses
    (user_api_key_auth.py:3230). A codec mapping NULL to "unknown, refuse" would lock out keys created before the
    column existed.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(blocked=True)
        sql(gateway, _BLOCKED_WRITES[state], (digest(key),))
        assert_ok(chat_as(gateway, key, model))


_OPTIONAL_KEY_COLUMN_WRITES: Final[dict[str, LiteralString]] = {
    "allowed_cache_controls": 'UPDATE "LiteLLM_VerificationToken" SET allowed_cache_controls = NULL WHERE token = %s',
    "access_group_ids": 'UPDATE "LiteLLM_VerificationToken" SET access_group_ids = NULL WHERE token = %s',
    "router_settings": 'UPDATE "LiteLLM_VerificationToken" SET router_settings = NULL WHERE token = %s',
    "budget_limits": 'UPDATE "LiteLLM_VerificationToken" SET budget_limits = NULL WHERE token = %s',
    "policies": 'UPDATE "LiteLLM_VerificationToken" SET policies = NULL WHERE token = %s',
}


@pytest.mark.parametrize("column", sorted(_OPTIONAL_KEY_COLUMN_WRITES))
def test_a_null_optional_key_column_still_serves_the_key(gateway: Gateway, column: str) -> None:
    """A never used key gets one optional column (`allowed_cache_controls`, `access_group_ids`, `router_settings`,
    `budget_limits` or `policies`) set to NULL by SQL.

    Its first request is served with 200: each field is `Optional` on `LiteLLM_VerificationToken`
    (verification_token.py:40-66), or, for `policies`, not a declared field at all. UNVERIFIED for `policies`: it
    is selected by `v.*` and assumed to be ignored by the model.

    For `router_settings` the column is set back to `'{}'` before the key's cleanup: `/key/delete` copies the row
    into `LiteLLM_DeletedVerificationToken` and answers 500 "`data.router_settings`: A value is required but not set"
    for a NULL (key_management_endpoints.py:6026), a product bug of the delete path reported in the triage, not of
    the request path this test is about.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        if column == "router_settings":
            scenario.cleanups.callback(
                sql,
                gateway,
                """UPDATE "LiteLLM_VerificationToken" SET router_settings = '{}'::jsonb WHERE token = %s""",
                (digest(key),),
            )
        sql(gateway, _OPTIONAL_KEY_COLUMN_WRITES[column], (digest(key),))
        assert_ok(chat_as(gateway, key, model))


def test_a_key_with_an_empty_models_array_written_by_sql_serves_any_model(gateway: Gateway) -> None:
    """A key created with `models: [<A>]` gets `models = '{}'` (the empty array) by SQL; it asks for model B.

    The request is served with 200: an empty key list restricts nothing. Paired with the NULL test above this
    pins that NULL and empty are different states for the key, which a store must not collapse.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        sql(gateway, 'UPDATE "LiteLLM_VerificationToken" SET models = ARRAY[]::TEXT[] WHERE token = %s', (digest(key),))
        assert_ok(chat_as(gateway, key, other))


def test_a_users_null_model_list_lets_their_personal_key_call_any_model(gateway: Gateway) -> None:
    """A user created with `models: [<A>]` gets `models = NULL` by SQL; their never used personal key asks for B.

    The request is served with 200: `LiteLLM_UserTable.set_model_info` turns a NULL list into `[]` (user.py:54-61),
    which means every model. Unlike the key, the user row treats NULL as empty.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        user: Final = scenario.user(models=[allowed])
        key: Final = scenario.key(user_id=user)
        sql(gateway, 'UPDATE "LiteLLM_UserTable" SET models = NULL WHERE user_id = %s', (user,))
        assert_ok(chat_as(gateway, key, other))


def test_a_team_with_an_empty_members_with_roles_object_serves_its_key(gateway: Gateway) -> None:
    """A team's `members_with_roles` is set to `'{}'::jsonb` (the column default) by SQL; its never used key calls.

    The request is served with 200: `TeamBase.set_model_info` turns an empty object into `[]` (team.py:121-122).
    Rows inserted by SQL or by old versions carry this default, so a typed codec must accept `{}` as an empty list.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team)
        sql(gateway, """UPDATE "LiteLLM_TeamTable" SET members_with_roles = '{}'::jsonb WHERE team_id = %s""", (team,))
        assert_ok(chat_as(gateway, key, model))


def test_a_team_with_null_router_settings_serves_its_key(gateway: Gateway) -> None:
    """A team's `router_settings` (a `Json?` column) is set to NULL by SQL; its never used key calls chat.

    The request is served with 200 (`router_settings: dict | None`, team.py). A codec that requires an object here
    would turn every legacy team into a 404 "Team doesn't exist in db".

    The column is set back to `'{}'` before the team's cleanup: `/team/delete` answers 500 "Internal server error"
    for a team whose `router_settings` is NULL (it copies the row into `LiteLLM_DeletedTeamTable`,
    team_endpoints.py:576), a product bug of the delete path reported in the triage.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        scenario.cleanups.callback(
            sql, gateway, """UPDATE "LiteLLM_TeamTable" SET router_settings = '{}'::jsonb WHERE team_id = %s""", (team,)
        )
        key: Final = scenario.key(team_id=team)
        sql(gateway, 'UPDATE "LiteLLM_TeamTable" SET router_settings = NULL WHERE team_id = %s', (team,))
        assert_ok(chat_as(gateway, key, model))


def test_a_team_with_a_null_model_list_serves_its_key_any_model(gateway: Gateway) -> None:
    """A team created with `models: [<A>]` gets `models = NULL` by SQL; its never used key asks for model B.

    Intended: served with 200, as for the empty list. The key row's copy is normalised (`team_models` NULL becomes
    `[]`, utils.py:6907-6908). UNVERIFIED: the team lookup goes through Prisma `find_unique`; if Prisma hands back
    None for a NULL scalar list, `LiteLLM_TeamTable.models: list[str]` rejects it, `get_team_object` reports 404
    "Team doesn't exist in db" (auth_checks.py:3896-3901) and the key cannot vouch with an empty team list. A 404
    here would be a product bug that the Rust team store must not copy.

    The answer is polled for up to `AFTER_TTL_SECONDS`: `/key/generate` reads the team to validate the key and
    leaves it in the auth cache with `models: [<A>]`, and the SQL write evicts nothing, so the first requests saw
    the cached list and were refused 403 `team_model_access_denied` (run 44176682961).
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        team: Final = scenario.team(models=[allowed])
        key: Final = scenario.key(team_id=team)
        sql(gateway, 'UPDATE "LiteLLM_TeamTable" SET models = NULL WHERE team_id = %s', (team,))
        assert_ok(settled(lambda: chat_as(gateway, key, other), 200, AFTER_TTL_SECONDS))


def test_a_blocked_project_with_a_null_model_list_still_refuses_its_key(gateway: Gateway) -> None:
    """A project created with `blocked: true` gets `models = NULL` by SQL; its never used key calls chat.

    Intended: 401 with "Authentication Error, Project=<id> is blocked. ..." exactly as for a blocked project with a
    list. UNVERIFIED: `get_project_object` validates the Prisma row into `LiteLLM_ProjectTableCachedObj`
    (`models: list[str]`, no NULL normaliser, litellm/models/project.py:24); if that raises, `_safe_fetch`
    swallows it (user_api_key_auth.py:3756-3763) and the block is lost for the request. Served would be a bug.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        project: Final = bare_project(gateway, team, blocked=True)
        scenario.cleanups.callback(scenario.delete_project, project)
        key: Final = scenario.key(team_id=team, project_id=project)
        sql(gateway, 'UPDATE "LiteLLM_ProjectTable" SET models = NULL WHERE project_id = %s', (project,))
        refused: Final = chat_as(gateway, key, model)
        assert refused.status_code == 401, refused.text
        assert error_of(refused)["message"] == _PROJECT_BLOCKED_TEMPLATE.format(project=project), refused.text


def test_a_team_metadata_object_without_require_trace_id_does_not_gate(gateway: Gateway) -> None:
    """A team's `metadata` is overwritten by SQL with `'{"require_trace_id": null}'::jsonb`; its key calls chat.

    The request is served with 200: only the JSON boolean `true` turns the gate on (`_team_metadata_requires_trace_id`,
    auth_checks.py:460-470). A codec that maps JSON null to a default of `true`, or drops unknown nulls into
    "missing" differently, must keep this answer.

    The answer is polled for up to `AFTER_TTL_SECONDS`: the team was cached with `require_trace_id: true` when the
    key was generated and the SQL write evicts nothing, so the first request got the trace ID 400 from the cached
    team (run 44176682961).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(metadata={"require_trace_id": True})
        key: Final = scenario.key(team_id=team)
        sql(
            gateway,
            """UPDATE "LiteLLM_TeamTable" SET metadata = '{"require_trace_id": null}'::jsonb WHERE team_id = %s""",
            (team,),
        )
        assert_ok(settled(lambda: chat_as(gateway, key, model), 200, AFTER_TTL_SECONDS))
