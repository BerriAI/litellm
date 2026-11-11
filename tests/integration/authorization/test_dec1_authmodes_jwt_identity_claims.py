"""JWT auth: how claims become the caller's user, email, team, organization and end user.

Contract (litellm/proxy/auth/handle_jwt.py `get_user_id`, `get_user_email`, `get_team_id`, `get_org_id`,
`get_end_user_id`, `find_and_validate_specific_team_id`, `get_objects`, `map_user_to_teams`;
litellm/proxy/auth/user_api_key_auth.py `_authorize_authenticated_request` for the body `user` fallback):

* Every `*_jwt_field` is a claim path in dot notation (`tenant.team` reads `{"tenant": {"team": ...}}`).
* `user_id_jwt_field` names the user; when the claim is missing the email claim stands in as the user id.
  `user_id_upsert` creates the user row on first sight, otherwise an unknown user is a 401.
* `team_id_jwt_field` makes a team claim required: a token without it is a 401 naming the field, a claim naming
  a team that does not exist is a 404, and a list claim uses its first element. `team_id_default` fills in a
  missing claim. `team_id_upsert` creates the team row on first sight.
* A caller whose user row exists is added to the claimed team's members as `user`.
* `org_id_jwt_field` and `end_user_id_jwt_field` attribute the spend row. A token end user wins over the
  request body's `user`; the body `user` is the end user only when the token names none, and it never
  replaces the token's user.

SpendLogs attribution (`user`, `team_id`, `organization_id`, `end_user`) is what customers bill from, so a
rewrite that resolves any of these differently is a billing change.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Final

import pytest

from tests.integration.authorization.dec1_authmodes_support import (
    JwtRig,
    chat_as,
    jwt_rig,
    served_body,
    spend_row,
    team_member_ids,
    unique,
    user_row,
)
from tests.integration.authorization.dec1_keys_support import assert_denied

DEFAULT_TEAM: Final = f"dec1-authmodes-default-team-{uuid.uuid4().hex[:12]}"


def _team_missing(team_id: str) -> str:
    return "{'error': \"Team doesn't exist in db. Team=" + team_id + '. Create team via `/team/new` call."}'


@pytest.fixture(scope="module")
def claims(tmp_path_factory: pytest.TempPathFactory) -> Iterator[JwtRig]:
    with jwt_rig(
        tmp_path_factory.mktemp("dec1-authmodes-claims"),
        user_id_jwt_field="sub",
        user_email_jwt_field="email",
        team_id_jwt_field="tenant.team",
        org_id_jwt_field="org",
        end_user_id_jwt_field="customer.id",
        user_id_upsert=True,
    ) as built:
        yield built


@pytest.fixture(scope="module")
def defaulted(tmp_path_factory: pytest.TempPathFactory) -> Iterator[JwtRig]:
    with jwt_rig(
        tmp_path_factory.mktemp("dec1-authmodes-team-default"),
        user_id_jwt_field="sub",
        team_id_jwt_field="team",
        team_id_default=DEFAULT_TEAM,
        user_id_upsert=True,
    ) as built:
        built.proxy.post("/team/new", {"team_id": DEFAULT_TEAM, "team_alias": DEFAULT_TEAM})
        yield built


@pytest.fixture(scope="module")
def strict_users(tmp_path_factory: pytest.TempPathFactory) -> Iterator[JwtRig]:
    with jwt_rig(
        tmp_path_factory.mktemp("dec1-authmodes-strict-users"),
        user_id_jwt_field="sub",
        team_id_jwt_field="team",
        team_id_upsert=True,
        user_id_upsert=False,
    ) as built:
        yield built


def test_a_nested_team_claim_attributes_the_spend_row_to_that_team(claims: JwtRig) -> None:
    """`team_id_jwt_field: tenant.team`; a token carries `{"tenant": {"team": <team id>}}` for a team that
    exists.

    The request must be served and its spend row's `team_id` must equal the nested claim value.
    """
    with claims.proxy.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        subject: Final = unique("nested-team")
        row: Final = spend_row(
            claims.proxy, chat_as(claims.proxy, claims.idp.token(sub=subject, tenant={"team": team}), model)
        )
        assert (row["user"], row["team_id"]) == (subject, team), row
        scenario.cleanups.callback(scenario.delete_user, subject)


def test_a_token_without_the_required_team_claim_is_a_401_naming_the_field(claims: JwtRig) -> None:
    """With `team_id_jwt_field: tenant.team` set, a correctly signed token carries no `tenant` claim.

    A configured team field makes the claim required, so the proxy must answer 401 `auth_error` with exactly
    "Authentication Error, No team found in token. Checked team_id field 'tenant.team' and team_alias field
    'None'." (handle_jwt.py:1697).
    """
    with claims.proxy.scenario() as scenario:
        model: Final = scenario.model()
        assert_denied(
            chat_as(claims.proxy, claims.idp.token(sub=unique("no-team")), model),
            status=401,
            error_type="auth_error",
            message=(
                "Authentication Error, No team found in token. Checked team_id field 'tenant.team' "
                "and team_alias field 'None'."
            ),
        )


def test_a_team_claim_naming_a_team_that_does_not_exist_is_a_404(claims: JwtRig) -> None:
    """A token's `tenant.team` names a team id that was never created, and `team_id_upsert` is off.

    Pinned current behaviour: the team lookup's 404 HTTPException passes through the auth handler with its
    dict detail stringified, so the proxy must answer 404 `auth_error` whose message is the Python repr
    `{'error': "Team doesn't exist in db. Team=<id>. Create team via `/team/new` call."}`.
    """
    with claims.proxy.scenario() as scenario:
        model: Final = scenario.model()
        missing: Final = unique("missing-team")
        token: Final = claims.idp.token(sub=unique("ghost-team"), tenant={"team": missing})
        assert_denied(
            chat_as(claims.proxy, token, model), status=404, error_type="auth_error", message=_team_missing(missing)
        )


def test_a_list_valued_team_claim_uses_its_first_element(claims: JwtRig) -> None:
    """A token's `tenant.team` is a list of two existing team ids.

    `team_id_jwt_field` is singular, so the proxy must take the first element: the spend row's `team_id`
    must be the first team, not the second.
    """
    with claims.proxy.scenario() as scenario:
        model: Final = scenario.model()
        first: Final = scenario.team()
        second: Final = scenario.team()
        subject: Final = unique("list-team")
        token: Final = claims.idp.token(sub=subject, tenant={"team": [first, second]})
        assert spend_row(claims.proxy, chat_as(claims.proxy, token, model))["team_id"] == first
        scenario.cleanups.callback(scenario.delete_user, subject)


def test_the_email_claim_is_stored_on_the_upserted_user_row(claims: JwtRig) -> None:
    """`user_email_jwt_field: email`; a token for a new `sub` carries an `email` claim.

    After the served request the upserted `LiteLLM_UserTable` row must carry exactly that `user_email`.
    """
    with claims.proxy.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        subject: Final = unique("email")
        email: Final = f"{subject}@dec1-authmodes.test"
        assert served_body(
            chat_as(claims.proxy, claims.idp.token(sub=subject, email=email, tenant={"team": team}), model)
        )
        row: Final = user_row(claims.proxy, subject)
        assert row is not None and row["user_email"] == email, row
        scenario.cleanups.callback(scenario.delete_user, subject)


def test_without_a_sub_claim_the_email_claim_becomes_the_user_id(claims: JwtRig) -> None:
    """A token carries an `email` but no `sub`.

    `get_user_id` falls back to the email claim (handle_jwt.py:1877), so the spend row's `user` and the
    upserted user row's `user_id` must both be the email address.
    """
    with claims.proxy.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        email: Final = f"{unique('mail-only')}@dec1-authmodes.test"
        row: Final = spend_row(
            claims.proxy, chat_as(claims.proxy, claims.idp.token(email=email, tenant={"team": team}), model)
        )
        assert row["user"] == email, row
        assert user_row(claims.proxy, email) is not None
        scenario.cleanups.callback(scenario.delete_user, email)


def test_the_org_claim_attributes_the_spend_row_to_that_organization(claims: JwtRig) -> None:
    """`org_id_jwt_field: org`; a token names an organization that exists.

    The spend row's `organization_id` must be that organization id.
    UNVERIFIED: assumes the spend writer reads the organization from the JWT identity's `org_id`
    (`user_api_key_org_id` metadata).
    """
    with claims.proxy.scenario() as scenario:
        model: Final = scenario.model()
        organization: Final = scenario.organization()
        team: Final = scenario.team()
        subject: Final = unique("org")
        token: Final = claims.idp.token(sub=subject, org=organization, tenant={"team": team})
        assert spend_row(claims.proxy, chat_as(claims.proxy, token, model))["organization_id"] == organization
        scenario.cleanups.callback(scenario.delete_user, subject)


def test_a_nested_end_user_claim_attributes_the_spend_row_to_that_end_user(claims: JwtRig) -> None:
    """`end_user_id_jwt_field: customer.id`; a token carries `{"customer": {"id": <customer>}}`.

    The spend row's `end_user` must be that customer id.
    """
    with claims.proxy.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        subject: Final = unique("end-user")
        customer: Final = unique("customer")
        token: Final = claims.idp.token(sub=subject, customer={"id": customer}, tenant={"team": team})
        assert spend_row(claims.proxy, chat_as(claims.proxy, token, model))["end_user"] == customer
        scenario.cleanups.callback(scenario.delete_user, subject)


def test_the_token_end_user_wins_over_the_request_body_user(claims: JwtRig) -> None:
    """A token names customer A in `customer.id`, and the request body sends `"user": "<customer B>"`.

    The JWT identity already carries an end user, so the body fallback never runs: the spend row's
    `end_user` must be A, and the `user` column must still be the token's `sub`.
    """
    with claims.proxy.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        subject: Final = unique("both")
        token_customer: Final = unique("token-customer")
        body_customer: Final = unique("body-customer")
        token: Final = claims.idp.token(sub=subject, customer={"id": token_customer}, tenant={"team": team})
        row: Final = spend_row(claims.proxy, chat_as(claims.proxy, token, model, user=body_customer))
        assert (row["end_user"], row["user"]) == (token_customer, subject), row
        scenario.cleanups.callback(scenario.delete_user, subject)


def test_without_a_token_end_user_the_body_user_becomes_the_end_user_but_not_the_user(claims: JwtRig) -> None:
    """A token has no `customer` claim, and the request body sends `"user": "<customer>"`.

    The spend row's `end_user` must be the body value, while `user` stays the token's `sub`: a caller can name
    an end user but can never impersonate another LiteLLM user through the body.
    """
    with claims.proxy.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        subject: Final = unique("body-user")
        body_customer: Final = unique("body-customer")
        token: Final = claims.idp.token(sub=subject, tenant={"team": team})
        row: Final = spend_row(claims.proxy, chat_as(claims.proxy, token, model, user=body_customer))
        assert (row["end_user"], row["user"]) == (body_customer, subject), row
        scenario.cleanups.callback(scenario.delete_user, subject)


def test_a_jwt_caller_is_added_to_the_claimed_teams_members(claims: JwtRig) -> None:
    """A new `sub` calls with a team claim for a team it is not a member of.

    `map_user_to_teams` must add the upserted user to the team: after the served request the team's
    `members_with_roles` must contain the `sub`, and the user row's `teams` must contain the team.
    """
    with claims.proxy.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        subject: Final = unique("joins")
        assert served_body(chat_as(claims.proxy, claims.idp.token(sub=subject, tenant={"team": team}), model))
        assert subject in team_member_ids(claims.proxy, team)
        row: Final = user_row(claims.proxy, subject)
        assert row is not None and team in (row["teams"] or []), row
        scenario.cleanups.callback(scenario.delete_user, subject)


def test_team_id_default_attributes_a_token_without_a_team_claim_to_the_default_team(defaulted: JwtRig) -> None:
    """`team_id_jwt_field: team` with `team_id_default` naming an existing team; a token has no `team` claim.

    Instead of the "No team found" 401, the request must be served and the spend row's `team_id` must be the
    default team.
    """
    with defaulted.proxy.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = unique("defaulted")
        row: Final = spend_row(defaulted.proxy, chat_as(defaulted.proxy, defaulted.idp.token(sub=subject), model))
        assert row["team_id"] == DEFAULT_TEAM, row
        scenario.cleanups.callback(scenario.delete_user, subject)


def test_a_team_claim_takes_precedence_over_team_id_default(defaulted: JwtRig) -> None:
    """Same proxy; a token carries a `team` claim naming another existing team.

    The default only fills a missing claim, so the spend row's `team_id` must be the claimed team.
    """
    with defaulted.proxy.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        subject: Final = unique("claimed")
        row: Final = spend_row(
            defaulted.proxy, chat_as(defaulted.proxy, defaulted.idp.token(sub=subject, team=team), model)
        )
        assert row["team_id"] == team, row
        scenario.cleanups.callback(scenario.delete_user, subject)


def test_team_id_upsert_creates_the_claimed_team_on_first_sight(strict_users: JwtRig) -> None:
    """`team_id_upsert: true`; an existing user calls with a `team` claim naming a team id that does not exist.

    The request must be served and a `LiteLLM_TeamTable` row with exactly that `team_id` must exist afterwards.
    """
    with strict_users.proxy.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = scenario.user()
        team: Final = unique("upserted-team")
        assert served_body(chat_as(strict_users.proxy, strict_users.idp.token(sub=subject, team=team), model))
        assert strict_users.proxy.rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,)) == [
            {"team_id": team}
        ]
        scenario.cleanups.callback(scenario.delete_team, team)


def test_without_user_id_upsert_an_unknown_sub_is_a_401_that_names_it(strict_users: JwtRig) -> None:
    """`user_id_upsert: false`; a token's `sub` names no user row.

    The proxy must answer 401 `auth_error` with exactly "Authentication Error, User doesn't exist in db.
    'user_id'=<sub>. Create user via `/user/new` call." and no user row may be created.
    """
    with strict_users.proxy.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        subject: Final = unique("stranger")
        assert_denied(
            chat_as(strict_users.proxy, strict_users.idp.token(sub=subject, team=team), model),
            status=401,
            error_type="auth_error",
            message=(
                f"Authentication Error, User doesn't exist in db. 'user_id'={subject}. "
                "Create user via `/user/new` call."
            ),
        )
        assert user_row(strict_users.proxy, subject) is None
