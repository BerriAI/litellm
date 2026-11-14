"""JWT auth: picking a team from group claims, and the team and user controls (model lists, budgets, rate
limits, blocking) that apply to JWT callers exactly as they apply to virtual keys.

Contract (litellm/proxy/auth/handle_jwt.py `find_team_with_model_access`, `resolve_team_from_header`;
litellm/proxy/auth/user_api_key_auth.py `run_centralized_common_checks`; litellm/proxy/hooks/
parallel_request_limiter_v3.py):

* `team_ids_jwt_field` lists candidate teams. The proxy walks them in sorted order and picks the first whose
  `models` allow the requested model and whose route is in `team_allowed_routes`. No such team is a 403 naming
  the model and the checked team set; a team that has the model but not the route is a 403 naming both.
  `x-litellm-team-id` (an id or a unique alias) picks one of the token's teams explicitly; any other value is a
  403 listing the allowed ids. `enforce_team_based_model_access` refuses a token with no teams at all.
* Once a team or user is resolved, the shared checks run: team model list (403 `team_model_access_denied`),
  team blocked (401), team budget (422 `budget_exceeded`), user budget for team-less callers (422), and the
  v3 limiter's team and user rpm limits (429 `throttling_error`).

A rewrite that authenticates the JWT but forgets to hand its team or user to these checks would silently
lift every spend control for SSO users, which is why each control is pinned for the JWT path here.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest

from tests.integration._support.client import eventually
from tests.integration.authorization.dec1_authmodes_support import (
    JwtRig,
    chat_as,
    jwt_rig,
    message_of,
    served_body,
    spend_row,
    unique,
)
from tests.integration.authorization.dec1_keys_support import assert_denied, assert_model_denied, error_of
from tests.integration.authorization.dec1_limits_support import assert_limited

pytestmark: Final = pytest.mark.timeout(240)


@pytest.fixture(scope="module")
def groups(tmp_path_factory: pytest.TempPathFactory) -> Iterator[JwtRig]:
    with jwt_rig(
        tmp_path_factory.mktemp("dec1-authmodes-groups"),
        user_id_jwt_field="sub",
        user_id_upsert=True,
        team_ids_jwt_field="groups",
    ) as built:
        yield built


@pytest.fixture(scope="module")
def narrow_routes(tmp_path_factory: pytest.TempPathFactory) -> Iterator[JwtRig]:
    with jwt_rig(
        tmp_path_factory.mktemp("dec1-authmodes-team-routes"),
        user_id_jwt_field="sub",
        user_id_upsert=True,
        team_ids_jwt_field="groups",
        team_allowed_routes=["/v1/embeddings"],
    ) as built:
        yield built


@pytest.fixture(scope="module")
def team_required(tmp_path_factory: pytest.TempPathFactory) -> Iterator[JwtRig]:
    with jwt_rig(
        tmp_path_factory.mktemp("dec1-authmodes-team-required"),
        user_id_jwt_field="sub",
        user_id_upsert=True,
        team_ids_jwt_field="groups",
        enforce_team_based_model_access=True,
    ) as built:
        yield built


@pytest.fixture(scope="module")
def team_claim(tmp_path_factory: pytest.TempPathFactory) -> Iterator[JwtRig]:
    with jwt_rig(
        tmp_path_factory.mktemp("dec1-authmodes-team-claim"),
        user_id_jwt_field="sub",
        user_id_upsert=True,
        team_id_jwt_field="team",
    ) as built:
        yield built


@pytest.fixture(scope="module")
def users_only(tmp_path_factory: pytest.TempPathFactory) -> Iterator[JwtRig]:
    with jwt_rig(tmp_path_factory.mktemp("dec1-authmodes-users"), user_id_jwt_field="sub") as built:
        yield built


def test_group_claims_pick_the_first_team_whose_models_allow_the_request(groups: JwtRig) -> None:
    """`team_ids_jwt_field: groups`; the token lists two teams, one whose `models` hold only another model and
    one whose `models` hold the requested model.

    The request must be served and attributed to the team that can call the model, whatever the claim order.
    """
    with groups.proxy.scenario() as scenario:
        model: Final = scenario.model()
        other: Final = scenario.model()
        without: Final = scenario.team(models=[other])
        holder: Final = scenario.team(models=[model])
        subject: Final = unique("groups")
        row: Final = spend_row(
            groups.proxy, chat_as(groups.proxy, groups.idp.token(sub=subject, groups=[without, holder]), model)
        )
        assert row["team_id"] == holder, row
        scenario.cleanups.callback(scenario.delete_user, subject)


def test_when_no_group_team_has_the_model_the_403_names_the_model_and_the_teams(groups: JwtRig) -> None:
    """The token lists one team whose `models` do not include the requested model.

    The proxy must answer 403 `auth_error` with exactly "No team has access to the requested model: <model>.
    Checked teams={'<team>'}. Check `/models` to see all available models." (a Python set repr).
    """
    with groups.proxy.scenario() as scenario:
        model: Final = scenario.model()
        other: Final = scenario.model()
        team: Final = scenario.team(models=[other])
        assert_denied(
            chat_as(groups.proxy, groups.idp.token(sub=unique("no-access"), groups=[team]), model),
            status=403,
            error_type="auth_error",
            message=(
                f"No team has access to the requested model: {model}. Checked teams={{'{team}'}}. "
                "Check `/models` to see all available models."
            ),
        )


def test_the_team_header_selects_one_of_the_tokens_group_teams(groups: JwtRig) -> None:
    """The token lists two teams that can both call the model, and the request sends `x-litellm-team-id`
    naming the one that sorts last.

    Without the header the proxy would pick the first sorted team; with it the spend row's `team_id` must be
    the header's team.
    """
    with groups.proxy.scenario() as scenario:
        model: Final = scenario.model()
        first, last = sorted((scenario.team(models=[model]), scenario.team(models=[model])))
        subject: Final = unique("header-team")
        response: Final = chat_as(
            groups.proxy,
            groups.idp.token(sub=subject, groups=[first, last]),
            model,
            headers={"x-litellm-team-id": last},
        )
        assert spend_row(groups.proxy, response)["team_id"] == last
        scenario.cleanups.callback(scenario.delete_user, subject)


def test_the_team_header_accepts_a_team_alias_from_the_tokens_groups(groups: JwtRig) -> None:
    """The token lists one team; the request sends `x-litellm-team-id` carrying that team's alias, not its id.

    A unique alias resolves to its team id, so the request must be served and attributed to that team.
    """
    with groups.proxy.scenario() as scenario:
        model: Final = scenario.model()
        alias: Final = unique("alias")
        team: Final = scenario.team(models=[model], team_alias=alias)
        subject: Final = unique("alias-header")
        response: Final = chat_as(
            groups.proxy, groups.idp.token(sub=subject, groups=[team]), model, headers={"x-litellm-team-id": alias}
        )
        assert spend_row(groups.proxy, response)["team_id"] == team
        scenario.cleanups.callback(scenario.delete_user, subject)


def test_a_team_header_outside_the_tokens_groups_is_a_403_listing_the_allowed_ids(groups: JwtRig) -> None:
    """The token lists one team; the request sends `x-litellm-team-id` naming a different existing team.

    The proxy must answer 403 `auth_error` with exactly "x-litellm-team-id '<other>' does not resolve to a team
    id or a unique team alias in your JWT's allowed teams. Allowed team ids: ['<team>']".
    """
    with groups.proxy.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        other: Final = scenario.team(models=[model])
        assert_denied(
            chat_as(
                groups.proxy,
                groups.idp.token(sub=unique("wrong-header"), groups=[team]),
                model,
                headers={"x-litellm-team-id": other},
            ),
            status=403,
            error_type="auth_error",
            message=(
                f"x-litellm-team-id '{other}' does not resolve to a team id or a unique team alias in your JWT's "
                f"allowed teams. Allowed team ids: ['{team}']"
            ),
        )


def test_a_group_team_with_the_model_but_not_the_route_is_a_403_naming_both(narrow_routes: JwtRig) -> None:
    """`team_allowed_routes: ["/v1/embeddings"]`; the token's only team can call the model, and the caller
    sends a chat completion.

    The proxy must answer 403 `auth_error` with exactly "Team <team> can access model <model> but route
    /v1/chat/completions is not in litellm_jwtauth.team_allowed_routes".
    """
    with narrow_routes.proxy.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        assert_denied(
            chat_as(narrow_routes.proxy, narrow_routes.idp.token(sub=unique("routes"), groups=[team]), model),
            status=403,
            error_type="auth_error",
            message=(
                f"Team {team} can access model {model} but route /v1/chat/completions is not in "
                "litellm_jwtauth.team_allowed_routes"
            ),
        )


def test_enforce_team_based_model_access_refuses_a_token_with_no_teams(team_required: JwtRig) -> None:
    """`enforce_team_based_model_access: true`; a token has a `sub` but no `groups` claim.

    The proxy must answer 403 `auth_error` with exactly "No teams found in token.
    `enforce_team_based_model_access` is set to True. Token must belong to a team."
    """
    with team_required.proxy.scenario() as scenario:
        model: Final = scenario.model()
        assert_denied(
            chat_as(team_required.proxy, team_required.idp.token(sub=unique("teamless")), model),
            status=403,
            error_type="auth_error",
            message=(
                "No teams found in token. `enforce_team_based_model_access` is set to True. "
                "Token must belong to a team."
            ),
        )


def test_enforce_team_based_model_access_serves_a_token_whose_team_has_the_model(team_required: JwtRig) -> None:
    """Same proxy; the token lists one team whose `models` include the requested model.

    The request must be served and attributed to that team.
    """
    with team_required.proxy.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        subject: Final = unique("team-member")
        row: Final = spend_row(
            team_required.proxy,
            chat_as(team_required.proxy, team_required.idp.token(sub=subject, groups=[team]), model),
        )
        assert row["team_id"] == team, row
        scenario.cleanups.callback(scenario.delete_user, subject)


def test_a_claimed_teams_model_list_refuses_another_model(team_claim: JwtRig) -> None:
    """`team_id_jwt_field: team`; the claimed team's `models` hold one model, and the caller asks for another
    served model.

    The shared team check must answer 403 `team_model_access_denied`, `param: model`, with the generic "The
    requested model '<model>' is not available for this API key ..." message.
    """
    with team_claim.proxy.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        team: Final = scenario.team(models=[allowed])
        refused: Final = chat_as(team_claim.proxy, team_claim.idp.token(sub=unique("team-models"), team=team), other)
        assert_model_denied(refused, other, "team_model_access_denied")


def test_a_blocked_claimed_team_refuses_its_jwt_callers(team_claim: JwtRig) -> None:
    """The claimed team is blocked through `POST /team/block` before any JWT call.

    The proxy must answer 401 `auth_error` with exactly "Authentication Error, Team=<team> is blocked. Update
    via `/team/unblock` if you're an admin." (auth_checks.py:1222).
    """
    with team_claim.proxy.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        team_claim.proxy.post("/team/block", {"team_id": team})
        assert_denied(
            chat_as(team_claim.proxy, team_claim.idp.token(sub=unique("blocked-team"), team=team), model),
            status=401,
            error_type="auth_error",
            message=f"Authentication Error, Team={team} is blocked. Update via `/team/unblock` if you're an admin.",
        )


def test_a_claimed_teams_rpm_limit_throttles_its_jwt_callers(team_claim: JwtRig) -> None:
    """The claimed team has `rpm_limit: 1`; two different subjects of that team each send one chat.

    The first must be served and the second must get the v3 limiter's 429 `throttling_error` "Rate limit
    exceeded for team: <team>. Limit type: requests. Current limit: 1, Remaining: 0. ...". The limit is the
    team's, shared by every JWT user of it.
    """
    with team_claim.proxy.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(rpm_limit=1)
        first: Final = unique("rpm-first")
        second: Final = unique("rpm-second")
        assert served_body(chat_as(team_claim.proxy, team_claim.idp.token(sub=first, team=team), model))
        assert_limited(
            chat_as(team_claim.proxy, team_claim.idp.token(sub=second, team=team), model), "team", team, "requests", 1
        )
        scenario.cleanups.callback(scenario.delete_user, first)
        scenario.cleanups.callback(scenario.delete_user, second)


def test_a_claimed_teams_budget_stops_its_jwt_callers_once_spent(team_claim: JwtRig) -> None:
    """The claimed team has `max_budget: 0.000000001`. One JWT call is served and charged, then the caller keeps
    calling.

    Polling, a later call must be refused with 422 `budget_exceeded` whose message starts "Budget has been
    exceeded! Team=<team> Current cost: ".
    """
    with team_claim.proxy.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(max_budget=0.000000001)
        subject: Final = unique("team-budget")
        assert served_body(chat_as(team_claim.proxy, team_claim.idp.token(sub=subject, team=team), model))
        refused: Final = eventually(
            lambda: chat_as(team_claim.proxy, team_claim.idp.token(sub=subject, team=team), model),
            lambda response: response.status_code == 422,
            seconds=70,
        )
        assert error_of(refused)["type"] == "budget_exceeded", refused.text
        assert message_of(refused).startswith(f"Budget has been exceeded! Team={team} Current cost: "), refused.text
        scenario.cleanups.callback(scenario.delete_user, subject)


def test_a_teamless_jwt_users_zero_budget_refuses_the_first_call(users_only: JwtRig) -> None:
    """Only `user_id_jwt_field: sub`; the user row named by `sub` exists with `max_budget: 0` and no spend.

    A user budget is spent when spend is greater than or equal to it, so the very first call must be refused
    with 422 `budget_exceeded` and exactly "ExceededBudget: User=<sub> over budget. Spend=0.0, Budget=0.0".
    """
    with users_only.proxy.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = scenario.user(max_budget=0)
        assert_denied(
            chat_as(users_only.proxy, users_only.idp.token(sub=subject), model),
            status=422,
            error_type="budget_exceeded",
            message=f"ExceededBudget: User={subject} over budget. Spend=0.0, Budget=0.0",
        )


def test_a_jwt_users_rpm_limit_throttles_the_second_call(users_only: JwtRig) -> None:
    """The user row named by `sub` has `rpm_limit: 1`; the same subject sends two chats with fresh prompts.

    The first must be served and the second must get the 429 `throttling_error` "Rate limit exceeded for
    user: <sub>. Limit type: requests. Current limit: 1, Remaining: 0. ...".
    """
    with users_only.proxy.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = scenario.user(rpm_limit=1)
        assert served_body(chat_as(users_only.proxy, users_only.idp.token(sub=subject), model))
        assert_limited(
            chat_as(users_only.proxy, users_only.idp.token(sub=subject), model), "user", subject, "requests", 1
        )


def test_a_jwt_users_model_list_refuses_another_model(users_only: JwtRig) -> None:
    """The user row named by `sub` has `models` holding one model; the caller asks for another served model.

    The proxy must answer 403 `user_model_access_denied`, `param: model`, with the generic model denial.
    UNVERIFIED: assumes the shared user model check (`can_user_call_model`) runs for team-less JWT callers.
    """
    with users_only.proxy.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        subject: Final = scenario.user(models=[allowed])
        assert_model_denied(
            chat_as(users_only.proxy, users_only.idp.token(sub=subject), other), other, "user_model_access_denied"
        )
