"""Organization, project and team boundaries on model lists, at write time and at request time.

Contract:

* An organization's `models` list is enforced when a team inside it is created or updated: every team model
  must be named in the organization list unless the organization holds `all-proxy-models`
  (litellm/proxy/management_endpoints/team_endpoints.py `_check_org_team_limits`). It is not re-checked per
  request (no request-time caller of `can_org_access_model`).
* A project's `models` list is enforced twice: a key created in the project may only name project models (or
  the inherited sentinels), and every request of a project key is checked against the project list with the
  `project_model_access_denied` type (key_management_endpoints.py `_check_project_key_limits`,
  auth_checks.py `can_project_access_model`).
* Moving a key to another team requires the new team to allow each of the key's models
  (`validate_key_team_change`), except `all-team-models`.
* Empty lists on a team or user mean "no restriction from this layer".

These rules sit between the Rust auth object stores (org, team, project) and the request path, so a store
port that loads a list but forgets where it is enforced shows up here.
"""

from __future__ import annotations

from typing import Final

from tests.integration._support.client import Gateway, eventually, object_value
from tests.integration.authorization.dec1_modelaccess_support import (
    KEY_DENIED,
    TEAM_DENIED,
    assert_model_denied,
    assert_served,
    body_of,
    chat,
    unique_name,
)

PROJECT_DENIED: Final = "project_model_access_denied"


def test_updating_an_org_team_with_a_model_outside_the_org_list_is_refused(gateway: Gateway) -> None:
    """An organization allows A; a team is created in it with `models: [A]` and then updated to `[A, B]`.

    The update must answer 400 and name the model with the exact text
    `Model '<B>' not in organization's allowed models. Organization allowed models=['<A>']. Organization:
    <org id>`. UNVERIFIED: only the status and message text are asserted, since the envelope depends on how
    `/team/update` re-raises the `HTTPException`.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        organization: Final = scenario.organization(models=[first])
        team: Final = scenario.team(models=[first], organization_id=organization)
        response: Final = gateway.request("POST", "/team/update", {"team_id": team, "models": [first, second]})
        assert response.status_code == 400, response.text
        expected: Final = (
            f"Model '{second}' not in organization's allowed models. "
            f"Organization allowed models={[first]!r}. Organization: {organization}"
        )
        assert expected in response.text, response.text


def test_an_org_with_all_proxy_models_accepts_any_team_model(gateway: Gateway) -> None:
    """An organization has `models: ["all-proxy-models"]`; a team is created in it with an arbitrary model.

    Creation must succeed and the team's key must be served on that model: the sentinel skips the
    organization subset check.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        organization: Final = scenario.organization(models=["all-proxy-models"])
        team: Final = scenario.team(models=[model], organization_id=organization)
        assert_served(chat(gateway, scenario.key(team_id=team), model))


def test_an_org_list_is_not_checked_on_requests_of_a_key_attached_only_to_the_org(gateway: Gateway) -> None:
    """An organization allows only A; a key is created with `organization_id: <org>`, no team and no list, and
    calls B.

    The call is served with 200 today: the organization list is only a write-time ceiling for its teams.
    Pinned as current behaviour (the keys index lists it as surprising); a Rust auth core that enforces the
    org list per request would change this to a 403 `org_model_access_denied`.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        organization: Final = scenario.organization(models=[first])
        key: Final = scenario.key(organization_id=organization)
        assert_served(chat(gateway, key, second))


def test_a_project_key_is_refused_a_model_outside_the_project_list(gateway: Gateway) -> None:
    """A team allows A and B; a project in that team allows only A; a key is created in the project with no
    list of its own and calls B.

    The answer must be 403 with exactly the OpenAI denial body of type `project_model_access_denied`, and A
    must be served.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        team: Final = scenario.team(models=[first, second])
        project: Final = scenario.project(team, models=[first])
        key: Final = scenario.key(team_id=team, project_id=project)
        assert_served(chat(gateway, key, first))
        assert_model_denied(chat(gateway, key, second), second, PROJECT_DENIED)


def test_creating_a_project_key_naming_a_model_outside_the_project_is_refused(gateway: Gateway) -> None:
    """A project allows only A; an admin calls `/key/generate` with `project_id` and `models: [B]`.

    The answer must be 400 and carry the exact text `Model '<B>' not in project's allowed models. Project
    allowed models=['<A>']. Project: <project id>`.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        team: Final = scenario.team(models=[first, second])
        project: Final = scenario.project(team, models=[first])
        response: Final = gateway.request(
            "POST", "/key/generate", {"team_id": team, "project_id": project, "models": [second]}
        )
        assert response.status_code == 400, response.text
        expected: Final = (
            f"Model '{second}' not in project's allowed models. Project allowed models={[first]!r}. Project: {project}"
        )
        assert expected in response.text, response.text


def test_a_project_key_may_name_all_team_models(gateway: Gateway) -> None:
    """A project allows only A; an admin creates a project key with `models: ["all-team-models"]`.

    Creation must succeed (the inherited sentinels are exempt from the subset check) and the key must still
    be refused B by the project with 403 `project_model_access_denied`.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        team: Final = scenario.team(models=[first, second])
        project: Final = scenario.project(team, models=[first])
        key: Final = scenario.key(team_id=team, project_id=project, models=["all-team-models"])
        assert_served(chat(gateway, key, first))
        assert_model_denied(chat(gateway, key, second), second, PROJECT_DENIED)


def test_a_key_denial_is_reported_before_the_project_denial(gateway: Gateway) -> None:
    """A project allows A and B; its key lists only A and calls a third model C that neither allows.

    The answer must be 403 `key_model_access_denied`, not the project type: the key list is checked first.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        third: Final = scenario.model()
        team: Final = scenario.team(models=[first, second, third])
        project: Final = scenario.project(team, models=[first, second])
        key: Final = scenario.key(team_id=team, project_id=project, models=[first])
        assert_model_denied(chat(gateway, key, third), third, KEY_DENIED)


def test_moving_a_key_to_a_team_that_lacks_its_model_is_refused(gateway: Gateway) -> None:
    """A key in team one lists model A; an admin calls `/key/update` with `team_id` of team two, which allows
    only B.

    The update must be refused (not 200) with `team_model_access_denied` in the body, and the key row must
    keep team one. UNVERIFIED: the status is assumed to be 403 from the `ProxyException` raised by
    `can_team_access_model` inside `validate_key_team_change`; only "not 200" and the type are asserted.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        team_one: Final = scenario.team(models=[first])
        team_two: Final = scenario.team(models=[second])
        key: Final = scenario.key(team_id=team_one, models=[first])
        response: Final = gateway.request("POST", "/key/update", {"key": key, "team_id": team_two})
        assert response.status_code != 200, response.text
        assert "team_model_access_denied" in response.text, response.text
        info: Final = object_value(gateway.get("/key/info", {"key": key})["info"])
        assert info["team_id"] == team_one, info


def test_moving_an_all_team_models_key_to_another_team_is_allowed(gateway: Gateway) -> None:
    """A key in team one has `models: ["all-team-models"]`; an admin moves it to team two, which allows only B.

    The update must answer 200 and the key must then be served on B, since the sentinel follows whichever
    team the key is in.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        team_one: Final = scenario.team(models=[first])
        team_two: Final = scenario.team(models=[second])
        key: Final = scenario.key(team_id=team_one, models=["all-team-models"])
        gateway.post("/key/update", {"key": key, "team_id": team_two})
        assert_served(eventually(lambda: chat(gateway, key, second), lambda response: response.status_code == 200))


def test_an_empty_team_list_does_not_restrict_its_keys(gateway: Gateway) -> None:
    """A team is created with `models: []` and its key has no list; the key calls a freshly added model.

    The call must be served: an empty team list means the team adds no restriction.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[])
        assert_served(chat(gateway, scenario.key(team_id=team), model))


def test_an_empty_user_list_does_not_restrict_a_personal_key(gateway: Gateway) -> None:
    """A user is created with `models: []` and owns a personal key with no list; the key calls a fresh model.

    The call must be served with 200.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(models=[])
        assert_served(chat(gateway, scenario.key(user_id=user), model))


def test_a_team_member_allowed_models_entry_may_be_a_wildcard(gateway: Gateway) -> None:
    """A team allows `<stem>-*`; a member's `allowed_models` is set to `["<stem>-a*"]`; deployments
    `<stem>-a1` and `<stem>-b1` exist and the member's team key calls both.

    `<stem>-a1` must be served and `<stem>-b1` refused with 403 `team_model_access_denied`. UNVERIFIED: the
    member scope is assumed to reuse `can_object_call_model`, which applies pattern matching.
    """
    with gateway.scenario() as scenario:
        stem: Final = unique_name()
        first: Final = scenario.model(name=f"{stem}-a1")
        second: Final = scenario.model(name=f"{stem}-b1")
        team: Final = scenario.team(models=[f"{stem}-*"])
        member: Final = scenario.member(team)
        gateway.post("/team/member_update", {"team_id": team, "user_id": member, "allowed_models": [f"{stem}-a*"]})
        key: Final = scenario.key(team_id=team, user_id=member)
        assert_served(chat(gateway, key, first))
        refused: Final = eventually(lambda: chat(gateway, key, second), lambda response: response.status_code == 403)
        assert_model_denied(refused, second, TEAM_DENIED)


def test_team_info_reports_the_models_list_it_enforces(gateway: Gateway) -> None:
    """A team is created with `models: [B, A, "<stem>-*"]` and an admin reads `GET /team/info`.

    `team_info.models` must be exactly `[B, A, "<stem>-*"]` in the same order: what the store returns is what
    the request path enforces and what `/v1/models` lists from.
    """
    with gateway.scenario() as scenario:
        first: Final = scenario.model()
        second: Final = scenario.model()
        pattern: Final = f"{unique_name()}-*"
        team: Final = scenario.team(models=[second, first, pattern])
        info: Final = body_of(gateway.request("GET", "/team/info", params={"team_id": team}))
        assert object_value(info["team_info"])["models"] == [second, first, pattern], info
