"""Tag and project management API contracts: /tag/new, update, info, list, delete and /project/new, info, list,
update, delete.

Tags and projects move behind a Rust store this week (draft PR #52178, thread D of the DB W3 brief). These tests
pin the management answers read from `litellm/proxy/management_endpoints/tag_management_endpoints.py` and
`litellm/enterprise/litellm_enterprise/proxy/management_endpoints/project_endpoints.py`: the tag response shapes,
the budget row a tag creates, the not found and duplicate answers (several of which are currently swallowed into
500 by a blanket `except Exception`, pinned here as BUG skips), who may write tags, project validation against the
owning team, the project not found and duplicate id messages, and the partial effect of a batch project delete.
Project persistence and key attachment refusals are in `test_project_lifecycle.py`.
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import Gateway, list_value, object_value, string_value
from tests.integration.management.dec1_mgmt_support import (
    PROXY_ADMIN_ONLY_PREFIX,
    body_of,
    budget_rows,
    caller,
    delete_budget_if_present,
    error_text,
    get,
    post,
    project_rows,
    proxy_error,
    tag,
    tag_rows,
    team_admin,
    unique,
)

pytestmark: Final = pytest.mark.shares_worker_state


def test_tag_new_answers_a_message_and_the_tag_config(gateway: Gateway) -> None:
    """/tag/new answers {"message": "Tag <name> created successfully", "tag": {...}} where the tag object carries
    name, description, models [] and model_info {}, and the LiteLLM_TagTable row has the description."""
    with gateway.scenario() as scenario:
        name: Final = unique("tag")
        response: Final = post(gateway, "/tag/new", {"name": name, "description": "for tests"})
        scenario.cleanups.callback(gateway.post, "/tag/delete", {"name": name})
        assert response.status_code == 200, response.text
        body: Final = body_of(response)
        assert body["message"] == f"Tag {name} created successfully"
        config: Final = object_value(body["tag"])
        assert (config["name"], config["description"], config["models"], config["model_info"]) == (
            name,
            "for tests",
            [],
            {},
        )
        assert tag_rows(name)[0]["description"] == "for tests"


def test_tag_new_with_max_budget_creates_and_links_a_budget_row(gateway: Gateway) -> None:
    """A tag created with max_budget 2.0 gets a new budget row with that max_budget, linked through the tag's
    budget_id, and /tag/info nests it as litellm_budget_table."""
    with gateway.scenario() as scenario:
        name: Final = tag(scenario, max_budget=2.0)
        budget_id: Final = tag_rows(name)[0]["budget_id"]
        assert isinstance(budget_id, str)
        scenario.cleanups.callback(delete_budget_if_present, gateway, budget_id)
        assert budget_rows(budget_id)[0]["max_budget"] == 2.0
        info: Final = gateway.post("/tag/info", {"names": [name]})
        assert object_value(object_value(info[name])["litellm_budget_table"])["max_budget"] == 2.0


def test_tag_new_records_the_caller_as_created_by(gateway: Gateway) -> None:
    """A proxy admin user's key creates a tag; the row's created_by and the response tag's created_by are that
    user's id."""
    with gateway.scenario() as scenario:
        admin: Final = caller(scenario, "proxy_admin")
        name: Final = unique("tag")
        response: Final = post(gateway, "/tag/new", {"name": name}, key=admin.key)
        scenario.cleanups.callback(gateway.post, "/tag/delete", {"name": name})
        assert response.status_code == 200, response.text
        assert object_value(body_of(response)["tag"])["created_by"] == admin.user_id
        assert tag_rows(name)[0]["created_by"] == admin.user_id


def test_tag_new_with_a_taken_name_is_400(gateway: Gateway) -> None:
    """A second /tag/new with an existing name should answer 400 "Tag <name> already exists". Today the handler's
    blanket `except Exception` turns that HTTPException into a 500 whose detail is "400: Tag <name> already exists"."""
    pytest.skip("BUG: duplicate /tag/new answers 500 with detail '400: Tag <name> already exists'")
    with gateway.scenario() as scenario:
        name: Final = tag(scenario)
        response: Final = post(gateway, "/tag/new", {"name": name})
        assert response.status_code == 400, response.text
        assert body_of(response)["detail"] == f"Tag {name} already exists"


def test_duplicate_tag_new_never_writes_a_second_row(gateway: Gateway) -> None:
    """Whatever status a duplicate /tag/new answers today, it must fail and leave exactly one LiteLLM_TagTable row
    with the first description."""
    with gateway.scenario() as scenario:
        name: Final = tag(scenario, description="first")
        response: Final = post(gateway, "/tag/new", {"name": name, "description": "second"})
        assert response.status_code >= 400, response.text
        assert f"Tag {name} already exists" in error_text(response)
        assert [row["description"] for row in tag_rows(name)] == ["first"]


@pytest.mark.parametrize(
    ("route", "body_for", "message_for"),
    [
        pytest.param(
            "/tag/update",
            lambda name: {"name": name, "description": "x"},
            lambda name: f"Tag {name} not found",
            id="update",
        ),
        pytest.param("/tag/delete", lambda name: {"name": name}, lambda name: f"Tag {name} not found", id="delete"),
        pytest.param(
            "/tag/info", lambda name: {"names": [name]}, lambda name: f"Tags not found: ['{name}']", id="info"
        ),
    ],
)
def test_tag_routes_answer_404_for_an_unknown_tag(gateway: Gateway, route: str, body_for, message_for) -> None:  # noqa: ANN001  # lambdas from the parametrize table
    """update, delete and info on a tag name with no row should answer 404 with detail "Tag <name> not found" (info:
    "Tags not found: ['<name>']"). Today the blanket `except Exception` in each handler rewraps that into a 500
    whose detail is "404: <message>"."""
    pytest.skip("BUG: /tag/update, /tag/delete and /tag/info answer 500 '404: ...' for an unknown tag instead of 404")
    name: Final = unique("ghost-tag")
    response: Final = post(gateway, route, body_for(name))
    assert response.status_code == 404, response.text
    assert body_of(response)["detail"] == message_for(name)


def test_tag_update_changes_description_and_keeps_the_budget_link(gateway: Gateway) -> None:
    """Updating only a tag's description must change the description and leave its budget_id unchanged."""
    with gateway.scenario() as scenario:
        name: Final = tag(scenario, max_budget=1.0, description="old")
        budget_id: Final = tag_rows(name)[0]["budget_id"]
        assert isinstance(budget_id, str)
        scenario.cleanups.callback(delete_budget_if_present, gateway, budget_id)
        response: Final = post(gateway, "/tag/update", {"name": name, "description": "new"})
        assert response.status_code == 200, response.text
        row: Final = tag_rows(name)[0]
        assert row["description"] == "new"
        assert row["budget_id"] == budget_id


def test_tag_delete_answers_a_message_and_keeps_the_budget_row(gateway: Gateway) -> None:
    """/tag/delete answers {"message": "Tag <name> deleted successfully"}, removes the tag row, and leaves its budget
    row in LiteLLM_BudgetTable, since the handler only deletes from LiteLLM_TagTable."""
    with gateway.scenario() as scenario:
        name: Final = unique("tag")
        gateway.post("/tag/new", {"name": name, "max_budget": 1.0})
        budget_id: Final = tag_rows(name)[0]["budget_id"]
        assert isinstance(budget_id, str)
        scenario.cleanups.callback(delete_budget_if_present, gateway, budget_id)
        response: Final = post(gateway, "/tag/delete", {"name": name})
        assert response.status_code == 200, response.text
        assert body_of(response) == {"message": f"Tag {name} deleted successfully"}
        assert tag_rows(name) == []
        assert len(budget_rows(budget_id)) == 1


def test_tag_info_is_keyed_by_tag_name(gateway: Gateway) -> None:
    """/tag/info for two tags answers an object keyed by tag name, each value carrying name, description, models,
    model_info, created_at, updated_at and created_by."""
    with gateway.scenario() as scenario:
        first: Final = tag(scenario, description="one")
        second: Final = tag(scenario, description="two")
        info: Final = gateway.post("/tag/info", {"names": [first, second]})
        assert set(info) == {first, second}
        entry: Final = object_value(info[first])
        assert {"name", "description", "models", "model_info", "created_at", "updated_at", "created_by"} <= set(entry)
        assert entry["description"] == "one"


@pytest.mark.parametrize("route", ["/tag/new", "/tag/update", "/tag/delete"])
def test_tag_writes_are_401_for_an_internal_user(gateway: Gateway, route: str) -> None:
    """Internal users may list tags but not write them: /tag/new, /tag/update and /tag/delete are in no internal
    user route list, so the route gate answers 401 naming the route, and an existing tag is unchanged."""
    with gateway.scenario() as scenario:
        internal: Final = caller(scenario, "internal_user")
        name: Final = tag(scenario, description="kept")
        response: Final = post(gateway, route, {"name": name, "description": "changed"}, key=internal.key)
        assert response.status_code == 401, response.text
        assert str(proxy_error(response)["message"]).startswith(f"{PROXY_ADMIN_ONLY_PREFIX}Route={route}.")
        assert tag_rows(name)[0]["description"] == "kept"


def test_tag_list_is_open_to_an_internal_user(gateway: Gateway) -> None:
    """/tag/list is an internal user route, so an internal user's key gets 200 rather than the route gate 401.
    UNVERIFIED: the list is scoped to tags produced by the user's own keys, so its contents are not pinned."""
    with gateway.scenario() as scenario:
        internal: Final = caller(scenario, "internal_user")
        response: Final = get(gateway, "/tag/list", key=internal.key)
        assert response.status_code == 200, response.text


def test_project_new_for_an_unknown_team_is_404_naming_the_team(gateway: Gateway) -> None:
    """Creating a project for a team id with no row answers 404 with message "Team not found, team_id=<id>", type
    not_found and param team_id."""
    team_id: Final = unique("ghost-team")
    response: Final = post(gateway, "/project/new", {"team_id": team_id, "project_alias": unique("p")})
    assert response.status_code == 404, response.text
    error: Final = proxy_error(response)
    assert error["message"] == f"Team not found, team_id={team_id}"
    assert error["param"] == "team_id"


def test_project_new_with_a_taken_project_id_is_400(gateway: Gateway) -> None:
    """Creating a project with an existing project_id answers 400 "Project id = <id> already exists. Please use a
    different project id." with param project_id, and the original project keeps its team."""
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        project_id: Final = scenario.project(team_id)
        other_team: Final = scenario.team()
        response: Final = post(gateway, "/project/new", {"team_id": other_team, "project_id": project_id})
        assert response.status_code == 400, response.text
        assert proxy_error(response)["message"] == (
            f"Project id = {project_id} already exists. Please use a different project id."
        )
        assert project_rows(project_id)[0]["team_id"] == team_id


@pytest.mark.parametrize(
    ("team_fields", "project_fields", "message"),
    [
        pytest.param({}, {"max_budget": -1.0}, "max_budget cannot be negative. Received: -1.0", id="negative_budget"),
        pytest.param(
            {},
            {"max_budget": 5.0, "soft_budget": 5.0},
            "soft_budget (5.0) must be strictly lower than max_budget (5.0)",
            id="soft_not_below_max",
        ),
        pytest.param(
            {"max_budget": 10.0},
            {"max_budget": 20.0},
            "Project max_budget (20.0) exceeds team's max_budget (10.0).",
            id="over_team_budget",
        ),
        pytest.param(
            {"tpm_limit": 100},
            {"tpm_limit": 200},
            "Project tpm_limit (200) exceeds team's tpm_limit (100).",
            id="over_team_tpm",
        ),
        pytest.param(
            {"models": ["m-allowed"]},
            {"models": ["m-other"]},
            "Model 'm-other' not in team's allowed models.",
            id="model_outside_team",
        ),
    ],
)
def test_project_new_is_400_when_it_exceeds_its_team(
    gateway: Gateway, team_fields: dict[str, object], project_fields: dict[str, object], message: str
) -> None:
    """A project may not be looser than its team: a negative budget, soft_budget not below max_budget, a budget or
    tpm limit above the team's, or a model outside the team's list is refused with 400 and the exact message
    from `_check_team_project_limits`, and no project row is written for the alias."""
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team(**team_fields)  # pyright: ignore[reportArgumentType]  # JsonValue kwargs
        alias: Final = unique("loose")
        response: Final = post(gateway, "/project/new", {"team_id": team_id, "project_alias": alias, **project_fields})  # pyright: ignore[reportArgumentType]  # JsonValue body
        assert response.status_code == 400, response.text
        assert message in error_text(response)
        assert gateway.rows('SELECT project_id FROM "LiteLLM_ProjectTable" WHERE project_alias = %s', (alias,)) == []


def test_project_new_by_a_team_admin_without_the_projects_grant_is_403(gateway: Gateway) -> None:
    """A team admin may create projects only when team_admin_editable_team_fields grants "projects". Without the
    grant the answer is 403 "Only admins or team admins can create projects. Your role is <role>". UNVERIFIED: the
    suite proxy's default grant list does not include projects."""
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        admin: Final = team_admin(scenario, team_id)
        alias: Final = unique("denied")
        response: Final = post(gateway, "/project/new", {"team_id": team_id, "project_alias": alias}, key=admin.key)
        assert response.status_code == 403, response.text
        assert "Only admins or team admins can create projects. Your role is" in error_text(response)


def test_project_info_for_an_unknown_project_is_404(gateway: Gateway) -> None:
    """/project/info for an id with no row answers 404 "Project not found, project_id=<id>" with param project_id."""
    project_id: Final = unique("ghost-project")
    response: Final = get(gateway, "/project/info", {"project_id": project_id})
    assert response.status_code == 404, response.text
    error: Final = proxy_error(response)
    assert error["message"] == f"Project not found, project_id={project_id}"
    assert error["param"] == "project_id"


def test_project_info_is_403_for_a_user_outside_the_team(gateway: Gateway) -> None:
    """/project/info is an internal user route, but an internal user who is not on the project's team gets 403 with
    "You don't have access to this project"; a member of that team gets 200."""
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        project_id: Final = scenario.project(team_id)
        outsider: Final = caller(scenario, "internal_user")
        refused: Final = get(gateway, "/project/info", {"project_id": project_id}, key=outsider.key)
        assert refused.status_code == 403, refused.text
        assert "You don't have access to this project" in error_text(refused)
        member_id: Final = scenario.member(team_id)
        member_key: Final = scenario.key(user_id=member_id)
        allowed: Final = get(gateway, "/project/info", {"project_id": project_id}, key=member_key)
        assert allowed.status_code == 200, allowed.text
        assert body_of(allowed)["project_id"] == project_id


def test_project_list_for_a_member_holds_only_their_teams_projects(gateway: Gateway) -> None:
    """An internal user on team A lists projects and sees team A's project but not team B's."""
    with gateway.scenario() as scenario:
        mine: Final = scenario.team()
        theirs: Final = scenario.team()
        my_project: Final = scenario.project(mine)
        their_project: Final = scenario.project(theirs)
        member_id: Final = scenario.member(mine)
        member_key: Final = scenario.key(user_id=member_id)
        response: Final = get(gateway, "/project/list", key=member_key)
        assert response.status_code == 200, response.text
        listed: Final = {object_value(row)["project_id"] for row in list_value(response.json())}
        assert my_project in listed
        assert their_project not in listed


def test_project_update_without_project_id_is_422(gateway: Gateway) -> None:
    """/project/update without project_id is refused by the request model, where `project_id: str` is required
    (litellm/proxy/_types.py:3517-3520), so FastAPI answers 422 with one `missing` error on `body.project_id`
    before the handler runs."""
    response: Final = post(gateway, "/project/update", {"description": "nameless"})
    assert response.status_code == 422, response.text
    assert [(item["type"], item["loc"]) for item in response.json()["detail"]] == [("missing", ["body", "project_id"])]


def test_project_delete_is_proxy_admin_only(gateway: Gateway) -> None:
    """Even a team admin with project rights may not delete a project: /project/delete answers 403 "Only admins can
    delete projects" (require_admin) for a team admin of the project's team, and the row stays."""
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        project_id: Final = scenario.project(team_id)
        admin: Final = team_admin(scenario, team_id)
        response: Final = gateway.request("DELETE", "/project/delete", {"project_ids": [project_id]}, key=admin.key)
        assert response.status_code in (401, 403), response.text
        assert len(project_rows(project_id)) == 1


def test_project_batch_delete_stops_at_the_first_missing_id_after_deleting_earlier_ones(gateway: Gateway) -> None:
    """/project/delete walks the ids in order without a transaction: with [existing, missing] the existing project
    is deleted, then the missing one answers 404 "Project not found, project_id=<missing>". Unlike /team/delete,
    which deletes nothing when one id is missing, the earlier project is gone afterwards."""
    with gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        created: Final = gateway.post("/project/new", {"team_id": team_id, "project_alias": unique("p")})
        project_id: Final = string_value(created["project_id"])
        missing: Final = unique("ghost-project")
        response: Final = gateway.request("DELETE", "/project/delete", {"project_ids": [project_id, missing]})
        assert response.status_code == 404, response.text
        assert proxy_error(response)["message"] == f"Project not found, project_id={missing}"
        assert project_rows(project_id) == []
