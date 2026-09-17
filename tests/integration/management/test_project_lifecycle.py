from hashlib import sha256
from typing import Final

import pytest
from pydantic import JsonValue

from integration._support.client import Gateway, string_value
from integration._support.database import read_rows


def _project_rows(project_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT p.project_id, p.project_alias, p.description, p.team_id, p.models, '
        'p.budget_id, b.max_budget FROM "LiteLLM_ProjectTable" AS p '
        'LEFT JOIN "LiteLLM_BudgetTable" AS b ON b.budget_id = p.budget_id '
        'WHERE p.project_id = %s',
        (project_id,),
    )


@pytest.mark.covers("mgmt.project.new.real_route_persists")
def test_project_new_persists_real_state(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model], description="new project", max_budget=7)
        rows: Final = _project_rows(project)
        assert rows != []
        assert len(rows) == 1
        row: Final = rows[0]
        assert row["project_id"] == project
        assert row["team_id"] == team
        assert row["description"] == "new project"
        assert row["models"] == [model]
        assert row["budget_id"] is not None
        assert row["max_budget"] == 7.0


@pytest.mark.covers("mgmt.project.update.real_route_persists")
def test_project_update_persists_real_state(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model], description="before", max_budget=3)
        updated: Final = gateway.post(
            "/project/update",
            {
                "project_id": project,
                "project_alias": "updated-project",
                "description": "after",
                "max_budget": 9,
            },
        )
        assert string_value(updated["project_id"]) == project
        rows: Final = _project_rows(project)
        assert rows != []
        assert len(rows) == 1
        row: Final = rows[0]
        assert row["project_alias"] == "updated-project"
        assert row["description"] == "after"
        assert row["team_id"] == team
        assert row["models"] == [model]
        assert row["max_budget"] == 9.0


@pytest.mark.covers("mgmt.project.delete.attached_key_refusal_preserves_state")
def test_project_delete_with_attached_key_refuses_and_preserves_state(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        created: Final = gateway.post(
            "/project/new",
            {"team_id": team, "project_alias": "delete-project", "models": [model]},
        )
        project: Final = string_value(created["project_id"])
        created_key: Final = gateway.post(
            "/key/generate",
            {"team_id": team, "project_id": project, "models": [model]},
        )
        key: Final = string_value(created_key["key"])
        digest: Final = sha256(key.encode()).hexdigest()
        project_before: Final = _project_rows(project)
        key_before: Final = read_rows(
            'SELECT project_id, team_id FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (digest,),
        )
        assert project_before != []
        assert key_before != []
        try:
            denied: Final = gateway.request("DELETE", "/project/delete", {"project_ids": [project]})
            assert denied.status_code == 400, denied.text
            assert _project_rows(project) == project_before
            assert read_rows(
                'SELECT project_id, team_id FROM "LiteLLM_VerificationToken" WHERE token = %s',
                (digest,),
            ) == key_before
        finally:
            if read_rows(
                'SELECT token FROM "LiteLLM_VerificationToken" WHERE token = %s',
                (digest,),
            ) != []:
                gateway.post("/key/delete", {"keys": [key]})
            if _project_rows(project) != []:
                cleanup: Final = gateway.request("DELETE", "/project/delete", {"project_ids": [project]})
                assert cleanup.status_code == 200, cleanup.text
