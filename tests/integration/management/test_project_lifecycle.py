from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import Gateway, object_value, string_value
from integration._support.database import read_rows
from pydantic import JsonValue


def _project_rows(project_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT p.project_id, p.project_alias, p.description, p.team_id, p.models, p.blocked, '
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
        budget: Final = scenario.budget(max_budget=7)
        project: Final = scenario.project(
            team, project_alias="new-project", budget_id=budget, models=[model], description="new project"
        )
        key: Final = scenario.key(team_id=team, project_id=project, models=[model])
        assert object_value(gateway.chat(model, key=key)["usage"])["total_tokens"] == 40
        rows: Final = _project_rows(project)
        assert rows != []
        assert len(rows) == 1
        row: Final = rows[0]
        assert row["project_id"] == project
        assert row["project_alias"] == "new-project"
        assert row["team_id"] == team
        assert row["description"] == "new project"
        assert row["models"] == [model]
        assert row["budget_id"] == budget
        assert row["blocked"] is False
        assert row["max_budget"] == 7.0


@pytest.mark.covers("mgmt.project.update.real_route_persists")
def test_project_update_persists_real_state(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        budget: Final = scenario.budget(max_budget=3)
        project: Final = scenario.project(team, budget_id=budget, models=[model], description="before")
        key: Final = scenario.key(team_id=team, project_id=project, models=[model])
        updated: Final = gateway.post(
            "/project/update",
            {
                "project_id": project,
                "project_alias": "updated-project",
                "description": "after",
                "max_budget": 9,
                "blocked": True,
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
        assert row["budget_id"] == budget
        assert row["blocked"] is True
        assert row["max_budget"] == 9.0
        blocked: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "blocked project"}]},
            key=key,
        )
        assert blocked.status_code == 401, blocked.text
        assert object_value(blocked.json()["error"])["type"] == "auth_error"
        gateway.post("/project/update", {"project_id": project, "blocked": False})
        assert object_value(gateway.chat(model, key=key)["usage"])["total_tokens"] == 40


@pytest.mark.covers("mgmt.project.delete.attached_key_refusal_preserves_state")
def test_project_delete_with_attached_key_refuses_and_preserves_state(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        budget: Final = scenario.budget()
        project: Final = scenario.project(
            team, budget_id=budget, project_alias="delete-project", models=[model]
        )
        key: Final = scenario.key(team_id=team, project_id=project, models=[model])
        digest: Final = sha256(key.encode()).hexdigest()
        project_before: Final = _project_rows(project)
        key_before: Final = read_rows(
            'SELECT token, key_alias, models, metadata, max_budget, team_id, project_id, budget_id '
            'FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (digest,),
        )
        assert len(project_before) == 1
        assert len(key_before) == 1
        assert key_before[0]["project_id"] == project
        assert key_before[0]["team_id"] == team
        denied: Final = gateway.request("DELETE", "/project/delete", {"project_ids": [project]})
        assert denied.status_code == 400, denied.text
        assert _project_rows(project) == project_before
        assert read_rows(
            'SELECT token, key_alias, models, metadata, max_budget, team_id, project_id, budget_id '
            'FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (digest,),
        ) == key_before
