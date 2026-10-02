from collections.abc import Iterator
from hashlib import sha256
from typing import Final

import httpx
import pytest
from integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment, object_value, string_value
from integration._support.database import read_rows, write_rows
from integration._support.process import owned_proxy
from pydantic import JsonValue


def _project_rows(project_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT p.project_id, p.project_alias, p.description, p.team_id, p.models, p.blocked, '
        'p.budget_id, b.max_budget FROM "LiteLLM_ProjectTable" AS p '
        'LEFT JOIN "LiteLLM_BudgetTable" AS b ON b.budget_id = p.budget_id '
        'WHERE p.project_id = %s',
        (project_id,),
    )


def _key_rows(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT token, key_alias, team_id, project_id FROM "LiteLLM_VerificationToken" WHERE token = %s',
        (sha256(key.encode()).hexdigest(),),
    )


def _discard_unexpected_key(candidate: Gateway, response: httpx.Response) -> None:
    if response.status_code != 200:
        return
    body: Final = JSON_OBJECT.validate_json(response.content)
    candidate.post("/key/delete", {"keys": [string_value(body["key"])]})


@pytest.fixture(scope="module")
def ownership_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with gateway_from_environment() as gateway:
        with owned_proxy(gateway, tmp_path_factory.mktemp("project-team-ownership"), {}, workers=2) as candidate:
            yield candidate


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


def test_key_generate_rejects_foreign_team_project(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        project_b: Final = scenario.project(team_b, models=[model])
        cross_team: Final = ownership_gateway.request(
            "POST",
            "/key/generate",
            {"team_id": team_a, "project_id": project_b, "models": [model]},
        )
        _discard_unexpected_key(ownership_gateway, cross_team)
        assert cross_team.status_code == 400, cross_team.text
        assert read_rows(
            'SELECT token FROM "LiteLLM_VerificationToken" WHERE project_id = %s AND team_id = %s',
            (project_b, team_a),
        ) == []


def test_key_generate_rejects_missing_team_for_owned_project(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_b: Final = scenario.team(models=[model])
        project_b: Final = scenario.project(team_b, models=[model])
        unbound: Final = ownership_gateway.request(
            "POST", "/key/generate", {"project_id": project_b, "models": [model]}
        )
        _discard_unexpected_key(ownership_gateway, unbound)
        assert unbound.status_code == 400, unbound.text
        assert read_rows(
            'SELECT token FROM "LiteLLM_VerificationToken" WHERE project_id = %s AND team_id IS NULL',
            (project_b,),
        ) == []


def test_key_generate_same_team_project_key_can_chat(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model])
        generated: Final = ownership_gateway.request(
            "POST", "/key/generate", {"team_id": team, "project_id": project, "models": [model]}
        )
        assert generated.status_code == 200, generated.text
        key: Final = string_value(JSON_OBJECT.validate_json(generated.content)["key"])
        scenario.cleanups.callback(scenario.delete_key, key)
        generated_rows: Final = _key_rows(key)
        assert len(generated_rows) == 1
        assert generated_rows[0]["team_id"] == team
        assert generated_rows[0]["project_id"] == project
        chat: Final = ownership_gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "control"}]},
            key=key,
        )
        assert chat.status_code == 200, chat.text
        assert object_value(JSON_OBJECT.validate_json(chat.content)["usage"])["total_tokens"] == 40


def test_key_update_rejects_team_change_and_allows_unchanged_values(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        project_a: Final = scenario.project(team_a, models=[model])
        key: Final = scenario.key(team_id=team_a, project_id=project_a, models=[model])
        aliased: Final = ownership_gateway.request("POST", "/key/update", {"key": key, "key_alias": "updated"})
        assert aliased.status_code == 200, aliased.text
        after_alias: Final = _key_rows(key)
        assert len(after_alias) == 1
        assert after_alias[0]["key_alias"] == "updated"
        assert after_alias[0]["team_id"] == team_a
        assert after_alias[0]["project_id"] == project_a
        unchanged_team: Final = ownership_gateway.request("POST", "/key/update", {"key": key, "team_id": team_a})
        assert unchanged_team.status_code == 200, unchanged_team.text
        after_unchanged_team: Final = _key_rows(key)
        assert after_unchanged_team == after_alias
        before_reassignment: Final = _key_rows(key)
        assert len(before_reassignment) == 1
        reassigned: Final = ownership_gateway.request("POST", "/key/update", {"key": key, "team_id": team_b})
        assert reassigned.status_code == 400, reassigned.text
        assert _key_rows(key) == before_reassignment


def test_key_update_can_detach_project_and_change_team(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        project_a: Final = scenario.project(team_a, models=[model])
        key: Final = scenario.key(team_id=team_a, project_id=project_a, models=[model])
        detached: Final = ownership_gateway.request(
            "POST", "/key/update", {"key": key, "project_id": None, "team_id": team_b}
        )
        assert detached.status_code == 200, detached.text
        after_detach: Final = _key_rows(key)
        assert len(after_detach) == 1
        assert after_detach[0]["project_id"] is None
        assert after_detach[0]["team_id"] == team_b
        assert after_detach[0]["key_alias"] is None


def test_key_regenerate_rejects_foreign_project_without_changing_key(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        project_b: Final = scenario.project(team_b, models=[model])
        generated: Final = ownership_gateway.request(
            "POST", "/key/generate", {"team_id": team_a, "models": [model]}
        )
        assert generated.status_code == 200, generated.text
        key: Final = string_value(JSON_OBJECT.validate_json(generated.content)["key"])
        before: Final = _key_rows(key)
        assert len(before) == 1
        response: Final = ownership_gateway.request(
            "POST", f"/key/{key}/regenerate", {"project_id": project_b}
        )
        _discard_unexpected_key(ownership_gateway, response)
        if response.status_code != 200:
            scenario.cleanups.callback(scenario.delete_key, key)
        assert response.status_code == 400, response.text
        assert _key_rows(key) == before


def test_project_update_rejects_moving_project_with_attached_key(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        attached_project: Final = scenario.project(team_a, models=[model])
        key: Final = scenario.key(team_id=team_a, project_id=attached_project, models=[model])
        project_before: Final = _project_rows(attached_project)
        key_before: Final = _key_rows(key)
        moved_with_key: Final = ownership_gateway.request(
            "POST", "/project/update", {"project_id": attached_project, "team_id": team_b}
        )
        assert moved_with_key.status_code == 400, moved_with_key.text
        assert _project_rows(attached_project) == project_before
        assert _key_rows(key) == key_before


def test_project_update_allows_moving_project_without_keys(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        unattached_project: Final = scenario.project(team_a, models=[model])
        moved_without_key: Final = ownership_gateway.request(
            "POST", "/project/update", {"project_id": unattached_project, "team_id": team_b}
        )
        assert moved_without_key.status_code == 200, moved_without_key.text
        moved_project: Final = _project_rows(unattached_project)
        assert len(moved_project) == 1
        assert moved_project[0]["team_id"] == team_b
        assert read_rows(
            'SELECT token FROM "LiteLLM_VerificationToken" WHERE project_id = %s',
            (unattached_project,),
        ) == []


def test_project_update_rejects_moving_project_with_teamless_key(ownership_gateway: Gateway) -> None:
    with ownership_gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        project: Final = scenario.project(team_a, models=[model])
        key: Final = scenario.key(team_id=team_a, project_id=project, models=[model])
        write_rows(
            'UPDATE "LiteLLM_VerificationToken" SET team_id = NULL WHERE token = %s',
            (sha256(key.encode()).hexdigest(),),
        )
        project_before: Final = _project_rows(project)
        key_before: Final = _key_rows(key)
        assert key_before[0]["team_id"] is None
        moved: Final = ownership_gateway.request("POST", "/project/update", {"project_id": project, "team_id": team_b})
        assert moved.status_code == 400, moved.text
        assert _project_rows(project) == project_before
        assert _key_rows(key) == key_before
