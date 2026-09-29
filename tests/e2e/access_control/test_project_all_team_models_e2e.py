"""Live e2e: a project allow-listed to "all-team-models" inherits the team's models.

The Admin UI writes ``all-team-models`` into a project's model list when a user picks
"All Team Models". The contract is that the project then allows exactly what the
parent team allows, so a project-scoped virtual key can call any model the team can
call and is still denied a model outside the team's list. Regression coverage for a
project-scoped key being denied with ``project_model_access_denied`` while naming a
model the team could reach.
"""

from __future__ import annotations

import pytest
from access_control_client import (
    ALL_PROXY_MODELS,
    ALL_TEAM_MODELS,
    PROJECT_MODEL_ACCESS_DENIED_MARKER,
    TEAM_MODEL_ACCESS_DENIED_MARKER,
    AccessControlClient,
)
from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import ChatResponse

pytestmark = pytest.mark.e2e

TEAM_MODEL = "gemini-2.5-flash"
OUTSIDE_MODEL = "gpt-5.5"


def _chat_assert_completion(client: AccessControlClient, key: str, model: str) -> None:
    result = client.chat_status(key, model, f"capital of France? {unique_marker()}")
    assert result.status_code == 200, (
        f"project key must be able to call {model!r}, got {result.status_code}: {result.body[:300]}"
    )
    assert ChatResponse.model_validate_json(result.body).choices, (
        f"200 must carry a real completion, not an error envelope: {result.body[:300]}"
    )


class TestProjectAllTeamModels:
    @pytest.mark.parametrize("team_models", [[], [ALL_PROXY_MODELS]])
    def test_all_team_models_project_calls_team_allowed_model(
        self, client: AccessControlClient, resources: ResourceManager, team_models: list[str]
    ) -> None:
        """A project set to all-team-models on an unrestricted team calls a served model."""
        marker = unique_marker()
        team_id = client.create_team(f"e2e-proj-team-{marker}", models=team_models)
        resources.defer(lambda: client.delete_team(team_id))
        project_id = client.create_project(team_id, f"e2e-proj-{marker}", models=[ALL_TEAM_MODELS])
        resources.defer(lambda: client.delete_project(project_id))
        key = client.project_key(team_id, project_id, models=[ALL_TEAM_MODELS])
        resources.defer(lambda: client.delete_key(key))

        _chat_assert_completion(client, key, TEAM_MODEL)

    def test_all_team_models_project_denied_outside_team_list(
        self, client: AccessControlClient, resources: ResourceManager
    ) -> None:
        """The inherited allowlist is the team's, not the proxy's."""
        marker = unique_marker()
        team_id = client.create_team(f"e2e-proj-team-{marker}", models=[])
        resources.defer(lambda: client.delete_team(team_id))
        project_id = client.create_project(team_id, f"e2e-proj-{marker}", models=[ALL_TEAM_MODELS])
        resources.defer(lambda: client.delete_project(project_id))
        key = client.project_key(team_id, project_id, models=[ALL_TEAM_MODELS])
        resources.defer(lambda: client.delete_key(key))

        client.set_team_models(team_id, f"e2e-proj-team-{marker}", [TEAM_MODEL])

        _chat_assert_completion(client, key, TEAM_MODEL)

        denied = client.chat_status(key, OUTSIDE_MODEL, f"capital of France? {unique_marker()}")
        assert denied.status_code == 403, (
            f"model outside the team's list must be denied 403, got {denied.status_code}: {denied.body[:300]}"
        )
        assert PROJECT_MODEL_ACCESS_DENIED_MARKER not in denied.body, (
            f"the denial must come from the key or team check, not the project check: {denied.body[:300]}"
        )
        assert TEAM_MODEL_ACCESS_DENIED_MARKER in denied.body, (
            f"403 body must be a team model-access denial, got: {denied.body[:300]}"
        )

    def test_project_explicit_model_list_calls_model(
        self, client: AccessControlClient, resources: ResourceManager
    ) -> None:
        """Control: an explicit project model list on the same topology succeeds."""
        marker = unique_marker()
        team_id = client.create_team(f"e2e-proj-team-{marker}", models=[])
        resources.defer(lambda: client.delete_team(team_id))
        project_id = client.create_project(team_id, f"e2e-proj-{marker}", models=[TEAM_MODEL])
        resources.defer(lambda: client.delete_project(project_id))
        key = client.project_key(team_id, project_id, models=[])
        resources.defer(lambda: client.delete_key(key))

        _chat_assert_completion(client, key, TEAM_MODEL)

        denied = client.chat_status(key, OUTSIDE_MODEL, f"capital of France? {unique_marker()}")
        assert denied.status_code == 403 and PROJECT_MODEL_ACCESS_DENIED_MARKER in denied.body, (
            f"a model outside the explicit project list must still be denied, got "
            f"{denied.status_code}: {denied.body[:300]}"
        )
