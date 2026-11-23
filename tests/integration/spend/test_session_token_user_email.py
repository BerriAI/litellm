import os
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows

from litellm.proxy._types import LiteLLM_UserTable
from litellm.proxy.auth.auth_checks import ExperimentalUIJWTToken


@pytest.mark.parametrize("user_role", ["proxy_admin", "internal_user"])
def test_a_session_token_logs_the_email_on_its_user_row(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch, user_role: str
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", os.environ.get("LITELLM_SALT_KEY", "sk-integration-salt"))
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        email: Final = f"integration-{uuid.uuid4().hex}@example.com"
        user_id: Final = scenario.user(user_role=user_role, user_email=email)
        team_id: Final = scenario.team(models=[model], members_with_roles=[{"role": "user", "user_id": user_id}])
        token: Final = ExperimentalUIJWTToken.get_cli_jwt_auth_token(
            user_info=LiteLLM_UserTable(user_id=user_id, user_role=user_role, teams=[team_id], models=[]),
            team_id=team_id,
            team_alias="cli-team",
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"session email {uuid.uuid4().hex}"}]},
            key=token,
        )
        assert response.status_code == 200, response.text
        logged: Final = eventually(
            lambda: read_rows(
                """SELECT "user", metadata->>'user_api_key_user_email' AS email
                FROM "LiteLLM_SpendLogs" WHERE request_id=%s""",
                (response.json()["id"],),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert [(row["user"], row["email"]) for row in logged] == [(user_id, email)]
