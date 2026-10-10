import os
import uuid
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually, string_value
from integration._support.database import read_rows

from litellm.proxy._types import LiteLLM_UserTable
from litellm.proxy.auth.auth_checks import ExperimentalUIJWTToken


def _cli_session_token(user_id: str, team_id: str) -> str:
    cli_user: Final = LiteLLM_UserTable(user_id=user_id, user_role="internal_user", teams=[team_id], models=[])
    return ExperimentalUIJWTToken.get_cli_jwt_auth_token(user_info=cli_user, team_id=team_id, team_alias="cli-team")


@pytest.mark.covers("quota_management.organization_budget.cli_session_token_without_org_id_charges_team_organization")
def test_cli_session_token_without_org_id_charges_and_caps_the_team_organization(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", os.environ.get("LITELLM_SALT_KEY", "sk-integration-salt"))
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        organization: Final = gateway.post(
            "/organization/new", {"organization_alias": f"integration-{uuid.uuid4().hex}", "max_budget": 0.06}
        )
        org_id: Final = string_value(organization["organization_id"])
        scenario.cleanups.callback(
            lambda: gateway.request("DELETE", "/organization/delete", {"organization_ids": [org_id]})
        )
        user_id: Final = scenario.user()
        team_id: Final = scenario.team(
            organization_id=org_id, models=[model], members_with_roles=[{"role": "user", "user_id": user_id}]
        )
        token: Final = _cli_session_token(user_id, team_id)
        prompt: Final = f"org budget {uuid.uuid4().hex}"
        upstream.get("/__observations").raise_for_status()
        first: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": prompt}]},
            key=token,
        )
        assert first.status_code == 200 and first.json()["usage"]["total_tokens"] == 40, first.text
        reached_upstream: Final = upstream.get("/__observations").json()["requests"]
        assert len(reached_upstream) == 1, reached_upstream
        assert reached_upstream[0]["body"]["model"] == "gpt-4o-mini", reached_upstream
        assert reached_upstream[0]["body"]["messages"] == [{"role": "user", "content": prompt}], reached_upstream
        logged: Final = eventually(
            lambda: read_rows(
                'SELECT organization_id, team_id, spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (first.json()["id"],),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert [(row["organization_id"], row["team_id"], float(row["spend"])) for row in logged] == [
            (org_id, team_id, pytest.approx(0.06))
        ]
        charged: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_OrganizationTable" WHERE organization_id=%s', (org_id,)),
            lambda values: len(values) == 1 and float(values[0]["spend"]) >= 0.06,
            seconds=70,
        )
        assert float(charged[0]["spend"]) == pytest.approx(0.06)
        assert float(gateway.get("/organization/info", {"organization_id": org_id})["spend"]) == pytest.approx(0.06)
        denied: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"over org budget {uuid.uuid4().hex}"}]},
            key=token,
        )
        assert denied.status_code == 422 and denied.json()["error"]["type"] == "budget_exceeded", denied.text
        assert f"Organization={org_id}" in denied.json()["error"]["message"], denied.text
        assert upstream.get("/__observations").json()["requests"] == []
