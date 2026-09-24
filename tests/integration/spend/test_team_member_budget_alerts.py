import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.mail import smtp_sink
from integration._support.process import owned_proxy

MEMBER_BUDGET: Final = 0.10
CALL_COST: Final = 20 * 0.001 + 20 * 0.002


def _membership_spend(user_id: str, team_id: str) -> float:
    rows: Final = read_rows(
        'SELECT spend FROM "LiteLLM_TeamMembership" WHERE user_id = %s AND team_id = %s', (user_id, team_id)
    )
    return float(str(rows[0]["spend"])) if rows else 0.0


def test_team_member_budget_thresholds_email_member_and_configured_recipients(gateway: Gateway, tmp_path: Path) -> None:
    member_email: Final = f"member-{uuid.uuid4().hex}@integration.test"
    finance_email: Final = f"finance-{uuid.uuid4().hex}@integration.test"
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["general_settings"]["alerting"] = ["email"]
    path: Final = tmp_path / "email-alerting.yaml"
    path.write_text(yaml.safe_dump(configuration))
    with smtp_sink() as mailbox:
        overrides: Final = {
            "SMTP_HOST": mailbox.host,
            "SMTP_PORT": str(mailbox.port),
            "SMTP_TLS": "False",
            "SMTP_SENDER_EMAIL": "alerts@integration.test",
        }
        with owned_proxy(gateway, tmp_path, overrides, config=path) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
            user_id: Final = scenario.user(user_email=member_email)
            team_id: Final = scenario.team(
                models=[model],
                team_member_budget=MEMBER_BUDGET,
                metadata={"team_member_max_budget_alert_emails": {"50": [], "100": [finance_email]}},
            )
            candidate.post("/team/member_add", {"team_id": team_id, "member": {"user_id": user_id, "role": "user"}})
            key: Final = scenario.key(team_id=team_id, user_id=user_id)

            first: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "first call"}]},
                key=key,
            )
            assert first.status_code == 200, first.text
            assert float(first.headers["x-litellm-response-cost"]) == pytest.approx(CALL_COST)
            eventually(
                lambda: _membership_spend(user_id, team_id), lambda spend: spend == pytest.approx(CALL_COST), seconds=70
            )
            assert mailbox.drain() == (), "no threshold is reached before the first call is recorded"

            second: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "second call"}]},
                key=key,
            )
            assert second.status_code == 200, second.text
            eventually(mailbox.pending, lambda count: count >= 1, seconds=30)
            halfway: Final = mailbox.drain()
            assert [delivery.recipients for delivery in halfway] == [(member_email,)], halfway
            assert "50%" in halfway[0].subject, halfway[0].subject
            assert f"${MEMBER_BUDGET}" in halfway[0].html, halfway[0].html
            eventually(
                lambda: _membership_spend(user_id, team_id),
                lambda spend: spend == pytest.approx(2 * CALL_COST),
                seconds=70,
            )

            third: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "third call"}]},
                key=key,
            )
            assert third.status_code == 422 and third.json()["error"]["type"] == "budget_exceeded", third.text
            eventually(mailbox.pending, lambda count: count >= 2, seconds=30)
            hundred: Final = mailbox.drain()
            assert all("100%" in delivery.subject for delivery in hundred), hundred
            assert {recipient for delivery in hundred for recipient in delivery.recipients} == {
                member_email,
                finance_email,
            }, hundred
            assert all(member_email in delivery.html and f"${MEMBER_BUDGET}" in delivery.html for delivery in hundred)
            assert len(hundred) == 2, hundred
