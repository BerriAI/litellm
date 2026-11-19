import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.mail import smtp_sink
from integration._support.process import OwnedProxy, owned_proxy

pytestmark: Final = pytest.mark.shares_worker_state

TEAM_BUDGET: Final = 0.14
SMALL_TEAM_BUDGET: Final = 0.10
CALL_COST: Final = 20 * 0.001 + 20 * 0.002


def _team_spend(team_id: str) -> float:
    rows: Final = read_rows('SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team_id,))
    return float(str(rows[0]["spend"])) if rows else 0.0


def _call(candidate: OwnedProxy, model: str, key: str) -> int:
    response: Final = candidate.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"call {uuid.uuid4().hex}"}]},
        key=key,
    )
    return response.status_code


def _spend_reaches(team_id: str, calls: int) -> None:
    eventually(lambda: _team_spend(team_id), lambda spend: spend == pytest.approx(calls * CALL_COST), seconds=70)


@pytest.mark.parametrize("alerting", [("email",), ("webhook", "email")])
def test_team_budget_thresholds_email_configured_recipients_once_across_rotating_keys(
    gateway: Gateway, tmp_path: Path, alerting: tuple[str, ...]
) -> None:
    owner_email: Final = f"owner-{uuid.uuid4().hex}@integration.test"
    recipients: Final = {pct: f"p{pct}-{uuid.uuid4().hex}@integration.test" for pct in ("50", "75", "90")}
    other_recipient: Final = f"other-{uuid.uuid4().hex}@integration.test"
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["general_settings"]["alerting"] = list(alerting)
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
            owner: Final = scenario.user(user_email=owner_email)
            alias: Final = f"shared-budget-{uuid.uuid4().hex}"
            team_id: Final = scenario.team(
                team_alias=alias,
                models=[model],
                max_budget=TEAM_BUDGET,
                metadata={"max_budget_alert_emails": {pct: [email] for pct, email in recipients.items()}},
            )
            first_key: Final = scenario.key(team_id=team_id, user_id=owner)
            rotated_key: Final = scenario.key(team_id=team_id, user_id=owner)

            assert _call(candidate, model, first_key) == 200
            _spend_reaches(team_id, 1)
            assert _call(candidate, model, first_key) == 200
            _spend_reaches(team_id, 2)
            assert mailbox.deliveries() == (), "no threshold is reached before the second call is recorded"

            assert _call(candidate, model, rotated_key) == 200
            crossed: Final = eventually(mailbox.deliveries, lambda found: len(found) >= 2, seconds=30)
            assert sorted(delivery.recipients for delivery in crossed) == [
                (recipients["50"],),
                (recipients["75"],),
            ], crossed
            _spend_reaches(team_id, 3)

            assert _call(candidate, model, first_key) == 422
            capped: Final = eventually(mailbox.deliveries, lambda found: len(found) >= 3, seconds=30)
            assert _call(candidate, model, rotated_key) == 422
            assert {delivery.recipients for delivery in capped} == {(email,) for email in recipients.values()}, capped
            assert all(alias in delivery.html and "Team Budget Alert" in delivery.subject for delivery in capped)
            assert all(owner_email not in delivery.html and first_key not in delivery.html for delivery in capped)
            assert len(mailbox.deliveries()) == 3, "each threshold is emailed once however many keys hit the team"

            other_team: Final = _small_team(scenario, model, other_recipient)
            other_key: Final = scenario.key(team_id=other_team, user_id=owner)
            assert _call(candidate, model, other_key) == 200
            _spend_reaches(other_team, 1)
            assert _call(candidate, model, other_key) == 200
            other: Final = eventually(mailbox.deliveries, lambda found: len(found) >= 4, seconds=30)
            assert [delivery.recipients for delivery in other[3:]] == [(other_recipient,)], other


def _small_team(scenario: Scenario, model: str, recipient: str) -> str:
    return scenario.team(
        models=[model], max_budget=SMALL_TEAM_BUDGET, metadata={"max_budget_alert_emails": {"50": [recipient]}}
    )
