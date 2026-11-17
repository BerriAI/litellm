import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.mail import smtp_sink
from integration._support.process import owned_proxy

pytestmark: Final = pytest.mark.shares_worker_state

KEY_BUDGET: Final = 0.10
CALL_COST: Final = 20 * 0.001 + 20 * 0.002


def _key_spend(key: str) -> float:
    rows: Final = read_rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE key_alias = %s', (key,))
    return float(str(rows[0]["spend"])) if rows else 0.0


def test_key_threshold_emails_owner_and_shared_mailbox_from_admin_ui_metadata(gateway: Gateway, tmp_path: Path) -> None:
    owner_email: Final = f"owner-{uuid.uuid4().hex}@integration.test"
    shared_mailbox: Final = f"shared-mailbox-{uuid.uuid4().hex}@integration.test"
    alias: Final = f"alert-key-{uuid.uuid4().hex}"
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["general_settings"]["alerting"] = ["email"]
    configuration["general_settings"]["user_api_key_cache_ttl"] = 1
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
            user_id: Final = scenario.user(user_email=owner_email)
            key: Final = scenario.key(
                key_alias=alias,
                user_id=user_id,
                models=[model],
                max_budget=KEY_BUDGET,
                metadata={"cost_center": "ops", "max_budget_alert_emails": {"50": [shared_mailbox]}},
            )

            first: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "first call"}]},
                key=key,
            )
            assert first.status_code == 200, first.text
            assert float(first.headers["x-litellm-response-cost"]) == pytest.approx(CALL_COST)
            eventually(lambda: _key_spend(alias), lambda spend: spend == pytest.approx(CALL_COST), seconds=70)
            assert mailbox.deliveries() == (), "no threshold is reached before the first call is recorded"

            second: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "second call"}]},
                key=key,
            )
            assert second.status_code == 200, second.text
            halfway: Final = eventually(mailbox.deliveries, lambda found: len(found) >= 2, seconds=30)
            assert {recipient for delivery in halfway for recipient in delivery.recipients} == {
                owner_email,
                shared_mailbox,
            }, halfway
            assert all("50%" in delivery.subject for delivery in halfway), halfway
            assert len(halfway) == 2, halfway
