import uuid
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy


@pytest.mark.covers("spend.user_budget.opted_in_team_key_is_denied_once_owner_budget_is_exhausted")
def test_team_key_is_denied_before_provider_once_owner_personal_budget_is_exhausted_when_opted_in(
    gateway: Gateway, tmp_path: Path
) -> None:
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["general_settings"]["apply_user_budget_to_team_keys"] = True
    path: Final = tmp_path / "apply-user-budget-to-team-keys.yaml"
    path.write_text(yaml.safe_dump(configuration))
    with (
        owned_proxy(gateway, tmp_path, {}, config=path) as candidate,
        candidate.scenario() as scenario,
        httpx.Client(base_url=candidate.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        user: Final = scenario.user(max_budget=0.06)
        team: Final = scenario.team(models=[model])
        added: Final = candidate.request(
            "POST", "/team/member_add", {"team_id": team, "member": {"user_id": user, "role": "user"}}
        )
        assert added.status_code == 200, added.text
        key: Final = scenario.key(team_id=team, user_id=user, models=[model])
        first: Final = candidate.chat(model, key=key, text=f"owner budget {uuid.uuid4().hex}")
        assert first["usage"]["total_tokens"] == 40
        spent: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_UserTable" WHERE user_id=%s', (user,)),
            lambda values: len(values) == 1 and float(values[0]["spend"]) >= 0.06,
            seconds=70,
        )
        assert float(spent[0]["spend"]) == pytest.approx(0.06)
        assert read_rows(
            'SELECT max_budget FROM "LiteLLM_VerificationToken" WHERE token=%s', (sha256(key.encode()).hexdigest(),)
        ) == [{"max_budget": None}]
        upstream.get("/__observations").raise_for_status()
        denied: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"over owner budget {uuid.uuid4().hex}"}]},
            key=key,
        )
        assert denied.status_code == 422 and denied.json()["error"]["type"] == "budget_exceeded", denied.text
        assert upstream.get("/__observations").json()["requests"] == []
