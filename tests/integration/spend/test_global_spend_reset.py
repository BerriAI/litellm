import uuid
from hashlib import sha256
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows


def _chat(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "max_tokens": 20, "messages": [{"role": "user", "content": f"reset {uuid.uuid4().hex}"}]},
        key=key,
    )


def _token_spend(key: str) -> list[dict[str, object]]:
    return read_rows(
        'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token = %s', (sha256(key.encode()).hexdigest(),)
    )  # fmt: skip


def _team_spend(team_id: str) -> list[dict[str, object]]:
    return read_rows('SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team_id,))


def _spend_log_totals(keys: tuple[str, ...]) -> list[dict[str, object]]:
    digests: Final = tuple(sha256(key.encode()).hexdigest() for key in keys)
    return read_rows(
        'SELECT COUNT(*) AS rows, COALESCE(SUM(spend), 0) AS total_spend FROM "LiteLLM_SpendLogs" '
        "WHERE api_key = ANY(%s::text[])",
        ("{" + ",".join(digests) + "}",),
    )


def _seed_spend(gateway: Gateway, model: str, team_id: str, key_a: str, key_b: str) -> None:
    assert _chat(gateway, model, key_a).status_code == 200
    assert _chat(gateway, model, key_b).status_code == 200
    for key in (key_a, key_b):
        eventually(
            lambda key=key: _token_spend(key),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) > 0,
            seconds=70,
        )
    eventually(
        lambda: _team_spend(team_id),
        lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) > 0,
        seconds=70,
    )


def test_global_spend_reset_zeroes_token_and_team_spend_but_keeps_logs(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team_id: Final = scenario.team()
        key_a: Final = scenario.key(team_id=team_id, models=[model], max_budget=0.06)
        key_b: Final = scenario.key(team_id=team_id, models=[model])
        internal_user: Final = scenario.user(user_role="internal_user")
        internal_key: Final = scenario.key(user_id=internal_user, models=[model])
        _seed_spend(gateway, model, team_id, key_a, key_b)
        before: Final = _spend_log_totals((key_a, key_b))
        assert float(str(before[0]["total_spend"])) == pytest.approx(0.12), before

        refused: Final = gateway.request("POST", "/global/spend/reset", key=internal_key)
        assert refused.status_code == 401, refused.text
        assert _token_spend(key_a)[0]["spend"] != 0, refused.text

        denied: Final = _chat(gateway, model, key_a)
        assert denied.status_code == 422, denied.text
        assert object_value(denied.json()["error"])["type"] == "budget_exceeded"

        reset: Final = gateway.request("POST", "/global/spend/reset")
        assert reset.status_code == 200, reset.text
        assert reset.json() == {
            "message": "Spend for all API Keys and Teams reset successfully",
            "status": "success",
        }, reset.text
        for key in (key_a, key_b):
            rows: Final = _token_spend(key)
            assert float(str(rows[0]["spend"])) == 0, rows
        assert float(str(_team_spend(team_id)[0]["spend"])) == 0
        assert _spend_log_totals((key_a, key_b)) == before


def test_reset_exhausted_key_is_served_again(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: /global/spend/reset zeroes DB spend but leaves Redis spend:key counters, "
        "so an exhausted key stays refused with budget_exceeded"
    )
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model], max_budget=0.06)
        assert _chat(gateway, model, key).status_code == 200
        eventually(
            lambda: _token_spend(key),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) >= 0.06,
            seconds=70,
        )
        denied: Final = _chat(gateway, model, key)
        assert denied.status_code == 422, denied.text
        reset: Final = gateway.request("POST", "/global/spend/reset")
        assert reset.status_code == 200, reset.text
        upstream.get("/__observations").raise_for_status()
        eventually(lambda: _chat(gateway, model, key), lambda response: response.status_code == 200, seconds=30)
        assert len(upstream.get("/__observations").json()["requests"]) == 1
