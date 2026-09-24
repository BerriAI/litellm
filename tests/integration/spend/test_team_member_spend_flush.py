"""Team member spend keeps landing after a flush in which every cost was a whole number.

The proxy runs on a one-connection pool so every spend flush reuses the same database
connection. A batch of $0 requests (a free model here) is the whole-number batch, and the
fractional batches that follow it must still land on that connection.

The $0 batch has to be flushed on its own before the paid request is sent. The spend log
row cannot prove that, since a separate monitor writes spend logs whenever they queue up,
but the daily user spend row is written by the flush cycle right after the member spend
statement, so its arrival means the whole-number batch has already been sent.
"""

from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from pydantic import JsonValue

SINGLE_CONNECTION_CONFIG: Final = """
model_list: []
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
  store_model_in_db: true
  proxy_batch_write_at: 1
  proxy_batch_polling_interval: 1
  database_connection_pool_limit: 1
router_settings:
  disable_cooldowns: true
"""


def _member_row(team_id: str, user_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT spend, total_spend FROM "LiteLLM_TeamMembership" WHERE team_id=%s AND user_id=%s',
        (team_id, user_id),
    )


def _member_spend_is(rows: list[dict[str, JsonValue]], amount: float) -> bool:
    return len(rows) == 1 and all(
        float(str(rows[0][column])) == pytest.approx(amount) for column in ("spend", "total_spend")
    )


def _daily_user_spend_rows(user_id: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT spend FROM "LiteLLM_DailyUserSpend" WHERE user_id=%s', (user_id,))


def _logged_spend(request_id: str) -> float:
    rows: Final = eventually(
        lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (request_id,)),
        lambda found: len(found) == 1,
        seconds=30,
    )
    return float(str(rows[0]["spend"]))


def _chat(gateway: Gateway, key: str, model: str) -> str:
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": "member spend control"}]},
        key=key,
    )
    assert response.status_code == 200, response.text
    return string_value(response.json()["id"])


def test_fractional_member_spend_lands_after_a_whole_number_flush_on_the_same_connection(
    gateway: Gateway, tmp_path: Path
) -> None:
    config: Final = tmp_path / "single_connection_proxy.yaml"
    config.write_text(SINGLE_CONNECTION_CONFIG)
    with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
        free: Final = scenario.model(input_cost_per_token=0, output_cost_per_token=0, num_retries=0)
        paid: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002, num_retries=0)
        team: Final = scenario.team(models=[free, paid])
        first: Final = scenario.user()
        second: Final = scenario.user()
        candidate.post(
            "/team/member_add",
            {"team_id": team, "member": [{"role": "user", "user_id": first}, {"role": "user", "user_id": second}]},
        )
        first_key: Final = scenario.key(team_id=team, user_id=first)
        second_key: Final = scenario.key(team_id=team, user_id=second)

        _chat(candidate, first_key, free)
        flushed: Final = eventually(lambda: _daily_user_spend_rows(first), lambda rows: len(rows) == 1, seconds=30)
        assert float(str(flushed[0]["spend"])) == 0
        assert _member_spend_is(_member_row(team, first), 0)

        paid_spend: Final = _logged_spend(_chat(candidate, second_key, paid))
        assert paid_spend > 0
        eventually(lambda: _member_row(team, second), lambda rows: _member_spend_is(rows, paid_spend), seconds=30)

        repeat_spend: Final = _logged_spend(_chat(candidate, first_key, paid))
        eventually(lambda: _member_row(team, first), lambda rows: _member_spend_is(rows, repeat_spend), seconds=30)
        eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id=%s', (team,)),
            lambda rows: float(str(rows[0]["spend"])) == pytest.approx(paid_spend + repeat_spend),
            seconds=30,
        )
