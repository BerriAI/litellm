import json
import os
import uuid
from collections.abc import Iterable
from hashlib import sha256
from typing import Final

import psycopg
import pytest
from integration._support.client import Gateway, delete_key_if_present, eventually, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from psycopg import sql


def _execute(statements: Iterable[sql.Composable]) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        for statement in statements:
            connection.execute(statement)


def _install_daily_user_rollup_fault(user_id: str) -> str:
    suffix: Final = f"fault-{uuid.uuid4().hex}"
    sequence: Final = sql.Identifier(f"{suffix}_attempts")
    function: Final = sql.Identifier(suffix)
    _execute(
        (
            sql.SQL("CREATE SEQUENCE {}").format(sequence),
            sql.SQL(
                "CREATE FUNCTION {}() RETURNS trigger LANGUAGE plpgsql AS $fault$ "
                "BEGIN PERFORM nextval({}); "
                "RAISE EXCEPTION 'synthetic daily rollup outage' USING ERRCODE = '55P03'; "
                "END $fault$"
            ).format(function, sql.Literal(f"{suffix}_attempts")),
            sql.SQL(
                'CREATE TRIGGER {} BEFORE INSERT ON "LiteLLM_DailyUserSpend" '
                "FOR EACH ROW WHEN (NEW.user_id = {}) EXECUTE FUNCTION {}()"
            ).format(sql.Identifier(suffix), sql.Literal(user_id), function),
        )
    )
    return suffix


def _lift_daily_user_rollup_fault(suffix: str) -> None:
    _execute(
        (
            sql.SQL('DROP TRIGGER IF EXISTS {} ON "LiteLLM_DailyUserSpend"').format(sql.Identifier(suffix)),
            sql.SQL("DROP FUNCTION IF EXISTS {}()").format(sql.Identifier(suffix)),
            sql.SQL("DROP SEQUENCE IF EXISTS {}").format(sql.Identifier(f"{suffix}_attempts")),
        )
    )


def _rollup_attempts(suffix: str) -> int:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        row: Final = connection.execute(
            sql.SQL("SELECT CASE WHEN is_called THEN last_value ELSE 0 END FROM {}").format(
                sql.Identifier(f"{suffix}_attempts")
            )
        ).fetchone()
    assert row is not None
    return int(row[0])


@pytest.mark.covers("spend.daily_rollup.failed_user_commit_is_retried_until_report_and_daily_activity_agree")
def test_failed_daily_user_rollup_commit_is_retried_so_spend_report_and_daily_activity_agree(
    gateway: Gateway,
) -> None:
    def provider(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/chat/completions"
        return Reply(
            body=json.dumps(
                {
                    "id": "chatcmpl-" + uuid.uuid4().hex,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "synthetic rollup answer"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 20, "total_tokens": 40},
                }
            ).encode()
        )

    with wire_server(provider) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            api_base=wire.url + "/v1", input_cost_per_token=0.001, output_cost_per_token=0.002, num_retries=0
        )
        user: Final = scenario.user()
        key: Final = string_value(gateway.post("/key/generate", {"user_id": user, "models": [model]})["key"])
        scenario.cleanups.callback(delete_key_if_present, gateway, key)
        digest: Final = sha256(key.encode()).hexdigest()
        suffix: Final = _install_daily_user_rollup_fault(user)
        scenario.cleanups.callback(_lift_daily_user_rollup_fault, suffix)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "rollup retry control"}]},
            key=key,
        )
        assert response.status_code == 200, response.text
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(20 * 0.001 + 20 * 0.002)
        body: Final = response.json()
        spend_rows: Final = eventually(
            lambda: read_rows(
                'SELECT spend, DATE("startTime")::text AS day, model FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (body["id"],),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert float(spend_rows[0]["spend"]) == pytest.approx(0.06)
        day: Final = string_value(spend_rows[0]["day"])
        stored_model: Final = string_value(spend_rows[0]["model"])
        eventually(lambda: _rollup_attempts(suffix), lambda attempts: attempts >= 1, seconds=70)
        _lift_daily_user_rollup_fault(suffix)
        activity: Final = eventually(
            lambda: gateway.request(
                "GET",
                "/user/daily/activity/aggregated",
                params={"start_date": day, "end_date": day, "api_key": digest},
            ),
            lambda polled: (
                polled.status_code == 200
                and len(polled.json().get("results", ())) > 0
                and polled.json()["results"][0]["breakdown"]["api_keys"]
                .get(digest, {})
                .get("metrics", {})
                .get("spend", 0)
                == pytest.approx(0.06)
            ),
            seconds=90,
        )
        assert activity.status_code == 200, activity.text
        metrics: Final = activity.json()["results"][0]["breakdown"]["api_keys"][digest]["metrics"]
        assert metrics == {
            "spend": pytest.approx(0.06),
            "flat_cost": pytest.approx(0.0),
            "prompt_tokens": 20,
            "completion_tokens": 20,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "compression_saved_tokens": 0,
            "compression_savings_spend": pytest.approx(0.0),
            "prompt_caching_savings_spend": pytest.approx(0.0),
            "gateway_injected_caching_savings_spend": pytest.approx(0.0),
            "autorouter_savings_spend": pytest.approx(0.0),
            "total_tokens": 40,
            "successful_requests": 1,
            "failed_requests": 0,
            "api_requests": 1,
            "total_response_time_ms": metrics["total_response_time_ms"],
            "timed_requests": metrics["timed_requests"],
        }
        report: Final = gateway.request(
            "GET",
            "/global/spend/report",
            params={"start_date": day, "end_date": day, "api_key": digest},
        )
        assert report.status_code == 200, report.text
        assert report.json() == [
            {
                "api_key": digest,
                "total_cost": pytest.approx(0.06),
                "total_input_tokens": 20,
                "total_output_tokens": 20,
                "model_details": [
                    {
                        "model": stored_model,
                        "total_cost": pytest.approx(0.06),
                        "total_input_tokens": 20,
                        "total_output_tokens": 20,
                    }
                ],
            }
        ]
        assert metrics["spend"] == pytest.approx(report.json()[0]["total_cost"])
