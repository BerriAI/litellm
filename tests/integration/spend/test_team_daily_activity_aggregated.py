import uuid
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows


@pytest.mark.covers("quota_management.spend_tracking.team_daily_activity_aggregated_reports_whole_range_team_spend")
def test_aggregated_team_activity_reports_the_whole_range_team_spend_in_one_page(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team: Final = scenario.team(models=[model])
        keys: Final = tuple(scenario.key(team_id=team, models=[model]) for _ in range(2))
        digests: Final = tuple(sha256(key.encode()).hexdigest() for key in keys)
        for key in keys:
            for _ in range(2):
                reply: Final = gateway.chat(model, key=key, text=f"team activity {uuid.uuid4().hex}")
                assert reply["usage"]["total_tokens"] == 40, reply
        logged: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE team_id=%s', (team,)),
            lambda values: len(values) == 4,
            seconds=70,
        )
        assert sum(float(row["spend"]) for row in logged) == pytest.approx(0.24)
        daily: Final = eventually(
            lambda: read_rows(
                'SELECT api_key, spend, successful_requests FROM "LiteLLM_DailyTeamSpend" WHERE team_id=%s', (team,)
            ),
            lambda values: sum(float(row["spend"]) for row in values) >= 0.24 - 1e-9,
            seconds=70,
        )
        assert sorted(row["api_key"] for row in daily) == sorted(digests), daily
        assert all(float(row["spend"]) == pytest.approx(0.12) and row["successful_requests"] == 2 for row in daily)
        today: Final = datetime.now(timezone.utc)
        response: Final = gateway.request(
            "GET",
            "/team/daily/activity/aggregated",
            params={
                "team_ids": team,
                "start_date": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
                "end_date": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
                "timezone": "0",
            },
        )
        assert response.status_code == 200, response.text
        body: Final = object_value(response.json())
        metadata: Final = object_value(body["metadata"])
        assert (
            metadata["total_spend"],
            metadata["total_prompt_tokens"],
            metadata["total_completion_tokens"],
            metadata["total_tokens"],
            metadata["total_api_requests"],
            metadata["total_successful_requests"],
            metadata["total_failed_requests"],
            metadata["page"],
            metadata["total_pages"],
            metadata["has_more"],
        ) == (pytest.approx(0.24), 80, 80, 160, 4, 4, 0, 1, 1, False), response.text
        results: Final = body["results"]
        assert isinstance(results, list) and len(results) == 1, response.text
        day: Final = object_value(results[0])
        assert object_value(day["metrics"])["spend"] == pytest.approx(0.24), response.text
        entities: Final = object_value(object_value(day["breakdown"])["entities"])
        assert set(entities) == {team}, response.text
        team_bucket: Final = object_value(entities[team])
        team_metrics: Final = object_value(team_bucket["metrics"])
        assert (team_metrics["spend"], team_metrics["api_requests"], team_metrics["successful_requests"]) == (
            pytest.approx(0.24),
            4,
            4,
        ), response.text
        per_key: Final = object_value(team_bucket["api_key_breakdown"])
        assert set(per_key) == set(digests), response.text
        assert tuple(object_value(object_value(per_key[digest])["metrics"])["spend"] for digest in digests) == (
            pytest.approx(0.12),
            pytest.approx(0.12),
        ), response.text


def test_model_top_keys_ranks_keys_by_spend_on_the_model_not_total_spend(gateway: Gateway) -> None:
    """Regression for the per-model Top Keys list: a key that spends more
    globally but less on the model must rank below the model's real top key."""
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        other_model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team: Final = scenario.team(models=[model, other_model])
        heavy: Final = scenario.key(team_id=team, models=[model, other_model])
        light: Final = scenario.key(team_id=team, models=[model])
        heavy_digest: Final = sha256(heavy.encode()).hexdigest()
        light_digest: Final = sha256(light.encode()).hexdigest()
        for _ in range(3):
            reply: Final = gateway.chat(other_model, key=heavy, text=f"top keys {uuid.uuid4().hex}")
            assert object_value(reply["usage"])["total_tokens"] == 40, reply
        reply = gateway.chat(model, key=heavy, text=f"top keys {uuid.uuid4().hex}")
        assert object_value(reply["usage"])["total_tokens"] == 40, reply
        for _ in range(2):
            reply = gateway.chat(model, key=light, text=f"top keys {uuid.uuid4().hex}")
            assert object_value(reply["usage"])["total_tokens"] == 40, reply
        daily: Final = eventually(
            lambda: read_rows(
                'SELECT api_key, model, spend FROM "LiteLLM_DailyTeamSpend" WHERE team_id=%s', (team,)
            ),
            lambda values: sum(float(row["spend"]) for row in values) >= 0.36 - 1e-9,
            seconds=70,
        )
        assert {row["api_key"] for row in daily} == {heavy_digest, light_digest}, daily
        today: Final = datetime.now(timezone.utc)
        response: Final = gateway.request(
            "GET",
            "/team/daily/activity/aggregated/model_top_keys",
            params={
                "team_ids": team,
                "model": model,
                "start_date": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
                "end_date": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
                "timezone": "0",
            },
        )
        assert response.status_code == 200, response.text
        body: Final = object_value(response.json())
        api_keys: Final = body["api_keys"]
        assert isinstance(api_keys, list), body
        assert tuple(object_value(row)["api_key"] for row in api_keys) == (light_digest, heavy_digest), response.text
        assert tuple(object_value(row)["spend"] for row in api_keys) == (pytest.approx(0.12), pytest.approx(0.06))
