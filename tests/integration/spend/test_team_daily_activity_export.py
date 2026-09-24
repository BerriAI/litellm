import csv
import io
import uuid
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows


def _export_range() -> dict[str, str]:
    today: Final = datetime.now(timezone.utc)
    return {
        "start_date": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
        "end_date": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
        "timezone": "0",
    }


def test_team_activity_export_returns_every_key_beyond_the_top_n_cap(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team: Final = scenario.team(models=[model])
        keys: Final = tuple(scenario.key(team_id=team, models=[model]) for _ in range(3))
        digests: Final = tuple(sha256(key.encode()).hexdigest() for key in keys)
        for key in keys:
            reply: Final = gateway.chat(model, key=key, text=f"team export {uuid.uuid4().hex}")
            assert reply["usage"]["total_tokens"] == 40, reply
        daily: Final = eventually(
            lambda: read_rows('SELECT api_key, spend FROM "LiteLLM_DailyTeamSpend" WHERE team_id=%s', (team,)),
            lambda values: len({row["api_key"] for row in values}) == 3,
            seconds=70,
        )
        spend_by_key: Final = {row["api_key"]: float(row["spend"]) for row in daily}
        response: Final = gateway.request(
            "GET",
            "/team/daily/activity/export",
            params={
                **_export_range(),
                "team_id": team,
                "export_type": "daily_with_keys",
                "format": "json",
            },
        )
        assert response.status_code == 200, response.text
        body: Final = object_value(response.json())
        rows: Final = tuple(object_value(row) for row in body["data"])
        assert sorted(string_value(row["api_key"]) for row in rows) == sorted(digests), response.text
        for row in rows:
            assert row["team_id"] == team, response.text
            assert float(row["spend"]) == pytest.approx(spend_by_key[string_value(row["api_key"])]), response.text
        metadata: Final = object_value(body["metadata"])
        assert (
            metadata["export_type"],
            metadata["team_ids"],
            metadata["total_api_requests"],
            metadata["total_successful_requests"],
            metadata["total_failed_requests"],
        ) == ("daily_with_keys", [team], 3, 3, 0), response.text
        assert float(metadata["total_spend"]) == pytest.approx(sum(spend_by_key.values())), response.text


def test_team_activity_export_csv_downloads_every_key(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team: Final = scenario.team(models=[model])
        keys: Final = tuple(scenario.key(team_id=team, models=[model]) for _ in range(3))
        digests: Final = tuple(sha256(key.encode()).hexdigest() for key in keys)
        for key in keys:
            reply: Final = gateway.chat(model, key=key, text=f"team export {uuid.uuid4().hex}")
            assert reply["usage"]["total_tokens"] == 40, reply
        daily: Final = eventually(
            lambda: read_rows('SELECT api_key, spend FROM "LiteLLM_DailyTeamSpend" WHERE team_id=%s', (team,)),
            lambda values: len({row["api_key"] for row in values}) == 3,
            seconds=70,
        )
        spend_by_key: Final = {row["api_key"]: float(row["spend"]) for row in daily}
        response: Final = gateway.request(
            "GET",
            "/team/daily/activity/export",
            params={
                **_export_range(),
                "team_id": team,
                "export_type": "daily_with_keys",
                "format": "csv",
            },
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/csv"), response.headers
        assert "attachment" in response.headers["content-disposition"], response.headers
        records: Final = tuple(csv.DictReader(io.StringIO(response.text)))
        assert len(records) == 3, response.text
        assert sorted(record["Key ID"] for record in records) == sorted(digests), response.text
        assert sorted(record["Team ID"] for record in records) == [team, team, team], response.text
        for record in records:
            assert record["Spend ($)"] == f"{spend_by_key[record['Key ID']]:.4f}", response.text


def test_team_activity_export_denies_a_member_another_team(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        member: Final = scenario.user(user_role="internal_user", teams=[team_a])
        member_key: Final = scenario.key(user_id=member, team_id=team_a, models=[model])
        reply: Final = gateway.chat(model, key=member_key, text=f"team export {uuid.uuid4().hex}")
        assert reply["usage"]["total_tokens"] == 40, reply
        daily: Final = eventually(
            lambda: read_rows('SELECT api_key, spend FROM "LiteLLM_DailyTeamSpend" WHERE team_id=%s', (team_a,)),
            lambda values: len(values) == 1,
            seconds=70,
        )
        denied: Final = gateway.request(
            "GET",
            "/team/daily/activity/export",
            params={**_export_range(), "team_id": team_b, "export_type": "daily", "format": "json"},
            key=member_key,
        )
        assert denied.status_code == 404, denied.text
        assert f"User does not belong to Team= {team_b}" in denied.text, denied.text
        allowed: Final = gateway.request(
            "GET",
            "/team/daily/activity/export",
            params={**_export_range(), "team_id": team_a, "export_type": "daily", "format": "json"},
            key=member_key,
        )
        assert allowed.status_code == 200, allowed.text
        rows: Final = tuple(object_value(row) for row in object_value(allowed.json())["data"])
        assert len(rows) == 1, allowed.text
        assert rows[0]["team_id"] == team_a, allowed.text
        assert float(rows[0]["spend"]) == pytest.approx(float(daily[0]["spend"])), allowed.text
