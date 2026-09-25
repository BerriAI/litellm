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


def _user_with_three_keys(
    gateway: Gateway, scenario, model: str
) -> tuple[str, tuple[str, ...], tuple[str, ...], dict[str, float]]:
    user: Final = scenario.user()
    keys: Final = tuple(scenario.key(user_id=user, models=[model]) for _ in range(3))
    digests: Final = tuple(sha256(key.encode()).hexdigest() for key in keys)
    for key in keys:
        reply: Final = gateway.chat(model, key=key, text=f"user export {uuid.uuid4().hex}")
        assert reply["usage"]["total_tokens"] == 40, reply
    daily: Final = eventually(
        lambda: read_rows('SELECT api_key, spend FROM "LiteLLM_DailyUserSpend" WHERE user_id=%s', (user,)),
        lambda values: len({row["api_key"] for row in values}) == 3,
        seconds=70,
    )
    spend_by_key: Final = {row["api_key"]: float(row["spend"]) for row in daily}
    return user, keys, digests, spend_by_key


def test_user_activity_export_returns_every_key_beyond_the_top_n_cap(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        user, keys, digests, spend_by_key = _user_with_three_keys(gateway, scenario, model)
        response: Final = gateway.request(
            "GET",
            "/user/daily/activity/export",
            params={
                **_export_range(),
                "user_id": user,
                "export_type": "daily_with_keys",
                "format": "json",
            },
        )
        assert response.status_code == 200, response.text
        body: Final = object_value(response.json())
        rows: Final = tuple(object_value(row) for row in body["data"])
        assert sorted(string_value(row["api_key"]) for row in rows) == sorted(digests), response.text
        for row in rows:
            assert row["user_id"] == user, response.text
            assert float(row["spend"]) == pytest.approx(spend_by_key[string_value(row["api_key"])]), response.text
        metadata: Final = object_value(body["metadata"])
        assert (
            metadata["export_type"],
            metadata["user_id"],
            metadata["total_api_requests"],
            metadata["total_successful_requests"],
            metadata["total_failed_requests"],
        ) == ("daily_with_keys", user, 3, 3, 0), response.text
        assert float(metadata["total_spend"]) == pytest.approx(sum(spend_by_key.values())), response.text


def test_user_activity_export_csv_downloads_every_key(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        user, keys, digests, spend_by_key = _user_with_three_keys(gateway, scenario, model)
        response: Final = gateway.request(
            "GET",
            "/user/daily/activity/export",
            params={
                **_export_range(),
                "user_id": user,
                "export_type": "daily_with_keys",
                "format": "csv",
            },
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/csv"), response.headers
        assert "attachment" in response.headers["content-disposition"], response.headers
        assert response.text.splitlines()[0].startswith("Date,User,User ID,Key Alias,Key ID,Spend ($),"), response.text
        records: Final = tuple(csv.DictReader(io.StringIO(response.text)))
        assert len(records) == 3, response.text
        assert sorted(record["Key ID"] for record in records) == sorted(digests), response.text
        assert sorted(record["User ID"] for record in records) == [user, user, user], response.text
        for record in records:
            assert record["Spend ($)"] == f"{spend_by_key[record['Key ID']]:.4f}", response.text


def test_user_activity_export_allows_a_non_admin_their_own_scope(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        member: Final = scenario.user(user_role="internal_user")
        member_key: Final = scenario.key(user_id=member, models=[model])
        reply: Final = gateway.chat(model, key=member_key, text=f"user export {uuid.uuid4().hex}")
        assert reply["usage"]["total_tokens"] == 40, reply
        eventually(
            lambda: read_rows('SELECT api_key FROM "LiteLLM_DailyUserSpend" WHERE user_id=%s', (member,)),
            lambda values: len(values) == 1,
            seconds=70,
        )
        response: Final = gateway.request(
            "GET",
            "/user/daily/activity/export",
            params={**_export_range(), "export_type": "daily", "format": "json"},
            key=member_key,
        )
        assert response.status_code == 200, response.text
        body: Final = object_value(response.json())
        assert object_value(body["metadata"])["user_id"] == member, response.text
        rows: Final = tuple(object_value(row) for row in body["data"])
        assert len(rows) == 1 and rows[0]["user_id"] == member, response.text


def test_user_activity_export_denies_a_non_admin_another_user(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        member: Final = scenario.user(user_role="internal_user")
        other: Final = scenario.user()
        member_key: Final = scenario.key(user_id=member, models=[model])
        denied: Final = gateway.request(
            "GET",
            "/user/daily/activity/export",
            params={**_export_range(), "user_id": other, "export_type": "daily", "format": "json"},
            key=member_key,
        )
        assert denied.status_code == 403, denied.text
