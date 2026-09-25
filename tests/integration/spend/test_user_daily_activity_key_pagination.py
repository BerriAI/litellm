import uuid
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from pydantic import JsonValue

from litellm.constants import USAGE_TOP_API_KEYS_LIMIT


def _key_breakdown(body: dict[str, JsonValue]) -> dict[str, JsonValue]:
    results: Final = body["results"]
    assert isinstance(results, list), body
    return {
        digest: metrics
        for day in results
        for digest, metrics in object_value(object_value(object_value(day)["breakdown"])["api_keys"]).items()
    }


def _range_params() -> dict[str, str]:
    today: Final = datetime.now(timezone.utc)
    return {
        "start_date": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
        "end_date": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
        "timezone": "0",
    }


def test_aggregated_key_pages_reach_every_key_through_cursor(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        expensive: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        cheap: Final = scenario.model(input_cost_per_token=0.00001, output_cost_per_token=0.00002)
        user: Final = scenario.user()
        needle: Final = scenario.key(user_id=user, key_alias=f"needle-{uuid.uuid4().hex}", models=[cheap])
        needle_digest: Final = sha256(needle.encode()).hexdigest()
        hay: Final = tuple(scenario.key(user_id=user, models=[expensive]) for _ in range(USAGE_TOP_API_KEYS_LIMIT))
        for key in (needle, *hay):
            reply: Final = gateway.chat(
                expensive if key in hay else cheap, key=key, text=f"key page {uuid.uuid4().hex}"
            )
            assert reply["usage"]["total_tokens"] == 40, reply
        daily: Final = eventually(
            lambda: read_rows('SELECT api_key FROM "LiteLLM_DailyUserSpend" WHERE user_id=%s', (user,)),
            lambda values: len(values) == USAGE_TOP_API_KEYS_LIMIT + 1,
            seconds=70,
        )
        assert len(daily) == USAGE_TOP_API_KEYS_LIMIT + 1, daily

        first: Final = gateway.request(
            "GET", "/user/daily/activity/aggregated", params={**_range_params(), "user_id": user}
        )
        assert first.status_code == 200, first.text
        first_body: Final = object_value(first.json())
        first_metadata: Final = object_value(first_body["metadata"])
        first_keys: Final = _key_breakdown(first_body)
        assert len(first_keys) == USAGE_TOP_API_KEYS_LIMIT, sorted(first_keys)
        assert needle_digest not in first_keys, "Precondition: the cheap key must fall outside page 1"
        next_cursor: Final = first_metadata["next_cursor"]
        assert isinstance(next_cursor, str) and next_cursor, first.text
        assert int(str(first_metadata["total_api_keys"])) == USAGE_TOP_API_KEYS_LIMIT + 1, first.text

        second: Final = gateway.request(
            "GET",
            "/user/daily/activity/aggregated",
            params={**_range_params(), "user_id": user, "cursor": next_cursor},
        )
        assert second.status_code == 200, second.text
        second_body: Final = object_value(second.json())
        second_metadata: Final = object_value(second_body["metadata"])
        assert set(_key_breakdown(second_body)) == {needle_digest}, second.text
        assert second_metadata["next_cursor"] is None, second.text
        assert int(str(second_metadata["total_api_keys"])) == USAGE_TOP_API_KEYS_LIMIT + 1, second.text
        assert float(str(second_metadata["total_spend"])) == pytest.approx(float(str(first_metadata["total_spend"]))), (
            first.text,
            second.text,
        )

        garbage: Final = gateway.request(
            "GET",
            "/user/daily/activity/aggregated",
            params={**_range_params(), "user_id": user, "cursor": "not-a-cursor"},
        )
        assert garbage.status_code == 400, garbage.text
