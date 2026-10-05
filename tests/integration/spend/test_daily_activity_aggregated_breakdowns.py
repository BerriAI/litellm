import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

from litellm.constants import PTU_SENTINEL_API_KEY, USAGE_TOP_API_KEYS_DEFAULT
from tests.integration._support.client import Gateway, object_value
from tests.integration._support.database import write_rows

_URL: Final = "/user/daily/activity/aggregated"
_RESULTS: Final = TypeAdapter(list[dict[str, JsonValue]])


def _unique_day() -> str:
    return str((datetime(1900, 1, 1) + timedelta(days=uuid.uuid4().int % 200000)).date())


def _seed(day: str, rows: Sequence[tuple[object, ...]]) -> None:
    for row in rows:
        write_rows(
            'INSERT INTO "LiteLLM_DailyUserSpend" (id, user_id, date, api_key, model, model_group,'
            " custom_llm_provider, mcp_namespaced_tool_name, endpoint, prompt_tokens, spend, api_requests,"
            " successful_requests, updated_at)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())",
            tuple(str(value) if isinstance(value, (int, float)) else value for value in row),
        )


def _clean(day: str) -> None:
    write_rows('DELETE FROM "LiteLLM_DailyUserSpend" WHERE date = %s', (day,))


def _activity(gateway: Gateway, day: str, **params: str) -> dict[str, JsonValue]:
    response: Final = gateway.request("GET", _URL, params={"start_date": day, "end_date": day, **params})
    assert response.status_code == 200, response.text
    return object_value(response.json())


def _row_id() -> str:
    return f"agg-{uuid.uuid4().hex}"


def _ranked_key_rows(day: str, count: int) -> list[tuple[object, ...]]:
    return [
        (
            _row_id(),
            f"user-{i:03d}",
            day,
            f"key-{i:03d}",
            "gpt-5",
            "",
            "openai",
            None,
            "/v1/chat/completions",
            10,
            6.0 if i == 4 else float(i + 1),
            1,
            1,
        )
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_get_daily_activity_aggregated_bounds_api_key_rollups(gateway: Gateway) -> None:
    """key-004 and key-005 tie on spend exactly at the default api_key_limit cutoff; the api_key
    tiebreaker keeps key-004 and drops key-005. The PTU sentinel outspends every key but takes no
    slot. Dropped keys and the sentinel still count toward the totals and the model rollup."""
    key_count: Final = USAGE_TOP_API_KEYS_DEFAULT + 5
    day: Final = _unique_day()
    _seed(
        day,
        [
            *_ranked_key_rows(day, key_count),
            (_row_id(), None, day, PTU_SENTINEL_API_KEY, "gpt-5", "", "azure", None, None, 0, 1000.0, 0, 0),
        ],
    )
    key_spend: Final = sum(6.0 if i == 4 else float(i + 1) for i in range(key_count))
    try:
        body: Final = _activity(gateway, day)
        metadata: Final = object_value(body["metadata"])
        assert metadata["total_spend"] == pytest.approx(key_spend + 1000.0)
        assert metadata["total_api_requests"] == key_count
        assert metadata["total_api_keys"] == key_count
        assert metadata["api_key_limit"] == USAGE_TOP_API_KEYS_DEFAULT
        results: Final = _RESULTS.validate_python(body["results"])
        assert len(results) == 1
        result_day: Final = object_value(results[0])
        assert object_value(result_day["metrics"])["spend"] == pytest.approx(key_spend + 1000.0)
        breakdown: Final = object_value(result_day["breakdown"])
        expected_top: Final = {f"key-{i:03d}" for i in range(6, key_count)} | {"key-004"}
        api_keys: Final = object_value(breakdown["api_keys"])
        assert set(api_keys) == expected_top
        assert object_value(object_value(api_keys["key-004"])["metrics"])["spend"] == 6.0
        assert PTU_SENTINEL_API_KEY not in api_keys
        models: Final = object_value(breakdown["models"])
        gpt5: Final = object_value(models["gpt-5"])
        assert object_value(gpt5["metrics"])["spend"] == pytest.approx(key_spend + 1000.0)
        assert set(object_value(gpt5["api_key_breakdown"])) == expected_top
        providers: Final = object_value(breakdown["providers"])
        openai: Final = object_value(providers["openai"])
        assert object_value(openai["metrics"])["spend"] == pytest.approx(key_spend)
        assert set(object_value(openai["api_key_breakdown"])) == expected_top
        endpoints: Final = object_value(breakdown["endpoints"])
        assert object_value(object_value(endpoints["/v1/chat/completions"])["metrics"])["api_requests"] == key_count
    finally:
        _clean(day)


@pytest.mark.asyncio
async def test_get_daily_activity_aggregated_reports_exact_limit_key_count_as_complete(gateway: Gateway) -> None:
    """With exactly USAGE_TOP_API_KEYS_DEFAULT keys nothing is dropped and total_api_keys equals the limit."""
    day: Final = _unique_day()
    _seed(day, _ranked_key_rows(day, USAGE_TOP_API_KEYS_DEFAULT))
    try:
        body: Final = _activity(gateway, day)
        metadata: Final = object_value(body["metadata"])
        assert metadata["total_api_keys"] == USAGE_TOP_API_KEYS_DEFAULT
        assert metadata["api_key_limit"] == USAGE_TOP_API_KEYS_DEFAULT
        results: Final = _RESULTS.validate_python(body["results"])
        api_keys: Final = object_value(object_value(object_value(results[0])["breakdown"])["api_keys"])
        assert set(api_keys) == {f"key-{i:03d}" for i in range(USAGE_TOP_API_KEYS_DEFAULT)}
    finally:
        _clean(day)


@pytest.mark.asyncio
async def test_get_daily_activity_aggregated_explicit_api_key_filter_scopes_results(
    gateway: Gateway,
) -> None:
    day: Final = _unique_day()
    _seed(
        day,
        [
            (
                _row_id(),
                f"user-{i}",
                day,
                f"key-{i}",
                "gpt-5",
                "",
                "openai",
                None,
                "/v1/chat/completions",
                10,
                float(i + 1),
                1,
                1,
            )
            for i in range(3)
        ],
    )
    try:
        body: Final = _activity(gateway, day, api_key="key-1")
        metadata: Final = object_value(body["metadata"])
        assert metadata["total_spend"] == 2.0
        assert metadata["total_api_keys"] == 1
        results: Final = _RESULTS.validate_python(body["results"])
        assert len(results) == 1
        breakdown: Final = object_value(object_value(results[0])["breakdown"])
        api_keys: Final = object_value(breakdown["api_keys"])
        assert set(api_keys) == {"key-1"}
        assert object_value(object_value(api_keys["key-1"])["metrics"])["spend"] == 2.0
        gpt5: Final = object_value(object_value(breakdown["models"])["gpt-5"])
        assert object_value(gpt5["metrics"])["spend"] == 2.0
        assert set(object_value(gpt5["api_key_breakdown"])) == {"key-1"}
    finally:
        _clean(day)


@pytest.mark.asyncio
async def test_get_daily_activity_aggregated_model_group_rollups_fall_back_to_model_name(
    gateway: Gateway,
) -> None:
    day: Final = _unique_day()
    _seed(
        day,
        [
            (
                _row_id(),
                "user-0",
                day,
                "key-0",
                "gpt-5",
                "gpt-5-eu",
                "openai",
                None,
                "/v1/chat/completions",
                10,
                7.0,
                1,
                1,
            ),
            (
                _row_id(),
                "user-1",
                day,
                "key-1",
                "gpt-5",
                "",
                "openai",
                None,
                "/v1/chat/completions",
                10,
                3.0,
                1,
                1,
            ),
            (
                _row_id(),
                "user-2",
                day,
                "key-2",
                "claude-x",
                None,
                "anthropic",
                None,
                "/v1/messages",
                10,
                2.0,
                1,
                1,
            ),
        ],
    )
    try:
        body: Final = _activity(gateway, day)
        results: Final = _RESULTS.validate_python(body["results"])
        assert len(results) == 1
        breakdown: Final = object_value(object_value(results[0])["breakdown"])
        model_groups: Final = object_value(breakdown["model_groups"])
        assert set(model_groups) == {"gpt-5-eu", "gpt-5", "claude-x"}
        assert object_value(object_value(model_groups["gpt-5-eu"])["metrics"])["spend"] == 7.0
        gpt5_group: Final = object_value(model_groups["gpt-5"])
        assert object_value(gpt5_group["metrics"])["spend"] == 3.0
        assert object_value(object_value(model_groups["claude-x"])["metrics"])["spend"] == 2.0
        assert set(object_value(gpt5_group["api_key_breakdown"])) == {"key-1"}
        models: Final = object_value(breakdown["models"])
        assert set(models) == {"gpt-5", "claude-x"}
        assert object_value(object_value(models["gpt-5"])["metrics"])["spend"] == 10.0
    finally:
        _clean(day)
