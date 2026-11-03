import csv
import io
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
_NUMBER: Final = TypeAdapter(float)


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


@pytest.mark.parametrize("suffix", ["/aggregated", ""])
def test_missing_attribution_reconciles_and_filters_without_key_collisions(gateway: Gateway, suffix: str) -> None:
    day: Final = _unique_day()
    rows: Final = (
        ("", None, None, 3.0),
        ("", "", "", 0.0),
        ("known-key", "unknown-model", None, -1.0),
        ("unattributed", "known-model", "", 2.0),
        ("raw:unattributed", "known-model", "named-group", 4.0),
        ("known-key", "", "unknown-model-group", 5.0),
    )
    _seed(
        day,
        [
            (_row_id(), "bucket-user", day, key, model, group, "", "tool", "/mcp", 10, spend, 1, 1)
            for key, model, group, spend in rows
        ],
    )
    params: Final = {"start_date": day, "end_date": day}
    url: Final = f"/user/daily/activity{suffix}"
    try:
        response: Final = gateway.request("GET", url, params=params)
        assert response.status_code == 200, response.text
        body: Final = object_value(response.json())
        results: Final = _RESULTS.validate_python(body["results"])
        assert len(results) == 1
        totals: Final = object_value(results[0]["metrics"])
        assert totals["spend"] == 13.0
        assert totals["api_requests"] == 6
        breakdown: Final = object_value(results[0]["breakdown"])
        for dimension in ("models", "model_groups", "api_keys", "providers", "mcp_servers", "endpoints"):
            buckets: Final = _RESULTS.validate_python(list(object_value(breakdown[dimension]).values()))
            for metric in ("spend", "api_requests", "prompt_tokens"):
                assert (
                    sum(_NUMBER.validate_python(object_value(bucket["metrics"])[metric]) for bucket in buckets)
                    == totals[metric]
                )
        models: Final = object_value(breakdown["models"])
        assert set(models) == {"known-model", "unknown-model"}
        assert object_value(object_value(models["unknown-model"])["metrics"])["api_requests"] == 4
        groups: Final = object_value(breakdown["model_groups"])
        assert set(groups) == {"unknown-model-group", "unknown-model", "known-model", "named-group"}
        assert object_value(object_value(groups["unknown-model-group"])["metrics"])["spend"] == 8.0
        keys: Final = object_value(breakdown["api_keys"])
        assert set(keys) == {"unattributed", "raw:unattributed", "raw:raw:unattributed", "known-key"}
        unknown_key: Final = object_value(keys["unattributed"])
        assert object_value(unknown_key["metadata"]) == {
            "key_alias": "Unattributed",
            "key_exists": False,
            "team_id": None,
            "user_id": None,
            "user_email": None,
        }
        for dimension in ("models", "model_groups", "providers", "mcp_servers", "endpoints"):
            for bucket in _RESULTS.validate_python(list(object_value(breakdown[dimension]).values())):
                nested: Final = _RESULTS.validate_python(list(object_value(bucket["api_key_breakdown"]).values()))
                assert (
                    sum(_NUMBER.validate_python(object_value(key["metrics"])["spend"]) for key in nested)
                    == object_value(bucket["metrics"])["spend"]
                )
        for key, spend, count in (
            ("unattributed", 3.0, 2),
            ("raw:unattributed", 2.0, 1),
            ("raw:raw:unattributed", 4.0, 1),
        ):
            filtered: Final = gateway.request("GET", url, params={**params, "api_key": key})
            assert filtered.status_code == 200, filtered.text
            metadata: Final = object_value(object_value(filtered.json())["metadata"])
            assert metadata["total_spend"] == spend
            assert metadata["total_api_requests"] == count
        model_response: Final = gateway.request("GET", url, params={**params, "model": "unknown-model"})
        assert model_response.status_code == 200, model_response.text
        model_metadata: Final = object_value(object_value(model_response.json())["metadata"])
        assert model_metadata["total_spend"] == 7.0
        assert model_metadata["total_api_requests"] == 4
    finally:
        _clean(day)


@pytest.mark.parametrize(
    ("route", "extra", "expected"),
    [
        ("keys", {}, {"unattributed": 3.0, "raw:unattributed": 2.0, "known-key": 7.0}),
        ("search", {"search": "Unattributed"}, {"unattributed": 3.0, "raw:unattributed": 2.0}),
        ("model_top_keys", {"model_group": "unknown-model-group"}, {"unattributed": 3.0}),
        (
            "model_top_keys",
            {"model_group": "unknown-model", "by_model_group": "false"},
            {"unattributed": 3.0, "raw:unattributed": 2.0},
        ),
    ],
)
def test_missing_attribution_key_drilldowns(
    gateway: Gateway,
    route: str,
    extra: dict[str, str],
    expected: dict[str, float],
) -> None:
    day: Final = _unique_day()
    _seed(
        day,
        [
            (_row_id(), "bucket-user", day, "", None, None, "", None, None, 0, 3.0, 1, 1),
            (_row_id(), "bucket-user", day, "unattributed", "unknown-model", "", "", None, None, 0, 2.0, 1, 1),
            (_row_id(), "bucket-user", day, "known-key", "known-model", "", "", None, None, 0, 7.0, 1, 1),
        ],
    )
    try:
        response: Final = gateway.request(
            "GET", f"{_URL}/{route}", params={"start_date": day, "end_date": day, **extra}
        )
        assert response.status_code == 200, response.text
        rows: Final = _RESULTS.validate_python(object_value(response.json())["api_keys"])
        assert {str(row["api_key"]): object_value(row["metrics"])["spend"] for row in rows} == expected
        unattributed: Final = next(row for row in rows if row["api_key"] == "unattributed")
        assert object_value(unattributed["metadata"])["key_alias"] == "Unattributed"
        assert object_value(unattributed["metadata"])["key_exists"] is False
    finally:
        _clean(day)


@pytest.mark.parametrize("format", ["json", "csv"])
@pytest.mark.parametrize(
    ("export_type", "field", "expected"),
    [
        ("daily_with_keys", "api_key", {"unattributed": 3.0, "raw:unattributed": 2.0, "raw:raw:unattributed": 4.0}),
        ("daily_with_models", "model", {"unknown-model": 5.0, "known-model": 4.0}),
    ],
)
def test_export_bucket_ids_round_trip_across_streaming_batches(
    gateway: Gateway, format: str, export_type: str, field: str, expected: dict[str, float]
) -> None:
    day: Final = _unique_day()
    _seed(
        day,
        [
            (_row_id(), "export-user", day, "", None, None, "", None, None, 0, 3.0, 1, 1),
            (_row_id(), "export-user", day, "", "", "", "", None, None, 0, 0.0, 1, 1),
            (_row_id(), "export-user", day, "unattributed", "unknown-model", "", "", None, None, 0, 2.0, 1, 1),
            (_row_id(), "export-user", day, "raw:unattributed", "known-model", "", "", None, None, 0, 4.0, 1, 1),
        ],
    )
    write_rows(
        'INSERT INTO "LiteLLM_DailyUserSpend" (id, user_id, date, api_key, model, spend, api_requests, updated_at) '
        "SELECT %s || i, 'export-user', %s, 'raw:a-key-' || i, 'a-model-' || i, 1, 1, now() "
        "FROM generate_series(1, 1001) i",
        (str(uuid.uuid4()), day),
    )
    params: Final = {"start_date": day, "end_date": day}
    try:
        response: Final = gateway.request(
            "GET", "/user/daily/activity/export", params={**params, "export_type": export_type, "format": format}
        )
        assert response.status_code == 200, response.text
        rows: Final = (
            _RESULTS.validate_python(response.json())
            if format == "json"
            else _RESULTS.validate_python(list(csv.DictReader(io.StringIO(response.text))))
        )
        buckets: Final = {str(row[field]): _NUMBER.validate_python(row["spend"]) for row in rows}
        filler: Final = "raw:raw:a-key-" if field == "api_key" else "a-model-"
        assert buckets == {**{f"{filler}{i}": 1.0 for i in range(1, 1002)}, **expected}
        assert len(rows) == len(buckets)
        assert sum(buckets.values()) == 1010.0
        for bucket, spend in expected.items():
            filtered: Final = _activity(gateway, day, **{field: bucket})
            assert object_value(filtered["metadata"])["total_spend"] == spend
        if field == "api_key":
            unattributed: Final = next(row for row in rows if row["api_key"] == "unattributed")
            assert unattributed["key_alias"] == "Unattributed"
            assert unattributed["user_id"] in (None, "")
    finally:
        _clean(day)


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


def _seed_guardrail_units(day: str, rows: Sequence[tuple[str, str, str, float | None, int, int]]) -> None:
    for guardrail_id, api_key, usage_unit, cost, units, untracked_units in rows:
        write_rows(
            'INSERT INTO "LiteLLM_DailyGuardrailUsageUnits" (guardrail_id, date, team_id, api_key, usage_unit,'
            " units, cost, untracked_units, updated_at) VALUES (%s, %s, '', %s, %s, %s, %s, %s, now())",
            (
                guardrail_id,
                day,
                api_key,
                usage_unit,
                str(units),
                None if cost is None else str(cost),
                str(untracked_units),
            ),
        )


def _guardrail_spend(breakdown: dict[str, JsonValue]) -> dict[str, tuple[float, dict[str, float]]]:
    return {
        guardrail_id: (
            float(object_value(object_value(entry)["metrics"])["spend"]),
            {
                api_key: float(object_value(object_value(key_entry)["metrics"])["spend"])
                for api_key, key_entry in object_value(object_value(entry)["api_key_breakdown"]).items()
            },
        )
        for guardrail_id, entry in object_value(breakdown["guardrails"]).items()
    }


def _day_breakdown(gateway: Gateway, day: str, **params: str) -> dict[str, JsonValue]:
    results: Final = _RESULTS.validate_python(_activity(gateway, day, **params)["results"])
    assert len(results) == 1
    return object_value(object_value(results[0])["breakdown"])


@pytest.mark.asyncio
async def test_get_daily_activity_aggregated_breaks_down_guardrail_spend(gateway: Gateway) -> None:
    day: Final = _unique_day()
    _seed(
        day,
        [
            (_row_id(), "user-a", day, "key-a", "gpt-5", "", "bedrock", None, "/v1/chat/completions", 10, 1.0, 1, 1),
            (_row_id(), "user-b", day, "key-b", "gpt-5", "", "bedrock", None, "/v1/chat/completions", 10, 2.0, 1, 1),
        ],
    )
    _seed_guardrail_units(
        day,
        [
            ("pii-filter", "key-a", "contentPolicyUnits", 0.10, 5, 2),
            ("pii-filter", "key-a", "sensitiveInformationPolicyUnits", 0.05, 1, 0),
            ("pii-filter", "key-b", "contentPolicyUnits", 0.20, 1, 0),
            ("pii-filter", "key-b", "legacyUnits", None, 7, 0),
            ("topic-filter", "key-a", "topicPolicyUnits", 0.01, 1, 0),
            ("topic-filter", "key-b", "topicPolicyUnits", None, 1, 0),
            ("pii-filter", "orphan-key", "contentPolicyUnits", 9.0, 1, 0),
        ],
    )
    try:
        everyone: Final = _day_breakdown(gateway, day)
        assert _guardrail_spend(everyone) == {
            "pii-filter": (pytest.approx(0.35), {"key-a": pytest.approx(0.15), "key-b": pytest.approx(0.20)}),
            "topic-filter": (pytest.approx(0.01), {"key-a": pytest.approx(0.01), "key-b": 0.0}),
        }
        api_keys: Final = object_value(everyone["api_keys"])
        pii_filter: Final = object_value(object_value(everyone["guardrails"])["pii-filter"])
        assert object_value(pii_filter["metrics"]) == {"spend": pytest.approx(0.35), "untracked_units": 9}
        pii_api_keys: Final = object_value(pii_filter["api_key_breakdown"])
        assert object_value(object_value(pii_api_keys["key-a"])["metrics"]) == {
            "spend": pytest.approx(0.15),
            "untracked_units": 2,
        }
        assert object_value(object_value(pii_api_keys["key-b"])["metrics"]) == {
            "spend": pytest.approx(0.20),
            "untracked_units": 7,
        }

        assert object_value(everyone["guardrails"])["topic-filter"] == {
            "metrics": {"spend": pytest.approx(0.01), "untracked_units": 1},
            "metadata": {"guardrail_name": "topic-filter"},
            "api_key_breakdown": {
                "key-a": {
                    "metrics": {"spend": pytest.approx(0.01), "untracked_units": 0},
                    "metadata": object_value(api_keys["key-a"])["metadata"],
                },
                "key-b": {
                    "metrics": {"spend": 0.0, "untracked_units": 1},
                    "metadata": object_value(api_keys["key-b"])["metadata"],
                },
            },
        }
        assert object_value(object_value(object_value(everyone["models"])["gpt-5"])["metrics"])["spend"] == 3.0

        assert _guardrail_spend(_day_breakdown(gateway, day, user_id="user-a")) == {
            "pii-filter": (pytest.approx(0.15), {"key-a": pytest.approx(0.15)}),
            "topic-filter": (pytest.approx(0.01), {"key-a": pytest.approx(0.01)}),
        }
        assert _guardrail_spend(_day_breakdown(gateway, day, api_key_limit="1")) == {
            "pii-filter": (pytest.approx(0.35), {"key-b": pytest.approx(0.20)}),
            "topic-filter": (pytest.approx(0.01), {"key-b": 0.0}),
        }
        limited_pii_filter: Final = object_value(
            object_value(_day_breakdown(gateway, day, api_key_limit="1")["guardrails"])["pii-filter"]
        )
        assert object_value(limited_pii_filter["metrics"])["untracked_units"] == 9
        assert set(object_value(limited_pii_filter["api_key_breakdown"])) == {"key-b"}
        assert _guardrail_spend(_day_breakdown(gateway, day, model="gpt-5")) == {}
        assert _guardrail_spend(_day_breakdown(gateway, day, model="")) == _guardrail_spend(everyone)
    finally:
        _clean(day)
        write_rows('DELETE FROM "LiteLLM_DailyGuardrailUsageUnits" WHERE date = %s', (day,))


def _entity_breakdown(gateway: Gateway, url: str, day: str, **params: str) -> dict[str, JsonValue]:
    response: Final = gateway.request("GET", url, params={"start_date": day, "end_date": day, **params})
    assert response.status_code == 200, response.text
    results: Final = _RESULTS.validate_python(object_value(response.json())["results"])
    assert len(results) == 1
    return object_value(object_value(results[0])["breakdown"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "table", "entity_column", "filter_param"),
    [
        ("/team/daily/activity/aggregated", "LiteLLM_DailyTeamSpend", "team_id", "team_ids"),
        (
            "/organization/daily/activity/aggregated",
            "LiteLLM_DailyOrganizationSpend",
            "organization_id",
            "organization_ids",
        ),
    ],
)
async def test_team_and_organization_activity_break_down_guardrail_spend_by_name(
    gateway: Gateway, url: str, table: str, entity_column: str, filter_param: str
) -> None:
    day: Final = _unique_day()
    db_guardrail_id: Final = str(uuid.uuid4())
    db_guardrail_name: Final = f"pii-filter-{db_guardrail_id[:8]}"
    for entity_id, api_key, spend in (("entity-a", "key-a", 1.0), ("entity-b", "key-b", 2.0)):
        write_rows(
            f'INSERT INTO "{table}" (id, {entity_column}, date, api_key, model, spend, api_requests,'
            " successful_requests, updated_at) VALUES (%s, %s, %s, %s, 'gpt-5', %s, 1, 1, now())",
            (_row_id(), entity_id, day, api_key, str(spend)),
        )
    write_rows(
        'INSERT INTO "LiteLLM_GuardrailsTable" (guardrail_id, guardrail_name, litellm_params, updated_at)'
        " VALUES (%s, %s, '{}', now())",
        (db_guardrail_id, db_guardrail_name),
    )
    _seed_guardrail_units(
        day,
        [
            (db_guardrail_id, "key-a", "contentPolicyUnits", 0.10, 1, 0),
            (db_guardrail_id, "key-b", "contentPolicyUnits", 0.20, 1, 0),
            ("config-topic-filter", "key-a", "topicPolicyUnits", 0.05, 1, 0),
            (db_guardrail_id, "orphan-key", "contentPolicyUnits", 9.0, 1, 0),
        ],
    )
    try:
        everyone: Final = _entity_breakdown(gateway, url, day)
        assert _guardrail_spend(everyone) == {
            db_guardrail_id: (pytest.approx(0.30), {"key-a": pytest.approx(0.10), "key-b": pytest.approx(0.20)}),
            "config-topic-filter": (pytest.approx(0.05), {"key-a": pytest.approx(0.05)}),
        }
        guardrails: Final = object_value(everyone["guardrails"])
        assert object_value(guardrails[db_guardrail_id])["metadata"] == {"guardrail_name": db_guardrail_name}
        assert object_value(guardrails["config-topic-filter"])["metadata"] == {"guardrail_name": "config-topic-filter"}

        assert _guardrail_spend(_entity_breakdown(gateway, url, day, **{filter_param: "entity-a"})) == {
            db_guardrail_id: (pytest.approx(0.10), {"key-a": pytest.approx(0.10)}),
            "config-topic-filter": (pytest.approx(0.05), {"key-a": pytest.approx(0.05)}),
        }
    finally:
        write_rows(f'DELETE FROM "{table}" WHERE date = %s', (day,))
        write_rows('DELETE FROM "LiteLLM_DailyGuardrailUsageUnits" WHERE date = %s', (day,))
        write_rows('DELETE FROM "LiteLLM_GuardrailsTable" WHERE guardrail_id = %s', (db_guardrail_id,))
