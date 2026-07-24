import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

import pytest
from fastapi import HTTPException

import litellm
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.spend_tracking.savings_endpoints import (
    _HourlySavingsRow,
    build_hourly_buckets,
    get_hourly_savings,
)

ADMIN = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-test")


class FakeDB:
    def __init__(self, rows, timezone_known=True):
        self._rows = rows
        self._timezone_known = timezone_known
        self.calls = []

    async def query_raw(self, query, *params):
        self.calls.append((query, params))
        if "pg_timezone_names" in query:
            return [{"ok": self._timezone_known}]
        return self._rows

    @property
    def savings_call(self):
        return next((c for c in self.calls if "LiteLLM_SpendLogs" in c[0]), None)


class FakePrismaClient:
    def __init__(self, rows=(), timezone_known=True):
        self.db = FakeDB(list(rows), timezone_known=timezone_known)


@pytest.fixture
def proxy_state(monkeypatch):
    from litellm.proxy import proxy_server

    def _apply(rows=(), disable_spend_logs=False, timezone_known=True):
        client = FakePrismaClient(rows, timezone_known=timezone_known)
        monkeypatch.setattr(proxy_server, "prisma_client", client, raising=False)
        monkeypatch.setattr(proxy_server, "disable_spend_logs", disable_spend_logs, raising=False)
        return client

    return _apply


def _anthropic_costs(model: str) -> tuple[float, float]:
    info = litellm.get_model_info(model=model, custom_llm_provider="anthropic")
    input_cost = info["input_cost_per_token"] or 0.0
    cache_read_cost = info.get("cache_read_input_token_cost") or input_cost
    return input_cost, cache_read_cost


def _row(bucket: str, model: str | None = "claude-sonnet-5", cache_read: int = 0, compression: int = 0):
    return _HourlySavingsRow(
        bucket_start=bucket,
        model=model,
        custom_llm_provider="anthropic" if model else None,
        cache_read_input_tokens=cache_read,
        compression_saved_tokens=compression,
    )


def _spine(*hours: int):
    return [_row(f"2026-07-23T{h:02d}:00", model=None) for h in hours]


def test_labels_and_order_come_from_the_zero_row_spine():
    buckets = build_hourly_buckets([*_spine(*range(24)), _row("2026-07-23T09:00", cache_read=8200)])

    assert [b.bucket_start for b in buckets] == [f"2026-07-23T{h:02d}:00" for h in range(24)]
    assert buckets[9].prompt_caching_savings_spend > 0
    assert all(b.prompt_caching_savings_spend == 0.0 for i, b in enumerate(buckets) if i != 9)


def test_a_short_dst_day_keeps_whatever_hours_the_spine_carries():
    # A spring-forward day is 23 hours: the query's spine simply omits the
    # missing local hour, and the folded series follows it without inventing one.
    hours = [h for h in range(24) if h != 2]
    buckets = build_hourly_buckets(_spine(*hours))

    assert len(buckets) == 23
    assert "2026-07-23T02:00" not in [b.bucket_start for b in buckets]


def test_savings_priced_per_model_before_being_summed_into_an_hour():
    sonnet_input, sonnet_cache_read = _anthropic_costs("claude-sonnet-5")
    opus_input, opus_cache_read = _anthropic_costs("claude-opus-4-6")
    assert sonnet_input != opus_input

    buckets = build_hourly_buckets(
        [
            _row("2026-07-23T09:00", model="claude-sonnet-5", cache_read=10_000),
            _row("2026-07-23T09:00", model="claude-opus-4-6", cache_read=10_000),
        ]
    )

    expected = 10_000 * (sonnet_input - sonnet_cache_read) + 10_000 * (opus_input - opus_cache_read)
    assert buckets[0].prompt_caching_savings_spend == pytest.approx(expected)
    assert buckets[0].prompt_caching_savings_spend != pytest.approx(20_000 * (sonnet_input - sonnet_cache_read))


def test_both_drivers_are_reported_separately():
    input_cost, cache_read_cost = _anthropic_costs("claude-sonnet-5")
    buckets = build_hourly_buckets([_row("2026-07-23T09:00", cache_read=8200, compression=4389)])
    assert buckets[0].compression_savings_spend == pytest.approx(4389 * input_cost)
    assert buckets[0].prompt_caching_savings_spend == pytest.approx(8200 * (input_cost - cache_read_cost))


def test_an_empty_spine_hour_prices_to_zero_not_an_error():
    buckets = build_hourly_buckets(_spine(0))
    assert buckets == [HourlyBucket(0.0, 0.0)]


def HourlyBucket(compression, caching):
    from litellm.types.savings import HourlySavingsBucket

    return HourlySavingsBucket(
        bucket_start="2026-07-23T00:00",
        compression_savings_spend=compression,
        prompt_caching_savings_spend=caching,
    )


@pytest.mark.asyncio
async def test_non_admin_is_refused(proxy_state):
    proxy_state()
    with pytest.raises(HTTPException) as exc:
        await get_hourly_savings(
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, api_key="sk-x"),
            start_date="2026-07-23",
            end_date="2026-07-23",
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_range_longer_than_the_cap_is_refused(proxy_state):
    client = proxy_state()
    with pytest.raises(HTTPException) as exc:
        await get_hourly_savings(user_api_key_dict=ADMIN, start_date="2026-07-01", end_date="2026-07-31")
    assert exc.value.status_code == 400
    assert client.db.calls == []


@pytest.mark.asyncio
async def test_backwards_range_is_refused(proxy_state):
    proxy_state()
    with pytest.raises(HTTPException) as exc:
        await get_hourly_savings(user_api_key_dict=ADMIN, start_date="2026-07-23", end_date="2026-07-21")
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_unparseable_date_is_refused(proxy_state):
    proxy_state()
    with pytest.raises(HTTPException) as exc:
        await get_hourly_savings(user_api_key_dict=ADMIN, start_date="07/23/2026", end_date="2026-07-23")
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_unknown_timezone_is_refused_before_the_savings_query(proxy_state):
    client = proxy_state(timezone_known=False)
    with pytest.raises(HTTPException) as exc:
        await get_hourly_savings(
            user_api_key_dict=ADMIN, start_date="2026-07-23", end_date="2026-07-23", timezone="Mars/Olympus_Mons"
        )
    assert exc.value.status_code == 400
    assert client.db.savings_call is None


@pytest.mark.asyncio
async def test_the_local_day_and_timezone_are_handed_to_postgres_verbatim(proxy_state):
    client = proxy_state(rows=[])
    response = await get_hourly_savings(
        user_api_key_dict=ADMIN,
        start_date="2026-07-23",
        end_date="2026-07-23",
        timezone="America/Los_Angeles",
    )

    # The endpoint passes the local day bounds and the zone; Postgres resolves
    # the DST-correct UTC window, so the offset is never computed in Python.
    _, params = client.db.savings_call
    assert params == ("2026-07-23T00:00:00", "2026-07-24T00:00:00", "America/Los_Angeles")
    assert response.timezone == "America/Los_Angeles"


@pytest.mark.asyncio
async def test_disabled_spend_logs_is_reported_without_touching_the_database(proxy_state):
    client = proxy_state(rows=[_row("2026-07-23T09:00", cache_read=8200)], disable_spend_logs=True)
    response = await get_hourly_savings(
        user_api_key_dict=ADMIN, start_date="2026-07-23", end_date="2026-07-23", timezone="America/Los_Angeles"
    )

    assert response.spend_logs_disabled is True
    assert response.timezone == "America/Los_Angeles"
    assert client.db.calls == []
    assert response.buckets == []


@pytest.mark.asyncio
async def test_rows_from_the_database_are_priced_into_the_response(proxy_state):
    input_cost, cache_read_cost = _anthropic_costs("claude-sonnet-5")
    proxy_state(
        rows=[
            *[
                {
                    "bucket_start": f"2026-07-23T{h:02d}:00",
                    "model": None,
                    "custom_llm_provider": None,
                    "cache_read_input_tokens": 0,
                    "compression_saved_tokens": 0,
                }
                for h in range(24)
            ],
            {
                "bucket_start": "2026-07-23T09:00",
                "model": "claude-sonnet-5",
                "custom_llm_provider": "anthropic",
                "cache_read_input_tokens": 8200,
                "compression_saved_tokens": 0,
            },
        ]
    )
    response = await get_hourly_savings(
        user_api_key_dict=ADMIN, start_date="2026-07-23", end_date="2026-07-23", timezone="America/Los_Angeles"
    )

    assert response.spend_logs_disabled is False
    assert len(response.buckets) == 24
    assert response.buckets[9].prompt_caching_savings_spend == pytest.approx(8200 * (input_cost - cache_read_cost))
