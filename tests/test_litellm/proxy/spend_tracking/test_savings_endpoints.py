import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../../../.."))

import pytest
from fastapi import HTTPException

import litellm
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.spend_tracking.savings_endpoints import (
    _HourlySavingsRow,
    build_hourly_buckets,
    get_hourly_savings,
    hour_bucket_labels,
)

ADMIN = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-test")


class FakeDB:
    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    async def query_raw(self, query, *params):
        self.calls.append((query, params))
        return self._rows


class FakePrismaClient:
    def __init__(self, rows=()):
        self.db = FakeDB(list(rows))


@pytest.fixture
def proxy_state(monkeypatch):
    from litellm.proxy import proxy_server

    def _apply(rows=(), disable_spend_logs=False):
        client = FakePrismaClient(rows)
        monkeypatch.setattr(proxy_server, "prisma_client", client, raising=False)
        monkeypatch.setattr(proxy_server, "disable_spend_logs", disable_spend_logs, raising=False)
        return client

    return _apply


def _anthropic_costs(model: str) -> tuple[float, float]:
    info = litellm.get_model_info(model=model, custom_llm_provider="anthropic")
    input_cost = info["input_cost_per_token"] or 0.0
    cache_read_cost = info.get("cache_read_input_token_cost") or input_cost
    return input_cost, cache_read_cost


def _row(bucket: str, model: str = "claude-sonnet-5", cache_read: int = 0, compression: int = 0) -> _HourlySavingsRow:
    return _HourlySavingsRow(
        bucket_start=bucket,
        model=model,
        custom_llm_provider="anthropic",
        cache_read_input_tokens=cache_read,
        compression_saved_tokens=compression,
    )


def test_hour_bucket_labels_covers_every_hour_of_a_single_day():
    labels = hour_bucket_labels(datetime(2026, 7, 23), datetime(2026, 7, 24))
    assert len(labels) == 24
    assert labels[0] == "2026-07-23T00:00"
    assert labels[13] == "2026-07-23T13:00"
    assert labels[-1] == "2026-07-23T23:00"


def test_hour_bucket_labels_spans_multiple_days():
    labels = hour_bucket_labels(datetime(2026, 7, 23), datetime(2026, 7, 25))
    assert len(labels) == 48
    assert labels[24] == "2026-07-24T00:00"


def test_quiet_hours_are_zero_filled_rather_than_dropped():
    labels = hour_bucket_labels(datetime(2026, 7, 23), datetime(2026, 7, 24))
    buckets = build_hourly_buckets([_row("2026-07-23T09:00", cache_read=8200)], labels)

    assert [b.bucket_start for b in buckets] == list(labels)
    assert buckets[9].prompt_caching_savings_spend > 0
    assert all(b.prompt_caching_savings_spend == 0.0 for i, b in enumerate(buckets) if i != 9)


def test_savings_priced_per_model_before_being_summed_into_an_hour():
    """
    The two models in this hour have different input rates, so pricing the
    summed tokens at either single rate gives the wrong answer. This is why the
    SQL groups by model and the dollars are computed here.
    """
    sonnet_input, sonnet_cache_read = _anthropic_costs("claude-sonnet-5")
    opus_input, opus_cache_read = _anthropic_costs("claude-opus-4-6")
    assert sonnet_input != opus_input

    buckets = build_hourly_buckets(
        [
            _row("2026-07-23T09:00", model="claude-sonnet-5", cache_read=10_000),
            _row("2026-07-23T09:00", model="claude-opus-4-6", cache_read=10_000),
        ],
        ("2026-07-23T09:00",),
    )

    expected = 10_000 * (sonnet_input - sonnet_cache_read) + 10_000 * (opus_input - opus_cache_read)
    assert buckets[0].prompt_caching_savings_spend == pytest.approx(expected)
    assert buckets[0].prompt_caching_savings_spend != pytest.approx(20_000 * (sonnet_input - sonnet_cache_read))


def test_both_drivers_are_reported_separately():
    input_cost, cache_read_cost = _anthropic_costs("claude-sonnet-5")
    buckets = build_hourly_buckets(
        [_row("2026-07-23T09:00", cache_read=8200, compression=4389)],
        ("2026-07-23T09:00",),
    )
    assert buckets[0].compression_savings_spend == pytest.approx(4389 * input_cost)
    assert buckets[0].prompt_caching_savings_spend == pytest.approx(8200 * (input_cost - cache_read_cost))


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
async def test_utc_offset_shifts_the_scanned_window_but_not_the_labels(proxy_state):
    client = proxy_state(rows=[])
    response = await get_hourly_savings(
        user_api_key_dict=ADMIN,
        start_date="2026-07-23",
        end_date="2026-07-23",
        utc_offset_minutes=-420,
    )

    _, params = client.db.calls[0]
    # 00:00 local on the US west coast is 07:00 UTC the same day.
    assert params[0] == "2026-07-23T07:00:00"
    assert params[1] == "2026-07-24T07:00:00"
    assert params[2] == -420
    # Buckets stay on the caller's clock, so midnight is midnight.
    assert response.buckets[0].bucket_start == "2026-07-23T00:00"
    assert len(response.buckets) == 24


@pytest.mark.asyncio
async def test_disabled_spend_logs_is_reported_instead_of_charting_zeroes(proxy_state):
    client = proxy_state(rows=[_row("2026-07-23T09:00", cache_read=8200)], disable_spend_logs=True)
    response = await get_hourly_savings(user_api_key_dict=ADMIN, start_date="2026-07-23", end_date="2026-07-23")

    assert response.spend_logs_disabled is True
    assert client.db.calls == []
    assert len(response.buckets) == 24
    assert all(b.prompt_caching_savings_spend == 0.0 for b in response.buckets)


@pytest.mark.asyncio
async def test_rows_from_the_database_are_priced_into_the_response(proxy_state):
    input_cost, cache_read_cost = _anthropic_costs("claude-sonnet-5")
    proxy_state(
        rows=[
            {
                "bucket_start": "2026-07-23T09:00",
                "model": "claude-sonnet-5",
                "custom_llm_provider": "anthropic",
                "cache_read_input_tokens": 8200,
                "compression_saved_tokens": 0,
            }
        ]
    )
    response = await get_hourly_savings(user_api_key_dict=ADMIN, start_date="2026-07-23", end_date="2026-07-23")

    assert response.spend_logs_disabled is False
    assert response.buckets[9].prompt_caching_savings_spend == pytest.approx(8200 * (input_cost - cache_read_cost))
