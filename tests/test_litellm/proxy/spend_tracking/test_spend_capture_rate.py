import json
import re
from collections.abc import Mapping
from datetime import date, datetime, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import psycopg
import pytest
from psycopg.rows import dict_row
from pydantic import ValidationError
from pytest_postgresql import factories

from litellm.constants import (
    SPEND_CAPTURE_RATE_CHECK_JOB_ID,
    SPEND_CAPTURE_RATE_DOCS_URL,
    SPEND_CAPTURE_RATE_MAX_RANGE_DAYS,
)
from litellm.llms.openai.organization_costs import OPENAI_ADMIN_KEY_ENV_VAR
from litellm.proxy.spend_tracking.spend_capture_rate import (
    ProviderBillingCredentialMissing,
    ProviderBillingRequestFailed,
    alert_message,
    captured_spend_by_day,
    compute_capture_rate,
    run_scheduled_spend_capture_rate_check,
    run_spend_capture_rate_check,
)
from litellm.types.proxy.spend_capture_rate import CaptureRateReport, SpendCaptureRateCheckSettings

_ADMIN_KEY: Final = "sk-admin-test"


def _utc_midnight(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())


def _bucket(day: str, *amounts: float) -> dict[str, object]:
    return {
        "object": "bucket",
        "start_time": _utc_midnight(day),
        "end_time": _utc_midnight(day) + 86400,
        "results": [
            {"object": "organization.costs.result", "amount": {"value": amount, "currency": "usd"}, "line_item": None}
            for amount in amounts
        ],
    }


class _FakeCostsApi:
    """Serves ``pages`` in order and records every request it saw."""

    def __init__(self, *pages: dict[str, object] | httpx.Response) -> None:
        self.responses = [
            page if isinstance(page, httpx.Response) else httpx.Response(200, json=page) for page in pages
        ]
        self.calls: list[tuple[str, Mapping[str, object], Mapping[str, str]]] = []

    async def __call__(self, url: str, params: Mapping[str, object], headers: Mapping[str, str]) -> httpx.Response:
        self.calls.append((url, dict(params), dict(headers)))
        return self.responses[len(self.calls) - 1]


def _page(*buckets: dict[str, object], next_page: str | None = None) -> dict[str, object]:
    return {"object": "page", "data": list(buckets), "has_more": next_page is not None, "next_page": next_page}


def _fake_prisma(rows: list[dict[str, object]]) -> MagicMock:
    prisma = MagicMock()
    prisma.db.query_raw = AsyncMock(return_value=rows)
    return prisma


def test_capture_rate_covers_every_day_in_the_range_and_flags_the_threshold():
    report = compute_capture_rate(
        provider="openai",
        start_date=date(2026, 9, 20),
        end_date=date(2026, 9, 22),
        captured_by_day={"2026-09-20": 8.0, "2026-09-22": 1.0},
        billed_by_day={"2026-09-20": 10.0, "2026-09-21": 5.0},
        threshold=0.9,
    )

    assert [day.date for day in report.days] == ["2026-09-20", "2026-09-21", "2026-09-22"]
    assert [day.capture_rate for day in report.days] == [0.8, 0.0, None]
    assert report.captured_spend == 9.0
    assert report.provider_spend == 15.0
    assert report.capture_rate == 0.6
    assert report.below_threshold is True


def test_capture_rate_at_or_above_the_threshold_is_not_flagged_and_a_zero_bill_has_no_rate():
    healthy = compute_capture_rate(
        provider="openai",
        start_date=date(2026, 9, 20),
        end_date=date(2026, 9, 20),
        captured_by_day={"2026-09-20": 9.5},
        billed_by_day={"2026-09-20": 10.0},
        threshold=0.9,
    )
    over = compute_capture_rate(
        provider="openai",
        start_date=date(2026, 9, 20),
        end_date=date(2026, 9, 20),
        captured_by_day={"2026-09-20": 12.0},
        billed_by_day={"2026-09-20": 10.0},
        threshold=0.9,
    )
    unbilled = compute_capture_rate(
        provider="openai",
        start_date=date(2026, 9, 20),
        end_date=date(2026, 9, 20),
        captured_by_day={"2026-09-20": 3.0},
        billed_by_day={},
        threshold=0.9,
    )

    assert (healthy.capture_rate, healthy.below_threshold) == (0.95, False)
    assert (over.capture_rate, over.below_threshold) == (1.2, False)
    assert (unbilled.capture_rate, unbilled.below_threshold) == (None, False)
    assert alert_message(healthy) is None
    assert alert_message(over) is None
    assert alert_message(unbilled) is None


def test_alert_messages_name_the_cause_and_link_the_docs():
    below = compute_capture_rate(
        provider="openai",
        start_date=date(2026, 9, 14),
        end_date=date(2026, 9, 20),
        captured_by_day={"2026-09-14": 700.0},
        billed_by_day={"2026-09-14": 1000.0},
        threshold=0.9,
    )

    below_message = alert_message(below)
    missing_message = alert_message(ProviderBillingCredentialMissing("openai", OPENAI_ADMIN_KEY_ENV_VAR))
    failed_message = alert_message(ProviderBillingRequestFailed("openai", "HTTP 401: nope"))

    assert below_message is not None and "70.0%" in below_message and "90%" in below_message
    assert "$700.00" in below_message and "$1,000.00" in below_message
    assert "2026-09-14 to 2026-09-20" in below_message
    assert missing_message is not None and OPENAI_ADMIN_KEY_ENV_VAR in missing_message
    assert failed_message is not None and "HTTP 401: nope" in failed_message
    assert all(SPEND_CAPTURE_RATE_DOCS_URL in m for m in (below_message, missing_message, failed_message))


@pytest.mark.asyncio
async def test_check_reads_the_closed_window_before_today_and_publishes_the_rate(monkeypatch):
    monkeypatch.setenv(OPENAI_ADMIN_KEY_ENV_VAR, _ADMIN_KEY)
    api = _FakeCostsApi(_page(_bucket("2026-09-21", 100.0), _bucket("2026-09-22", 100.0)))
    prisma = _fake_prisma([{"date": "2026-09-21", "spend": 95.0}, {"date": "2026-09-22", "spend": 91.0}])
    alert = AsyncMock()
    publish = MagicMock()

    results = await run_spend_capture_rate_check(
        prisma,
        SpendCaptureRateCheckSettings(lookback_days=2, threshold=0.9),
        alert=alert,
        publish=publish,
        today=date(2026, 9, 23),
        http_get=api,
    )

    (report,) = results
    assert isinstance(report, CaptureRateReport)
    assert (report.start_date, report.end_date) == ("2026-09-21", "2026-09-22")
    assert report.capture_rate == 0.93
    publish.assert_called_once_with("openai", 0.93)
    alert.assert_not_awaited()
    assert api.calls[0][1]["start_time"] == _utc_midnight("2026-09-21")
    assert api.calls[0][1]["end_time"] == _utc_midnight("2026-09-23")
    sql, start, end, providers = prisma.db.query_raw.await_args.args
    assert (start, end) == ("2026-09-21", "2026-09-22")
    assert set(providers) == {"openai", "text-completion-openai"}
    assert '"LiteLLM_DailyUserSpend"' in sql


@pytest.mark.asyncio
async def test_check_alerts_under_the_threshold_and_still_publishes_the_rate(monkeypatch):
    monkeypatch.setenv(OPENAI_ADMIN_KEY_ENV_VAR, _ADMIN_KEY)
    api = _FakeCostsApi(_page(_bucket("2026-09-22", 200.0)))
    prisma = _fake_prisma([{"date": "2026-09-22", "spend": 50.0}])
    alert = AsyncMock()
    publish = MagicMock()

    await run_spend_capture_rate_check(
        prisma,
        SpendCaptureRateCheckSettings(lookback_days=1),
        alert=alert,
        publish=publish,
        today=date(2026, 9, 23),
        http_get=api,
    )

    publish.assert_called_once_with("openai", 0.25)
    alert.assert_awaited_once()
    assert "25.0%" in alert.await_args.args[0]


@pytest.mark.asyncio
async def test_check_alerts_on_a_missing_admin_key_and_publishes_nothing(monkeypatch):
    monkeypatch.delenv(OPENAI_ADMIN_KEY_ENV_VAR, raising=False)
    api = _FakeCostsApi()
    prisma = _fake_prisma([])
    alert = AsyncMock()
    publish = MagicMock()

    (result,) = await run_spend_capture_rate_check(
        prisma, SpendCaptureRateCheckSettings(), alert=alert, publish=publish, today=date(2026, 9, 23), http_get=api
    )

    assert result == ProviderBillingCredentialMissing("openai", OPENAI_ADMIN_KEY_ENV_VAR)
    publish.assert_called_once_with("openai", None)
    alert.assert_awaited_once()
    assert api.calls == []
    prisma.db.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_check_alerts_on_an_unreadable_bill_and_publishes_no_rate(monkeypatch):
    monkeypatch.setenv(OPENAI_ADMIN_KEY_ENV_VAR, _ADMIN_KEY)
    api = _FakeCostsApi(httpx.Response(401, json={"error": {"message": "Incorrect API key provided"}}))
    prisma = _fake_prisma([{"date": "2026-09-22", "spend": 5.0}])
    alert = AsyncMock()
    publish = MagicMock()

    (result,) = await run_spend_capture_rate_check(
        prisma, SpendCaptureRateCheckSettings(), alert=alert, publish=publish, today=date(2026, 9, 23), http_get=api
    )

    assert result == ProviderBillingRequestFailed("openai", "HTTP 401: " + api.responses[0].text[:300])
    publish.assert_called_once_with("openai", None)
    alert.assert_awaited_once()
    assert "HTTP 401" in alert.await_args.args[0]
    prisma.db.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_check_publishes_no_rate_when_the_provider_billed_nothing(monkeypatch):
    monkeypatch.setenv(OPENAI_ADMIN_KEY_ENV_VAR, _ADMIN_KEY)
    api = _FakeCostsApi(_page())
    prisma = _fake_prisma([{"date": "2026-09-22", "spend": 5.0}])
    publish = MagicMock()
    alert = AsyncMock()

    (report,) = await run_spend_capture_rate_check(
        prisma,
        SpendCaptureRateCheckSettings(lookback_days=1),
        alert=alert,
        publish=publish,
        today=date(2026, 9, 23),
        http_get=api,
    )

    assert isinstance(report, CaptureRateReport) and report.capture_rate is None
    publish.assert_called_once_with("openai", None)
    alert.assert_not_awaited()


def _pod_lock(acquired: bool) -> MagicMock:
    lock = MagicMock()
    lock.redis_cache = MagicMock()
    lock.redis_cache.async_get_cache = AsyncMock(return_value="other-pod")
    lock.get_redis_lock_key = MagicMock(return_value="lock-key")
    lock.acquire_lock = AsyncMock(return_value=acquired)
    lock.release_lock = AsyncMock()
    return lock


async def _scheduled_run(
    lock: MagicMock, monkeypatch, *, captured: float, prisma: MagicMock | None = None
) -> tuple[AsyncMock, MagicMock]:
    monkeypatch.setenv(OPENAI_ADMIN_KEY_ENV_VAR, _ADMIN_KEY)
    alert = AsyncMock()
    publish = MagicMock()
    await run_scheduled_spend_capture_rate_check(
        prisma or _fake_prisma([{"date": "2026-09-22", "spend": captured}]),
        SpendCaptureRateCheckSettings(lookback_days=1),
        pod_lock_manager=lock,
        alert=alert,
        publish=publish,
        today=date(2026, 9, 23),
        http_get=_FakeCostsApi(_page(_bucket("2026-09-22", 200.0))),
    )
    return alert, publish


async def _scheduled_run_under_threshold(lock: MagicMock, monkeypatch) -> tuple[AsyncMock, MagicMock]:
    return await _scheduled_run(lock, monkeypatch, captured=50.0)


@pytest.mark.asyncio
async def test_a_healthy_scheduled_check_publishes_and_never_touches_the_alert_lock(monkeypatch):
    lock = _pod_lock(acquired=False)

    alert, publish = await _scheduled_run(lock, monkeypatch, captured=190.0)

    publish.assert_called_once_with("openai", 0.95)
    alert.assert_not_awaited()
    lock.acquire_lock.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_scheduled_check_that_fails_never_claims_the_alert_window(monkeypatch):
    lock = _pod_lock(acquired=True)
    prisma = MagicMock()
    prisma.db.query_raw = AsyncMock(side_effect=RuntimeError("database gone"))

    with pytest.raises(RuntimeError, match="database gone"):
        await _scheduled_run(lock, monkeypatch, captured=0.0, prisma=prisma)

    lock.acquire_lock.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduled_check_publishes_but_stays_quiet_when_another_pod_holds_the_alert_window(monkeypatch):
    lock = _pod_lock(acquired=False)

    alert, publish = await _scheduled_run_under_threshold(lock, monkeypatch)

    publish.assert_called_once_with("openai", 0.25)
    alert.assert_not_awaited()
    lock.release_lock.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduled_check_alerts_and_keeps_the_lock_until_it_expires_when_it_wins(monkeypatch):
    lock = _pod_lock(acquired=True)

    alert, publish = await _scheduled_run_under_threshold(lock, monkeypatch)

    lock.acquire_lock.assert_awaited_once_with(cronjob_id=SPEND_CAPTURE_RATE_CHECK_JOB_ID, ttl=900)
    lock.release_lock.assert_not_awaited()
    publish.assert_called_once_with("openai", 0.25)
    alert.assert_awaited_once()


@pytest.mark.asyncio
async def test_scheduled_check_alerts_when_the_lock_cannot_be_acquired_or_read(monkeypatch):
    lock = _pod_lock(acquired=False)
    lock.redis_cache.async_get_cache = AsyncMock(side_effect=ConnectionError("redis down"))

    alert, publish = await _scheduled_run_under_threshold(lock, monkeypatch)

    publish.assert_called_once_with("openai", 0.25)
    alert.assert_awaited_once()


@pytest.mark.asyncio
async def test_scheduled_check_alerts_without_a_lock_manager(monkeypatch):
    lock = _pod_lock(acquired=False)
    lock.redis_cache = None

    alert, publish = await _scheduled_run_under_threshold(lock, monkeypatch)

    lock.acquire_lock.assert_not_awaited()
    publish.assert_called_once_with("openai", 0.25)
    alert.assert_awaited_once()


def test_settings_reject_typos_and_out_of_range_values():
    with pytest.raises(ValidationError, match="threshhold"):
        SpendCaptureRateCheckSettings.model_validate({"threshhold": 0.9})
    with pytest.raises(ValidationError, match="threshold"):
        SpendCaptureRateCheckSettings.model_validate({"threshold": 1.5})
    with pytest.raises(ValidationError, match="providers"):
        SpendCaptureRateCheckSettings.model_validate({"providers": []})
    with pytest.raises(ValidationError, match="providers"):
        SpendCaptureRateCheckSettings.model_validate({"providers": ["anthropic"]})
    with pytest.raises(ValidationError, match="lookback_days"):
        SpendCaptureRateCheckSettings.model_validate({"lookback_days": SPEND_CAPTURE_RATE_MAX_RANGE_DAYS + 1})
    parsed = SpendCaptureRateCheckSettings.model_validate(
        json.loads('{"providers": ["openai"], "threshold": 0.8, "lookback_days": 3, "openai_project_ids": ["p"]}')
    )
    assert (parsed.threshold, parsed.lookback_days, parsed.openai_project_ids) == (0.8, 3, ("p",))


_capture_postgresql_proc: Final = factories.postgresql_proc()
_capture_postgresql: Final = factories.postgresql("_capture_postgresql_proc")

_DAILY_USER_SPEND_DDL: Final = """
    CREATE TABLE "LiteLLM_DailyUserSpend" (
        id TEXT PRIMARY KEY,
        date TEXT NOT NULL,
        custom_llm_provider TEXT,
        spend DOUBLE PRECISION DEFAULT 0
    )
"""


class _PsycopgPrisma:
    """``prisma_client.db.query_raw`` on a real connection, with ``$n`` placeholders converted for psycopg."""

    def __init__(self, conn: psycopg.Connection) -> None:
        self.db = self
        self._conn = conn

    async def query_raw(self, sql: str, *params: object) -> list[dict[str, object]]:
        converted: Final = re.sub(r"\$(\d+)", r"%(p\1)s", sql)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                converted,  # pyright: ignore[reportArgumentType]  # psycopg stubs want a literal-typed query
                {f"p{i}": list(v) if isinstance(v, tuple) else v for i, v in enumerate(params, start=1)},
            )
            return cur.fetchall()


@pytest.mark.asyncio
async def test_captured_spend_sums_only_the_openai_billed_providers_inside_the_window(
    _capture_postgresql: psycopg.Connection,
):
    conn: Final = _capture_postgresql
    conn.execute(_DAILY_USER_SPEND_DDL)  # pyright: ignore[reportArgumentType]  # DDL literal
    rows: Final = (
        ("2026-09-19", "openai", 1.0),
        ("2026-09-20", "openai", 2.0),
        ("2026-09-20", "openai", 3.0),
        ("2026-09-20", "text-completion-openai", 0.5),
        ("2026-09-20", "anthropic", 100.0),
        ("2026-09-21", "azure", 100.0),
        ("2026-09-22", "openai", 4.0),
    )
    for index, (day, provider, spend) in enumerate(rows):
        conn.execute(
            'INSERT INTO "LiteLLM_DailyUserSpend" (id, date, custom_llm_provider, spend) VALUES (%s, %s, %s, %s)',
            (f"row-{index}", day, provider, spend),
        )
    conn.commit()

    captured = await captured_spend_by_day(
        _PsycopgPrisma(conn),  # pyright: ignore[reportArgumentType]  # duck-typed prisma for the raw query
        litellm_providers=("openai", "text-completion-openai"),
        start_date=date(2026, 9, 20),
        end_date=date(2026, 9, 21),
    )

    assert dict(captured) == {"2026-09-20": 5.5}
