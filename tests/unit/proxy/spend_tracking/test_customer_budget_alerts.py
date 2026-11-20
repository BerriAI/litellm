import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
import pytest_asyncio
import respx
from pydantic import TypeAdapter

import litellm
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.common_utils.callback_utils import initialize_callbacks_on_proxy
from litellm.proxy.db.budget_window_spend_writer import CommittedWindowSpend
from litellm.proxy.management_helpers.apps import AppStore
from litellm.proxy.management_helpers.customer_hierarchy import PrismaRecord
from litellm.proxy.spend_tracking.customer_budget_alerts import CustomerBudgetAlerting
from litellm.types.integrations.customer_budget import CustomerBudgetEvent, CustomerBudgetScope
from litellm.types.proxy.management_endpoints.apps import App
from tests.unit.proxy.management_helpers import test_apps
from tests.unit.proxy.management_helpers.test_apps import AppDatabase

app_store: Final = test_apps.app_store

_NOW: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
_PAYLOAD: Final = TypeAdapter(Mapping[str, object])
_URL: Final = "https://alerts.example/customer-budget"
_ROWS: Final[Mapping[CustomerBudgetScope, Mapping[str, object]]] = {
    "customer_org": {"customer_org_id": "org-1", "alias": "customer-test"},
    "customer_team": {
        "customer_team_id": "team-1",
        "alias": "customer-team",
        "parent_org": {"customer_org_id": "org-1", "alias": "customer-test"},
    },
    "customer_member": {
        "customer_member_id": "member-1",
        "alias": "customer-member",
        "parent_org": {"customer_org_id": "org-1", "alias": "customer-test"},
        "parent_team_ids": ["team-1", "team-2"],
    },
}


@pytest.mark.parametrize("scope", ("app", "app_customer_org", "app_customer_member"))
@pytest.mark.parametrize(
    "duration,reset_at,charged_reset",
    (
        (None, None, None),
        ("1mo", None, None),
        ("12mo", None, None),
        ("7d", None, None),
        ("1mo", _NOW.replace(year=2028), None),
        ("12mo", _NOW.replace(year=2028), None),
        ("1mo", _NOW.replace(year=2028), _NOW.replace(month=3, day=31)),
        ("12mo", _NOW.replace(year=2028), _NOW.replace(year=2028, month=2, day=29)),
    ),
)
@pytest.mark.parametrize("hard_limit", (None, 10.0))
@pytest.mark.parametrize("delayed", (False, True))
@pytest.mark.parametrize("legacy", (False, True))
@pytest.mark.asyncio
async def test_app_alerts_use_inherited_limits_and_current_spend_without_repeating(
    app_store: tuple[AppDatabase, AppStore, App],
    alerts: DualCache,
    respx_mock: respx.MockRouter,
    scope: CustomerBudgetScope,
    duration: str | None,
    reset_at: datetime | None,
    charged_reset: datetime | None,
    hard_limit: float | None,
    delayed: bool,
    legacy: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, store, app = app_store
    receiver: Final = respx_mock.post(_URL).respond(200)
    await database.litellm_budgettable.create(
        {
            "budget_id": "policy",
            "max_budget": hard_limit,
            "soft_budget": 5,
            "budget_duration": duration,
            "budget_limits": [{"max_budget": hard_limit, "budget_duration": duration, "reset_at": reset_at.isoformat()}]
            if hard_limit is not None and reset_at is not None
            else [],
        }
    )
    await database.litellm_apptable.update(
        {"app_id": app.app_id},
        {"budget_id": "policy", "default_tier_budget_id": "policy", "default_customer_member_budget_id": "policy"},
    )
    await database.litellm_appcustomerorgtable.create({"app_id": app.app_id, "customer_org_id": "org", "spend": 1000})
    await database.litellm_appcustomermembertable.create(
        {"app_id": app.app_id, "customer_member_id": "member", "spend": 1000}
    )
    entity_id: Final = app.app_id + (
        ":org" if scope == "app_customer_org" else ":member" if scope == "app_customer_member" else ""
    )
    charged_start: Final = (
        _NOW.replace(year=2027 if duration == "12mo" else 2026, month=2, day=28)
        if charged_reset is not None
        else _NOW - timedelta(days=3)
        if duration == "7d"
        else _NOW
    )
    for spend in (4, 6, 11, 12):
        if duration is None:
            for table in (
                database.litellm_apptable,
                database.litellm_appcustomerorgtable,
                database.litellm_appcustomermembertable,
            ):
                await table.update_many({}, {"spend": spend})
        windows: Final = (
            (
                CommittedWindowSpend(
                    entity_type=scope,
                    entity_id=entity_id,
                    window_duration=duration,
                    window_start=charged_start,
                    spend=spend,
                    reset_at=charged_reset,
                ),
            )
            if duration is not None
            else ()
        )
        instant: Final = _NOW + timedelta(days=400 if delayed else 0, seconds=spend)
        monkeypatch.setattr("time.time", instant.timestamp)
        if legacy and duration is not None:
            await CustomerBudgetAlerting.detect_committed_windows(
                windows, _RoutedClient(_database({}), database), alerts, now=instant
            )
        else:
            await CustomerBudgetAlerting.detect_committed_batch(
                {f"{scope}_list_transactions": {entity_id: 1}},
                _RoutedClient(_database({}), database),
                alerts,
                now=instant,
                windows=windows,
            )
    payloads: Final = tuple(_PAYLOAD.validate_json(call.request.content) for call in receiver.calls)
    expected: Final = (("soft_budget_crossed", 6, 5),) + (
        (("budget_crossed", 11, 10),) if hard_limit is not None else ()
    )
    assert tuple((item["event"], item["spend"], item["budget"]) for item in payloads) == expected
    assert {(item["scope"], item["app_id"], item["app_alias"], item["budget_duration"]) for item in payloads} == {
        (scope, app.app_id, "Chat", duration)
    }
    assert {item["customer_org_id"] for item in payloads} == ({"org"} if scope == "app_customer_org" else {None})
    assert {item["customer_member_id"] for item in payloads} == (
        {"member"} if scope == "app_customer_member" else {None}
    )
    assert {item["period"] for item in payloads} == {
        charged_reset.isoformat()
        if charged_reset is not None
        else {
            "1mo": "2026-02-01T00:00:00+00:00",
            "12mo": "2027-01-01T00:00:00+00:00",
            "7d": "2026-01-05T00:00:00+00:00",
        }.get(duration, "lifetime")
    }
    if duration is not None:
        await CustomerBudgetAlerting.detect_committed_batch(
            {f"{scope}_list_transactions": {entity_id: 1}}, database, alerts, now=_NOW + timedelta(days=400)
        )
        assert len(receiver.calls) == len(expected)
    if duration is not None and delayed:
        rolled: Final = windows[0].model_copy(
            update={
                "window_start": _NOW.replace(year=2030 if charged_reset else 2027, day=4 if duration == "7d" else 1),
                "reset_at": None,
            }
        )
        for seconds in (15, 18):
            delivered_at: Final = _NOW + timedelta(days=400, seconds=seconds)
            monkeypatch.setattr("time.time", delivered_at.timestamp)
            await CustomerBudgetAlerting.detect_committed_windows(
                (windows[0], rolled), database, alerts, now=delivered_at
            )
        latest: Final = tuple(_PAYLOAD.validate_json(call.request.content) for call in receiver.calls)
        assert len(latest) == len(expected) * 2
        assert len({(item["event"], item["period"]) for item in latest}) == len(expected) * 2


@pytest.mark.parametrize("scope", ("app_customer_org", "app_customer_member"))
@pytest.mark.parametrize("expired", (False, True))
@pytest.mark.asyncio
async def test_app_alerts_respect_custom_budgets_and_temporary_increase_expiry(
    app_store: tuple[AppDatabase, AppStore, App],
    alerts: DualCache,
    respx_mock: respx.MockRouter,
    scope: CustomerBudgetScope,
    expired: bool,
) -> None:
    database, store, app = app_store
    receiver: Final = respx_mock.post(_URL).respond(200)
    for identity, maximum in (("default", 1), ("custom", 10), ("increase", 20)):
        await database.litellm_budgettable.create(
            {"budget_id": identity, "max_budget": maximum, "soft_budget": maximum / 2}
        )
    await database.litellm_apptable.update(
        {"app_id": app.app_id}, {"default_tier_budget_id": "default", "default_customer_member_budget_id": "default"}
    )
    fields: Final = {
        "app_id": app.app_id,
        "spend": 12,
        "budget_id": "custom",
        "budget_override_id": "increase",
        "budget_override_expires_at": _NOW + timedelta(seconds=-1 if expired else 1),
    }
    await database.litellm_appcustomerorgtable.create({**fields, "customer_org_id": "org"})
    await database.litellm_appcustomermembertable.create({**fields, "customer_member_id": "member"})
    await CustomerBudgetAlerting.detect_committed_batch(
        {f"{scope}_list_transactions": {app.app_id + (":org" if scope == "app_customer_org" else ":member"): 1}},
        database,
        alerts,
        now=_NOW,
    )
    payloads: Final = tuple(_PAYLOAD.validate_json(call.request.content) for call in receiver.calls)
    assert {(item["event"], item["budget"]) for item in payloads} == (
        {("soft_budget_crossed", 5), ("budget_crossed", 10)} if expired else {("soft_budget_crossed", 10)}
    )


@dataclass(frozen=True, slots=True)
class _Record:
    fields: Mapping[str, object]

    def model_dump(self) -> Mapping[str, object]:
        return self.fields


@dataclass(frozen=True, slots=True)
class _Table:
    records: Mapping[str, PrismaRecord | Exception | None]
    relations: frozenset[str]

    async def find_unique(
        self, where: Mapping[str, object], include: Mapping[str, object] | None = None
    ) -> PrismaRecord | None:
        if include is not None and not include.keys() <= self.relations:
            raise ValueError("Unknown customer relation")
        entity_id: Final = next(iter(where.values()))
        record: Final = self.records.get(entity_id) if isinstance(entity_id, str) else None
        if isinstance(record, Exception):
            raise record
        return record


@dataclass(frozen=True, slots=True)
class _Database:
    litellm_customerorganizationtable: _Table
    litellm_customerteamtable: _Table
    litellm_customermembertable: _Table


@dataclass(frozen=True, slots=True)
class _RoutedClient:
    db: _Database | AppDatabase
    writer_db: _Database | AppDatabase


def _database(records: Mapping[str, PrismaRecord | Exception | None]) -> _Database:
    return _Database(
        _Table(records, frozenset(("budget",))),
        _Table(records, frozenset(("budget", "budget_override", "parent_org"))),
        _Table(records, frozenset(("budget", "budget_override", "parent_org"))),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("customer_org", "customer_team", "customer_member"))
async def test_committed_crossing_reads_writer_spend_when_the_replica_lags(
    scope: CustomerBudgetScope, alerts: DualCache, respx_mock: respx.MockRouter
) -> None:
    receiver: Final = respx_mock.post(_URL).respond(200)
    fields: Final = {
        **_ROWS[scope],
        "budget": {"max_budget": 1.0},
    }
    database: Final = _RoutedClient(
        db=_database({"entity-1": _Record({**fields, "spend": 0.0})}),
        writer_db=_database({"entity-1": _Record({**fields, "spend": 2.0})}),
    )
    await CustomerBudgetAlerting.detect_committed_batch(
        {f"{scope}_list_transactions": {"entity-1": 2.0}}, database, alerts, now=_NOW
    )
    payloads: Final = tuple(_PAYLOAD.validate_json(call.request.content) for call in receiver.calls)
    assert tuple((event["event"], event["scope"], event["spend"], event["budget"]) for event in payloads) == (
        ("budget_crossed", scope, 2.0, 1.0),
    )


@pytest_asyncio.fixture
async def alerts(monkeypatch: pytest.MonkeyPatch) -> DualCache:
    monkeypatch.setenv("CUSTOMER_BUDGET_WEBHOOK_URL", _URL)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "callbacks", [])
    initialize_callbacks_on_proxy(["customer_budget_webhook"], False, "config.yaml", {})
    return DualCache()


@pytest.mark.parametrize("expires", (None, _NOW - timedelta(seconds=1)))
@pytest.mark.parametrize("customer_member_id", ("customer-member-1", "<!channel> <http://x|y> &"))
@pytest.mark.asyncio
async def test_customer_member_alert_uses_effective_override_and_customer_identity(
    alerts: DualCache, respx_mock: respx.MockRouter, expires: datetime | None, customer_member_id: str
) -> None:
    receiver: Final = respx_mock.post(_URL).respond(200)
    record: Final = _Record(
        {
            "spend": 3.0,
            "budget": {"max_budget": 10.0, "soft_budget": 8.0},
            "budget_override": {"max_budget": 2.0, "soft_budget": 1.0},
            "budget_override_expires_at": expires,
            "parent_org": {"customer_org_id": "customer-org-1", "alias": "customer-test"},
            "parent_team_ids": ["customer-team-1", "customer-team-2"],
        }
    )
    await CustomerBudgetAlerting.detect_committed_batch(
        {"customer_member_list_transactions": {customer_member_id: 3.0}},
        _database({customer_member_id: record}),
        alerts,
        now=_NOW,
    )
    payloads: Final = tuple(_PAYLOAD.validate_json(call.request.content) for call in receiver.calls)
    if expires is not None:
        assert payloads == ()
        return
    assert {item["event"] for item in payloads} == {"budget_crossed", "soft_budget_crossed"}
    for payload in payloads:
        assert payload["budget"] == (2.0 if payload["event"] == "budget_crossed" else 1.0)
        assert payload["scope"] == "customer_member"
        assert payload["period"] == "lifetime"
        assert payload["customer_org_id"] == "customer-org-1"
        assert payload["customer_org_alias"] == "customer-test"
        assert payload["customer_team_id"] is None
        assert payload["customer_team_alias"] is None
        assert payload["customer_member_id"] == customer_member_id
        assert "team_id" not in payload
        assert "organization_id" not in payload


@pytest.mark.parametrize("cost", (0.0, -1.0))
@pytest.mark.asyncio
async def test_free_or_refunded_customer_spend_does_not_send_alerts(
    alerts: DualCache, respx_mock: respx.MockRouter, cost: float
) -> None:
    receiver: Final = respx_mock.post(_URL).respond(200)
    await CustomerBudgetAlerting.detect_committed_batch(
        {"customer_org_list_transactions": {"customer-org-1": cost}},
        _database({"customer-org-1": RuntimeError("No lookup should be needed")}),
        alerts,
        now=_NOW,
    )
    assert receiver.call_count == 0


@pytest.mark.parametrize("bad", (None, RuntimeError("database unavailable"), _Record({"spend": "invalid"})))
@pytest.mark.asyncio
async def test_one_unreadable_customer_does_not_hide_another_customers_alerts(
    alerts: DualCache, respx_mock: respx.MockRouter, bad: PrismaRecord | Exception | None
) -> None:
    receiver: Final = respx_mock.post(_URL).respond(200)
    good: Final = _Record(
        {
            "customer_org_id": "customer-org-good",
            "alias": "customer-test",
            "spend": 2.0,
            "budget": {"max_budget": 1.0},
        }
    )
    await CustomerBudgetAlerting.detect_committed_batch(
        {"customer_org_list_transactions": {"customer-org-bad": 1.0, "customer-org-good": 1.0}},
        _database({"customer-org-bad": bad, "customer-org-good": good}),
        alerts,
        now=_NOW,
    )
    payloads: Final = tuple(_PAYLOAD.validate_json(call.request.content) for call in receiver.calls)
    assert len(payloads) == 1
    assert payloads[0]["entity_id"] == "customer-org-good"


@pytest.mark.parametrize("scope", ("customer_org", "customer_team", "customer_member"))
@pytest.mark.asyncio
async def test_customer_budget_reset_allows_the_next_periods_crossing(
    scope: CustomerBudgetScope, alerts: DualCache, respx_mock: respx.MockRouter
) -> None:
    receiver: Final = respx_mock.post(_URL).respond(200)
    for reset in (_NOW + timedelta(days=1), _NOW + timedelta(days=2)):
        record: Final = _Record(
            {
                **_ROWS[scope],
                "spend": 2.0,
                "budget": {
                    "max_budget": 1.0,
                    "soft_budget": 0.5,
                    "budget_duration": "1d",
                    "budget_reset_at": reset,
                },
            }
        )
        for _ in range(2):
            await CustomerBudgetAlerting.detect_committed_batch(
                {f"{scope}_list_transactions": {"entity-1": 1.0}},
                _database({"entity-1": record}),
                alerts,
                now=_NOW,
            )
    payloads: Final = tuple(_PAYLOAD.validate_json(call.request.content) for call in receiver.calls)
    assert len(payloads) == 4
    assert {item["period"] for item in payloads} == {
        (_NOW + timedelta(days=1)).isoformat(),
        (_NOW + timedelta(days=2)).isoformat(),
    }
    assert {item["customer_org_id"] for item in payloads} == {"org-1"}
    assert {item["customer_team_id"] for item in payloads} == ({"team-1"} if scope == "customer_team" else {None})


@pytest.mark.parametrize("scope", ("customer_team", "customer_member"))
@pytest.mark.parametrize("expires", (None, _NOW + timedelta(hours=1), _NOW - timedelta(seconds=1)))
@pytest.mark.parametrize("override_duration", (None, "1d"))
@pytest.mark.asyncio
async def test_override_uses_its_effective_budget_and_period(
    scope: CustomerBudgetScope,
    expires: datetime | None,
    override_duration: str | None,
    alerts: DualCache,
    respx_mock: respx.MockRouter,
) -> None:
    receiver: Final = respx_mock.post(_URL).respond(200)
    template_reset: Final = _NOW + timedelta(days=30)
    override_reset: Final = _NOW + timedelta(days=1)
    record: Final = _Record(
        {
            **_ROWS[scope],
            "spend": 20.0,
            "budget": {
                "max_budget": 10.0,
                "budget_duration": "30d",
                "budget_reset_at": template_reset,
            },
            "budget_override": {
                "max_budget": 1.0,
                "budget_duration": override_duration,
                "budget_reset_at": override_reset,
            },
            "budget_override_expires_at": expires,
        }
    )
    await CustomerBudgetAlerting.detect_committed_batch(
        {f"{scope}_list_transactions": {"entity-1": 1.0}}, _database({"entity-1": record}), alerts, now=_NOW
    )
    expired: Final = expires is not None and expires <= _NOW
    expected: Final = template_reset if expired or override_duration is None else override_reset
    assert tuple(
        (payload["period"], payload["budget"])
        for call in receiver.calls
        for payload in (_PAYLOAD.validate_json(call.request.content),)
    ) == ((expected.isoformat(), 10.0 if expired else 1.0),)


@pytest.mark.parametrize("transactions", ({}, {"customer_org_list_transactions": {"customer-org-1": 1.0}}))
@pytest.mark.asyncio
async def test_no_customer_budget_does_not_send_an_alert(
    alerts: DualCache, respx_mock: respx.MockRouter, transactions: Mapping[str, object]
) -> None:
    receiver: Final = respx_mock.post(_URL).respond(200)
    await CustomerBudgetAlerting.detect_committed_batch(
        transactions,
        _database({"customer-org-1": _Record({"spend": 2.0})}),
        alerts,
        now=_NOW,
    )
    assert receiver.call_count == 0


@pytest.mark.parametrize("reset", (None, _NOW + timedelta(days=30, microseconds=1)))
@pytest.mark.asyncio
async def test_crossings_do_not_repeat_after_a_day_or_near_the_end_of_the_budget_period(
    alerts: DualCache, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, reset: datetime | None
) -> None:
    receiver: Final = respx_mock.post(_URL).respond(200)
    database: Final = _database(
        {
            "org": _Record(
                {
                    "customer_org_id": "org",
                    "spend": 2.0,
                    "budget": {
                        "max_budget": 1.0,
                        "soft_budget": 0.5,
                        "budget_duration": "30d" if reset is not None else None,
                        "budget_reset_at": reset,
                    },
                }
            )
        }
    )
    last_instant: Final = reset - timedelta(microseconds=1) if reset else _NOW + timedelta(days=365)
    for instant in (_NOW, _NOW + timedelta(days=2), last_instant):
        monkeypatch.setattr("time.time", instant.timestamp)
        await CustomerBudgetAlerting.detect_committed_batch(
            {"customer_org_list_transactions": {"org": 1.0}}, database, alerts, now=instant
        )
    assert receiver.call_count == 2
    assert tuple(_PAYLOAD.validate_json(call.request.content)["event"] for call in receiver.calls) == (
        "soft_budget_crossed",
        "budget_crossed",
    )


class _BrokenCallback(CustomLogger):
    async def async_on_customer_budget_event(self, event: CustomerBudgetEvent) -> None:
        raise RuntimeError("callback unavailable")


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("app", "app_customer_org", "app_customer_member"))
async def test_app_warning_claim_survives_reset_after_an_early_period_warning(
    scope: CustomerBudgetScope, alerts: DualCache, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    receiver: Final = respx_mock.post(_URL).respond(200)
    reset: Final = _NOW.replace(month=2)
    for instant in (_NOW, reset + timedelta(seconds=2), reset + timedelta(hours=2)):
        monkeypatch.setattr("time.time", instant.timestamp)
        await CustomerBudgetAlerting.emit(
            CustomerBudgetEvent(
                event="soft_budget_crossed",
                scope=scope,
                entity_id="app:member",
                spend=6,
                budget=5,
                budget_duration="1mo",
                period=reset.isoformat(),
                timestamp=instant,
            ),
            alerts,
        )
    assert receiver.call_count == 1


@pytest.mark.asyncio
async def test_a_broken_callback_does_not_hide_another_callback_or_repeat_a_crossing(
    alerts: DualCache, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    receiver: Final = respx_mock.post(_URL).respond(200)
    webhook: Final = litellm.callbacks[0]
    monkeypatch.setattr(litellm, "callbacks", [_BrokenCallback(), webhook])
    monkeypatch.setattr(litellm, "success_callback", [webhook])
    event: Final = CustomerBudgetEvent(
        event="budget_crossed",
        scope="customer_org",
        entity_id="org",
        customer_org_id="org",
        spend=2,
        budget=1,
        period="lifetime",
        timestamp=_NOW,
    )
    await asyncio.gather(*(CustomerBudgetAlerting.emit(event, alerts) for _ in range(20)))
    assert receiver.call_count == 1
    assert _PAYLOAD.validate_json(receiver.calls[0].request.content) == event.model_dump(mode="json")


@pytest.mark.asyncio
async def test_no_interested_callback_does_not_read_customer_rows(
    alerts: DualCache, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [CustomLogger()])
    receiver: Final = respx_mock.post(_URL).respond(200)
    await CustomerBudgetAlerting.detect_committed_batch(
        {"customer_org_list_transactions": {"org": 2.0}},
        _database({"org": RuntimeError("must not look up customers without a callback")}),
        alerts,
        now=_NOW,
    )
    assert receiver.call_count == 0
