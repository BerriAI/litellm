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
from litellm.proxy.management_helpers.customer_hierarchy import PrismaRecord
from litellm.proxy.spend_tracking.customer_budget_alerts import CustomerBudgetAlerting
from litellm.types.integrations.customer_budget import CustomerBudgetEvent

_NOW: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
_PAYLOAD: Final = TypeAdapter(Mapping[str, object])
_URL: Final = "https://alerts.example/customer-budget"


@dataclass(frozen=True, slots=True)
class _Record:
    fields: Mapping[str, object]

    def model_dump(self) -> Mapping[str, object]:
        return self.fields


@dataclass(frozen=True, slots=True)
class _Table:
    records: Mapping[str, PrismaRecord | Exception | None]

    async def find_unique(
        self, where: Mapping[str, object], include: Mapping[str, object] | None = None
    ) -> PrismaRecord | None:
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
    db: _Database
    writer_db: _Database


def _database(records: Mapping[str, PrismaRecord | Exception | None]) -> _Database:
    table: Final = _Table(records)
    return _Database(table, table, table)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("customer_org", "customer_team", "customer_member"))
async def test_committed_crossing_reads_writer_spend_when_the_replica_lags(
    scope: str, alerts: DualCache, respx_mock: respx.MockRouter
) -> None:
    receiver: Final = respx_mock.post(_URL).respond(200)
    fields: Final = {
        "customer_org_id": "org-1",
        "customer_team_id": "team-1",
        "parent_org": {"customer_org_id": "org-1"},
        "parent_team": {"customer_team_id": "team-1", "parent_org": {"customer_org_id": "org-1"}},
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
            "parent_team": {
                "customer_team_id": "customer-team-1",
                "parent_org": {"customer_org_id": "customer-org-1", "alias": "customer-test"},
            },
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
        assert payload["customer_team_id"] == "customer-team-1"
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


@pytest.mark.asyncio
async def test_customer_budget_reset_allows_the_next_periods_crossing(
    alerts: DualCache, respx_mock: respx.MockRouter
) -> None:
    receiver: Final = respx_mock.post(_URL).respond(200)
    for reset in (_NOW, _NOW + timedelta(days=1)):
        record: Final = _Record(
            {
                "customer_org_id": "customer-org-1",
                "alias": "customer-test",
                "budget_reset_at": reset,
                "spend": 2.0,
                "budget": {"max_budget": 1.0, "soft_budget": 0.5},
            }
        )
        for _ in range(2):
            await CustomerBudgetAlerting.detect_committed_batch(
                {"customer_org_list_transactions": {"customer-org-1": 1.0}},
                _database({"customer-org-1": record}),
                alerts,
                now=_NOW,
            )
    payloads: Final = tuple(_PAYLOAD.validate_json(call.request.content) for call in receiver.calls)
    assert len(payloads) == 4
    assert {item["period"] for item in payloads} == {
        _NOW.isoformat(),
        (_NOW + timedelta(days=1)).isoformat(),
    }


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


class _BrokenCallback(CustomLogger):
    async def async_on_customer_budget_event(self, event: CustomerBudgetEvent) -> None:
        raise RuntimeError("callback unavailable")


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
