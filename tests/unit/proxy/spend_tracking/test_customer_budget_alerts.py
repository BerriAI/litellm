from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
import pytest_asyncio
import respx
from pydantic import TypeAdapter

import litellm
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.management_helpers.customer_hierarchy import PrismaRecord
from litellm.proxy.spend_tracking.customer_budget_alerts import send_customer_budget_alerts
from litellm.proxy.utils import ProxyLogging
from litellm.types.integrations.slack_alerting import AlertType

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


def _database(records: Mapping[str, PrismaRecord | Exception | None]) -> _Database:
    table: Final = _Table(records)
    return _Database(table, table, table)


@pytest_asyncio.fixture
async def alerts(monkeypatch: pytest.MonkeyPatch) -> ProxyLogging:
    monkeypatch.setenv("WEBHOOK_URL", _URL)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    logging: Final = ProxyLogging(user_api_key_cache=UserApiKeyCache())
    logging.update_values(alerting=["webhook"], alert_types=[AlertType.budget_alerts], redis_cache=None)
    return logging


@pytest.mark.parametrize("expires", (None, _NOW - timedelta(seconds=1)))
@pytest.mark.asyncio
async def test_customer_member_alert_uses_effective_override_and_customer_identity(
    alerts: ProxyLogging, respx_mock: respx.MockRouter, expires: datetime | None
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
    await send_customer_budget_alerts(
        {"customer_member_list_transactions": {"customer-member-1": 3.0}},
        _database({"customer-member-1": record}),
        alerts,
        now=_NOW,
    )
    payloads: Final = tuple(_PAYLOAD.validate_json(call.request.content) for call in receiver.calls)
    if expires is not None:
        assert payloads == ()
        return
    assert {item["event"] for item in payloads} == {"budget_crossed", "soft_budget_crossed"}
    for payload in payloads:
        assert payload["max_budget"] == 2.0
        assert payload["soft_budget"] == 1.0
        assert payload["customer_org_id"] == "customer-org-1"
        assert payload["customer_org_alias"] == "customer-test"
        assert payload["customer_team_id"] == "customer-team-1"
        assert payload["customer_member_id"] == "customer-member-1"
        assert "team_id" not in payload
        assert "organization_id" not in payload


@pytest.mark.parametrize("cost", (0.0, -1.0))
@pytest.mark.asyncio
async def test_free_or_refunded_customer_spend_does_not_send_alerts(
    alerts: ProxyLogging, respx_mock: respx.MockRouter, cost: float
) -> None:
    receiver: Final = respx_mock.post(_URL).respond(200)
    await send_customer_budget_alerts(
        {"customer_org_list_transactions": {"customer-org-1": cost}},
        _database({"customer-org-1": RuntimeError("No lookup should be needed")}),
        alerts,
        now=_NOW,
    )
    assert receiver.call_count == 0


@pytest.mark.parametrize("bad", (None, RuntimeError("database unavailable"), _Record({"spend": "invalid"})))
@pytest.mark.asyncio
async def test_one_unreadable_customer_does_not_hide_another_customers_alerts(
    alerts: ProxyLogging, respx_mock: respx.MockRouter, bad: PrismaRecord | Exception | None
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
    await send_customer_budget_alerts(
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
    alerts: ProxyLogging, respx_mock: respx.MockRouter
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
            await send_customer_budget_alerts(
                {"customer_org_list_transactions": {"customer-org-1": 1.0}},
                _database({"customer-org-1": record}),
                alerts,
                now=_NOW,
            )
    payloads: Final = tuple(_PAYLOAD.validate_json(call.request.content) for call in receiver.calls)
    assert len(payloads) == 4
    assert {item["customer_budget_reset_at"] for item in payloads} == {
        _NOW.isoformat().replace("+00:00", "Z"),
        (_NOW + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
    }


@pytest.mark.parametrize("transactions", ({}, {"customer_org_list_transactions": {"customer-org-1": 1.0}}))
@pytest.mark.asyncio
async def test_no_customer_budget_does_not_send_an_alert(
    alerts: ProxyLogging, respx_mock: respx.MockRouter, transactions: Mapping[str, object]
) -> None:
    receiver: Final = respx_mock.post(_URL).respond(200)
    await send_customer_budget_alerts(
        transactions,
        _database({"customer-org-1": _Record({"spend": 2.0})}),
        alerts,
        now=_NOW,
    )
    assert receiver.call_count == 0
