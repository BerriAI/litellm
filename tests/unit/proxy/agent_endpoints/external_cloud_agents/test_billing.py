import base64
from datetime import timedelta
from typing import Final

import httpx
import pytest

from litellm.proxy.agent_endpoints.external_cloud_agents.billing import (
    BillingItem,
    SpendSync,
    aggregate,
    fetch_page,
    window_start,
)
from litellm.proxy.agent_endpoints.external_cloud_agents.models import DailySpend, Failure, SyncResult
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from tests.unit.proxy.agent_endpoints.external_cloud_agents.conftest import DatabaseDouble, Harness, NOW


def item(**overrides: object) -> BillingItem:
    return BillingItem.model_validate(
        {
            "eventTime": (NOW - timedelta(hours=1)).isoformat(),
            "chargeCategory": "usage",
            "principal": {"type": "user", "id": "vendor-user"},
            "pricing": {"contracted": "0.1234567891"},
            "usage": {"prompt-text-tokens": "40", "completion-text-tokens": "6"},
            "displayExtras": {"userEmail": "Member@example.test", "displayLabel": "Member"},
            **overrides,
        }
    )


def test_aggregate_filters_and_sums_decimal_cost_strings_and_request_fallback() -> None:
    rows: Final = aggregate(
        (
            item(),
            item(usage={"requests": "5", "prompt-text-tokens": "3", "completion-text-tokens": "4"}),
            item(chargeCategory="credit"),
            item(principal={"type": "team", "id": "team"}),
            item(principal=None),
            item(eventTime=(NOW - timedelta(days=40)).isoformat()),
            item(eventTime=NOW.isoformat()),
        ),
        window_start(NOW),
        NOW,
    )
    assert rows == (
        DailySpend(
            vendor_user_id="vendor-user",
            email="Member@example.test",
            display_name="Member",
            day=NOW.date(),
            cost_micros=246914,
            prompt_tokens=43,
            completion_tokens=10,
            requests=6,
        ),
    )


def test_aggregate_uses_utc_day_and_accepts_empty_usage() -> None:
    rows: Final = aggregate((item(eventTime="2026-09-18T01:00:00+02:00", usage={}),), window_start(NOW), NOW)
    assert rows[0].day == (NOW - timedelta(days=1)).date()
    assert (rows[0].prompt_tokens, rows[0].completion_tokens, rows[0].requests) == (0, 0, 1)
    assert aggregate((), window_start(NOW), NOW) == ()


@pytest.mark.parametrize(
    "cost,micros", [("0.00000049", 0), ("0.0000005", 1), ("1.2345675", 1234568), ("-0.0000005", -1)]
)
def test_cost_is_rounded_exactly_to_nearest_micro_usd(cost: str, micros: int) -> None:
    rows: Final = aggregate((item(pricing={"contracted": cost}),), window_start(NOW), NOW)
    assert rows[0].cost_micros == micros


class Vendor:
    def __init__(self, pages: tuple[httpx.Response, ...], key: str = "vendor-test-key") -> None:
        self.pages = iter(pages)
        self.key = key
        self.requests: tuple[httpx.Request, ...] = ()

    def respond(self, request: httpx.Request) -> httpx.Response:
        self.requests = (*self.requests, request)
        assert request.headers["Authorization"] == "Basic " + base64.b64encode(f"{self.key}:".encode()).decode()
        assert request.url.params["product"] == "grok-bot"
        assert int(request.url.params["pageSize"]) == 500
        assert request.extensions["timeout"]["read"] > 0
        return next(self.pages)


def page(*items: BillingItem, more: bool = False, cursor: str | None = None) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": [row.model_dump(mode="json", by_alias=True) for row in items],
            "pagination": {"hasMore": more, **({"nextCursor": cursor} if cursor else {})},
        },
    )


@pytest.mark.asyncio
async def test_key_save_validates_once_encrypts_and_syncs_all_pages(db: DatabaseDouble) -> None:
    assert db.source is not None
    vendor: Final = Vendor((page(item(), more=True, cursor="page-2"), page(item(usage={"requests": "3"}))))
    async with httpx.AsyncClient(transport=httpx.MockTransport(vendor.respond)) as client:
        result: Final = await SpendSync(db, client, lambda: NOW, base_url="https://billing.example.test/").save_key(
            db.source.source_id, vendor.key
        )
    assert result == SyncResult(rows=1)
    assert db.source.admin_api_key != vendor.key
    assert decrypt_value_helper(db.source.admin_api_key, "admin_api_key") == vendor.key
    assert db.persisted[0].cost_micros == 246914
    assert db.persisted[0].requests == 4
    assert len(vendor.requests) == 2
    assert all(request.url.host == "billing.example.test" for request in vendor.requests)
    assert "cursor" not in vendor.requests[0].url.params
    assert vendor.requests[1].url.params["cursor"] == "page-2"
    assert vendor.requests[0].url.params["startTime"] == vendor.requests[1].url.params["startTime"]
    assert vendor.requests[0].url.params["endTime"] == vendor.requests[1].url.params["endTime"]


@pytest.mark.asyncio
async def test_invalid_key_returns_clear_failure_and_does_not_replace_existing_key(db: DatabaseDouble) -> None:
    assert db.source is not None
    original: Final = encrypt_value_helper("existing-valid-key")
    db.source = db.source.model_copy(update={"admin_api_key": original})
    vendor: Final = Vendor((httpx.Response(401, text="private-vendor-response"),))
    async with httpx.AsyncClient(transport=httpx.MockTransport(vendor.respond)) as client:
        result: Final = await SpendSync(db, client, lambda: NOW).save_key(db.source.source_id, vendor.key)
    assert result == Failure(status=400, message="Invalid GrokBot Admin API key")
    assert db.source.admin_api_key == original
    assert db.persisted == ()
    assert len(vendor.requests) == 1


def test_invalid_key_route_returns_400_without_storing_key(harness: Harness) -> None:
    response: Final = harness.client.put(
        f"/v1/external_cloud_agents/sources/{harness.source_id}/admin_api_key", json={"api_key": "bad-key"}
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid GrokBot Admin API key"}
    assert harness.db.source is not None and harness.db.source.admin_api_key is None


@pytest.mark.asyncio
async def test_rate_limit_stops_pagination_preserves_data_and_records_sanitized_retry_after(db: DatabaseDouble) -> None:
    assert db.source is not None
    key: Final = "vendor-test-key"
    db.source = db.source.model_copy(update={"admin_api_key": encrypt_value_helper(key)})
    vendor: Final = Vendor(
        (page(item(), more=True, cursor="next"), httpx.Response(429, headers={"Retry-After": "60"}, text=key))
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(vendor.respond)) as client:
        result: Final = await SpendSync(db, client, lambda: NOW).run(db.source.source_id)
    assert result == Failure(status=429, message="Vendor rate limit. Retry-After 60")
    assert db.persisted == ()
    assert db.writes[-1] == (db.source.source_id, result.message)
    assert key not in str(result)
    assert len(vendor.requests) == 2


@pytest.mark.asyncio
async def test_repeated_cursor_aborts_without_committing_partial_totals(db: DatabaseDouble) -> None:
    assert db.source is not None
    db.source = db.source.model_copy(update={"admin_api_key": encrypt_value_helper("vendor-test-key")})
    vendor: Final = Vendor((page(item(), more=True, cursor="same"), page(item(), more=True, cursor="same")))
    async with httpx.AsyncClient(transport=httpx.MockTransport(vendor.respond)) as client:
        result: Final = await SpendSync(db, client, lambda: NOW).run(db.source.source_id)
    assert result == Failure(status=502, message="Vendor billing API returned invalid pagination")
    assert len(vendor.requests) == 2
    assert db.persisted == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response", [httpx.Response(500, text="private"), httpx.Response(200, json={"data": "invalid"})]
)
async def test_vendor_error_messages_do_not_expose_response_or_key(response: httpx.Response) -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
        result: Final = await fetch_page(client, "private-key", window_start(NOW), NOW)
    assert isinstance(result, Failure)
    assert result.status == 502
    assert "private" not in result.message


@pytest.mark.asyncio
async def test_empty_sync_replaces_window_with_empty_totals(db: DatabaseDouble) -> None:
    assert db.source is not None
    vendor: Final = Vendor((page(),))
    async with httpx.AsyncClient(transport=httpx.MockTransport(vendor.respond)) as client:
        result: Final = await SpendSync(db, client, lambda: NOW).save_key(db.source.source_id, vendor.key)
    assert result == SyncResult(rows=0)
    assert db.persisted == ()


@pytest.mark.asyncio
async def test_injected_base_url_can_redirect_billing_to_a_scripted_server() -> None:
    vendor: Final = Vendor((page(),))
    async with httpx.AsyncClient(transport=httpx.MockTransport(vendor.respond)) as client:
        result: Final = await fetch_page(client, vendor.key, window_start(NOW), NOW, base_url="http://127.0.0.1:19199/")
    assert not isinstance(result, Failure)
    assert str(vendor.requests[0].url).startswith("http://127.0.0.1:19199/v2/teams/billing-items?")


@pytest.mark.asyncio
async def test_admin_key_cannot_be_redirected_by_environment_or_request_input(
    db: DatabaseDouble, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_EXTERNAL_CLOUD_AGENTS_BILLING_BASE_URL", "https://untrusted.example.test")
    vendor: Final = Vendor((page(), page()))
    async with httpx.AsyncClient(transport=httpx.MockTransport(vendor.respond)) as client:
        harness: Final = Harness(db, client)
        with harness.client:
            saved: Final = harness.client.put(
                f"/v1/external_cloud_agents/sources/{harness.source_id}/admin_api_key",
                params={"base_url": "https://untrusted.example.test"},
                json={"api_key": vendor.key, "base_url": "https://untrusted.example.test"},
            )
            assert saved.status_code == 200, saved.text
            synced: Final = harness.client.post(
                f"/v1/external_cloud_agents/sources/{harness.source_id}/sync_spend",
                json={"base_url": "https://untrusted.example.test"},
            )
            assert synced.status_code == 200, synced.text
    assert len(vendor.requests) == 2
    assert all(
        str(request.url).startswith("https://api.cursor.com/v2/teams/billing-items?") for request in vendor.requests
    )
