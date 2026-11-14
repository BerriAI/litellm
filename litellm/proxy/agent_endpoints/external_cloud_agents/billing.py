import json
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from itertools import groupby
from operator import attrgetter
from types import MappingProxyType
from typing import Final, Protocol

import httpx
from pydantic import AwareDatetime, Field, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.proxy.agent_endpoints.external_cloud_agents.database import Database
from litellm.proxy.agent_endpoints.external_cloud_agents.inventory import get_source
from litellm.proxy.agent_endpoints.external_cloud_agents.models import DailySpend, Failure, FrozenModel, SyncResult
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper

BILLING_BASE_URL: Final = "https://api.cursor.com"
BILLING_TIMEOUT: Final = 20.0


class HTTPClient(Protocol):
    async def get(
        self, url: str, *, params: Mapping[str, str | int], auth: httpx.BasicAuth, timeout: float
    ) -> httpx.Response: ...


class Principal(FrozenModel):
    type: str
    id: str


class Pricing(FrozenModel):
    contracted: Decimal


class Usage(FrozenModel):
    prompt_tokens: int = Field(default=0, alias="prompt-text-tokens")
    completion_tokens: int = Field(default=0, alias="completion-text-tokens")
    requests: int = 1


class Display(FrozenModel):
    userEmail: str | None = None
    displayLabel: str | None = None


class BillingItem(FrozenModel):
    eventTime: AwareDatetime
    chargeCategory: str
    principal: Principal | None = None
    pricing: Pricing
    usage: Usage = Usage()
    displayExtras: Display = Display()


class Pagination(FrozenModel):
    hasMore: bool
    nextCursor: str | None = None


class BillingPage(FrozenModel):
    data: tuple[BillingItem, ...]
    pagination: Pagination


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def window_start(end: datetime, days: int = 30) -> datetime:
    return end.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days - 1)


def combine(rows: tuple[DailySpend, ...]) -> tuple[DailySpend, ...]:
    return tuple(
        combine_group(tuple(group))
        for _, group in groupby(
            sorted(rows, key=attrgetter("vendor_user_id", "day")), attrgetter("vendor_user_id", "day")
        )
    )


def combine_group(rows: tuple[DailySpend, ...]) -> DailySpend:
    latest: Final = rows[-1]
    return latest.model_copy(
        update={
            "email": next((row.email for row in reversed(rows) if row.email), None),
            "display_name": next((row.display_name for row in reversed(rows) if row.display_name), None),
            "cost_micros": sum(row.cost_micros for row in rows),
            "prompt_tokens": sum(row.prompt_tokens for row in rows),
            "completion_tokens": sum(row.completion_tokens for row in rows),
            "requests": sum(row.requests for row in rows),
        }
    )


def aggregate(items: tuple[BillingItem, ...], start: datetime, end: datetime) -> tuple[DailySpend, ...]:
    return combine(
        tuple(
            DailySpend(
                vendor_user_id=item.principal.id,
                email=item.displayExtras.userEmail,
                display_name=item.displayExtras.displayLabel,
                day=item.eventTime.astimezone(timezone.utc).date(),
                cost_micros=int((item.pricing.contracted * 1_000_000).to_integral_value(rounding=ROUND_HALF_UP)),
                prompt_tokens=item.usage.prompt_tokens,
                completion_tokens=item.usage.completion_tokens,
                requests=item.usage.requests,
            )
            for item in items
            if item.chargeCategory == "usage"
            and item.principal is not None
            and item.principal.type == "user"
            and start <= item.eventTime < end
        )
    )


async def fetch_page(
    client: HTTPClient,
    key: str,
    start: datetime,
    end: datetime,
    cursor: str | None = None,
    *,
    base_url: str = BILLING_BASE_URL,
) -> BillingPage | Failure:
    params: Final = MappingProxyType(
        {
            "product": "grok-bot",
            "startTime": start.isoformat().replace("+00:00", "Z"),
            "endTime": end.isoformat().replace("+00:00", "Z"),
            "pageSize": 500,
            **({"cursor": cursor} if cursor else {}),
        }
    )
    try:
        url: Final = base_url.rstrip("/") + "/v2/teams/billing-items"
        response: Final = await client.get(url, params=params, auth=httpx.BasicAuth(key, ""), timeout=BILLING_TIMEOUT)
        if response.status_code == 429:
            retry_header: Final = TypeAdapter(str).validate_python(response.headers.get("Retry-After", "unspecified"))
            retry_after: Final = retry_header.replace(key, "[redacted]")[:100]
            return Failure(status=429, message=f"Vendor rate limit. Retry-After {retry_after}")
        if response.status_code in (401, 403):
            return Failure(status=400, message="Invalid GrokBot Admin API key")
        if response.status_code != 200:
            return Failure(status=502, message=f"Vendor billing API returned HTTP {response.status_code}")
        return BillingPage.model_validate(response.json())
    except (httpx.HTTPError, ValueError):
        return Failure(status=502, message="Vendor billing API request failed or returned invalid billing data")


class BillingScan:
    def __init__(
        self,
        client: HTTPClient,
        key: str,
        start: datetime,
        end: datetime,
        first_page: BillingPage | None,
        base_url: str = BILLING_BASE_URL,
    ) -> None:
        self.client = client
        self.base_url = base_url
        self.key = key
        self.start = start
        self.end = end
        self.preloaded = first_page
        self.cursor: str | None = None
        self.seen: frozenset[str] = frozenset()
        self.rows: tuple[DailySpend, ...] = ()
        self.finished = False

    async def advance(self) -> Failure | None:
        page: Final = self.preloaded or await fetch_page(
            self.client, self.key, self.start, self.end, self.cursor, base_url=self.base_url
        )
        self.preloaded = None
        if isinstance(page, Failure):
            return page
        self.rows = combine((*self.rows, *aggregate(page.data, self.start, self.end)))
        if not page.pagination.hasMore:
            self.finished = True
            return None
        cursor: Final = page.pagination.nextCursor
        if not cursor or cursor in self.seen:
            return Failure(status=502, message="Vendor billing API returned invalid pagination")
        self.seen = self.seen | {cursor}
        self.cursor = cursor
        return None


class SpendSync:
    def __init__(
        self,
        db: Database,
        client: HTTPClient,
        clock: Callable[[], datetime] = utc_now,
        *,
        base_url: str = BILLING_BASE_URL,
    ) -> None:
        self.db = db
        self.client = client
        self.clock = clock
        self.base_url = base_url

    async def save_key(self, source_id: str, key: str) -> SyncResult | Failure:
        source: Final = await get_source(self.db, source_id)
        if source is None:
            return Failure(status=404, message="Unknown source")
        end: Final = self.clock()
        first: Final = await fetch_page(self.client, key, window_start(end), end, base_url=self.base_url)
        if isinstance(first, Failure):
            await self.record_error(source_id, first)
            return first
        await self.db.execute_raw(
            'UPDATE "LiteLLM_ExternalCloudAgentSource" SET admin_api_key=$2,last_spend_sync_error=NULL WHERE id=$1',
            source_id,
            encrypt_value_helper(key),
        )
        return await self.run(source_id, key=key, first_page=first, end=end)

    async def run(
        self,
        source_id: str,
        *,
        key: str | None = None,
        first_page: BillingPage | None = None,
        end: datetime | None = None,
    ) -> SyncResult | Failure:
        source: Final = await get_source(self.db, source_id)
        if source is None:
            return Failure(status=404, message="Unknown source")
        api_key: Final = key or (
            decrypt_value_helper(source.admin_api_key, "admin_api_key") if source.admin_api_key else None
        )
        if not api_key:
            return Failure(status=400, message="Set a GrokBot Admin API key before syncing spend")
        until: Final = end or self.clock()
        scan: Final = BillingScan(self.client, api_key, window_start(until), until, first_page, self.base_url)
        try:
            while not scan.finished:
                failure = await scan.advance()
                if failure is not None:
                    await self.record_error(source_id, failure)
                    return failure
            await self.persist(source_id, scan, source.admin_api_key)
            return SyncResult(rows=len(scan.rows))
        except Exception as exc:  # noqa: BLE001  # sync failures must preserve persisted spend and redact credentials
            verbose_proxy_logger.error("External cloud agent spend sync failed (%s)", type(exc).__name__)
            failed: Final = Failure(status=503, message="Spend sync failed. Previously synced data is still available")
            await self.record_error(source_id, failed)
            return failed

    async def record_error(self, source_id: str, failure: Failure) -> None:
        try:
            await self.db.execute_raw(
                'UPDATE "LiteLLM_ExternalCloudAgentSource" SET last_spend_sync_error=$2 WHERE id=$1',
                source_id,
                failure.message,
            )
        except Exception as exc:  # noqa: BLE001  # recording a sync failure must not mask the original failure
            verbose_proxy_logger.error("External cloud agent sync error could not be stored (%s)", type(exc).__name__)

    async def persist(self, source_id: str, scan: BillingScan, encrypted_key: str | None) -> None:
        await self.db.execute_raw(
            'WITH eligible AS (UPDATE "LiteLLM_ExternalCloudAgentSource" '
            "SET last_spend_sync_at=$5::timestamp,last_spend_sync_error=NULL WHERE id=$1 AND admin_api_key=$6 "
            "AND (last_spend_sync_at IS NULL OR last_spend_sync_at<=$5::timestamp) RETURNING id), "
            "incoming AS (SELECT x.* FROM jsonb_to_recordset($2::jsonb) AS x("
            "vendor_user_id text,email text,display_name text,day date,cost_micros bigint,"
            "prompt_tokens bigint,completion_tokens bigint,requests bigint) CROSS JOIN eligible), "
            'removed AS (DELETE FROM "LiteLLM_ExternalCloudAgentSpend" s WHERE source_id=$1 '
            "AND day BETWEEN $3::date AND $4::date AND NOT EXISTS "
            "(SELECT 1 FROM incoming i WHERE i.vendor_user_id=s.vendor_user_id AND i.day=s.day) "
            "AND EXISTS (SELECT 1 FROM eligible)) "
            'INSERT INTO "LiteLLM_ExternalCloudAgentSpend" '
            "(source_id,vendor_user_id,email,display_name,day,cost_micros,prompt_tokens,completion_tokens,requests) "
            "SELECT $1,vendor_user_id,email,display_name,day,cost_micros,prompt_tokens,completion_tokens,requests FROM incoming "
            "ON CONFLICT (source_id,vendor_user_id,day) DO UPDATE SET email=EXCLUDED.email,"
            "display_name=EXCLUDED.display_name,cost_micros=EXCLUDED.cost_micros,prompt_tokens=EXCLUDED.prompt_tokens,"
            "completion_tokens=EXCLUDED.completion_tokens,requests=EXCLUDED.requests",
            source_id,
            json.dumps(tuple(row.model_dump(mode="json") for row in scan.rows)),
            scan.start.date().isoformat(),
            scan.end.date().isoformat(),
            scan.end.replace(tzinfo=None),
            encrypted_key,
        )
