"""Authorization-scoped access to aggregated usage/spend data for Ask AI.

The caller's authorization is baked into the provider at construction. A
non-admin caller receives a provider that can only ever read its own
``user_id``, so an out-of-scope query is unrepresentable rather than something
each tool handler has to remember to guard. Team and tag breakdowns are
admin-only and the provider refuses them for a user scope as defense in depth.

Data is read through the same ``common_daily_activity`` helpers that back the
``/spend`` REST endpoints, so there is a single source of truth for the
aggregation logic.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

from litellm.types.proxy.management_endpoints.common_daily_activity import (
    DailySpendData,
    MetricWithMetadata,
    SpendAnalyticsPaginatedResponse,
    SpendMetrics,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

TABLE_DAILY_USER_SPEND = "litellm_dailyuserspend"
TABLE_DAILY_TEAM_SPEND = "litellm_dailyteamspend"
TABLE_DAILY_TAG_SPEND = "litellm_dailytagspend"

ENTITY_FIELD_USER = "user_id"
ENTITY_FIELD_TEAM = "team_id"
ENTITY_FIELD_TAG = "tag"

PAGINATED_PAGE_SIZE = 200
TOP_N_MODELS = 15
TOP_N_PROVIDERS = 10


@dataclass(frozen=True, slots=True)
class AdminScope:
    """Global view. ``caller_user_id`` is the admin's own id (may be None) and
    is not used to filter; admins may optionally pass an explicit user filter."""

    caller_user_id: Optional[str]


@dataclass(frozen=True, slots=True)
class UserScope:
    """Non-admin view. Every query is forced to this ``user_id``."""

    user_id: str


AiChatScope = Union[AdminScope, UserScope]


def _parse_csv(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [t.strip() for t in raw.split(",") if t.strip()]


class ScopedUsageDataProvider:
    """Reads daily activity data within the bounds of an ``AiChatScope``."""

    def __init__(self, scope: AiChatScope, prisma_client: Optional["PrismaClient"]) -> None:
        self._scope = scope
        self._prisma_client = prisma_client

    @property
    def is_admin(self) -> bool:
        return isinstance(self._scope, AdminScope)

    async def usage(
        self, start_date: str, end_date: str, user_id_filter: Optional[str]
    ) -> SpendAnalyticsPaginatedResponse:
        scope = self._scope
        effective_user_id = scope.user_id if isinstance(scope, UserScope) else user_id_filter
        from litellm.proxy.management_endpoints.common_daily_activity import (
            get_daily_activity_aggregated,
        )

        return await get_daily_activity_aggregated(
            prisma_client=self._prisma_client,
            table_name=TABLE_DAILY_USER_SPEND,
            entity_id_field=ENTITY_FIELD_USER,
            entity_id=effective_user_id,
            entity_metadata_field=None,
            start_date=start_date,
            end_date=end_date,
            model=None,
            api_key=None,
        )

    async def team(self, start_date: str, end_date: str, team_ids: Optional[str]) -> SpendAnalyticsPaginatedResponse:
        self._require_admin("team usage")
        return await self._paginated(
            TABLE_DAILY_TEAM_SPEND, ENTITY_FIELD_TEAM, _parse_csv(team_ids), start_date, end_date
        )

    async def tag(self, start_date: str, end_date: str, tags: Optional[str]) -> SpendAnalyticsPaginatedResponse:
        self._require_admin("tag usage")
        return await self._paginated(TABLE_DAILY_TAG_SPEND, ENTITY_FIELD_TAG, _parse_csv(tags), start_date, end_date)

    def _require_admin(self, what: str) -> None:
        if not isinstance(self._scope, AdminScope):
            raise PermissionError(f"{what} data is only available to admin callers")

    async def _paginated(
        self,
        table_name: str,
        entity_id_field: str,
        entity_id: Optional[List[str]],
        start_date: str,
        end_date: str,
    ) -> SpendAnalyticsPaginatedResponse:
        from litellm.proxy.management_endpoints.common_daily_activity import (
            get_daily_activity,
        )

        return await get_daily_activity(
            prisma_client=self._prisma_client,
            table_name=table_name,
            entity_id_field=entity_id_field,
            entity_id=entity_id,
            entity_metadata_field=None,
            start_date=start_date,
            end_date=end_date,
            model=None,
            api_key=None,
            page=1,
            page_size=PAGINATED_PAGE_SIZE,
        )


@dataclass(frozen=True, slots=True)
class _Totals:
    spend: float
    api_requests: int
    total_tokens: int

    def plus(self, m: "SpendMetrics") -> "_Totals":
        return _Totals(
            spend=self.spend + m.spend,
            api_requests=self.api_requests + m.api_requests,
            total_tokens=self.total_tokens + m.total_tokens,
        )


def _accumulate(
    days: List[DailySpendData],
    pick: Callable[[DailySpendData], Dict[str, MetricWithMetadata]],
) -> Dict[str, _Totals]:
    """Sum one breakdown dimension across days into per-name totals."""
    totals: Dict[str, _Totals] = {}
    for day in days:
        for name, entry in pick(day).items():
            totals[name] = totals.get(name, _Totals(0.0, 0, 0)).plus(entry.metrics)
    return totals


def _ranked(totals: Dict[str, _Totals], limit: int) -> List[Tuple[str, _Totals]]:
    return sorted(totals.items(), key=lambda item: -item[1].spend)[:limit]


def summarise_usage_data(resp: SpendAnalyticsPaginatedResponse) -> str:
    """Render global/user usage into concise text the LLM can reason over."""
    meta = resp.metadata
    header = (
        f"Total Spend: ${meta.total_spend:.4f}\n"
        f"Total Requests: {meta.total_api_requests}\n"
        f"Successful: {meta.total_successful_requests} | Failed: {meta.total_failed_requests}\n"
        f"Total Tokens: {meta.total_tokens}"
    )

    models = _accumulate(resp.results, lambda d: d.breakdown.models)
    providers = _accumulate(resp.results, lambda d: d.breakdown.providers)

    model_lines = [
        f"  - {name}: ${t.spend:.4f} ({t.api_requests} reqs, {t.total_tokens} tokens)"
        for name, t in _ranked(models, TOP_N_MODELS)
    ]
    provider_lines = [
        f"  - {name}: ${t.spend:.4f} ({t.api_requests} reqs)" for name, t in _ranked(providers, TOP_N_PROVIDERS)
    ]

    sections = [header, ""]
    sections += ["Top Models by Spend:"] + (model_lines or ["  (no data)"]) + [""]
    sections += ["Top Providers by Spend:"] + (provider_lines or ["  (no data)"])
    return "\n".join(sections)


def _entity_alias(entry: MetricWithMetadata, entity_id: str) -> str:
    raw = entry.metadata.get("alias")
    return raw if isinstance(raw, str) and raw else entity_id


def summarise_entity_data(resp: SpendAnalyticsPaginatedResponse, entity_label: str) -> str:
    """Render team/tag entity usage into concise text."""
    if not resp.results:
        return f"No {entity_label} usage data found for the given date range."

    totals = _accumulate(resp.results, lambda d: d.breakdown.entities)
    aliases = {eid: _entity_alias(entry, eid) for day in resp.results for eid, entry in day.breakdown.entities.items()}

    lines = [f"{entity_label} Usage ({len(totals)} {entity_label.lower()}s):", ""]
    for eid, t in _ranked(totals, len(totals)):
        lines.append(
            f"- {aliases.get(eid, eid)} (ID: {eid}): ${t.spend:.4f} | {t.api_requests} reqs | {t.total_tokens} tokens"
        )
    return "\n".join(lines)
