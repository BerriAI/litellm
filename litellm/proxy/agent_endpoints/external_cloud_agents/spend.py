from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from itertools import groupby
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

from litellm.proxy.agent_endpoints.external_cloud_agents.billing import utc_now, window_start
from litellm.proxy.agent_endpoints.external_cloud_agents.database import Database
from litellm.proxy.agent_endpoints.external_cloud_agents.models import (
    MatchedSpend,
    MemberSpend,
    SpendReport,
    SpendTotals,
    TeamSpend,
)


def totals(rows: tuple[MemberSpend, ...]) -> SpendTotals:
    gateways: Final = MappingProxyType(
        {row.litellm_user_id: row.litellm_gateway_spend_usd for row in rows if row.litellm_user_id is not None}
    )
    return SpendTotals(
        grok_cost_usd=sum((row.grok_cost_usd for row in rows), Decimal(0)),
        litellm_gateway_spend_usd=sum(gateways.values(), Decimal(0)),
        tokens=sum(row.tokens for row in rows),
        requests=sum(row.requests for row in rows),
    )


def split_units(value: int, index: int, count: int) -> int:
    quotient, remainder = divmod(value, count)
    return quotient + int(index < remainder)


def split_money(value: Decimal, index: int, count: int) -> Decimal:
    exponent: Final = value.as_tuple().exponent
    precision: Final = max(6, -exponent) if isinstance(exponent, int) else 6
    return Decimal(split_units(int(value.scaleb(precision)), index, count)).scaleb(-precision)


def allocated(row: MatchedSpend, index: int, count: int) -> MatchedSpend:
    return row.model_copy(
        update={
            "grok_cost_usd": split_money(row.grok_cost_usd, index, count),
            "litellm_gateway_spend_usd": split_money(row.litellm_gateway_spend_usd, index, count),
            "tokens": split_units(row.tokens, index, count),
            "requests": split_units(row.requests, index, count),
        }
    )


def team_members(row: MatchedSpend) -> tuple[tuple[str, str, MatchedSpend], ...]:
    if not row.team_ids:
        return (("", "Unassigned", row),)
    teams: Final = MappingProxyType(dict(zip(row.team_ids, row.team_names)))
    return tuple(
        (team_id, name, allocated(row, index, len(teams)))
        for index, (team_id, name) in enumerate(sorted(teams.items()))
    )


def rollup(key: str, rows: tuple[tuple[str, str, MatchedSpend], ...]) -> TeamSpend:
    values: Final = totals(tuple(row[2] for row in rows))
    return TeamSpend(
        team_id=key or None,
        team_name=rows[0][1],
        grok_cost_usd=values.grok_cost_usd,
        litellm_gateway_spend_usd=values.litellm_gateway_spend_usd,
        tokens=values.tokens,
        requests=values.requests,
    )


def report(source_id: str, days: int, end: datetime, rows: tuple[MatchedSpend, ...]) -> SpendReport:
    pairs: Final = sum(map(team_members, rows), ())
    return SpendReport(
        source_id=source_id,
        days=days,
        start_day=window_start(end, days).date(),
        end_day=end.date(),
        members=tuple(MemberSpend.model_validate(row.model_dump()) for row in rows),
        teams=tuple(
            rollup(key, tuple(group)) for key, group in groupby(sorted(pairs, key=lambda p: p[0]), lambda p: p[0])
        ),
        totals=totals(rows),
    )


class SpendReader:
    def __init__(self, db: Database, clock: Callable[[], datetime] = utc_now) -> None:
        self.db = db
        self.clock = clock

    async def read(self, source_id: str, days: int) -> SpendReport:
        end: Final = self.clock()
        rows: Final = TypeAdapter(tuple[MatchedSpend, ...]).validate_python(
            await self.db.query_raw(
                "WITH vendor AS (SELECT vendor_user_id,"
                "(array_agg(email ORDER BY day DESC) FILTER (WHERE email IS NOT NULL))[1] AS email,"
                "(array_agg(display_name ORDER BY day DESC) FILTER (WHERE display_name IS NOT NULL))[1] AS display_name,"
                "SUM(cost_micros)/1000000::numeric AS grok_cost_usd,SUM(prompt_tokens+completion_tokens)::bigint AS tokens,"
                'SUM(requests)::bigint AS requests FROM "LiteLLM_ExternalCloudAgentSpend" '
                "WHERE source_id=$1 AND day BETWEEN $2::text::date AND $3::text::date GROUP BY vendor_user_id), "
                'emails AS (SELECT LOWER(user_email) AS email,MIN(user_id) AS user_id FROM "LiteLLM_UserTable" '
                "WHERE user_email IS NOT NULL GROUP BY LOWER(user_email) HAVING COUNT(*)=1), "
                "matched AS (SELECT v.*,u.user_id AS litellm_user_id,u.teams FROM vendor v "
                'LEFT JOIN emails e ON e.email=LOWER(v.email) LEFT JOIN "LiteLLM_UserTable" u ON u.user_id=e.user_id), '
                'gateway AS (SELECT d.user_id,SUM(d.spend::numeric) AS spend FROM "LiteLLM_DailyUserSpend" d '
                "JOIN (SELECT DISTINCT litellm_user_id FROM matched) ids ON ids.litellm_user_id=d.user_id "
                "WHERE d.date BETWEEN $2::text AND $3::text GROUP BY d.user_id) "
                "SELECT m.*,COALESCE(g.spend,0) AS litellm_gateway_spend_usd,COALESCE(m.teams,ARRAY[]::text[]) AS team_ids,"
                "ARRAY(SELECT COALESCE(t.team_alias,member_team.team_id) "
                "FROM unnest(m.teams) WITH ORDINALITY AS member_team(team_id,position) "
                'LEFT JOIN "LiteLLM_TeamTable" t ON t.team_id=member_team.team_id ORDER BY position) AS team_names '
                "FROM matched m LEFT JOIN gateway g ON g.user_id=m.litellm_user_id ORDER BY m.vendor_user_id",
                source_id,
                window_start(end, days).date().isoformat(),
                end.date().isoformat(),
            )
        )
        return report(source_id, days, end, rows)
