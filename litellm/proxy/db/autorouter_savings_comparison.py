from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from datetime import timedelta
from math import isclose
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.constants import MAX_SPENDLOG_ROWS_TO_QUERY
from litellm.proxy.db.autorouter_session_rollup import AUTOROUTER_SESSION_WINDOW_SQL
from litellm.proxy.db.create_views import SupportsRawQueries

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


class SessionSavingsComparison(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    router_name: str
    router_type: str
    turns: int
    estimated_turns: int
    actual_spend: float
    saved_spend: float
    complete: bool

    def coverage_fields(self, recorded_savings: float, recorded_turns: int) -> Mapping[str, float | int]:
        if self.turns != recorded_turns or not self.complete:
            return MappingProxyType({})
        if not isclose(self.saved_spend, recorded_savings, rel_tol=1e-9, abs_tol=1e-9):
            return MappingProxyType({})
        return MappingProxyType(
            {
                "savings_estimated_turns": self.estimated_turns,
                "savings_estimated_actual_spend": self.actual_spend,
                "savings_estimated_saved_spend": self.saved_spend,
            }
        )


class _ReadTransactions(Protocol):
    def tx(self, *, timeout: timedelta, max_wait: timedelta) -> AbstractAsyncContextManager[SupportsRawQueries]: ...


_COMPARISONS: Final = TypeAdapter(tuple[SessionSavingsComparison, ...])


async def historical_session_comparisons(
    prisma_client: "PrismaClient",
    start_date: str,
    end_date: str,
    api_key: str | None,
    user_id: str | None,
    session_id: str | None = None,
) -> Mapping[tuple[str, str], SessionSavingsComparison]:
    try:
        reader: Final = cast(_ReadTransactions, prisma_client.read_db)  # cast-ok: untyped Prisma transaction delegate
        async with reader.tx(timeout=timedelta(seconds=3), max_wait=timedelta(seconds=1)) as transaction:
            await transaction.execute_raw("SET TRANSACTION READ ONLY")
            await transaction.execute_raw("SET LOCAL statement_timeout = 2000")
            rows: Final = await transaction.query_raw(
                HISTORICAL_SESSION_COMPARISONS_SQL,
                start_date,
                end_date,
                api_key,
                user_id,
                session_id,
            )
            comparisons: Final = _COMPARISONS.validate_python(rows or ())
            return MappingProxyType({(row.router_name, row.router_type): row for row in comparisons})
    except Exception:  # noqa: BLE001  # missing retained logs must not discard recorded dollar savings
        verbose_proxy_logger.warning("Historical auto-router cost comparison unavailable; preserving recorded savings")
        return MappingProxyType({})


HISTORICAL_SESSION_COMPARISONS_SQL: Final = f"""
WITH {AUTOROUTER_SESSION_WINDOW_SQL}, scoped AS MATERIALIZED (
    SELECT * FROM windowed WHERE $5::text IS NULL OR session_id = $5::text
), limited_logs AS MATERIALIZED (
    SELECT session.api_key, session.session_id, session.router_name, session.router_type, session.comparison_user_id,
        logs.spend, logs.prompt_tokens + logs.completion_tokens AS tokens,
        logs.metadata::jsonb -> 'routing_decision' AS decision,
        logs.metadata::jsonb -> 'autorouter_savings' AS savings,
        logs.metadata::jsonb -> 'autorouter_savings_estimate' AS estimate
    FROM scoped AS session JOIN "LiteLLM_SpendLogs" AS logs
      ON logs.api_key = session.api_key
      AND CASE WHEN char_length(logs.session_id) > 256
          THEN 'sha256:' || encode(sha256(convert_to(logs.session_id, 'UTF8')), 'hex')
          ELSE logs.session_id END = session.session_id
      AND (session.comparison_user_id IS NULL OR logs."user" = session.comparison_user_id)
      AND logs."startTime" BETWEEN session.first_turn_at AND session.last_turn_at
      AND COALESCE(logs.metadata::jsonb #>> '{{routing_decision,router_model_name}}', logs.model_group)
          = session.router_name
    WHERE session.savings_estimated_turns < session.turns
      AND logs.status = 'success' AND COALESCE(logs.metadata::jsonb ->> 'internal_call_origin', '') = ''
    LIMIT {MAX_SPENDLOG_ROWS_TO_QUERY + 1}
), facts AS (
    SELECT *,
        CASE WHEN jsonb_typeof(decision -> 'classifier_cost') = 'number'
            THEN (decision ->> 'classifier_cost')::float8 ELSE 0 END AS classifier,
        CASE WHEN jsonb_typeof(savings) = 'number' AND (
            estimate IS NULL OR estimate = 'null'::jsonb OR (
                jsonb_typeof(estimate -> 'version') = 'number' AND estimate ->> 'version' IN ('1', '2', '3')
                AND estimate ->> 'status' = 'estimated'
            )
        ) THEN savings::text::float8 END AS saved
    FROM limited_logs
), compared AS (
    SELECT api_key, session_id, router_name, router_type, comparison_user_id,
        COUNT(*) AS turns, SUM(spend + classifier) AS spend, SUM(tokens) AS total_tokens,
        COUNT(saved) AS estimated_turns,
        COALESCE(SUM(spend + classifier) FILTER (WHERE saved IS NOT NULL), 0)::float8 AS actual_spend,
        COALESCE(SUM(saved), 0)::float8 AS saved_spend
    FROM facts GROUP BY 1, 2, 3, 4, 5
), reconciled AS (
    SELECT session.*, logs.estimated_turns, logs.actual_spend,
        COALESCE((SELECT COUNT(*) FROM limited_logs) <= {MAX_SPENDLOG_ROWS_TO_QUERY}
            AND logs.turns = session.turns AND logs.total_tokens = session.total_tokens
            AND ABS(logs.spend - session.spend) <= GREATEST(1e-9, ABS(session.spend) * 1e-9)
            AND ABS(logs.saved_spend - session.saved_spend) <= GREATEST(1e-9, ABS(session.saved_spend) * 1e-9), FALSE
        ) AS recovered
    FROM scoped AS session LEFT JOIN compared AS logs
        ON logs.api_key = session.api_key AND logs.session_id = session.session_id
        AND logs.router_name = session.router_name AND logs.router_type = session.router_type
        AND logs.comparison_user_id IS NOT DISTINCT FROM session.comparison_user_id
)
SELECT router_name, router_type,
    SUM(turns)::bigint AS turns,
    SUM(CASE WHEN recovered THEN estimated_turns ELSE savings_estimated_turns END)::bigint AS estimated_turns,
    SUM(CASE WHEN recovered THEN actual_spend ELSE savings_estimated_actual_spend END)::float8 AS actual_spend,
    SUM(saved_spend)::float8 AS saved_spend,
    BOOL_AND(recovered OR savings_estimated_turns = turns) AS complete
FROM reconciled GROUP BY router_name, router_type
"""
