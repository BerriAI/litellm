"""``LiteLLM_SpendLogs`` has no index on ``"user"`` or ``team_id``, so a read filtered by owner scans the
whole date window. The owner's ``LiteLLM_Daily*Spend`` rows (kept after key deletion) plus their live keys
name every api_key that produced those logs, and ``(api_key, "startTime")`` is indexed."""

from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from typing import Final

from pydantic import ConfigDict, TypeAdapter

from litellm.proxy.utils import PrismaClient, query_raw_rows
from litellm.types.llms.base import LiteLLMBaseModel

SPEND_LOG_API_KEY_SCOPE_MAX_KEYS: Final = 5000

_USER_API_KEYS_SQL: Final = """
SELECT DISTINCT api_key FROM (
    SELECT api_key FROM "LiteLLM_DailyUserSpend"
    WHERE user_id = $1 AND date >= $2 AND date <= $3
    UNION ALL
    SELECT token AS api_key FROM "LiteLLM_VerificationToken" WHERE user_id = $1
) owned
LIMIT $4
"""

_TEAM_API_KEYS_SQL: Final = """
SELECT DISTINCT api_key FROM (
    SELECT api_key FROM "LiteLLM_DailyTeamSpend"
    WHERE team_id = ANY($1::text[]) AND date >= $2 AND date <= $3
    UNION ALL
    SELECT token AS api_key FROM "LiteLLM_VerificationToken" WHERE team_id = ANY($1::text[])
) owned
LIMIT $4
"""


class _ApiKeyRow(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    api_key: str


_API_KEY_ROWS: Final = TypeAdapter(tuple[_ApiKeyRow, ...])


def _unless_too_many(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...] | None:
    keys: Final = tuple(row.api_key for row in _API_KEY_ROWS.validate_python(rows))
    return None if len(keys) > SPEND_LOG_API_KEY_SCOPE_MAX_KEYS else keys


def _daily_dates_around(start: date, end: date) -> tuple[str, str]:
    return (start - timedelta(days=1)).isoformat(), (end + timedelta(days=1)).isoformat()


async def spend_log_api_keys_for_user(
    prisma_client: PrismaClient, user_id: str, start: date, end: date
) -> tuple[str, ...] | None:
    """The api_keys that can carry this user's spend logs in the window, or None when there are too many to help."""
    first, last = _daily_dates_around(start, end)
    return _unless_too_many(
        await query_raw_rows(
            prisma_client.db, _USER_API_KEYS_SQL, user_id, first, last, SPEND_LOG_API_KEY_SCOPE_MAX_KEYS + 1
        )
    )


async def spend_log_api_keys_for_teams(
    prisma_client: PrismaClient, team_ids: Sequence[str], start: date, end: date
) -> tuple[str, ...] | None:
    first, last = _daily_dates_around(start, end)
    return _unless_too_many(
        await query_raw_rows(
            prisma_client.db, _TEAM_API_KEYS_SQL, list(team_ids), first, last, SPEND_LOG_API_KEY_SCOPE_MAX_KEYS + 1
        )
    )
