"""
Latest health-check row per model, deduplicated by Postgres.

prisma-client-py's ``find_many(distinct=...)`` dedups client-side: the emitted
SQL carries no DISTINCT, so the whole append-only history table streams to the
worker on every call. ``SELECT DISTINCT ON`` keeps the transfer at one row per
(model_id, model_name) and is served by the matching descending index.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, field_validator

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

LATEST_HEALTH_CHECKS_SQL: Final = """
SELECT DISTINCT ON ("model_id", "model_name")
       "health_check_id", "model_name", "model_id", "status",
       "healthy_count", "unhealthy_count", "error_message",
       "response_time_ms", "details", "checked_by",
       "checked_at", "created_at", "updated_at"
FROM "LiteLLM_HealthCheckTable"
ORDER BY "model_id" ASC, "model_name" ASC, "checked_at" DESC
"""

LATEST_HEALTH_CHECKS_FOR_MODELS_SQL: Final = """
SELECT DISTINCT ON ("model_id", "model_name")
       "health_check_id", "model_name", "model_id", "status",
       "healthy_count", "unhealthy_count", "error_message",
       "response_time_ms", "details", "checked_by",
       "checked_at", "created_at", "updated_at"
FROM "LiteLLM_HealthCheckTable"
WHERE "model_name" = ANY($1)
ORDER BY "model_id" ASC, "model_name" ASC, "checked_at" DESC
"""


class LatestHealthCheckRow(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    health_check_id: str
    model_name: str
    model_id: str | None = None
    status: str
    healthy_count: int = 0
    unhealthy_count: int = 0
    error_message: str | None = None
    response_time_ms: float | None = None
    details: JsonValue | None = None
    checked_by: str | None = None
    checked_at: datetime
    created_at: datetime
    updated_at: datetime

    @field_validator("details", mode="before")
    @classmethod
    def _decode_json_text(cls, value: object) -> object:
        return json.loads(value) if isinstance(value, str) else value

    @field_validator("checked_at", "created_at", "updated_at")
    @classmethod
    def _assume_utc(cls, value: datetime) -> datetime:
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


_ROWS_ADAPTER: Final = TypeAdapter(tuple[LatestHealthCheckRow, ...])


async def fetch_latest_health_checks(prisma_client: PrismaClient) -> tuple[LatestHealthCheckRow, ...]:
    try:
        rows: Final = await prisma_client.db.query_raw(LATEST_HEALTH_CHECKS_SQL)
        return _ROWS_ADAPTER.validate_python(rows)
    except Exception as query_err:  # noqa: BLE001  # health decorates other reads; a driver error must not fail them
        verbose_proxy_logger.error("Error getting all latest health checks: %s", query_err)
        return ()


async def fetch_latest_health_checks_for_models(
    prisma_client: PrismaClient, model_names: Sequence[str]
) -> tuple[LatestHealthCheckRow, ...]:
    if not model_names:
        return ()
    try:
        rows: Final = await prisma_client.db.query_raw(LATEST_HEALTH_CHECKS_FOR_MODELS_SQL, list(model_names))
        return _ROWS_ADAPTER.validate_python(rows)
    except Exception as query_err:  # noqa: BLE001  # a paged model list must not fail on its health decoration
        verbose_proxy_logger.error("Error getting latest health checks for models: %s", query_err)
        return ()
