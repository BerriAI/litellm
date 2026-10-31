from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from litellm.models.gateway_activity import GatewayRouteActivity, GatewayStatusCodeActivity
from litellm.repositories.gateway_activity_repository import GatewayActivityRepository


@pytest.mark.asyncio
async def test_gateway_activity_decodes_both_row_kinds_with_nullable_union_columns() -> None:
    prisma: Final = MagicMock()
    prisma.db.query_raw = AsyncMock(return_value=[
        {
            "kind": "route", "date": "2026-01-02", "category": "llm", "route": "/chat/completions",
            "successful_requests": 7, "failed_requests": 2, "status_code": None,
        },
        {
            "kind": "status", "date": None, "category": None, "route": None,
            "successful_requests": 0, "failed_requests": 2, "status_code": 429,
        },
    ])
    result: Final = await GatewayActivityRepository(prisma).get_activity("2026-01-01", "2026-01-31")
    assert result == (
        GatewayRouteActivity(
            kind="route", date="2026-01-02", category="llm", route="/chat/completions",
            successful_requests=7, failed_requests=2,
        ),
        GatewayStatusCodeActivity(kind="status", status_code=429, failed_requests=2),
    )


@pytest.mark.asyncio
async def test_gateway_activity_rejects_malformed_database_rows() -> None:
    prisma: Final = MagicMock()
    prisma.db.query_raw = AsyncMock(return_value=[{"kind": "status", "status_code": None, "failed_requests": 2}])
    with pytest.raises(ValidationError):
        await GatewayActivityRepository(prisma).get_activity("2026-01-01", "2026-01-31")


@pytest.mark.asyncio
async def test_gateway_activity_does_not_hide_database_failure() -> None:
    prisma: Final = MagicMock()
    prisma.db.query_raw = AsyncMock(side_effect=RuntimeError("reader unavailable"))
    with pytest.raises(RuntimeError, match="reader unavailable"):
        await GatewayActivityRepository(prisma).get_activity("2026-01-01", "2026-01-31")
