from collections.abc import Mapping
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.prompt_caching_requests import router
from tests.unit.proxy.db.fake_prisma_engine import statement_timeout_error, statement_timeout_reads

pytestmark = pytest.mark.usefixtures("local_model_cost_map")

_START: Final = "2026-09-01T00:00:00Z"
_END: Final = "2026-09-02T00:00:00Z"
_URL: Final = "/cost_optimization/prompt_caching/requests"


def _app(role: LitellmUserRoles | None) -> FastAPI:
    application: Final = FastAPI()
    application.include_router(router)

    def caller() -> UserAPIKeyAuth:
        return UserAPIKeyAuth(user_role=role)

    application.dependency_overrides[user_api_key_auth] = caller
    return application


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [None, LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY])
async def test_non_admin_is_denied_before_database_access(
    role: LitellmUserRoles | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app(role)), base_url="http://test") as client:
        response: Final = await client.get(_URL, params={"start_date": _START, "end_date": _END})
    assert response.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("params", [
    {"filter": "savings"}, {"page_size": 0}, {"page_size": 101}, {"start_date": "invalid"},
    {"cursor_start_time": "invalid", "cursor_request_id": "request"},
    {"cursor_start_time": _START, "cursor_request_id": ""},
])
async def test_invalid_request_is_rejected(params: Mapping[str, str | int]) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(LitellmUserRoles.PROXY_ADMIN)), base_url="http://test"
    ) as client:
        response: Final = await client.get(_URL, params={"start_date": _START, "end_date": _END, **params})
    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("params", [{"cursor_start_time": _START}, {"cursor_request_id": "request"}])
async def test_incomplete_cursor_is_rejected(
    params: Mapping[str, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", None)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(LitellmUserRoles.PROXY_ADMIN)), base_url="http://test"
    ) as client:
        response: Final = await client.get(_URL, params={"start_date": _START, "end_date": _END, **params})
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_the_requests_read_is_cancelled_by_postgres_after_30s(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    prisma: Final = MagicMock()
    transaction: Final = statement_timeout_reads(prisma, AsyncMock(return_value=[]))
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(LitellmUserRoles.PROXY_ADMIN)), base_url="http://test"
    ) as client:
        response: Final = await client.get(_URL, params={"start_date": _START, "end_date": _END})

    assert response.status_code == 200, response.text
    assert transaction.statements == ("SET LOCAL statement_timeout = 30000",)


@pytest.mark.asyncio
async def test_a_cancelled_requests_read_is_a_503_with_a_date_range_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    prisma: Final = MagicMock()
    statement_timeout_reads(prisma, AsyncMock(side_effect=statement_timeout_error()))
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(LitellmUserRoles.PROXY_ADMIN)), base_url="http://test"
    ) as client:
        response: Final = await client.get(_URL, params={"start_date": _START, "end_date": _END})

    assert response.status_code == 503, response.text
    assert response.json() == {
        "detail": {"error": "Prompt caching requests took longer than 30s. Pick a shorter date range"}
    }
