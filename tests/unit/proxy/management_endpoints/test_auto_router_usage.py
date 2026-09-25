from datetime import date, datetime
from typing import Final
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.auto_router_usage import get_auto_router_usage


class Database:
    def __init__(self) -> None:
        self.get_auto_router_usage = AsyncMock(return_value=())


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY])
async def test_admin_filters_reach_prisma_client(role: LitellmUserRoles) -> None:
    db = Database()
    await get_auto_router_usage(
        date(2026, 1, 1),
        date(2026, 1, 2),
        UserAPIKeyAuth(user_role=role),
        db,
        router_name="router'quoted",
        router_type="complexity",
        user_id="owner",
        api_key="key-hash",
    )
    db.get_auto_router_usage.assert_awaited_once_with(
        start_time=datetime(2026, 1, 1),
        end_time=datetime(2026, 1, 3),
        user_id="owner",
        api_key="key-hash",
        router_name="router'quoted",
        router_type="complexity",
        destination_model=None,
    )


@pytest.mark.asyncio
async def test_non_admin_is_scoped_to_own_user_when_filter_omitted() -> None:
    db = Database()
    await get_auto_router_usage(
        date(2026, 1, 1),
        date(2026, 1, 1),
        UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="own-user"),
        db,
        destination_model="model-a",
    )
    db.get_auto_router_usage.assert_awaited_once_with(
        start_time=datetime(2026, 1, 1),
        end_time=datetime(2026, 1, 2),
        user_id="own-user",
        api_key=None,
        router_name=None,
        router_type=None,
        destination_model="model-a",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", [None, "own-user"])
async def test_non_admin_cannot_read_other_users_or_unbound_service_account(caller: str | None) -> None:
    db = Database()
    with pytest.raises(HTTPException) as error:
        await get_auto_router_usage(
            date(2026, 1, 1),
            date(2026, 1, 1),
            UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id=caller),
            db,
            destination_model="model-a",
            user_id="another-user",
        )
    assert error.value.status_code == 403
    db.get_auto_router_usage.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("model,router_name", [(None, None), ("model-a", "router-a")])
async def test_query_requires_one_specific_model_or_router(model: str | None, router_name: str | None) -> None:
    db = Database()
    with pytest.raises(HTTPException) as error:
        await get_auto_router_usage(
            date(2026, 1, 1),
            date(2026, 1, 1),
            UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
            db,
            destination_model=model,
            router_name=router_name,
        )
    assert error.value.status_code == 400
    db.get_auto_router_usage.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("end", [date(2025, 12, 31), date.max, date(2026, 4, 4)])
async def test_invalid_or_overlong_ranges_never_query_spend_logs(end: date) -> None:
    db: Final = Database()
    with pytest.raises(HTTPException) as error:
        await get_auto_router_usage(
            date(2026, 1, 1), end, UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN), db, destination_model="fast"
        )
    assert error.value.status_code == 400
    db.get_auto_router_usage.assert_not_called()


@pytest.mark.asyncio
async def test_maximum_range_includes_its_last_day() -> None:
    db: Final = Database()
    await get_auto_router_usage(
        date(2026, 1, 1), date(2026, 4, 3), UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN), db,
        destination_model="fast",
    )
    assert db.get_auto_router_usage.call_args.kwargs["start_time"] == datetime(2026, 1, 1)
    assert db.get_auto_router_usage.call_args.kwargs["end_time"] == datetime(2026, 4, 4)


@pytest.mark.asyncio
async def test_router_type_cannot_filter_a_destination_model() -> None:
    db: Final = Database()
    with pytest.raises(HTTPException) as error:
        await get_auto_router_usage(
            date(2026, 1, 1), date(2026, 1, 1), UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN), db,
            destination_model="fast", router_type="complexity",
        )
    assert error.value.status_code == 400
    db.get_auto_router_usage.assert_not_called()
