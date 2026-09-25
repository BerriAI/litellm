from datetime import date
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.auto_router_usage import get_auto_router_usage


class Database:
    def __init__(self) -> None:
        self.query_raw = AsyncMock(return_value=[])


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY])
async def test_admin_filters_reach_query_as_parameters(role: LitellmUserRoles) -> None:
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
    assert db.query_raw.call_args.args[1:] == (
        "2026-01-01T00:00:00",
        "2026-01-03T00:00:00",
        "owner",
        "key-hash",
        "router'quoted",
        "complexity",
        None,
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
    assert db.query_raw.call_args.args[1:] == (
        "2026-01-01T00:00:00",
        "2026-01-02T00:00:00",
        "own-user",
        None,
        None,
        None,
        "model-a",
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
    db.query_raw.assert_not_called()


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
    db.query_raw.assert_not_called()
