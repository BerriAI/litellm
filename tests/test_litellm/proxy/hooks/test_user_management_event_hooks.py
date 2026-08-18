import asyncio
import json
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm.proxy._types import NewUserRequest, NewUserResponse, UserAPIKeyAuth
from litellm.proxy.hooks.user_management_event_hooks import UserManagementEventHooks


class FakeUserTable:
    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows

    async def find_unique(self, where: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return next(
            (row for row in self._rows if all(row.get(key) == value for key, value in where.items())),
            None,
        )


class FakePrismaClient:
    def __init__(self, rows: List[Dict[str, Any]]):
        self.db = SimpleNamespace(litellm_usertable=FakeUserTable(rows))


async def _run_created_hook(prisma_client: FakePrismaClient, audit_log: AsyncMock) -> None:
    proxy_server = SimpleNamespace(
        prisma_client=prisma_client,
        litellm_proxy_admin_name="admin-user",
    )
    with (
        patch.dict(sys.modules, {"litellm.proxy.proxy_server": proxy_server}),
        patch.object(litellm, "store_audit_logs", True),
        patch(
            "litellm.proxy.hooks.user_management_event_hooks.create_audit_log_for_update",
            audit_log,
        ),
        patch.object(
            UserManagementEventHooks,
            "async_send_user_invitation_email",
            AsyncMock(),
        ),
    ):
        await UserManagementEventHooks.async_user_created_hook(
            data=NewUserRequest(user_email="new@example.com", send_invite_email=False),
            response=NewUserResponse(
                user_id="user-1",
                user_email="new@example.com",
                key="sk-test",
            ),
            user_api_key_dict=UserAPIKeyAuth(user_id="admin-user", api_key="sk-admin"),
        )
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_created_hook_audit_logs_the_row_read_back_from_the_database():
    """The audit entry must describe the persisted user row, not the /user/new response."""
    prisma_client = FakePrismaClient(
        [
            {
                "user_id": "user-1",
                "user_email": "new@example.com",
                "user_role": "proxy_admin",
                "models": ["gpt-4"],
                "teams": ["team-a"],
                "metadata": '{"source": "api"}',
            }
        ]
    )
    audit_log = AsyncMock()

    await _run_created_hook(prisma_client, audit_log)

    audit_log.assert_awaited_once()
    request_data = audit_log.await_args.kwargs["request_data"]
    assert request_data.object_id == "user-1"
    assert request_data.action == "created"

    updated_values = json.loads(request_data.updated_values)
    assert updated_values["user_role"] == "proxy_admin"
    assert updated_values["models"] == ["gpt-4"]
    assert updated_values["teams"] == ["team-a"]
    assert updated_values["metadata"] == {"source": "api"}


@pytest.mark.asyncio
async def test_created_hook_skips_the_audit_log_when_no_user_row_exists():
    """A user id that resolves to nothing must not produce an audit entry."""
    audit_log = AsyncMock()

    await _run_created_hook(FakePrismaClient([]), audit_log)

    audit_log.assert_not_awaited()
