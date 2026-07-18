from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._experimental.mcp_server.db import (
    create_draft_mcp_server,
    delete_expired_draft_mcp_servers,
    get_draft_mcp_server,
    _delete_draft_mcp_server,
)
from litellm.proxy._types import (
    LiteLLM_MCPServerTable,
    MCPApprovalStatus,
    MCPTransport,
    NewMCPServerRequest,
)


def _make_prisma_row(
    server_id: str = "draft-1",
    approval_status: str = "draft",
    created_at: datetime | None = None,
) -> MagicMock:
    now = created_at or datetime.now(timezone.utc)
    row = MagicMock()
    row.server_id = server_id
    row.approval_status = approval_status
    row.created_at = now
    row.updated_at = now
    row.env_vars = None
    row.model_dump = MagicMock(
        return_value={
            "server_id": server_id,
            "server_name": server_id,
            "alias": server_id,
            "url": "https://example.com",
            "transport": MCPTransport.http,
            "approval_status": approval_status,
            "created_at": now,
            "updated_at": now,
            "created_by": "admin",
            "updated_by": "admin",
            "teams": [],
            "mcp_access_groups": [],
            "allowed_tools": [],
            "extra_headers": [],
            "env_vars": None,
            "credentials": None,
            "auth_type": None,
            "description": None,
            "mcp_info": None,
            "static_headers": None,
            "command": None,
            "args": [],
            "env": {},
            "authorization_url": None,
            "token_url": None,
            "registration_url": None,
            "allow_all_keys": False,
            "available_on_public_internet": True,
            "timeout": None,
            "max_concurrent_requests": None,
            "is_byok": False,
            "byok_description": [],
            "byok_api_key_help_url": None,
            "source_url": None,
            "instructions": None,
            "submitted_by": None,
            "submitted_at": None,
            "delegate_auth_to_upstream": False,
            "oauth_passthrough": False,
            "oauth2_flow": None,
            "spec_path": None,
            "tool_name_to_display_name": None,
            "tool_name_to_description": None,
        },
    )
    return row


def _mock_repo(find_first_return=None, delete_many_return=0):
    mock_table = MagicMock()
    mock_table.find_first = AsyncMock(return_value=find_first_return)
    mock_table.delete_many = AsyncMock(return_value=delete_many_return)
    mock_repo_instance = MagicMock()
    mock_repo_instance.table = mock_table
    return mock_repo_instance, mock_table


class TestCreateDraftMcpServer:
    @pytest.mark.asyncio
    async def test_sets_approval_status_to_draft(self):
        prisma = MagicMock()
        payload = NewMCPServerRequest(
            server_id="draft-new",
            alias="Draft",
            url="https://example.com",
            transport=MCPTransport.http,
        )
        draft_row = _make_prisma_row(server_id="draft-new")

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.db._delete_draft_mcp_server",
                AsyncMock(),
            ) as del_mock,
            patch(
                "litellm.proxy._experimental.mcp_server.db.create_mcp_server",
                AsyncMock(return_value=draft_row),
            ) as create_mock,
        ):
            result = await create_draft_mcp_server(prisma, payload, touched_by="admin")

        assert payload.approval_status == MCPApprovalStatus.draft
        del_mock.assert_awaited_once_with(prisma, "draft-new")
        create_mock.assert_awaited_once_with(prisma, payload, "admin")
        assert result is draft_row

    @pytest.mark.asyncio
    async def test_generates_server_id_when_none(self):
        prisma = MagicMock()
        payload = NewMCPServerRequest(
            alias="NoId",
            url="https://example.com",
            transport=MCPTransport.http,
        )
        assert payload.server_id is None

        draft_row = _make_prisma_row()
        with (
            patch(
                "litellm.proxy._experimental.mcp_server.db._delete_draft_mcp_server",
                AsyncMock(),
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.db.create_mcp_server",
                AsyncMock(return_value=draft_row),
            ),
        ):
            await create_draft_mcp_server(prisma, payload, touched_by="admin")

        assert payload.server_id is not None


class TestGetDraftMcpServer:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_row(self):
        repo_instance, mock_table = _mock_repo(find_first_return=None)
        prisma = MagicMock()

        with patch(
            "litellm.proxy._experimental.mcp_server.db.MCPServerRepository",
            return_value=repo_instance,
        ):
            result = await get_draft_mcp_server(prisma, "missing", ttl_seconds=300)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_expired(self):
        old_time = datetime.now(timezone.utc) - timedelta(seconds=600)
        row = _make_prisma_row(created_at=old_time)
        repo_instance, _ = _mock_repo(find_first_return=row)
        prisma = MagicMock()

        with patch(
            "litellm.proxy._experimental.mcp_server.db.MCPServerRepository",
            return_value=repo_instance,
        ):
            result = await get_draft_mcp_server(prisma, "draft-1", ttl_seconds=300)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_table_when_valid(self):
        recent_time = datetime.now(timezone.utc) - timedelta(seconds=10)
        row = _make_prisma_row(created_at=recent_time)
        repo_instance, _ = _mock_repo(find_first_return=row)
        prisma = MagicMock()

        with patch(
            "litellm.proxy._experimental.mcp_server.db.MCPServerRepository",
            return_value=repo_instance,
        ):
            result = await get_draft_mcp_server(prisma, "draft-1", ttl_seconds=300)

        assert result is not None
        assert isinstance(result, LiteLLM_MCPServerTable)
        assert result.server_id == "draft-1"

    @pytest.mark.asyncio
    async def test_handles_naive_created_at(self):
        naive_time = datetime.utcnow() - timedelta(seconds=10)
        row = _make_prisma_row(created_at=naive_time)
        repo_instance, _ = _mock_repo(find_first_return=row)
        prisma = MagicMock()

        with patch(
            "litellm.proxy._experimental.mcp_server.db.MCPServerRepository",
            return_value=repo_instance,
        ):
            result = await get_draft_mcp_server(prisma, "draft-1", ttl_seconds=300)

        assert result is not None


class TestDeleteExpiredDraftMcpServers:
    @pytest.mark.asyncio
    async def test_deletes_old_drafts(self):
        repo_instance, mock_table = _mock_repo(delete_many_return=3)
        prisma = MagicMock()

        with patch(
            "litellm.proxy._experimental.mcp_server.db.MCPServerRepository",
            return_value=repo_instance,
        ):
            count = await delete_expired_draft_mcp_servers(prisma, ttl_seconds=300)

        assert count == 3
        mock_table.delete_many.assert_awaited_once()
        where = mock_table.delete_many.call_args.kwargs["where"]
        assert where["approval_status"] == MCPApprovalStatus.draft
        assert "lt" in where["created_at"]

    @pytest.mark.asyncio
    async def test_returns_zero_when_none_expired(self):
        repo_instance, _ = _mock_repo(delete_many_return=0)
        prisma = MagicMock()

        with patch(
            "litellm.proxy._experimental.mcp_server.db.MCPServerRepository",
            return_value=repo_instance,
        ):
            count = await delete_expired_draft_mcp_servers(prisma, ttl_seconds=300)

        assert count == 0


class TestDeleteDraftMcpServer:
    @pytest.mark.asyncio
    async def test_deletes_by_server_id_and_status(self):
        repo_instance, mock_table = _mock_repo(delete_many_return=1)
        prisma = MagicMock()

        with patch(
            "litellm.proxy._experimental.mcp_server.db.MCPServerRepository",
            return_value=repo_instance,
        ):
            await _delete_draft_mcp_server(prisma, "target-id")

        mock_table.delete_many.assert_awaited_once()
        where = mock_table.delete_many.call_args.kwargs["where"]
        assert where["server_id"] == "target-id"
        assert where["approval_status"] == MCPApprovalStatus.draft

    @pytest.mark.asyncio
    async def test_propagates_delete_errors(self):
        repo_instance, mock_table = _mock_repo()
        mock_table.delete_many = AsyncMock(side_effect=Exception("db error"))
        prisma = MagicMock()

        with (
            patch(
                "litellm.proxy._experimental.mcp_server.db.MCPServerRepository",
                return_value=repo_instance,
            ),
            pytest.raises(Exception, match="db error"),
        ):
            await _delete_draft_mcp_server(prisma, "target-id")
