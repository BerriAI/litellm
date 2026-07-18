from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.caching.caching import DualCache
from litellm.proxy._experimental.mcp_server import delegation_db


def _mock_prisma_returning(rows):
    """A prisma client whose delegation table.find_first pops from `rows`."""
    table = MagicMock()
    table.find_first = AsyncMock(side_effect=list(rows))
    client = MagicMock()
    client.db.litellm_useragentdelegationtable = table
    return client, table


def _row(user_id="alice", agent_id="agent-1", revoked_at=None):
    # revoked_at is a real attribute (not an auto-Mock) because production reads
    # row.revoked_at to decide active vs already-revoked.
    return MagicMock(
        revoked_at=revoked_at,
        model_dump=lambda: {
            "delegation_id": "d1",
            "user_id": user_id,
            "agent_id": agent_id,
            "granted_at": "2026-07-17T00:00:00Z",
            "granted_by": "admin",
            "revoked_at": revoked_at,
            "revoked_by": None,
        },
    )


class TestActiveDelegationCache:
    @pytest.mark.asyncio
    async def test_positive_result_served_from_cache_second_call(self):
        client, table = _mock_prisma_returning([_row()])
        with patch("litellm.proxy.proxy_server.user_api_key_cache", DualCache()):
            first = await delegation_db.get_active_user_agent_delegation(client, "alice", "agent-1")
            second = await delegation_db.get_active_user_agent_delegation(client, "alice", "agent-1")
        assert first is not None and second is not None
        assert second.user_id == "alice"
        table.find_first.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_negative_result_cached_so_repeated_denial_skips_db(self):
        client, table = _mock_prisma_returning([None])
        with patch("litellm.proxy.proxy_server.user_api_key_cache", DualCache()):
            first = await delegation_db.get_active_user_agent_delegation(client, "bob", "agent-1")
            second = await delegation_db.get_active_user_agent_delegation(client, "bob", "agent-1")
        assert first is None and second is None
        table.find_first.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_revoke_busts_cache_so_next_read_hits_db_and_denies(self):
        """Immediate revocation: after a cached positive, revoke must invalidate
        the entry so the next resolve fails closed rather than serving stale
        consent until TTL."""
        cache = DualCache()
        active_table = MagicMock()
        active_table.find_first = AsyncMock(side_effect=[_row(), None])
        active_table.find_unique = AsyncMock(return_value=_row())
        active_table.update = AsyncMock(
            return_value=MagicMock(
                model_dump=lambda: {
                    "delegation_id": "d1",
                    "user_id": "alice",
                    "agent_id": "agent-1",
                    "granted_at": "2026-07-17T00:00:00Z",
                    "granted_by": "admin",
                    "revoked_at": "2026-07-17T01:00:00Z",
                    "revoked_by": "admin",
                }
            )
        )
        client = MagicMock()
        client.db.litellm_useragentdelegationtable = active_table
        with patch("litellm.proxy.proxy_server.user_api_key_cache", cache):
            assert await delegation_db.get_active_user_agent_delegation(client, "alice", "agent-1") is not None
            await delegation_db.revoke_user_agent_delegation(client, "alice", "agent-1", revoked_by="admin")
            after = await delegation_db.get_active_user_agent_delegation(client, "alice", "agent-1")
        assert after is None
        assert active_table.find_first.await_count == 2

    @pytest.mark.asyncio
    async def test_grant_busts_negative_cache_so_new_consent_is_immediate(self):
        cache = DualCache()
        table = MagicMock()
        table.find_first = AsyncMock(side_effect=[None, _row()])
        table.upsert = AsyncMock(return_value=_row())
        client = MagicMock()
        client.db.litellm_useragentdelegationtable = table
        with patch("litellm.proxy.proxy_server.user_api_key_cache", cache):
            assert await delegation_db.get_active_user_agent_delegation(client, "alice", "agent-1") is None
            await delegation_db.grant_user_agent_delegation(client, "alice", "agent-1", granted_by="admin")
            after = await delegation_db.get_active_user_agent_delegation(client, "alice", "agent-1")
        assert after is not None


class TestListDelegationScoping:
    """A non-admin key with no user association must not fall through to the
    unscoped 'list all' view (which would leak every user's consent records)."""

    @pytest.mark.asyncio
    async def test_list_all_with_none_user_id_returns_all_rows(self):
        """Pins the footgun the endpoint guard defends against: the store treats
        user_id=None as no filter (admin-only view)."""
        table = MagicMock()
        table.find_many = AsyncMock(return_value=[_row("alice"), _row("bob")])
        client = MagicMock()
        client.db.litellm_useragentdelegationtable = table
        result = await delegation_db.list_user_agent_delegations(client, user_id=None)
        assert len(result) == 2
        assert table.find_many.await_args.kwargs["where"] == {}


class TestDelegationRequestValidation:
    def test_grant_request_rejects_both_targets(self):
        import pytest
        from pydantic import ValidationError

        from litellm.types.mcp_server.user_agent_delegation import NewUserAgentDelegationRequest

        with pytest.raises(ValidationError):
            NewUserAgentDelegationRequest(user_id="u1", user_email="u@x.com", agent_id="a1")

    def test_grant_request_rejects_neither_target(self):
        import pytest
        from pydantic import ValidationError

        from litellm.types.mcp_server.user_agent_delegation import NewUserAgentDelegationRequest

        with pytest.raises(ValidationError):
            NewUserAgentDelegationRequest(agent_id="a1")

    def test_grant_request_rejects_blank_agent(self):
        import pytest
        from pydantic import ValidationError

        from litellm.types.mcp_server.user_agent_delegation import NewUserAgentDelegationRequest

        with pytest.raises(ValidationError):
            NewUserAgentDelegationRequest(user_id="u1", agent_id="   ")

    def test_grant_request_accepts_single_target(self):
        from litellm.types.mcp_server.user_agent_delegation import NewUserAgentDelegationRequest

        assert NewUserAgentDelegationRequest(user_email="u@x.com", agent_id="a1").agent_id == "a1"


class TestRevokeAuditIntegrity:
    @pytest.mark.asyncio
    async def test_revoke_already_revoked_returns_none_and_does_not_restamp(self):
        """Re-revoking an already-revoked consent must be a no-op: return None
        (so the endpoint 404s) and never overwrite the original revoked_at/by."""
        already_revoked = MagicMock(revoked_at="2026-07-17T00:00:00Z")
        table = MagicMock()
        table.find_unique = AsyncMock(return_value=already_revoked)
        table.update = AsyncMock()
        client = MagicMock()
        client.db.litellm_useragentdelegationtable = table

        with patch("litellm.proxy.proxy_server.user_api_key_cache", DualCache()):
            result = await delegation_db.revoke_user_agent_delegation(
                client, "alice", "agent-1", revoked_by="admin2"
            )

        assert result is None
        table.update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_returns_none(self):
        table = MagicMock()
        table.find_unique = AsyncMock(return_value=None)
        table.update = AsyncMock()
        client = MagicMock()
        client.db.litellm_useragentdelegationtable = table

        with patch("litellm.proxy.proxy_server.user_api_key_cache", DualCache()):
            result = await delegation_db.revoke_user_agent_delegation(client, "ghost", "agent-1", revoked_by="a")

        assert result is None
        table.update.assert_not_awaited()
