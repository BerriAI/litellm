"""Unit tests for litellm.proxy.shunt_endpoints.endpoints's capability-token auth dependency."""

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.guardrails.shunt_capability_token import mint_shunt_capability_token
from litellm.proxy.shunt_endpoints.endpoints import _caller_from_capability_token


@pytest.fixture(autouse=True)
def _salt_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-1234-test-salt-key")


class TestMissingOrMalformedHeader:
    @pytest.mark.asyncio
    async def test_no_header_is_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            await _caller_from_capability_token(authorization=None)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_non_bearer_header_is_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            await _caller_from_capability_token(authorization="Basic abc123")
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_malformed_token_is_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            await _caller_from_capability_token(authorization="Bearer not-a-real-token")
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_expired_token_is_rejected(self, monkeypatch):
        token = mint_shunt_capability_token(key_hash="deadbeef", master_key=None, now=1_000_000)
        # 121s after mint, one second past the 120s TTL.
        import litellm.proxy.guardrails.shunt_capability_token as token_mod

        monkeypatch.setattr(token_mod.time, "time", lambda: 1_000_121)
        with pytest.raises(HTTPException) as exc_info:
            await _caller_from_capability_token(authorization=f"Bearer {token}")
        assert exc_info.value.status_code == 401


class TestKeyHashGrant:
    @pytest.mark.asyncio
    async def test_resolves_the_key_object_for_the_grants_hash(self, monkeypatch):
        token = mint_shunt_capability_token(key_hash="deadbeef", master_key=None)
        resolved = UserAPIKeyAuth(api_key="deadbeef", team_id="team-1")

        async def _fake_get_key_object(**kwargs):
            assert kwargs["hashed_token"] == "deadbeef"
            return resolved

        import litellm.proxy.auth.auth_checks as auth_checks

        monkeypatch.setattr(auth_checks, "get_key_object", _fake_get_key_object)
        result = await _caller_from_capability_token(authorization=f"Bearer {token}")
        assert result is resolved

    @pytest.mark.asyncio
    async def test_a_lookup_failure_is_rejected_not_propagated(self, monkeypatch):
        token = mint_shunt_capability_token(key_hash="deadbeef", master_key=None)

        async def _raising_get_key_object(**kwargs):
            raise Exception("key not found")

        import litellm.proxy.auth.auth_checks as auth_checks

        monkeypatch.setattr(auth_checks, "get_key_object", _raising_get_key_object)
        with pytest.raises(HTTPException) as exc_info:
            await _caller_from_capability_token(authorization=f"Bearer {token}")
        assert exc_info.value.status_code == 401


class TestMasterKeyGrant:
    @pytest.mark.asyncio
    async def test_matching_master_key_resolves_as_proxy_admin(self, monkeypatch):
        monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-the-real-master-key")
        token = mint_shunt_capability_token(key_hash=None, master_key="sk-the-real-master-key")
        result = await _caller_from_capability_token(authorization=f"Bearer {token}")
        assert result.user_role == LitellmUserRoles.PROXY_ADMIN

    @pytest.mark.asyncio
    async def test_master_key_mismatch_is_rejected(self, monkeypatch):
        monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-the-current-master-key")
        token = mint_shunt_capability_token(key_hash=None, master_key="sk-a-stale-master-key")
        with pytest.raises(HTTPException) as exc_info:
            await _caller_from_capability_token(authorization=f"Bearer {token}")
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_no_configured_master_key_rejects_a_master_key_grant(self, monkeypatch):
        monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
        token = mint_shunt_capability_token(key_hash=None, master_key="sk-anything")
        with pytest.raises(HTTPException) as exc_info:
            await _caller_from_capability_token(authorization=f"Bearer {token}")
        assert exc_info.value.status_code == 401
