"""Unit tests for litellm.proxy.shunt_endpoints.endpoints."""

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.guardrails.shunt_capability_token import mint_shunt_capability_token
from litellm.proxy.shunt_endpoints.endpoints import _caller_from_capability_token, _worker_text


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

    # Regression: a resolved master-key caller carried the raw master key as its own api_key,
    # which _worker_text later places in the outbound request's metadata["user_api_key"] --
    # reachable by any raw-metadata logging callback. Normal master-key auth substitutes a
    # stable alias there specifically to keep the real key out of that sink; this must match.
    @pytest.mark.asyncio
    async def test_resolved_caller_never_carries_the_raw_master_key(self, monkeypatch):
        from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS

        monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-the-real-master-key")
        token = mint_shunt_capability_token(key_hash=None, master_key="sk-the-real-master-key")
        result = await _caller_from_capability_token(authorization=f"Bearer {token}")
        assert result.api_key == LITELLM_PROXY_MASTER_KEY_ALIAS
        assert result.api_key != "sk-the-real-master-key"

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


class _FakeRouter:
    def __init__(self, response_text: str):
        self._response_text = response_text

    async def acompletion(self, **kwargs):
        from litellm.types.utils import Choices, Message, ModelResponse

        return ModelResponse(choices=[Choices(index=0, message=Message(role="assistant", content=self._response_text))])


class _FakeProxyLogging:
    def __init__(self, *, blocks: bool):
        self._blocks = blocks
        self.calls = []

    async def pre_call_hook(self, *, user_api_key_dict, data, call_type):
        self.calls.append((user_api_key_dict, data, call_type))
        if self._blocks:
            raise HTTPException(status_code=429, detail="rate limited")
        return data


# Regression: the worker call went straight to llm_router.acompletion, skipping every
# registered rate-limit/budget callback (they run as async_pre_call_hook, which only
# proxy_logging_obj.pre_call_hook walks). A caller already over budget or rate-limited could
# keep spending through this endpoint indefinitely.
class TestWorkerTextEnforcesRateLimitsAndBudget:
    @pytest.mark.asyncio
    async def test_calls_pre_call_hook_before_the_worker_model(self, monkeypatch):
        fake_logging = _FakeProxyLogging(blocks=False)
        monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", fake_logging)
        holder = UserAPIKeyAuth(api_key="fakehash1234567890")
        text = await _worker_text(
            _FakeRouter("the worker's answer"),
            model="claude-haiku-4-5",
            system_prompt="be precise",
            message="what does this do",
            user_api_key_dict=holder,
            label="bulk_read",
        )
        assert text == "the worker's answer"
        assert len(fake_logging.calls) == 1
        called_key, _, called_type = fake_logging.calls[0]
        assert called_key is holder
        assert called_type == "acompletion"

    @pytest.mark.asyncio
    async def test_a_blocked_pre_call_hook_prevents_the_worker_call(self, monkeypatch):
        fake_logging = _FakeProxyLogging(blocks=True)
        monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", fake_logging)
        holder = UserAPIKeyAuth(api_key="fakehash1234567890")
        with pytest.raises(HTTPException) as exc_info:
            await _worker_text(
                _FakeRouter("should never be reached"),
                model="claude-haiku-4-5",
                system_prompt="be precise",
                message="what does this do",
                user_api_key_dict=holder,
                label="bulk_read",
            )
        assert exc_info.value.status_code == 429
